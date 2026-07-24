import os
import json
import logging
import numpy as np
import pandas as pd
from tqdm import tqdm
import SimpleITK as sitk

from dataset.utility.rotation import *
from dataset.utility.standardization import gtpp

def perform_preprocessing(dataframe, params, logger=None):
    """
    Processes ultrasound data by loading images and applying transformations to produce 
    pre-processed, rotated, and translated ultrasound images along with associated landmark points.

    Args:
        dataframe (pd.DataFrame): DataFrame containing data paths and metadata.
        params: Parameter object containing processing settings and paths.

    Returns:
        None. Saves processed images and landmarks in specified `out_dir`.
    """
    if logger is None:
        logger = logging.getLogger('my_project_logger') 
    # Extract all filenames from the dataframe for processing
    filenames = list(dataframe['fcaso'].values)
    filedirs = list(dataframe['fscan'].values)

    # Load ground truth data for standard planes from JSON file
    dicto = json.loads(get_file_list('doc/info//gt.txt')[0])
    filedirs_ = []
    for filedir in filedirs:
        parts = filedir.strip('/').split('/')
        filedirs_.append('/'.join(parts[3:]))
    ct = 0
    ct_not_found = 0
    dataframe.loc[:, 'csvfound'] = True
    # txt_path = os.path.join(params.preprocessing.params.gtpp.file_paths.list_files.test)
    # with open(txt_path, "w", encoding="utf-8") as f:
    #     f.write("\n".join(map(str, filedirs_)))
    print(params.preprocessing)
    if params.preprocessing.gtpp:
        gtpp(dataframe, params.preprocessing.params.gtpp)
    else: # Usual routine
        # Iterate over each file for processing
        for i, filename in tqdm(enumerate(filenames), total=len(filenames)):
            # Prepare file paths and identifiers from dataframe row
            image_path = os.path.join(dataframe.loc[i, 'fscan'])
            name = dataframe.loc[i, 'osid']
            
            save_im_path = os.path.join(params.ds.sys, params.preprocessing.save_dir, dataframe.loc[i, 'mscan'])
            save_csv_path = os.path.join(params.ds.sys, params.preprocessing.save_dir, dataframe.loc[i, 'mcsv'])
            save_lmk_path = os.path.join(params.ds.sys, params.preprocessing.save_dir, dataframe.loc[i, 'mlmk'])

        # Skip processing if image already exists
            if os.path.exists(save_im_path):
                
                # logger.info('Image seems to be processed already!!!')
                # logger.info(save_im_path)
                continue

            # Load ultrasound image volume and header metadata
            V, header = extract_image(image_path)
            V = np.array(V)

            lmk_path = dataframe.loc[i, '_fcsv']

            # Attempt to load landmarks CSV file
            try:
                lmk = pd.read_csv(lmk_path)
            except Exception as e:
                logger.error(str(e))
                ct_not_found += 1
                logger.warning(f"Could not load landmarks from {lmk_path} for {name}! Skipping!! Count: {ct_not_found}")
                dataframe.loc[i, 'csvfound'] = False
                continue

            # Verify landmarks count
            if len(lmk) != 19:
                logger.warning('Something wrong!! Landmark count mismatch: %s', filename)

            # Convert landmark points to matrix format in millimeters
            L_in_mm = get_matrix_of_lmks(lmk)

            # Log header info for debugging
            # logger.info(header)

            # Determine pixel spacing information from header; fallback if keys missing
            try:
                p = header.get('spacings')[:3]
            except Exception as e:
                p = np.array([header['space directions'][0, 0],
                            header['space directions'][1, 1],
                            header['space directions'][2, 2]])

            img_size = np.array(V.shape)

            # --------- #
            # STEP 1: Apply low-pass filter to the 3D ultrasound image volume
            # --------- #
            if params.preprocessing.filter:
                V = filter_3d_image(V, 
                                    params.preprocessing.params.filter.filter_size, 
                                    params.preprocessing.params.filter.mode)

            # ----------  #
            # STEP 2: Interpolate image to desired spacing and size
            # ----------  #

            if params.preprocessing.bspline:
                image = sitk.GetImageFromArray(V)
                image.SetSpacing(p)  # Set original spacing

                new_spacing = params.preprocessing.params.bspline.spacing
                new_size = params.preprocessing.params.bspline.size

                # Configure the resampler filter for image interpolation
                resampler = sitk.ResampleImageFilter()
                resampler.SetOutputSpacing(new_spacing)
                resampler.SetSize(new_size)
                resampler.SetInterpolator(sitk.sitkBSpline)
                resampler.SetOutputDirection(image.GetDirection())
                resampler.SetOutputOrigin(image.GetOrigin())

                resampled_image = resampler.Execute(image)
                V = sitk.GetArrayFromImage(resampled_image)
                header['sizes'] = params.preprocessing.params.bspline.size

                p = new_spacing
                # Determine pixel spacing information from header; fallback if keys missing
                try:
                    header['spacings'] = p
                except Exception:
                    header['space directions'] = np.array([ [p[0], 0, 0],
                                                            [0, p[1], 0],
                                                            [0, 0, p[2]]
                                                            ]
                                                        )
                    
            # Convert landmarks from mm to pixel units
            L_in_pix = L_in_mm / p

            # ------ -------------- #
            # STEP 5: Apply affine transformation to image and landmarks
            # ------ -------------- #
            if params.preprocessing.affine:
                # Retrieve ground truth plane data from dictionary for current image
                plane = dicto.get(name)
                if plane is None:
                    logger.warning(f"Ground truth for {name} not found!")
                    # ct_not_found += 1
                    dataframe.loc[i, 'rot_found'] = False
                    pass
                ct += 1
                # Assumption: Centers are in RAS coordinate system
                center_gt_in_pix = np.array([-1 * plane["center"][0], -1 * plane["center"][1], plane["center"][2]]) / p
                img_size = np.array(V.shape).astype(np.float32)
                # Compute translation vector for affine transform
                translation_vector = (center_gt_in_pix - img_size / 2.0) * (2.0 / img_size).astype(np.float32)
                # Compute rotation matrix from plane parameters
                rotation_3x3 = affine3Dmatrix(plane)

                # Construct full 3x4 affine transformation matrix
                transform_matrix = np.zeros((3, 4))
                transform_matrix[:3, :3] = rotation_3x3
                transform_matrix[:3, 3] = translation_vector

                # Apply affine transformation to the image volume
                Vhat = grid_transform_3d(V.astype(np.float32), transform_matrix.astype(np.float32))
                
                if params.preprocessing.swap:
                    Vhat = np.transpose(Vhat, (2, 1, 0))
                    L_in_pix = swap_xz_coordinates(L_in_pix)

                L_in_pix_norm = (L_in_pix - img_size / 2.0) * (2.0 / img_size).astype(np.float32)
                inv_translation_vector = -1 * translation_vector

                Lhat_in_pix_norm = affine_transform(L_in_pix_norm, np.linalg.inv(rotation_3x3), inv_translation_vector)
                Lhat_in_pix = (Lhat_in_pix_norm / (2.0 / img_size).astype(np.float32)) + img_size / 2.0
                V = Vhat.copy()
                L_in_pix = Lhat_in_pix
            # Save processed image and transformed landmarks
            save_3d_image(V, header, img_size, save_im_path)
            save_transformed_landmarks(L_in_pix * p, lmk, save_csv_path, save_lmk_path)

        logger.info("Number of images with missing csvs: %d", ct_not_found)
        dataframe.to_csv('tmp.csv')
