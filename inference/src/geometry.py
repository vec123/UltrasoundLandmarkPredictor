"""Stage 1 - deterministic geometric preprocessing.

Runs only the two *annotation-free* steps of the real pipeline
(``dataset/preprocess.py``): a low-pass smoothing filter and a B-spline resample
to a fixed isotropic grid. Pose standardisation (the affine step, which needs
manual annotations) is deliberately NOT here - see ``INFERENCE_PLAN.md`` Stage 3.

Torch-free and repo-import-free on purpose: this module must run in the imaging
env (``medim_env``, no torch), so the two trivial deterministic steps are
reimplemented faithfully here rather than imported from the training repo (which
pulls in torch). Both are byte-faithful ports of the upstream behaviour, with the
source cited; read the ``resample_bspline`` warning before changing anything.
"""
from __future__ import annotations

from typing import Sequence, Tuple

import numpy as np
from scipy import ndimage
import SimpleITK as sitk

# Canonical target grid the network expects: 1 mm isotropic, 128^3.
DEFAULT_OUT_SPACING: Tuple[float, float, float] = (1.0, 1.0, 1.0)
DEFAULT_OUT_SIZE: Tuple[int, int, int] = (128, 128, 128)

# NRRD per-axis header fields not rewritten explicitly by build_output_header;
# each must have exactly one entry per axis.
_PER_AXIS_FIELDS = ("labels", "units", "centers", "centerings", "axis mins", "axis maxs", "thicknesses")


def smooth(volume: np.ndarray, filter_size: int = 3, mode: str = "reflect") -> np.ndarray:
    """Uniform-mean low-pass filter (speckle reduction).

    Faithful, torch-free reimplementation of the pipeline's ``filter_3d_image``
    (``dataset/utility/rotation.py:75-80``): an ``N**3`` averaging kernel applied
    with ``scipy.ndimage.convolve``. NOTE: convolution runs in the input's native
    dtype (uint8 in -> uint8 out), exactly as the repo does; do **not** cast to
    float beforehand or the rounding is changed.
    """
    kernel = np.ones((filter_size,) * volume.ndim) / (filter_size ** volume.ndim)
    return ndimage.convolve(volume, kernel, mode=mode)


def resample_bspline(
    volume: np.ndarray,
    spacing: Sequence[float],
    out_spacing: Sequence[float] = DEFAULT_OUT_SPACING,
    out_size: Sequence[int] = DEFAULT_OUT_SIZE,
) -> np.ndarray:
    """B-spline resample to a fixed grid. Verbatim port of ``preprocess.py:111-139``.

    .. warning::
       **Corner-anchored fixed box - NOT scale-to-fit.** The output is an
       ``out_size`` box at ``out_spacing`` mm, sampled from the array's
       ``(0, 0, 0)`` corner. A field of view larger than
       ``out_size * out_spacing`` mm is **cropped**; a smaller one is silently
       zero-filled. There is no warning either way. This is intentional: it keeps
       mm coordinates numerically equal to voxel indices downstream
       (``preprocess.py:142``). Do not "fix" it to centre-crop, or Stage-1 output
       stops matching the training distribution.

    The output pixel type follows the input (uint8 in -> uint8 out, with B-spline
    overshoot clamped), matching the upstream pipeline.
    """
    image = sitk.GetImageFromArray(volume)
    image.SetSpacing([float(s) for s in spacing])

    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing([float(s) for s in out_spacing])
    resampler.SetSize([int(s) for s in out_size])
    resampler.SetInterpolator(sitk.sitkBSpline)
    resampler.SetOutputDirection(image.GetDirection())
    resampler.SetOutputOrigin(image.GetOrigin())

    return sitk.GetArrayFromImage(resampler.Execute(image))


def centering_pad(
    shape: Sequence[int],
    spacing: Sequence[float],
    out_spacing: Sequence[float] = DEFAULT_OUT_SPACING,
    out_size: Sequence[int] = DEFAULT_OUT_SIZE,
) -> Tuple[Tuple[int, ...], np.ndarray]:
    """Voxels to zero-pad *before* each axis so the volume's FOV sits in the
    middle of the ``out_size * out_spacing`` box instead of at its corner.

    Translation only - no rotation, no flip. Axes whose FOV already exceeds the
    box get 0 (they are cropped either way, as in :func:`resample_bspline`).

    Returns ``(pad_voxels, shift_mm)``: the pad in input voxels and the shift
    that pad actually applies in mm. Add ``shift_mm`` to raw-frame coordinates to
    get grid coordinates, or subtract it from grid coordinates to get back to
    the raw frame.
    """
    spacing = np.asarray(spacing, dtype=float)
    box_mm = np.asarray(out_size, dtype=float) * np.asarray(out_spacing, dtype=float)
    fov_mm = np.asarray(shape, dtype=float) * spacing
    pad = np.round(np.clip(box_mm - fov_mm, 0, None) / 2.0 / spacing).astype(int)
    return tuple(int(p) for p in pad), pad * spacing


def preprocess_geometry(
    volume: np.ndarray,
    spacing: Sequence[float],
    *,
    do_filter: bool = True,
    do_bspline: bool = True,
    filter_size: int = 3,
    filter_mode: str = "reflect",
    out_spacing: Sequence[float] = DEFAULT_OUT_SPACING,
    out_size: Sequence[int] = DEFAULT_OUT_SIZE,
    pad_before: Sequence[int] = None,
) -> np.ndarray:
    """Full Stage-1 chain: optional smoothing, then optional B-spline resample.

    Order matches the pipeline (filter first, then resample). Either step can be
    switched off for debugging via ``do_filter`` / ``do_bspline``.

    ``pad_before`` (voxels per axis, e.g. from :func:`centering_pad`) zero-pads
    the volume between the two steps. Padding *after* the filter keeps the
    filter's edge behaviour identical to the unpadded pipeline.
    """
    out = volume
    if do_filter:
        out = smooth(out, filter_size=filter_size, mode=filter_mode)
    if pad_before is not None and any(pad_before):
        out = np.pad(out, [(int(p), 0) for p in pad_before])
    if do_bspline:
        out = resample_bspline(out, spacing, out_spacing=out_spacing, out_size=out_size)
    return out


def affine3Dmatrix(plane_dict: dict) -> np.ndarray:
    """Convert plane normals (sagittal, coronal, axial) to rotation matrix.

    Matches the training pipeline's ``dataset/utility/rotation.py:affine3Dmatrix()``.

    Args:
        plane_dict: Dictionary with 'sagittal', 'coronal', 'axial' plane normal vectors.

    Returns:
        3x3 rotation matrix.
    """
    normal_sagittal = np.array([plane_dict['sagittal'][0], plane_dict['sagittal'][1], -plane_dict['sagittal'][2]])
    normal_coronal = np.array([plane_dict['coronal'][0], plane_dict['coronal'][1], -plane_dict['coronal'][2]])
    normal_axial = np.array([plane_dict['axial'][0], plane_dict['axial'][1], -plane_dict['axial'][2]])

    V0 = np.array([[1, 0, 0], [0, 1, 0], [0, 0, -1]], dtype=float)
    V1 = np.array([normal_sagittal, normal_axial, normal_coronal], dtype=float)

    M = np.transpose(np.linalg.solve(V0, V1))
    return M


def affine_transform_scipy(
    volume: np.ndarray,
    rotation_matrix: np.ndarray,
    translation_vector: np.ndarray,
) -> np.ndarray:
    """Apply 3D affine transformation using scipy (no torch needed).

    Implements pose standardization by applying rotation and translation.
    Uses scipy.ndimage.affine_transform which expects the *inverse* transform.

    Args:
        volume: 3D uint8 array to transform
        rotation_matrix: 3x3 rotation matrix (RAS convention)
        translation_vector: 3D translation vector in voxel units

    Returns:
        Transformed uint8 volume (same size as input, zero-filled padding)
    """
    # scipy expects the inverse transformation (backward mapping)
    affine_matrix = np.eye(4)
    affine_matrix[:3, :3] = rotation_matrix
    affine_matrix[:3, 3] = translation_vector

    try:
        affine_inv = np.linalg.inv(affine_matrix)
    except np.linalg.LinAlgError:
        raise ValueError("Affine matrix is singular and cannot be inverted")

    matrix = affine_inv[:3, :3]
    offset = affine_inv[:3, 3]

    # Apply affine transform with linear interpolation (order=1)
    transformed = ndimage.affine_transform(
        volume.astype(np.float32),
        matrix,
        offset=offset,
        order=1,
        mode="constant",
        cval=0.0,
    )

    # Clamp to original dtype range and convert back
    return np.clip(transformed, 0, 255).astype(np.uint8)


def build_output_header(
    header: dict,
    out_shape: Sequence[int],
    out_spacing: Sequence[float],
) -> dict:
    """Copy the input header and update geometry fields for the resampled volume.

    Sets ``sizes`` and ``spacings`` (as the repo does at ``preprocess.py:128,133``)
    and, additionally, refreshes ``space directions`` / ``kinds`` when present so a
    stale transform can't leak into the output and confuse Slicer. Pixel ``type``,
    ``dimension`` and ``endian`` are left for ``nrrd.write`` to derive from the
    array, keeping header and data consistent.
    """
    out = dict(header)
    spacing = [float(s) for s in out_spacing]

    out["sizes"] = np.asarray(list(out_shape))
    out["spacings"] = np.asarray(spacing)
    if "space directions" in out:
        out["space directions"] = np.diag(spacing)
    if "kinds" in out:
        out["kinds"] = ["space"] * len(out_shape)
    # A 4D input (x, y, z, t) is collapsed to 3D by load_volume, but its other
    # per-axis fields keep 4 entries (e.g. labels "x" "y" "z" "t") - an invalid
    # NRRD that Slicer/teem refuses to load. Trim them to the output axes.
    ndim = len(out_shape)
    for key in _PER_AXIS_FIELDS:
        if key in out and len(out[key]) != ndim:
            if len(out[key]) > ndim:
                out[key] = list(out[key])[:ndim]
            else:
                del out[key]
    return out
