
src.  
These modules are imported and called in tools.

Geometry:
    Processing Modules:

    -B-spline resampling for uniform resolution
    -kernel smoothing for denoising
    -preprocessing (smooth + bspline)
    -affine transformation with scipy instead of torch

Nrrd_io:
     Input Output Nrrd:
     -Load volume
     -Get spacing
     -Get size (field ov view)

Probe:
    uses nrrd_io to test the nrrd:
    - get widths 
    (i.e. is it within the 128x128x128 box that the algoritm enforces)
    - Is it mentioned in gt.txt


