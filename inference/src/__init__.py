"""Library modules for the inference package (importable, testable core).

The CLI entry points in :mod:`inference.tools` are thin wrappers over these:

    nrrd_io.py   pure .nrrd / gt.txt helpers (numpy + pynrrd)
    geometry.py  Stage-1 transforms (smooth, resample_bspline, ...)
    probe.py     Stage-0 logic (VolumeProbe, probe_folder, render_table)
    model.py     Stage-2 prediction (checkpoint load, forward pass, peak extraction)

``nrrd_io``, ``geometry`` and ``probe`` are torch-free and repo-import-free so
they run in the imaging env (medim_env). ``model`` requires torch and imports
``net.model.backbone.ResUNet3D`` - it belongs in the model env (torch_gpu_env).
"""

__all__ = ["nrrd_io", "geometry", "probe", "model"]
