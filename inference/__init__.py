"""FetusNet inference package.

Standalone, annotation-free tooling for turning raw fetal-ultrasound ``.nrrd``
volumes into landmark predictions. Built around the plan in ``INFERENCE_PLAN.md``.

Layer overview (dependencies point downward only):

    tools/            thin CLI entry points (argparse + presentation)
      probe_nrrd.py       Stage 0 - inspect volumes (no torch, no checkpoint)
      preprocess_geom.py  Stage 1 - deterministic geometric preprocessing
      infer_landmarks.py  Stage 0+1+2 - end to end: raw .nrrd -> landmarks
    src/              importable, testable library core
      probe.py            Stage-0 domain logic (VolumeProbe, probe_folder, ...)
      geometry.py         Stage-1 transforms (smooth, resample_bspline, ...)
      nrrd_io.py          pure .nrrd / gt.txt helpers (numpy + pynrrd only)
      model.py            Stage-2 prediction (checkpoint load, forward pass, peaks)

Stage 0 and Stage 1 are deliberately **torch-free and repo-import-free** so they
run in the imaging env (``medim_env``: numpy, scipy, SimpleITK, pynrrd - no
torch). The two deterministic preprocessing steps are faithfully reimplemented in
:mod:`inference.src.geometry` rather than imported from the training repo, which
pulls in torch. Stage 2 (:mod:`inference.src.model`) is the deep-learning stage -
it requires torch and belongs in the model env (``torch_gpu_env``).
``infer_landmarks.py`` ties all three stages together for convenience; pass
``input.already_preprocessed: true`` in its config to skip Stage 1 when running
it inside a torch-only environment that lacks the imaging libs.
"""

__all__ = ["src", "tools"]
