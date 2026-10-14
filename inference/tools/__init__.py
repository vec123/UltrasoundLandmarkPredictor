"""Command-line entry points for the inference package.

Each module here is a thin argparse wrapper over the library in
:mod:`inference`; all real work lives in the library so it stays importable and
testable. Run either as a module from the repo root::

    python -m inference.tools.probe_nrrd --folder ...
    python -m inference.tools.preprocess_geom --folder ... --out ...

or directly as a file (a small bootstrap puts the repo root on ``sys.path``).
"""
