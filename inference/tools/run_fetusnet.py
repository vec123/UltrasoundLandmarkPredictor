"""Run the unmodified ``fetusnet.py`` on Windows.

``fetusnet.py`` has no ``if __name__ == "__main__":`` guard and its loaders
start worker processes (``get_test_dl`` hard-codes ``num_workers=1``). On
Windows a worker re-runs the whole script and dies ("DataLoader worker exited
unexpectedly"); on Linux (his machine) workers are forked and it works.

This launcher only forces DataLoader workers to 0, so data are loaded in the
main process (same results), then runs ``fetusnet.py`` with the arguments
given - exactly as ``python fetusnet.py <args>`` would.

    python -m inference.tools.run_fetusnet --mode=test --checkpoint=... <other fetusnet.py args>
"""
import os
import runpy
import sys
from pathlib import Path

import torchio as tio

REPO = Path(__file__).resolve().parents[2]


class _InProcessSubjectsLoader(tio.SubjectsLoader):
    def __init__(self, *args, **kwargs):
        kwargs["num_workers"] = 0
        super().__init__(*args, **kwargs)


if __name__ == "__main__":
    tio.SubjectsLoader = _InProcessSubjectsLoader
    os.chdir(REPO)                      # fetusnet.py uses repo-relative paths
    sys.path.insert(0, str(REPO))
    sys.argv = [str(REPO / "fetusnet.py")] + sys.argv[1:]
    runpy.run_path(str(REPO / "fetusnet.py"), run_name="__main__")
