"""Set up a run of the ORIGINAL ``fetusnet.py --mode=test`` on processed volumes.

His test mode does not take .nrrd paths directly: it reads a dataframe CSV
(``<ds.sys><ds.root><ds.dataframe>.csv``) listing processed volumes (128^3,
1 mm - what ``--mode=preprocess`` writes to ``DATA/volumes/``) and a processed
landmark CSV for every scan (``dataset/MyDataset.py`` raises if one is missing).
This tool writes those inputs for any processed .nrrd files - yours or his -
and prints his exact command. Nothing in his code is changed.

Scans without a real landmark CSV get a placeholder (all 19 points at 0), so
their rows in his ``test_scores.csv`` are meaningless; the predictions in
``<out>/ckpt/eval/<scan>.fcsv`` are unaffected.

Example
-------
    python -m inference.tools.prepare_fetusnet_test inference/results/original/1 inference/results/original/2 \
        --landmarks inference/results/original/1/targets
"""
from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

import nrrd
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
LMKS = ["prn", "enL", "enR", "pg", "n", "sn", "sl", "li", "ls", "acL",
        "acR", "cphL", "cphR", "chR", "chL", "aR", "aL", "exL", "exR"]
TORCH_ENV_PYTHON = r"C:\Users\vic-b\Documents\Victors\Projects\environments\torch_gpu_env\Scripts\python.exe"


def collect(paths):
    files = []
    for p in map(Path, paths):
        files += sorted(p.glob("*.nrrd")) if p.is_dir() else [p]
    return files


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("inputs", nargs="+", help="Processed .nrrd files and/or folders of them.")
    ap.add_argument("--landmarks", type=Path, default=None,
                    help="Folder of processed landmark CSVs named <scan>.csv (optional, enables real scores).")
    ap.add_argument("--out", type=Path, default=REPO / "inference" / "results" / "fetusnet_test")
    ap.add_argument("--checkpoint", type=Path, default=REPO / "best_model" / "best.pt")
    args = ap.parse_args(argv)

    data_dir, lmk_dir, ckpt_dir = args.out / "data", args.out / "data" / "landmarks", args.out / "ckpt"
    for d in (lmk_dir, ckpt_dir):
        d.mkdir(parents=True, exist_ok=True)

    rows, n_real = [], 0
    for f in collect(args.inputs):
        shape = tuple(int(s) for s in nrrd.read_header(str(f))["sizes"][:3])
        if shape != (128, 128, 128):
            print(f"skip {f.name}: shape {shape} is not a processed 128^3 volume")
            continue
        scan = f.stem[: -len("_preprocessed")] if f.stem.endswith("_preprocessed") else f.stem
        real = args.landmarks / f"{scan}.csv" if args.landmarks else None
        if real is not None and real.exists():
            mcsv, n_real = real.resolve(), n_real + 1
        else:
            mcsv = lmk_dir / f"{scan}.csv"
            pd.DataFrame({"x": 0.0, "y": 0.0, "z": 0.0, "label": LMKS}).to_csv(mcsv)
        rows.append(dict(nsid=scan, npid=1, set=2, nonfrontal_after_rot=False, visibles=str(LMKS),
                         mscan=str(f.resolve()), mcsv=str(Path(mcsv).resolve())))
    if not rows:
        print("No processed volumes found.")
        return 1
    pd.DataFrame(rows).to_csv(data_dir / "dataframe.csv", index=False)

    ckpt = ckpt_dir / "best.pt"
    if not ckpt.exists():
        try:
            os.link(args.checkpoint, ckpt)      # no 400 MB copy
        except OSError:
            shutil.copy2(args.checkpoint, ckpt)

    print(f"{len(rows)} scan(s) listed in {data_dir / 'dataframe.csv'} ({n_real} with real landmarks)")
    print("\nRun his test mode from the repo root (PowerShell, torch_gpu_env). run_fetusnet = plain")
    print("`python fetusnet.py <args>`, only with DataLoader workers off so it works on Windows:\n")
    print('$env:MPLBACKEND = "Agg"')
    print(f'& "{TORCH_ENV_PYTHON}" -m inference.tools.run_fetusnet --mode=test --checkpoint="{ckpt_dir.as_posix()}" '
          f'--validation.use_model=best --ds.sys="{args.out.resolve().as_posix()}/" --ds.root="data/" '
          f'--ds.dataframe=dataframe --test_patients="[1]" --validation.save_outputs=False '
          f'--validation.save_targets=False --validation.show_figures=False --wandb.log=False')
    print(f"\nPredictions: {ckpt_dir / 'eval'}\\<scan>.fcsv   Scores: {ckpt_dir / 'test_scores.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
