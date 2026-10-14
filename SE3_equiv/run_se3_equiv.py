"""SE(3) equivariance of a landmark model: augment -> infer_landmarks -> compare.

For one volume on the model's input grid, every configured rigid transform ``g``:

1. writes the augmented volume ``gV``                -> augmented/<stem>_<g>.nrrd
2. runs infer_landmarks on it                       -> landmarks/<stem>_<g>.fcsv/.csv
3. writes where an equivariant model would put them -> expected/<stem>_<g>_expected.fcsv
4. scores ``|| f(gV) - g f(V) ||`` per landmark, ``f(V)`` = prediction on the
   untouched volume (no annotations needed)

and finally per_transform.csv, per_landmark.csv, summary_*.csv, transforms.json
and error_vs_angle.png. Load an augmented .nrrd with its two .fcsv in Slicer to
see a failure.

Example (repo root)
-------------------
    python -m SE3_equiv.run_se3_equiv SE3_equiv/config_se3_equiv.yaml
    python -m SE3_equiv.run_se3_equiv SE3_equiv/config_se3_equiv.yaml --center volume --output SE3_equiv/results/10142001_center_volume
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import nrrd
import numpy as np
import pandas as pd
import yaml

# --- bootstrap: allow running as a bare file, not just `python -m` -------
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from inference.src.model import left_right_permutation
from inference.src.nrrd_io import load_volume, read_spacing
from net.postprocess.save_fscv_csv import save_fscv_csv
from SE3_equiv.metrics import evaluate_transform, plot, summarize
from SE3_equiv.predictor import InferLandmarksPredictor
from SE3_equiv.se3 import build_transforms, resolve_center, transform_volume

logger = logging.getLogger("SE3_equiv")


def run(cfg: dict) -> int:
    volume_path = Path(cfg["input"]["volume"])
    out_dir = Path(cfg["output"]["folder"])
    tcfg = cfg.get("transforms", {})
    margin = float(cfg.get("evaluation", {}).get("fov_margin_vox", 3))
    order = int(tcfg.get("interpolation_order", 1))
    for sub in ("augmented", "landmarks", "expected"):
        (out_dir / sub).mkdir(parents=True, exist_ok=True)
    (out_dir / "config_used.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))

    predictor = InferLandmarksPredictor(cfg["infer_config"], cfg.get("infer_overrides"))
    names = predictor.landmark_names
    lr_perm = left_right_permutation(names)
    logger.info("model: %s", predictor.describe())

    volume, header = load_volume(volume_path)
    spacing = read_spacing(header)
    stem = volume_path.stem
    logger.info("volume: %s  %s %s  spacing %s", volume_path, "x".join(map(str, volume.shape)), volume.dtype, spacing)

    reference = predictor.predict(volume_path, out_dir / "landmarks")
    center = resolve_center(tcfg.get("center", "landmarks"), reference, volume.shape, spacing)
    transforms = build_transforms(tcfg)
    logger.info("reference f(V) -> landmarks/%s.fcsv; rotation centre %s mm; %d transforms",
                stem, center.tolist(), len(transforms))

    per_transform, per_landmark, manifest = [], [], []
    t_start = time.time()
    for k, g in enumerate(transforms, 1):
        augmented, exact = transform_volume(volume, spacing, g, center, order=order)
        aug_path = out_dir / "augmented" / "{}_{}.nrrd".format(stem, g.name)
        nrrd.write(str(aug_path), augmented, header=header)

        predicted = predictor.predict(aug_path, out_dir / "landmarks")
        expected = g.apply_points(reference, center)
        save_fscv_csv(out=str(out_dir / "expected" / "{}_{}_expected".format(stem, g.name)),
                      coords=expected, selected_lmks=names, spacing=1.0)

        err, in_fov, summary = evaluate_transform(reference, predicted, expected, g.rotation,
                                                  volume.shape, spacing, lr_perm, margin)
        rv = g.rotvec_deg
        per_transform.append(dict(
            name=g.name, family=g.family, angle_deg=g.angle_deg, shift_mm=g.shift_mm, exact=exact,
            rotvec_i_deg=rv[0], rotvec_j_deg=rv[1], rotvec_k_deg=rv[2],
            t_i_mm=g.translation[0], t_j_mm=g.translation[1], t_k_mm=g.translation[2], **summary))
        for i, lmk in enumerate(names):
            per_landmark.append(dict(
                name=g.name, family=g.family, angle_deg=g.angle_deg, landmark=lmk,
                pred_x=predicted[i, 0], pred_y=predicted[i, 1], pred_z=predicted[i, 2],
                exp_x=expected[i, 0], exp_y=expected[i, 1], exp_z=expected[i, 2],
                err_mm=err[i], in_fov=bool(in_fov[i])))
        manifest.append(dict(name=g.name, family=g.family, rotation=g.rotation.tolist(),
                             translation_mm=g.translation.tolist(), center_mm=center.tolist(), exact=exact))

        logger.info("[%3d/%d] %-26s angle %6.1f  shift %5.1f  %s  err mean %5.2f  median %5.2f  max %6.2f mm  "
                    "rot %5.1f deg  shape %4.2f mm  in-box %2d%s",
                    k, len(transforms), g.name, g.angle_deg, g.shift_mm, "exact " if exact else "interp",
                    summary["err_mean"], summary["err_median"], summary["err_max"], summary["rot_err_deg"],
                    summary["shape_rms_mm"], summary["n_in_fov"], "  L/R SWAPPED" if summary["lr_swapped"] else "")

    per_transform = pd.DataFrame(per_transform)
    per_landmark = pd.DataFrame(per_landmark)
    per_transform.to_csv(out_dir / "per_transform.csv", index=False)
    per_landmark.to_csv(out_dir / "per_landmark.csv", index=False)
    (out_dir / "transforms.json").write_text(json.dumps(manifest, indent=1))
    report(per_transform, per_landmark, out_dir)
    logger.info("done in %.0f s -> %s", time.time() - t_start, out_dir)
    return 0


def report(per_transform: pd.DataFrame, per_landmark: pd.DataFrame, out_dir: Path) -> None:
    """Summary tables + plot; rerun alone with --report-only."""
    tables = summarize(per_transform, per_landmark)
    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 30)
    for key, table in tables.items():
        table.to_csv(out_dir / "summary_{}.csv".format(key), index=False)
        print("\n== {} ==\n{}".format(key, table.round(2).to_string(index=False)))
    if plot(per_transform, out_dir / "error_vs_angle.png"):
        logger.info("plot -> %s", out_dir / "error_vs_angle.png")


def main(argv=None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    p = argparse.ArgumentParser(prog="run_se3_equiv", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("config", type=Path, help="SE(3) evaluation YAML.")
    p.add_argument("--center", default=None, help="Override transforms.center (landmarks | volume).")
    p.add_argument("--output", default=None, help="Override output.folder.")
    p.add_argument("--report-only", action="store_true",
                   help="Rebuild summaries/plot from the output folder's per_*.csv; no inference.")
    parsed = p.parse_args(argv)
    with open(parsed.config, "r") as f:
        cfg = yaml.safe_load(f)
    if parsed.center:
        cfg.setdefault("transforms", {})["center"] = parsed.center
    if parsed.output:
        cfg.setdefault("output", {})["folder"] = parsed.output
    if parsed.report_only:
        out_dir = Path(cfg["output"]["folder"])
        report(pd.read_csv(out_dir / "per_transform.csv"), pd.read_csv(out_dir / "per_landmark.csv"), out_dir)
        return 0
    return run(cfg)


if __name__ == "__main__":
    raise SystemExit(main())
