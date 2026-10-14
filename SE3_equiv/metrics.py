"""Equivariance error of one prediction, and summaries over many transforms.

Per landmark: ``|| f(gV) - g f(V) ||`` in mm. Per transform, additionally:

- ``rot_err_deg``: angle between the rotation that best maps the reference
  landmarks onto the new prediction (Kabsch) and the applied rotation - did the
  face turn by the right amount as a whole?
- ``shape_rms_mm``: residual after that best rigid fit - did the face's shape
  survive, independent of where it ended up?
- ``lr_swapped``: the prediction matches better with left/right names exchanged.
"""
from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

from SE3_equiv.se3 import rotation_angle_deg

ANGLE_BINS = [0, 5, 15, 30, 45, 60, 90, 120, 150, 180]


def kabsch(P: np.ndarray, Q: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
    """Proper rigid fit ``Q ~ R P + t``. Returns ``(R, t, rms residual)``."""
    p0, q0 = P.mean(axis=0), Q.mean(axis=0)
    U, _, Vt = np.linalg.svd((P - p0).T @ (Q - q0))
    d = 1.0 if np.linalg.det(Vt.T @ U.T) >= 0 else -1.0
    R = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T
    t = q0 - R @ p0
    rms = float(np.sqrt(np.mean(np.sum((P @ R.T + t - Q) ** 2, axis=1))))
    return R, t, rms


def evaluate_transform(
    reference_mm: np.ndarray,
    predicted_mm: np.ndarray,
    expected_mm: np.ndarray,
    rotation: np.ndarray,
    shape: Sequence[int],
    spacing: np.ndarray,
    lr_perm: Sequence[int],
    margin_vox: float,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
    """Returns ``(err_mm, in_fov, summary)``. Landmarks whose expected position
    leaves the box (minus ``margin_vox``) are excluded from the summary."""
    err = np.linalg.norm(predicted_mm - expected_mm, axis=1)
    vox = expected_mm / spacing
    in_fov = np.all((vox >= margin_vox) & (vox <= np.asarray(shape) - 1 - margin_vox), axis=1)

    s = {"n_in_fov": int(in_fov.sum()), "err_mean": np.nan, "err_median": np.nan, "err_max": np.nan,
         "sdr_2mm": np.nan, "sdr_4mm": np.nan, "rot_err_deg": np.nan, "centroid_err_mm": np.nan,
         "shape_rms_mm": np.nan, "lr_swapped": False}
    if in_fov.sum() == 0:
        return err, in_fov, s

    e = err[in_fov]
    s.update(err_mean=float(e.mean()), err_median=float(np.median(e)), err_max=float(e.max()),
             sdr_2mm=float(np.mean(e <= 2.0)), sdr_4mm=float(np.mean(e <= 4.0)),
             centroid_err_mm=float(np.linalg.norm(predicted_mm[in_fov].mean(0) - expected_mm[in_fov].mean(0))))
    if in_fov.sum() >= 3:
        R_hat, _, rms = kabsch(reference_mm[in_fov], predicted_mm[in_fov])
        s.update(rot_err_deg=rotation_angle_deg(R_hat @ rotation.T), shape_rms_mm=rms)
    if list(lr_perm) != list(range(len(lr_perm))):
        swapped = np.linalg.norm(predicted_mm[list(lr_perm)] - expected_mm, axis=1)[in_fov]
        s["lr_swapped"] = bool(swapped.mean() < e.mean())
    return err, in_fov, s


def summarize(per_transform: pd.DataFrame, per_landmark: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """Tables by transform family, by rotation angle, and by landmark."""
    agg = dict(n=("name", "count"), err_mean=("err_mean", "mean"), err_median=("err_median", "median"),
               worst_mean=("err_mean", "max"), sdr_2mm=("sdr_2mm", "mean"), sdr_4mm=("sdr_4mm", "mean"),
               rot_err_deg=("rot_err_deg", "mean"), shape_rms_mm=("shape_rms_mm", "mean"),
               n_lr_swapped=("lr_swapped", "sum"), n_exact=("exact", "sum"))
    by_family = per_transform.groupby("family", sort=False).agg(**agg).reset_index()

    rot = per_transform[per_transform["family"].isin(["cube_rotation", "axis_sweep", "random_se3"])].copy()
    # round: a 15-degree rotation comes back from its matrix as 15.0000001
    rot["angle_bin"] = pd.cut(rot["angle_deg"].round(3), ANGLE_BINS, include_lowest=True)
    by_angle = rot.groupby("angle_bin", observed=True).agg(**agg).reset_index()
    by_angle["angle_bin"] = by_angle["angle_bin"].astype(str)

    moved = per_landmark[(per_landmark["family"] != "identity") & per_landmark["in_fov"]]
    by_landmark = moved.groupby("landmark", sort=False)["err_mm"].agg(
        err_mean="mean", err_median="median", err_max="max",
        sdr_2mm=lambda e: float(np.mean(e <= 2.0)), sdr_4mm=lambda e: float(np.mean(e <= 4.0))).reset_index()
    return {"by_family": by_family, "by_angle": by_angle, "by_landmark": by_landmark}


def plot(per_transform: pd.DataFrame, path) -> bool:
    """Error vs rotation angle and vs shift. Skipped (False) without matplotlib."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return False

    fig, (ax_rot, ax_shift) = plt.subplots(1, 2, figsize=(12, 4.5))
    for family, marker in [("axis_sweep", "o"), ("cube_rotation", "s"), ("random_se3", "^")]:
        g = per_transform[per_transform["family"] == family]
        if len(g):
            ax_rot.scatter(g["angle_deg"], g["err_mean"], marker=marker, label=family, alpha=0.8)
    ax_rot.set_xlabel("rotation angle (deg)")
    ax_rot.set_ylabel("mean equivariance error (mm)")
    ax_rot.set_title("|| f(gV) - g f(V) || vs rotation")
    ax_rot.grid(alpha=0.3)
    ax_rot.legend()

    g = per_transform[per_transform["family"] == "translation"]
    for axis in sorted(set(n.split("_")[1][0] for n in g["name"])) if len(g) else []:
        ga = g[g["name"].str.startswith("shift_" + axis)]
        ax_shift.plot(ga["shift_mm"], ga["err_mean"], "o-", label="axis " + axis)
    ax_shift.set_xscale("log", base=2)
    ax_shift.set_xlabel("shift (mm)")
    ax_shift.set_ylabel("mean equivariance error (mm)")
    ax_shift.set_title("pure translations")
    ax_shift.grid(alpha=0.3)
    ax_shift.legend()

    fig.tight_layout()
    fig.savefig(str(path), dpi=130)
    plt.close(fig)
    return True
