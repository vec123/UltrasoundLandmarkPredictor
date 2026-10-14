"""nrrd -> landmarks, by running ``inference/tools/infer_landmarks`` as-is.

The harness only needs ``landmark_names`` and ``predict(path, out_dir) -> (L, 3)
mm``. A different checkpoint is evaluated by editing the inference config; a
different backbone by giving the harness any object with those two members.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from inference.src.model import load_model
from inference.tools.infer_landmarks import load_config_from_yaml, process_one

# infer_landmarks Config attributes the SE(3) config may override
OVERRIDABLE = {
    "already_preprocessed", "swap_xz", "save_heatmaps", "save_preprocessed",
    "device", "checkpoint", "detector",
}


class InferLandmarksPredictor:
    def __init__(self, infer_config, overrides: dict = None):
        self.args = load_config_from_yaml(Path(infer_config))
        for key, value in (overrides or {}).items():
            if key not in OVERRIDABLE:
                raise KeyError("infer_overrides.{} is not supported (allowed: {})".format(key, sorted(OVERRIDABLE)))
            setattr(self.args, key, Path(value) if key == "checkpoint" else value)
        self.args.overwrite = True
        if self.args.device == "auto":
            self.args.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.bundle = load_model(self.args.checkpoint, device=self.args.device)
        self.landmark_names = list(self.args.landmark_names)
        # process_one logs every file it writes; keep the harness output readable
        logging.getLogger("inference.infer_landmarks").setLevel(logging.WARNING)

    def describe(self) -> str:
        a = self.args
        return "{} (epoch {}, device {}, detector {}, already_preprocessed={}, swap_xz={}, swap_LR={})".format(
            a.checkpoint, self.bundle.epoch, a.device, a.detector, a.already_preprocessed, a.swap_xz, a.swap_lr)

    def predict(self, volume_path: Path, out_dir: Path) -> np.ndarray:
        """Run infer_landmarks on one .nrrd; returns ``(L, 3)`` mm in its array frame."""
        volume_path, out_dir = Path(volume_path), Path(out_dir)
        if not process_one(volume_path, self.bundle, self.args, out_dir):
            raise RuntimeError("infer_landmarks failed on {}".format(volume_path))
        df = pd.read_csv(out_dir / (volume_path.stem + ".csv"), index_col=0).set_index("label")
        return df.loc[self.landmark_names, ["x", "y", "z"]].to_numpy(dtype=float)
