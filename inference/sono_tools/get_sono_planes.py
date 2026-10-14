"""Extract sono planes from NRRD and run SonoNet inference.

Loads 3D ultrasound volumes, extracts 3 orthogonal slices (sagittal, axial, coronal)
from the center, and passes through the SonoNet model to estimate translation and rotation.

Configuration via YAML (single source of truth). Saves results to JSON.

Examples
--------
    python -m inference.tools.get_sono_planes sono.yaml
"""

from __future__ import annotations

import sys
import argparse
import json
import logging
from pathlib import Path

import numpy as np
import torch
import yaml

# --- bootstrap: allow running as a bare file
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from inference.src.nrrd_io import load_volume, read_spacing
from inference.src.geometry import preprocess_geometry
from dataset.utility.gtpp.utils.SonoNet import SonoNet

logger = logging.getLogger("inference.get_sono_planes")



def extract_center_slices(volume: np.ndarray) -> torch.Tensor:
    """Extract 3 orthogonal slices from center of volume.

    Args:
        volume: 3D numpy array (preprocessed, typically 128x128x128)

    Returns:
        Tensor of shape (1, 3, H, W) with 3 orthogonal center slices
    """
    volume_torch = torch.from_numpy(volume.astype(np.float32)).unsqueeze(0)  # (1, x, y, z)

    x, y, z = volume_torch.shape[1:]
    cx, cy, cz = x // 2, y // 2, z // 2

    # Extract 3 orthogonal slices from center
    slices = torch.stack([
        volume_torch[0, cx, :, :],   # sagittal: (y, z)
        volume_torch[0, :, cy, :],   # axial: (x, z)
        volume_torch[0, :, :, cz],   # coronal: (x, y)
    ], dim=0).unsqueeze(0)  # (1, 3, H, W)

    return slices


def load_model(checkpoint_path: str, device: str = "cpu") -> SonoNet:
    """Load pretrained SonoNet model.

    Args:
        checkpoint_path: Path to model checkpoint
        device: torch device

    Returns:
        SonoNet model in eval mode
    """
    # Create dummy config object with required fields
    class Config:
        net = "SN16"
        latent = 128
        latent2 = 128
        last_bn = True
        limit_translation = 1.0

    config = Config()

    model = SonoNet(
        config,
        input_dim=1,
        num_output_t=3,
        num_output_r=4,
        dropout_p=0.0,
    ).to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    # Extract model_state_dict if checkpoint is a training dict
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
    model.eval()

    return model


def get_sono_planes(
    nrrd_path: str,
    checkpoint_path: str,
    device: str = "cpu",
) -> dict:
    """Load NRRD, preprocess, extract planes, run SonoNet inference.

    Args:
        nrrd_path: Path to .nrrd file
        checkpoint_path: Path to model checkpoint
        device: torch device ("cpu" or "cuda")

    Returns:
        dict with keys: tran, rot, g1, g2 (as numpy arrays)
    """
    volume, header = load_volume(nrrd_path)
    logger.info("  Raw volume shape: %s, dtype: %s", volume.shape, volume.dtype)

    # Preprocess: smooth + b-spline resample to 128³ at 1mm isotropic
    spacing = read_spacing(header)
    logger.info("  Spacing: %s mm", spacing)

    volume = preprocess_geometry(
        volume,
        spacing,
        do_filter=True,
        do_bspline=True,
        filter_size=3,
        filter_mode="reflect",
        out_spacing=(1.0, 1.0, 1.0),
        out_size=(128, 128, 128),
    )
    logger.info("  Preprocessed volume shape: %s", volume.shape)

    # Extract center slices
    slices = extract_center_slices(volume)
    slices = slices.to(device)
    logger.info("  Slices shape: %s", slices.shape)

    model = load_model(checkpoint_path, device=device)

    logger.info("Running SonoNet inference...")
    with torch.no_grad():
        tran, rot, g1, g2 = model(slices)

    logger.info("  tran shape: %s, rot shape: %s", tran.shape, rot.shape)

    return {
        "tran": tran.cpu().numpy().tolist(),
        "rot": rot.cpu().numpy().tolist(),
        "g1": g1.cpu().numpy().tolist(),
        "g2": g2.cpu().numpy().tolist(),
    }


def load_config_from_yaml(config_path: Path) -> dict:
    """Load sono config from YAML.

    Expected structure:
        checkpoint: "path/to/checkpoint"
        input:
          folder: "path/to/folder"  OR  volume: "path/to/file.nrrd"
        output:
          json: "path/to/output.json"
        inference:
          device: "cpu"     (optional, default)
          pattern: "*.nrrd" (for folder input, optional)
    """
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    config = {
        "checkpoint": cfg.get("checkpoint"),
        "input_folder": None,
        "input_volume": None,
        "output_json": None,
        "device": "cpu",
        "pattern": "*.nrrd",
    }

    if not config["checkpoint"]:
        raise ValueError("checkpoint path is required")

    # Input
    if "input" in cfg:
        inp = cfg["input"]
        if "folder" in inp and inp["folder"]:
            config["input_folder"] = Path(inp["folder"])
        if "volume" in inp and inp["volume"]:
            config["input_volume"] = Path(inp["volume"])

    if not config["input_folder"] and not config["input_volume"]:
        raise ValueError("Must provide input.folder or input.volume")

    # Output
    if "output" in cfg and "json" in cfg["output"]:
        config["output_json"] = Path(cfg["output"]["json"])

    if not config["output_json"]:
        raise ValueError("output.json path is required")

    # Inference
    if "inference" in cfg:
        inf = cfg["inference"]
        config["device"] = inf.get("device", "cpu")
        config["pattern"] = inf.get("pattern", "*.nrrd")

    return config


def collect_inputs(config: dict) -> list:
    """Collect input volumes to process."""
    if config["input_volume"]:
        volume = config["input_volume"]
        if not volume.is_file():
            logger.error("Not a file: %s", volume)
            return []
        return [volume]

    if not config["input_folder"] or not config["input_folder"].is_dir():
        logger.error("Not a folder: %s", config["input_folder"])
        return []

    return sorted(config["input_folder"].glob(config["pattern"]))


def process_one(nrrd_path: Path, config: dict) -> tuple:
    """Process one volume. Returns (success, results_dict or None)."""
    try:
        logger.info("Processing: %s", nrrd_path.name)
        results = get_sono_planes(
            nrrd_path=str(nrrd_path),
            checkpoint_path=config["checkpoint"],
            device=config["device"],
        )
        logger.info("  -> Done")
        return True, results
    except Exception as e:
        logger.error("  -> Failed: %s", e)
        return False, None


def main(argv=None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    parser = argparse.ArgumentParser(
        prog="get_sono_planes",
        description="Run SonoNet inference on ultrasound volumes, save poses to JSON.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m inference.tools.get_sono_planes sono.yaml
        """,
    )
    parser.add_argument("config", type=Path, help="YAML config file")
    args = parser.parse_args(argv)

    logger.info("Loading config from %s", args.config)
    try:
        config = load_config_from_yaml(args.config)
    except Exception as e:
        logger.error("Failed to load config: %s", e)
        return 1

    inputs = collect_inputs(config)
    if not inputs:
        logger.error("No input volumes found")
        return 1

    logger.info("Processing %d volume(s)...", len(inputs))

    results_dict = {}
    ok = 0

    for path in inputs:
        success, results = process_one(path, config)
        if success:
            ok += 1
            filename = path.stem  # Filename without extension
            results_dict[filename] = {"pose": results}

    # Save results to JSON
    config["output_json"].parent.mkdir(parents=True, exist_ok=True)
    with open(config["output_json"], "w") as f:
        json.dump(results_dict, f, indent=2)

    logger.info("Done: %d/%d volume(s) processed", ok, len(inputs))
    logger.info("Results saved to %s", config["output_json"])

    return 0 if ok == len(inputs) else 1


if __name__ == "__main__":
    raise SystemExit(main())
