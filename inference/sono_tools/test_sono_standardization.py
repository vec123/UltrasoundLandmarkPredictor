"""Test SonoNet using training pipeline's standardization.py iterative refinement logic.

This script directly uses the exact iterative refinement from standardization.py
to test whether the network predictions are correct when using training's exact code.

Usage:
    python -m inference.sono_tools.test_sono_standardization config.yaml
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

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from inference.sono_tools.get_sono_planes import load_model, extract_center_slices
from inference.src.nrrd_io import load_volume, read_spacing
from inference.src.geometry import preprocess_geometry, affine3Dmatrix
from dataset.utility.gtpp.utils.pytorch3d import quaternion_to_matrix
from dataset.utility.gtpp.utils import utils

logger = logging.getLogger("inference.test_sono_standardization")


def quaternion_angular_distance(q1: np.ndarray, q2: np.ndarray) -> float:
    """Angular distance (in degrees) between two quaternions."""
    dot = np.clip(abs(np.dot(q1, q2)), -1.0, 1.0)
    return 2 * np.arccos(dot) * 180 / np.pi


def rotation_matrix_to_quaternion(R: np.ndarray) -> np.ndarray:
    """Convert 3x3 rotation matrix to quaternion [w, x, y, z]."""
    trace = R[0, 0] + R[1, 1] + R[2, 2]
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s

    q = np.array([w, x, y, z])
    if q[0] < 0:
        q = -q
    return q


def load_gt_poses(gt_path: Path) -> dict:
    """Load ground truth poses from gt.txt."""
    with open(gt_path, 'r') as f:
        lines = f.read().splitlines()
    if lines:
        return json.loads(lines[0])
    return {}


def test_one_volume_standardization(
    nrrd_path: Path,
    gt_poses: dict,
    model,
    device: str,
    max_iterations: int = 10,
) -> tuple:
    """Test one volume using standardization.py's exact iterative refinement.

    Directly ports the iterative loop from standardization.py lines 141-157.
    """
    scan_name = nrrd_path.stem
    if scan_name not in gt_poses:
        logger.warning("Not in GT: %s", scan_name)
        return False, None

    try:
        logger.info("Testing: %s (standardization.py iterative mode)", nrrd_path.name)

        # Load and preprocess volume
        volume, header = load_volume(str(nrrd_path))
        spacing = read_spacing(header)

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

        # Compute ground truth
        gt = gt_poses[scan_name]
        gt_rotation = affine3Dmatrix(gt)
        gt_rot_quat = rotation_matrix_to_quaternion(gt_rotation)

        # Convert volume to torch (1, 128, 128, 128) - NO channel dim, training uses 4D
        images = torch.from_numpy(volume.astype(np.float32)).unsqueeze(0).to(device)

        # Initialize with identity (standardization.py lines 141-143)
        R_forward = torch.eye(3, dtype=torch.float32).unsqueeze(0).to(device)  # Forward (1, 3, 3)
        T_forward = torch.zeros(1, 3, dtype=torch.float32).to(device)  # (1, 3)

        iteration_errors = []

        # Build minimal config for get_slices
        class MinimalConfig:
            class Params:
                desired_size = [128, 128, 128]
            params = Params()

        config = MinimalConfig()

        # Iterative refinement loop - maintain FORWARD rotation, use INVERTED for extraction only
        for iteration in range(max_iterations):
            # For extraction: use inverted rotation (needed for backward mapping in grid_sample)
            R_inverted = torch.linalg.inv(R_forward[0]).unsqueeze(0)
            T_inverted = -1 * T_forward
            slices_in, _ = utils.get_slices(config, images, R_inverted, T_inverted)

            with torch.no_grad():
                ytr_es, yrr_es, _, _ = model(slices_in.to(device))

            # Accumulate rotation and translation IN FORWARD FRAME
            rot = quaternion_to_matrix(yrr_es)  # (1, 3, 3)

            # Accumulate translation in forward frame
            t_current = torch.bmm(R_forward, ytr_es.unsqueeze(2))
            T_forward = T_forward + t_current[:, :, 0]

            # Accumulate rotation in forward frame
            R_forward = torch.bmm(R_forward, rot)

            # For error: use forward rotation
            pred_quat = rotation_matrix_to_quaternion(R_forward[0].cpu().numpy())

            rot_error = quaternion_angular_distance(pred_quat, gt_rot_quat)
            tran_error = torch.norm(T_forward).cpu().item()

            iteration_errors.append({
                "iter": iteration,
                "rot_error": rot_error,
                "tran_error": tran_error,
                "pred_quat": pred_quat.tolist(),
            })

            logger.info(
                "  Iter %d: rot_error=%.2f°, tran_error=%.4f, pred_quat=%s",
                iteration,
                rot_error,
                tran_error,
                np.round(pred_quat, 3),
            )

            # Early exit if converged
            if rot_error < 5.0 and tran_error < 0.1:
                logger.info("  -> Converged at iteration %d", iteration)
                break

        final_rot_error = iteration_errors[-1]["rot_error"]
        final_tran_error = iteration_errors[-1]["tran_error"]

        return True, {
            "scan_name": scan_name,
            "mode": "standardization_iterative",
            "num_iterations": len(iteration_errors),
            "final_rot_error": float(final_rot_error),
            "final_tran_error": float(final_tran_error),
            "iteration_log": iteration_errors,
        }

    except Exception as e:
        logger.error("  -> Failed: %s", e)
        import traceback
        logger.error(traceback.format_exc())
        return False, None


def load_config_from_yaml(config_path: Path) -> dict:
    """Load test config from YAML."""
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)

    config = {
        "checkpoint": cfg.get("checkpoint"),
        "input_folder": None,
        "input_volume": None,
        "output_json": None,
        "gt_path": None,
        "device": "cpu",
        "pattern": "*.nrrd",
    }

    if not config["checkpoint"]:
        raise ValueError("checkpoint path is required")

    if "input" in cfg:
        inp = cfg["input"]
        if "folder" in inp and inp["folder"]:
            config["input_folder"] = Path(inp["folder"])
        if "volume" in inp and inp["volume"]:
            config["input_volume"] = Path(inp["volume"])

    if not config["input_folder"] and not config["input_volume"]:
        raise ValueError("Must provide input.folder or input.volume")

    if "ground_truth" in cfg and "pose_path" in cfg["ground_truth"]:
        gt_p = cfg["ground_truth"]["pose_path"]
        if gt_p:
            config["gt_path"] = Path(gt_p)

    if not config["gt_path"]:
        raise ValueError("ground_truth.pose_path is required")

    if "output" in cfg and "json" in cfg["output"]:
        config["output_json"] = Path(cfg["output"]["json"])

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


def main(argv=None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    parser = argparse.ArgumentParser(
        prog="test_sono_standardization",
        description="Test SonoNet using standardization.py's exact iterative refinement.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m inference.sono_tools.test_sono_standardization config_test.yaml
        """,
    )
    parser.add_argument("config", type=Path, help="YAML config file")
    parser.add_argument("--max-iterations", type=int, default=10, help="Max iterations for iterative mode")
    args = parser.parse_args(argv)

    logger.info("Loading config from %s", args.config)
    try:
        config = load_config_from_yaml(args.config)
    except Exception as e:
        logger.error("Failed to load config: %s", e)
        return 1

    logger.info("Loading ground truth from %s", config["gt_path"])
    try:
        gt_poses = load_gt_poses(config["gt_path"])
    except Exception as e:
        logger.error("Failed to load ground truth: %s", e)
        return 1

    inputs = collect_inputs(config)
    if not inputs:
        logger.error("No input volumes found")
        return 1

    logger.info("Loading model from %s", config["checkpoint"])
    model = load_model(config["checkpoint"], device=config["device"])

    logger.info("Testing %d volume(s) with standardization.py iterative refinement...", len(inputs))

    results = []
    ok = 0

    for path in inputs:
        success, metrics = test_one_volume_standardization(
            path, gt_poses, model, config["device"],
            max_iterations=args.max_iterations,
        )
        if success:
            ok += 1
            results.append(metrics)

    # Summary
    if results:
        rot_errors = [r["final_rot_error"] for r in results]
        logger.info("\n=== Summary ===")
        logger.info("Final rotation error (degrees): mean=%.2f, std=%.2f, min=%.2f, max=%.2f",
                   np.mean(rot_errors), np.std(rot_errors),
                   np.min(rot_errors), np.max(rot_errors))
        logger.info("Converged: %d/%d volumes", sum(1 for r in results if r["final_rot_error"] < 5.0), len(results))

    if config["output_json"]:
        config["output_json"].parent.mkdir(parents=True, exist_ok=True)
        with open(config["output_json"], "w") as f:
            json.dump(results, f, indent=2)
        logger.info("Results saved to %s", config["output_json"])

    logger.info("Done: %d/%d volume(s) tested", ok, len(inputs))
    return 0 if ok == len(inputs) else 1


if __name__ == "__main__":
    raise SystemExit(main())
