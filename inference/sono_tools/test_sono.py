"""Test SonoNet predictions against ground truth poses from gt.txt.

Loads NRRD volumes, runs SonoNet inference, and compares outputs to ground truth.
Reports error metrics: translation MAE, rotation angular error, etc.

Usage:
    python -m inference.tools.test_sono config.yaml
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
from inference.src.nrrd_io import load_volume, read_spacing, load_gt_keys
from inference.src.geometry import preprocess_geometry
from dataset.utility.rotation import affine3Dmatrix, grid_transform_3d

logger = logging.getLogger("inference.test_sono")


def load_gt_poses(gt_path: Path) -> dict:
    """Load ground truth poses from gt.txt (JSONL format, one JSON per line)."""
    with open(gt_path, 'r') as f:
        lines = f.read().splitlines()
    if lines:
        return json.loads(lines[0])
    return {}


def quaternion_to_rotation_matrix(q: np.ndarray) -> np.ndarray:
    """Convert quaternion [w, x, y, z] (pytorch3d format) to 3x3 rotation matrix."""
    w, x, y, z = q
    return np.array([
        [1 - 2*(y**2 + z**2), 2*(x*y - w*z), 2*(x*z + w*y)],
        [2*(x*y + w*z), 1 - 2*(x**2 + z**2), 2*(y*z - w*x)],
        [2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x**2 + y**2)],
    ])


def rotation_matrix_to_quaternion(R: np.ndarray) -> np.ndarray:
    """Convert 3x3 rotation matrix to quaternion [w, x, y, z] (pytorch3d format).

    Returns quaternion with real part (w) first, matching training pipeline format.
    """
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
    # Standardize: ensure w (real part) is non-negative
    if q[0] < 0:
        q = -q
    return q


def quaternion_angular_distance(q1: np.ndarray, q2: np.ndarray) -> float:
    """Angular distance (in degrees) between two quaternions."""
    dot = np.clip(abs(np.dot(q1, q2)), -1.0, 1.0)
    return 2 * np.arccos(dot) * 180 / np.pi


def test_one_volume(
    nrrd_path: Path,
    gt_poses: dict,
    model,
    device: str,
    mode: str = "iterative",
    max_iterations: int = 10,
) -> tuple:
    """Test one volume. Returns (success, metrics_dict or None).

    Args:
        mode: "aligned" (single-pass, volume pre-aligned) or "iterative" (start from raw, refine)
        max_iterations: Max refinement steps (only for iterative mode)
    """
    scan_name = nrrd_path.stem
    if scan_name not in gt_poses:
        logger.warning("Not in GT: %s", scan_name)
        return False, None

    try:
        logger.info("Testing: %s (mode=%s)", nrrd_path.name, mode)

        # Load and preprocess volume
        volume, header = load_volume(str(nrrd_path))
        spacing = read_spacing(header)
        original_shape = volume.shape

        # Preprocess (filter + B-spline)
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

        gt = gt_poses[scan_name]
        gt_rotation = affine3Dmatrix(gt)
        gt_rot_quat = rotation_matrix_to_quaternion(gt_rotation)

        if mode == "aligned":
            # Original mode: apply gt.txt upfront, then single inference
            rotation_3x3 = gt_rotation
            center_gt_pix = np.array([-1 * gt["center"][0], -1 * gt["center"][1], gt["center"][2]]) / spacing
            img_size = np.array(original_shape, dtype=np.float32)
            translation_vector = (center_gt_pix - img_size / 2.0) * (2.0 / img_size)

            transform_matrix = np.zeros((3, 4))
            transform_matrix[:3, :3] = rotation_3x3
            transform_matrix[:3, 3] = translation_vector
            volume = grid_transform_3d(volume.astype(np.float32), transform_matrix.astype(np.float32))
            logger.info("  Applied affine transformation using gt.txt")

            slices = extract_center_slices(volume)
            slices = slices.to(device)

            with torch.no_grad():
                pred_tran_norm, pred_rot, _, _ = model(slices)

            pred_tran_norm = pred_tran_norm.cpu().numpy()[0]
            pred_rot = pred_rot.cpu().numpy()[0]

            gt_tran_norm_aligned = np.array([0.0, 0.0, 0.0])
            gt_rot_aligned = np.array([0.0, 0.0, 0.0, 1.0])

            tran_error_aligned = np.linalg.norm(pred_tran_norm - gt_tran_norm_aligned)
            rot_error_aligned = quaternion_angular_distance(pred_rot, gt_rot_aligned)

            logger.info("  Pred tran: %s, Error=%.4f", np.round(pred_tran_norm, 3), tran_error_aligned)
            logger.info("  Pred rot: %s, Angular error: %.2f degrees", np.round(pred_rot, 3), rot_error_aligned)

            return True, {
                "scan_name": scan_name,
                "mode": "aligned",
                "pred_tran_norm": pred_tran_norm.tolist(),
                "pred_rot": pred_rot.tolist(),
                "gt_tran_norm_aligned": gt_tran_norm_aligned.tolist(),
                "gt_rot_aligned": gt_rot_aligned.tolist(),
                "tran_error_aligned": float(tran_error_aligned),
                "rot_error_aligned": float(rot_error_aligned),
            }

        else:  # iterative mode
            # Start from identity pose, iteratively refine using numpy transformations
            current_rotation = np.eye(3)
            current_tran_norm = np.array([0.0, 0.0, 0.0])
            current_volume = volume.copy()

            iteration_errors = []

            for iteration in range(max_iterations):
                # Extract slices from current volume state
                slices = extract_center_slices(current_volume)
                slices = slices.to(device)

                with torch.no_grad():
                    pred_tran_norm, pred_rot, _, _ = model(slices)

                pred_tran_norm = pred_tran_norm.cpu().numpy()[0]
                pred_rot = pred_rot.cpu().numpy()[0]

                # Convert quaternion to rotation matrix and accumulate
                pred_rotation = quaternion_to_rotation_matrix(pred_rot)
                current_rotation = pred_rotation @ current_rotation
                current_tran_norm = current_tran_norm + pred_tran_norm

                # Apply accumulated transformation to volume for next iteration
                # CRITICAL: Use INVERSE rotation for resampling (matching standardization.py line 155)
                img_size = np.array(current_volume.shape, dtype=np.float32)
                translation_voxels = current_tran_norm * (img_size / 2.0)

                transform_matrix = np.zeros((3, 4))
                transform_matrix[:3, :3] = np.linalg.inv(current_rotation)
                transform_matrix[:3, 3] = -translation_voxels

                current_volume = grid_transform_3d(
                    current_volume.astype(np.float32),
                    transform_matrix.astype(np.float32),
                )

                # Compute errors vs ground truth
                rot_error = quaternion_angular_distance(
                    rotation_matrix_to_quaternion(current_rotation),
                    gt_rot_quat,
                )
                tran_error = np.linalg.norm(current_tran_norm)

                iteration_errors.append({
                    "iter": iteration,
                    "rot_error": rot_error,
                    "tran_error": tran_error
                })

                logger.info(
                    "  Iter %d: rot_error=%.2f°, tran_error=%.4f, pred_rot=%s",
                    iteration,
                    rot_error,
                    tran_error,
                    np.round(pred_rot, 3),
                )

                # Early exit if converged
                if rot_error < 5.0 and tran_error < 0.1:
                    logger.info("  -> Converged at iteration %d", iteration)
                    break

            final_rot_error = iteration_errors[-1]["rot_error"]
            final_tran_error = iteration_errors[-1]["tran_error"]

            return True, {
                "scan_name": scan_name,
                "mode": "iterative",
                "num_iterations": len(iteration_errors),
                "final_rot_error": float(final_rot_error),
                "final_tran_error": float(final_tran_error),
                "iteration_log": iteration_errors,
            }

    except Exception as e:
        logger.error("  -> Failed: %s", e)
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

    # Input
    if "input" in cfg:
        inp = cfg["input"]
        if "folder" in inp and inp["folder"]:
            config["input_folder"] = Path(inp["folder"])
        if "volume" in inp and inp["volume"]:
            config["input_volume"] = Path(inp["volume"])

    if not config["input_folder"] and not config["input_volume"]:
        raise ValueError("Must provide input.folder or input.volume")

    # Ground truth
    if "ground_truth" in cfg and "pose_path" in cfg["ground_truth"]:
        gt_p = cfg["ground_truth"]["pose_path"]
        if gt_p:
            config["gt_path"] = Path(gt_p)

    if not config["gt_path"]:
        raise ValueError("ground_truth.pose_path is required for testing")

    # Output
    if "output" in cfg and "json" in cfg["output"]:
        config["output_json"] = Path(cfg["output"]["json"])

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


def main(argv=None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    parser = argparse.ArgumentParser(
        prog="test_sono",
        description="Test SonoNet predictions against ground truth poses.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m inference.tools.test_sono config_test.yaml
  python -m inference.tools.test_sono config_test.yaml --mode iterative
        """,
    )
    parser.add_argument("config", type=Path, help="YAML config file")
    parser.add_argument("--mode", choices=["aligned", "iterative"], default="iterative",
                        help="Test mode: aligned (pre-aligned volumes) or iterative (raw volumes, refine)")
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

    # Load model
    logger.info("Loading model from %s", config["checkpoint"])
    model = load_model(config["checkpoint"], device=config["device"])

    logger.info("Testing %d volume(s)...", len(inputs))

    results = []
    ok = 0

    for path in inputs:
        success, metrics = test_one_volume(
            path, gt_poses, model, config["device"],
            mode=args.mode,
            max_iterations=args.max_iterations,
        )
        if success:
            ok += 1
            results.append(metrics)

    # Summary statistics
    if results:
        if args.mode == "aligned":
            tran_errors = [r["tran_error_aligned"] for r in results]
            rot_errors = [r["rot_error_aligned"] for r in results]

            logger.info("\n=== Summary (Aligned mode: pre-aligned volumes) ===")
            logger.info("Translation error (L2 norm, [-1,1] normalized): mean=%.4f, std=%.4f, min=%.4f, max=%.4f",
                       np.mean(tran_errors), np.std(tran_errors),
                       np.min(tran_errors), np.max(tran_errors))
            logger.info("Rotation error (degrees): mean=%.2f, std=%.2f, min=%.2f, max=%.2f",
                       np.mean(rot_errors), np.std(rot_errors),
                       np.min(rot_errors), np.max(rot_errors))
        else:
            rot_errors = [r["final_rot_error"] for r in results]
            num_iters = [r["num_iterations"] for r in results]

            logger.info("\n=== Summary (Iterative mode: refining from raw volumes) ===")
            logger.info("Final rotation error (degrees): mean=%.2f, std=%.2f, min=%.2f, max=%.2f",
                       np.mean(rot_errors), np.std(rot_errors),
                       np.min(rot_errors), np.max(rot_errors))
            logger.info("Iterations to convergence: mean=%.1f, min=%d, max=%d",
                       np.mean(num_iters), np.min(num_iters), np.max(num_iters))

    # Save results to JSON if requested
    if config["output_json"]:
        config["output_json"].parent.mkdir(parents=True, exist_ok=True)
        with open(config["output_json"], "w") as f:
            json.dump(results, f, indent=2)
        logger.info("Results saved to %s", config["output_json"])

    logger.info("Done: %d/%d volume(s) tested", ok, len(inputs))
    return 0 if ok == len(inputs) else 1


if __name__ == "__main__":
    raise SystemExit(main())
