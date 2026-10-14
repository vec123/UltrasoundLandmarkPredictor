"""Stage 0+1: probe raw volumes, preprocess to canonical grid.

Probes shape, spacing, FOV, and whether each volume has a manual pose annotation.
Prints summary table.
 Applies smoothing + B-spline resample to 1 mm
isotropic 128³ grid, with affine pose standardization if pose annotation exists.

Configuration is entirely via YAML (single source of truth). No CLI args except config path (and --probe-only).

Examples
--------
    # Probe only
    python -m inference.tools.preprocess_geom config_preprocess.yaml --probe-only

    # Probe and preprocess
    python -m inference.tools.preprocess_geom config_preprocess.yaml
"""

from __future__ import annotations
import os
import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import nrrd
import yaml

# --- bootstrap: allow running as a bare file, not just `python -m` -------(Should be removed-make package instead)
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from inference.src.geometry import (
    affine3Dmatrix,
    affine_transform_scipy,
    build_output_header,
    preprocess_geometry,
)
from inference.src.nrrd_io import load_gt_keys, load_volume, physical_fov, read_spacing
from inference.src.probe import probe_volume, render_table, summarize
from dataset.utility.rotation import affine3Dmatrix

logger = logging.getLogger("inference.preprocess")

# YAML config sets these; these are fallback defaults only!
DEFAULT_OUT_SPACING = (1.0, 1.0, 1.0)
DEFAULT_OUT_SIZE = (128, 128, 128)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="preprocess_geom",
        description="Stage 0+1: probe raw volumes, apply geometric preprocessing.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Probe volumes (from config)
  python -m inference.tools.preprocess_geom config_preprocess.yaml --probe-only

  # Probe and preprocess (from config)
  python -m inference.tools.preprocess_geom config_preprocess.yaml
        """,
    )
    p.add_argument("config", type=Path, help="YAML config file (single source of truth).")
    p.add_argument("--probe-only", action="store_true", help="Only probe volumes, skip preprocessing.")
    return p


def _collect_inputs(args) -> list:
    volume = getattr(args, 'volume', None)
    if volume is not None:
        if not volume.is_file():
            logger.error("Not a file: %s", volume)
            return []
        return [volume]
    if not args.folder or not args.folder.is_dir():
        logger.error("Not a folder: %s", args.folder)
        return []
    return sorted(args.folder.glob(args.pattern))


def process_one(path: Path, out_dir: Path, args) -> bool:
    """Preprocess one volume and write ``<out_dir>/<name>.nrrd``. Returns success."""
    out_path = os.path.join(str(out_dir), path.name)
    if os.path.exists(out_path) and not args.overwrite:
        logger.info("skip (exists): %s", os.path.basename(out_path))
        return True

    data, header = load_volume(path)
    spacing = read_spacing(header)
    box_mm = np.asarray(args.spacing, dtype=float) * np.asarray(args.size, dtype=float)
    fov = physical_fov(data.shape, spacing)

    crop = bool(np.any(fov > box_mm))
    logger.info(
        "%-14s in=%s spacing=%.3f FOV=%s%s",
        path.name,
        "x".join(map(str, data.shape)),
        float(spacing[0]),
        np.round(fov, 1).tolist(),
        "  [WILL CROP: FOV exceeds box]" if crop else "",
    )

    prepared = preprocess_geometry(
        data,
        spacing,
        do_filter=not args.no_filter,
        do_bspline=not args.no_bspline,
        filter_size=args.filter_size,
        filter_mode=args.filter_mode,
        out_spacing=args.spacing,
        out_size=args.size,
    )

    # Apply affine transformation if ground truth pose is provided
    if args.gt_pose_path and not args.no_affine:
        try:
            with open(args.gt_pose_path, 'r') as f:
                gt_dict = json.load(f)

            # Look up this volume's pose in the gt.txt by scanning name
            scan_name = path.stem  # Filename without extension
            if scan_name in gt_dict:
                pose_data = gt_dict[scan_name]

                # Extract plane vectors (sagittal, coronal, axial) to compute rotation
                if 'sagittal' in pose_data and 'coronal' in pose_data and 'axial' in pose_data:
                    rotation = affine3Dmatrix(pose_data)
                else:
                    rotation = np.eye(3)

                # Compute translation from center (in mm) to grid center
                # Canonical grid: 128³ at 1mm spacing, center at voxel index 64
                if 'center' in pose_data:
                    center_mm = np.array(pose_data['center'])
                    # Convert from RAS to grid coordinates (flip x, y for RAS→LPS)
                    center_voxel = np.array([-center_mm[0], -center_mm[1], center_mm[2]])
                    grid_center = np.array([64.0, 64.0, 64.0])
                    translation = center_voxel - grid_center
                else:
                    translation = np.array([0, 0, 0])

                logger.info("  applying affine from gt.txt for %s", scan_name)
                prepared = affine_transform_scipy(prepared, rotation, translation)
            else:
                logger.warning("  no pose found in gt.txt for %s (will skip affine)", scan_name)
        except Exception as e:
            logger.error("  affine failed for %s: %s (continuing without affine)", path.name, e)

    out_header = build_output_header(header, prepared.shape, args.spacing)
    out_dir.mkdir(parents=True, exist_ok=True)
    nrrd.write(out_path, prepared, header=out_header)
    logger.info("  -> wrote %s  out=%s %s", os.path.basename(out_path), "x".join(map(str, prepared.shape)), prepared.dtype)
    return True


def load_config_from_yaml(config_path: Path):
    """Load config from YAML, return namespace with all settings."""
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)

    class Config:
        pass

    c = Config()
    # Defaults
    c.folder = None
    c.volume = None
    c.out = None
    c.gt_pose_path = None
    c.pattern = "*.nrrd"
    c.filter_size = 3
    c.filter_mode = "reflect"
    c.no_filter = False
    c.no_bspline = False
    c.no_affine = False
    c.spacing = list(DEFAULT_OUT_SPACING)
    c.size = list(DEFAULT_OUT_SIZE)
    c.overwrite = False

    # Input
    if 'input' in cfg:
        inp = cfg['input']
        if 'folder' in inp and inp['folder']:
            c.folder = Path(inp['folder'])
        if 'volume' in inp and inp['volume']:
            c.volume = Path(inp['volume'])

    # Output
    if 'output' in cfg and 'folder' in cfg['output']:
        c.out = Path(cfg['output']['folder'])

    # Ground truth
    if 'ground_truth' in cfg and 'pose_path' in cfg['ground_truth']:
        pose_p = cfg['ground_truth']['pose_path']
        if pose_p:
            c.gt_pose_path = Path(pose_p)

    # Preprocessing
    if 'preprocessing' in cfg:
        preproc = cfg['preprocessing']
        c.no_filter = not preproc.get('filter', True)
        c.filter_size = preproc.get('filter_size', 3)
        c.filter_mode = preproc.get('filter_mode', 'reflect')
        c.no_bspline = not preproc.get('bspline', True)
        c.no_affine = not preproc.get('affine', False)

        spacing = preproc.get('target_spacing', DEFAULT_OUT_SPACING)
        size = preproc.get('target_size', DEFAULT_OUT_SIZE)
        c.spacing = list(spacing) if isinstance(spacing, (list, tuple)) else [spacing] * 3
        c.size = list(size) if isinstance(size, (list, tuple)) else [size] * 3

    # Batch
    if 'batch' in cfg:
        batch = cfg['batch']
        c.pattern = batch.get('pattern', '*.nrrd')
        c.overwrite = batch.get('overwrite', False)

    return c


def main(argv=None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parsed_args = build_parser().parse_args(argv)

    logger.info("Loading config from %s", parsed_args.config)
    args = load_config_from_yaml(parsed_args.config)

    # Validate required fields
    if not args.folder and not args.volume:
        logger.error("Must provide input.folder or input.volume in config")
        return 1

    inputs = _collect_inputs(args)
    if not inputs:
        logger.error("No input volumes found.")
        return 1

    # Load gt keys if gt.txt is specified
    gt_keys = None
    if args.gt_pose_path:
        try:
            gt_keys = load_gt_keys(args.gt_pose_path)
            logger.info("Loaded %d gt.txt keys from %s", len(gt_keys), args.gt_pose_path)
        except FileNotFoundError:
            logger.warning("gt.txt not found at %s - skipping annotation check", args.gt_pose_path)

    # Probe all volumes
    logger.info("Probing %d volume(s)...", len(inputs))
    probes = []
    for path in inputs:
        try:
            probe = probe_volume(path, gt_keys=gt_keys, box_mm=128.0)
            probes.append(probe)
        except Exception as exc:
            logger.error("Failed to probe %s: %s", path.name, exc)

    if probes:
        print("\n" + render_table(probes) + "\n")
        print(summarize(probes, box_mm=128.0) + "\n")

    # Early return if probe-only
    if parsed_args.probe_only:
        return 0 if probes else 1

    # Otherwise, preprocess
    if not args.out:
        logger.error("Must provide output.folder in config for preprocessing")
        return 1

    logger.info("Preprocessing %d volume(s)...", len(inputs))
    ok = 0
    for path in inputs:
        try:
            ok += bool(process_one(path, args.out, args))
        except Exception as exc:  # noqa: BLE001 - one bad file shouldn't stop the batch
            logger.error("Failed on %s: %s", path.name, exc)

    logger.info("Done: %d/%d volume(s) prepared into %s", ok, len(inputs), args.out)
    return 0 if ok == len(inputs) else 1


if __name__ == "__main__":
    raise SystemExit(main())
