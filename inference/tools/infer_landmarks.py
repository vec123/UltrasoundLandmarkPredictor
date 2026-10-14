"""Stage 0+1+2: raw (or preprocessed) .nrrd -> 19 facial landmark coordinates.

Runs the full pipeline against a trained checkpoint (``best_model/best.pt``):
optionally probes + geometrically preprocesses a raw volume (Stage 0-1, same
code as ``preprocess_geom.py``), then runs the network and writes Slicer-
readable ``.fcsv``/``.csv`` landmarks (Stage 2, ``documentation/INFERENCE_PLAN.md``).

Ground-truth pose standardization (the affine step) is OPTIONAL: the model's
author confirmed inference does not require it. It only runs if
``ground_truth.pose_path`` is set in the config AND the volume's name is a key
in that file - leave it unset for the common case of new, unannotated scans.

Configuration is entirely via YAML (single source of truth), matching
``preprocess_geom.py``.

Examples
--------
    # Single volume, using the bundled default config
    python -m inference.tools.infer_landmarks inference/config_infer_landmarks.yaml

    # Skip Stage 1 (input already sits on the canonical 1mm/128^3 grid)
    python -m inference.tools.infer_landmarks my_config.yaml --already-preprocessed

    # Run over several subjects: input.folder in the config contains a
    # "{subject_id}" placeholder (e.g. ".../Casos_Mar/{subject_id}/20_semanas"),
    # substituted in turn for each ID. output.folder may use {subject_id} and
    # {scan_dir} (the scan's parent folder, e.g. 20_semanas); without either,
    # each subject gets its own output.folder/<subject_id>/ subfolder. Optional
    # - omit --subject-ids and input.folder/volume is used literally.
    python -m inference.tools.infer_landmarks my_config.yaml --subject-ids 1 2 3 4 5 6
"""
from __future__ import annotations

import argparse
import copy
import json
import logging
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import nrrd
import torch
import yaml

# --- bootstrap: allow running as a bare file, not just `python -m` -------
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from inference.src.geometry import (
    affine3Dmatrix,
    affine_transform_scipy,
    build_output_header,
    centering_pad,
    preprocess_geometry,
)
from inference.src.nrrd_io import load_volume, read_spacing
from inference.src.probe import probe_volume
from inference.src.model import (
    DEFAULT_LANDMARK_NAMES,
    get_peak_location,
    left_right_permutation,
    load_model,
    predict_heatmaps,
)
from net.postprocess.save_fscv_csv import save_fscv_csv

logger = logging.getLogger("inference.infer_landmarks")

DEFAULT_OUT_SPACING = (1.0, 1.0, 1.0)
DEFAULT_OUT_SIZE = (128, 128, 128)

# Placeholder substituted by --subject-ids into input.folder / input.volume,
# e.g. ".../Casos_Mar/{subject_id}/20_semanas" -> ".../Casos_Mar/3/20_semanas".
SUBJECT_PLACEHOLDER = "{subject_id}"
# Placeholder in output.folder for the input scan's parent folder name,
# e.g. ".../Casos_Mar/1/20_semanas/1-20s-01.nrrd" -> "20_semanas".
SCAN_DIR_PLACEHOLDER = "{scan_dir}"


class Config:
    """Plain settings bag populated by :func:`load_config_from_yaml`."""


def load_config_from_yaml(config_path: Path) -> Config:
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f) or {}

    c = Config()
    # input
    inp = cfg.get("input", {})
    c.volume = Path(inp["volume"]) if inp.get("volume") else None
    c.folder = Path(inp["folder"]) if inp.get("folder") else None
    c.already_preprocessed = bool(inp.get("already_preprocessed", False))

    # model
    mdl = cfg.get("model", {})
    c.checkpoint = Path(mdl.get("checkpoint", "best_model/best.pt"))
    c.device = mdl.get("device", "cpu")
    c.detector = mdl.get("detector", "argmax")

    # output
    out = cfg.get("output", {})
    c.out_dir = Path(out.get("folder", "inference/results/landmarks"))
    c.landmarks_subfolder = out.get("landmarks_subfolder") or ""
    c.save_preprocessed = bool(out.get("save_preprocessed", True))
    c.save_heatmaps = bool(out.get("save_heatmaps", False))

    # ground truth (optional pose standardization)
    gt = cfg.get("ground_truth", {})
    c.gt_pose_path = Path(gt["pose_path"]) if gt.get("pose_path") else None
    c.affine = bool(gt.get("affine", False))

    # preprocessing (Stage 1)
    pre = cfg.get("preprocessing", {})
    c.do_filter = bool(pre.get("filter", True))
    c.filter_size = int(pre.get("filter_size", 3))
    c.filter_mode = pre.get("filter_mode", "reflect")
    c.do_bspline = bool(pre.get("bspline", True))
    c.centering = bool(pre.get("centering", False))
    c.swap_xz = bool(pre.get("swap_xz", False))
    c.target_spacing = list(pre.get("target_spacing", DEFAULT_OUT_SPACING))
    c.target_size = list(pre.get("target_size", DEFAULT_OUT_SIZE))

    # landmarks
    lmk = cfg.get("landmarks", {})
    c.landmark_names = list(lmk["names"]) if lmk.get("names") else list(DEFAULT_LANDMARK_NAMES)
    c.swap_lr = bool(lmk.get("swap_LR", False))
    c.lr_permutation = left_right_permutation(c.landmark_names) if c.swap_lr else None

    # batch
    batch = cfg.get("batch", {})
    c.pattern = batch.get("pattern", "*.nrrd")
    c.overwrite = bool(batch.get("overwrite", False))

    return c


def _collect_inputs(args: Config) -> list:
    if args.volume is not None:
        if not args.volume.is_file():
            logger.error("Not a file: %s", args.volume)
            return []
        return [args.volume]
    if not args.folder or not args.folder.is_dir():
        logger.error("Not a folder: %s", args.folder)
        return []
    return sorted(args.folder.glob(args.pattern))


def _expand_subject_ids(args: Config, subject_ids: List[str]) -> List[Tuple[Path, str]]:
    """Substitute each subject ID into whichever of input.folder / input.volume
    holds the ``{subject_id}`` placeholder, and collect that subject's inputs.

    Returns a flat list of ``(path, subject_id)`` pairs across all subjects.
    """
    template_attr = None
    for attr in ("folder", "volume"):
        value = getattr(args, attr)
        if value is not None and SUBJECT_PLACEHOLDER in str(value):
            template_attr = attr
            break
    if template_attr is None:
        raise ValueError(
            "--subject-ids given but neither input.folder nor input.volume "
            f"contains a {SUBJECT_PLACEHOLDER!r} placeholder to substitute into."
        )

    template = str(getattr(args, template_attr))
    other_attr = "volume" if template_attr == "folder" else "folder"

    resolved: List[Tuple[Path, str]] = []
    for sid in subject_ids:
        sub_args = copy.copy(args)
        setattr(sub_args, template_attr, Path(template.format(subject_id=sid)))
        setattr(sub_args, other_attr, None)

        subject_inputs = _collect_inputs(sub_args)
        if not subject_inputs:
            logger.warning("Subject %s: no volumes found (resolved path had none)", sid)
        resolved.extend((p, sid) for p in subject_inputs)
    return resolved


def _resolve_out_dir(out_dir: Path, path: Path, subject_id: Optional[str]) -> Path:
    """Per-volume output folder: substitute {subject_id} / {scan_dir} if the
    config's output.folder uses them. With --subject-ids but no placeholder at
    all, fall back to a <subject_id>/ subfolder so subjects never collide."""
    template = str(out_dir)
    has_placeholder = SUBJECT_PLACEHOLDER in template or SCAN_DIR_PLACEHOLDER in template
    resolved = Path(
        template.replace(SUBJECT_PLACEHOLDER, subject_id or "")
        .replace(SCAN_DIR_PLACEHOLDER, path.parent.name)
    )
    if subject_id and not has_placeholder:
        return resolved / subject_id
    return resolved


def _preprocess(path: Path, header: dict, data: np.ndarray, args: Config) -> tuple:
    """Stage 0-1: probe + geometric preprocessing, optional gt affine.

    Returns ``(volume, out_header, shift_vox)`` on the canonical grid. ``shift_vox``
    is how far the content was moved by ``preprocessing.centering`` (grid voxels);
    subtract it from predicted coordinates to get back to the raw scan's frame.
    """
    spacing = read_spacing(header)
    probe = probe_volume(path)
    logger.info(
        "%-14s in=%s spacing=%.3f FOV=%s%s",
        path.name,
        "x".join(map(str, probe.shape)),
        spacing[0],
        list(probe.fov_mm),
        "  [WILL CROP: FOV exceeds box]" if probe.exceeds_box else "",
    )

    pad, shift_mm = None, np.zeros(3)
    if args.centering:
        if args.gt_pose_path and args.affine:
            logger.warning("  centering skipped: gt affine already positions the volume")
        else:
            pad, shift_mm = centering_pad(data.shape, spacing, args.target_spacing, args.target_size)
            logger.info("  centering: pad %s vox (shift %s mm)", pad, np.round(shift_mm, 1).tolist())

    prepared = preprocess_geometry(
        data,
        spacing,
        do_filter=args.do_filter,
        do_bspline=args.do_bspline,
        filter_size=args.filter_size,
        filter_mode=args.filter_mode,
        out_spacing=args.target_spacing,
        out_size=args.target_size,
        pad_before=pad,
    )
    shift_vox = shift_mm / np.asarray(args.target_spacing, dtype=float)

    if args.gt_pose_path and args.affine:
        try:
            gt_dict = json.loads(Path(args.gt_pose_path).read_text())
            scan_name = path.stem
            if scan_name in gt_dict:
                pose_data = gt_dict[scan_name]
                if {"sagittal", "coronal", "axial"} <= pose_data.keys():
                    rotation = affine3Dmatrix(pose_data)
                else:
                    rotation = np.eye(3)
                if "center" in pose_data:
                    center_mm = np.array(pose_data["center"])
                    center_voxel = np.array([-center_mm[0], -center_mm[1], center_mm[2]])
                    translation = center_voxel - np.array([64.0, 64.0, 64.0])
                else:
                    translation = np.zeros(3)
                logger.info("  applying gt.txt pose for %s", scan_name)
                prepared = affine_transform_scipy(prepared, rotation, translation)
            else:
                logger.info("  no pose found in gt.txt for %s (skipping affine)", scan_name)
        except Exception as exc:  # noqa: BLE001 - affine is best-effort, never fatal
            logger.error("  affine failed for %s: %s (continuing without it)", path.name, exc)

    out_header = build_output_header(header, prepared.shape, args.target_spacing)
    return prepared, out_header, shift_vox


def process_one(path: Path, bundle, args: Config, out_dir: Path) -> bool:
    """Run Stage 0-2 for one volume and write its landmarks. Returns success."""
    name = path.stem
    lmk_dir = out_dir / args.landmarks_subfolder
    out_stem = str(lmk_dir / name)
    if Path(out_stem + ".fcsv").exists() and not args.overwrite:
        logger.info("skip (exists): %s", name)
        return True

    data, header = load_volume(path)

    if args.already_preprocessed:
        volume, out_header, spacing, shift_vox = data, header, args.target_spacing, np.zeros(3)
    else:
        volume, out_header, shift_vox = _preprocess(path, header, data, args)
        spacing = args.target_spacing

    lmk_dir.mkdir(parents=True, exist_ok=True)

    if args.save_preprocessed and not args.already_preprocessed:
        pre_path = out_dir / f"{name}.nrrd"
        if pre_path.resolve() == path.resolve():
            pre_path = out_dir / f"{name}_preprocessed.nrrd"  # never overwrite the raw input
        nrrd.write(str(pre_path), volume, header=out_header)

    if args.swap_xz:
        # best.pt expects x<->z swapped input; swap its heatmaps back to the scan's frame
        heatmaps = predict_heatmaps(bundle, np.ascontiguousarray(volume.transpose(2, 1, 0)))
        heatmaps = heatmaps.permute(0, 3, 2, 1)
    else:
        heatmaps = predict_heatmaps(bundle, volume)
    if args.lr_permutation is not None:
        heatmaps = heatmaps[args.lr_permutation]  # landmarks.swap_LR: rename left <-> right
    coords = get_peak_location(heatmaps, method=args.detector).cpu().numpy() - shift_vox

    save_fscv_csv(
        out=out_stem,
        coords=coords,
        selected_lmks=args.landmark_names,
        spacing=spacing,
    )
    logger.info("  -> wrote %s.fcsv / .csv (%d landmarks)", out_stem, len(args.landmark_names))

    if args.save_heatmaps:
        heatmap_header = build_output_header(out_header, volume.shape, spacing)
        probs = torch.sigmoid(heatmaps).cpu().numpy()
        heatmap_dir = out_dir / "heatmaps"
        heatmap_dir.mkdir(exist_ok=True)
        for i, lmk in enumerate(args.landmark_names):
            nrrd.write(str(heatmap_dir / f"{name}_{lmk}_heatmap.nrrd"), probs[i], header=heatmap_header)

    return True


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="infer_landmarks",
        description="Stage 0+1+2: raw .nrrd -> facial landmark coordinates.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("config", type=Path, help="YAML config file (single source of truth).")
    p.add_argument(
        "--already-preprocessed",
        action="store_true",
        help="Override config: skip Stage 1, input is already on the canonical grid.",
    )
    p.add_argument(
        "--subject-ids",
        nargs="+",
        default=None,
        metavar="ID",
        help=(
            "Run once per subject ID, substituting each into the "
            f"{SUBJECT_PLACEHOLDER!r} placeholder in input.folder (or input.volume). "
            "Output goes to output.folder (placeholders substituted), or to "
            "output.folder/<subject_id>/ if it has none. Optional - "
            "omit to use input.folder/volume literally, as a plain path."
        ),
    )
    return p


def main(argv=None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parsed = build_parser().parse_args(argv)

    logger.info("Loading config from %s", parsed.config)
    args = load_config_from_yaml(parsed.config)
    if parsed.already_preprocessed:
        args.already_preprocessed = True

    if not args.volume and not args.folder:
        logger.error("Must provide input.volume or input.folder in config")
        return 1

    if parsed.subject_ids:
        try:
            resolved_inputs = _expand_subject_ids(args, parsed.subject_ids)
        except ValueError as exc:
            logger.error(str(exc))
            return 1
        if not resolved_inputs:
            logger.error("No input volumes found for any subject id.")
            return 1
    else:
        if SUBJECT_PLACEHOLDER in str(args.out_dir):
            logger.error(
                "output.folder contains %r but --subject-ids was not given - "
                "pass --subject-ids or remove the placeholder from output.folder.",
                SUBJECT_PLACEHOLDER,
            )
            return 1
        plain_inputs = _collect_inputs(args)
        if not plain_inputs:
            logger.error("No input volumes found.")
            return 1
        resolved_inputs = [(p, None) for p in plain_inputs]

    logger.info("Loading checkpoint from %s (device=%s)", args.checkpoint, args.device)
    bundle = load_model(args.checkpoint, device=args.device)
    logger.info(
        "Model ready: base_features=%d, n_landmarks=%d, epoch=%s, val_loss=%s",
        bundle.base_features, bundle.n_landmarks, bundle.epoch, bundle.val_loss,
    )
    if bundle.n_landmarks != len(args.landmark_names):
        logger.error(
            "Checkpoint has %d output landmarks but config lists %d names - fix landmarks.names.",
            bundle.n_landmarks, len(args.landmark_names),
        )
        return 1
    if args.swap_lr:
        pairs = [(args.landmark_names[i], args.landmark_names[j])
                 for i, j in enumerate(args.lr_permutation) if i < j]
        if not pairs:
            logger.warning("landmarks.swap_LR is on but no L/R name pairs were found - nothing to swap")
        else:
            logger.info("landmarks.swap_LR: swapping %d pairs %s", len(pairs), pairs)

    if parsed.subject_ids:
        n_subjects = len(set(sid for _, sid in resolved_inputs))
        logger.info("Running inference on %d volume(s) across %d subject(s)...", len(resolved_inputs), n_subjects)
    else:
        logger.info("Running inference on %d volume(s)...", len(resolved_inputs))

    jobs = [(path, subject_id, _resolve_out_dir(args.out_dir, path, subject_id))
            for path, subject_id in resolved_inputs]
    seen = {}
    for path, _, out_dir in jobs:
        key = (out_dir / path.stem).resolve()
        if key in seen:
            logger.error("Output collision: %s and %s both write %s - add {subject_id} to output.folder.",
                         seen[key], path, out_dir / path.stem)
            return 1
        seen[key] = path

    ok = 0
    for path, subject_id, out_dir in jobs:
        try:
            ok += bool(process_one(path, bundle, args, out_dir))
        except Exception as exc:  # noqa: BLE001 - one bad file shouldn't stop the batch
            tag = f" (subject {subject_id})" if subject_id else ""
            logger.error("Failed on %s%s: %s", path.name, tag, exc)

    logger.info("Done: %d/%d volume(s) -> %s", ok, len(resolved_inputs), args.out_dir)
    return 0 if ok == len(resolved_inputs) else 1


if __name__ == "__main__":
    raise SystemExit(main())
