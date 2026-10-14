"""Stage 0 - probe raw ``.nrrd`` volumes.

Domain logic for inspecting a folder of volumes *before* any checkpoint or
annotation exists. Answers the questions ``INFERENCE_PLAN.md`` Stage 0 asks:
what shape / spacing / field-of-view are these, will the 128 mm cube crop them,
and does a manual pose annotation already exist in ``gt.txt``?

Torch-free by design (imports only :mod:`inference.nrrd_io`).
"""
from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np

from inference.src.nrrd_io import is_4d, load_volume, physical_fov, read_spacing

logger = logging.getLogger(__name__)

# Default edge of the fixed preprocessing cube (mm). FOV beyond this gets cropped.
DEFAULT_BOX_MM: float = 128.0


@dataclass(frozen=True)
class VolumeProbe:
    """Everything Stage 0 measures about one volume."""

    name: str
    path: str
    shape: Tuple[int, ...]
    dtype: str
    is_4d: bool
    spacing: Tuple[float, float, float]
    fov_mm: Tuple[float, float, float]
    intensity_min: float
    intensity_max: float
    in_gt: bool  # filename in gt.txt
    exceeds_box: bool  # any FOV axis > box_mm -> the B-spline step will crop it

    def to_dict(self) -> dict:
        return asdict(self)


def probe_volume(
    path,
    gt_keys: Optional[set] = None,
    box_mm: float = DEFAULT_BOX_MM,
) -> VolumeProbe:
    """Measure a single ``.nrrd`` file."""
    path = Path(path)
    data, header = load_volume(path)
    spacing = read_spacing(header)
    fov = physical_fov(data.shape, spacing)

    stem = path.stem
    in_gt = stem in gt_keys if gt_keys else False

    return VolumeProbe(
        name=stem,
        path=str(path),
        shape=tuple(int(s) for s in data.shape),
        dtype=str(data.dtype),
        is_4d=is_4d(header),
        spacing=tuple(round(float(s), 4) for s in spacing),
        fov_mm=tuple(round(float(s), 1) for s in fov),
        intensity_min=float(data.min()),
        intensity_max=float(data.max()),
        in_gt=in_gt,
        exceeds_box=bool(np.any(fov > box_mm)),
    )


def probe_folder(
    folder,
    gt_keys: Optional[set] = None,
    box_mm: float = DEFAULT_BOX_MM,
    pattern: str = "*.nrrd",
) -> List[VolumeProbe]:
    """Probe every matching volume in ``folder`` (sorted). Bad files are skipped."""
    probes: List[VolumeProbe] = []
    for path in sorted(Path(folder).glob(pattern)):
        try:
            probes.append(probe_volume(path, gt_keys=gt_keys, box_mm=box_mm))
        except Exception as exc:  # noqa: BLE001 - one bad file shouldn't stop the run
            logger.error("Failed to probe %s: %s", path.name, exc)
    return probes


def render_table(probes: Sequence[VolumeProbe]) -> str:
    """Format probes as an aligned, human-readable table."""
    if not probes:
        return "(no volumes found)"

    headers = ["name", "shape", "spacing", "FOV mm (x,y,z)", "min/max", "GT", "box"]
    rows = []
    for p in probes:
        rows.append(
            [
                p.name,
                "x".join(str(s) for s in p.shape),
                f"{p.spacing[0]:.3f}",
                ", ".join(f"{v:.1f}" for v in p.fov_mm),
                f"{p.intensity_min:.0f}/{p.intensity_max:.0f}",
                "yes" if p.in_gt else "-",
                "CROP" if p.exceeds_box else "pad",
            ]
        )

    widths = [max(len(h), *(len(r[i]) for r in rows)) for i, h in enumerate(headers)]
    fmt = "  ".join(f"{{:<{w}}}" for w in widths)
    lines = [fmt.format(*headers), fmt.format(*("-" * w for w in widths))]
    lines += [fmt.format(*r) for r in rows]
    return "\n".join(lines)


def summarize(probes: Sequence[VolumeProbe], box_mm: float = DEFAULT_BOX_MM) -> str:
    """One-paragraph conclusion keyed to the Stage-0 decision table."""
    if not probes:
        return "No volumes probed."

    n = len(probes)
    in_gt = sum(p.in_gt for p in probes)
    cropped = sum(p.exceeds_box for p in probes)
    four_d = sum(p.is_4d for p in probes)

    parts = [f"{n} volume(s) probed."]
    if in_gt:
        parts.append(
            f"{in_gt} already in gt.txt -> manual pose exists (Stage 3 Option A; the easy path)."
        )
    else:
        parts.append(
            "None in gt.txt -> no manual pose; pose standardisation is Stage 3 Option B/C/D."
        )
    if cropped:
        parts.append(
            f"{cropped} exceed the {box_mm:.0f} mm cube -> the B-spline step WILL crop them; "
            "check the face is inside the box."
        )
    else:
        parts.append(f"All within the {box_mm:.0f} mm cube -> zero-padded, no cropping.")
    if four_d:
        parts.append(f"{four_d} are 4D (channel dropped to [...,0]).")
    return " ".join(parts)
