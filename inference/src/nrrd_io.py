"""Pure ``.nrrd`` and ``gt.txt`` helpers.

Intentionally dependency-light: ``numpy`` + ``pynrrd`` only, no torch, no
SimpleITK. This keeps Stage 0 (probing) runnable without the heavy stack.

The spacing and 4D-collapse behaviour here mirror the training pipeline exactly
(see the docstrings) so that what we measure and preprocess is what the network
actually consumed.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import nrrd

PathLike = "os.PathLike | str"


def load_volume(path) -> Tuple[np.ndarray, dict]:
    """Read a ``.nrrd`` volume, collapsing a trailing 4th channel to 3D.

    Mirrors ``dataset/utility/rotation.py:extract_image`` (the 4D drop at
    ``rotation.py:36``) but without importing torch, so the probe stays light.

    Returns the raw array (native dtype, unchanged) and the pynrrd header dict.
    """
    data, header = nrrd.read(str(path))
    if data.ndim == 4:
        data = data[:, :, :, 0]
    return data, header


def read_spacing(header: dict) -> np.ndarray:
    """Voxel spacing ``(x, y, z)`` in mm, read exactly as the pipeline reads it.

    Primary: ``header['spacings'][:3]``. Fallback: the diagonal of
    ``header['space directions']``. Source: ``dataset/preprocess.py:90-95`` and
    ``dataset/MyDataset.py:90-94``.
    """
    try:
        return np.asarray(header["spacings"][:3], dtype=float)
    except Exception:  # noqa: BLE001 - mirror the pipeline's broad fallback
        sd = np.asarray(header["space directions"], dtype=float)
        return np.asarray([sd[0, 0], sd[1, 1], sd[2, 2]], dtype=float)


def physical_fov(shape, spacing) -> np.ndarray:
    """Physical field of view in mm per axis: ``shape * spacing``."""
    return np.asarray(shape, dtype=float) * np.asarray(spacing, dtype=float)


def is_4d(header: dict) -> bool:
    """Whether the on-disk volume carried a 4th (channel) dimension."""
    dim = header.get("dimension")
    if dim is not None:
        return int(dim) == 4
    sizes = header.get("sizes")
    return sizes is not None and len(sizes) == 4


def load_gt_keys(path) -> set:
    """Load gt.txt keys (scan names that have ground truth poses).

    Raises ``FileNotFoundError`` if missing; caller decides if fatal.
    """
    raw = Path(path).read_text()
    mapping = json.loads(raw) if raw.strip() else {}
    return set(mapping.keys())
