"""Stage 2 - landmark prediction from a preprocessed volume.

Loads a trained ``ResUNet3D`` checkpoint, runs a forward pass on a canonical
(1 mm isotropic, 128^3) volume, and extracts one voxel-coordinate peak per
landmark heatmap.

Requires torch - this is the deep-learning stage of the pipeline (see
``inference/__init__.py`` and ``documentation/INFERENCE_PLAN.md`` Stage 2),
as opposed to :mod:`inference.src.geometry` / :mod:`inference.src.nrrd_io`,
which stay torch-free.

Peak extraction is a small, faithful reimplementation of
``net/postprocess/where_is_landmark.py`` rather than an import of it: that
module also drags in ``net.plot.matrices_3d`` (unused there, but it hard-imports
``imageio``), which would force a plotting dependency onto a prediction-only
tool. Same rationale as the reimplementations in ``inference/src/geometry.py``.
Only ``argmax`` and ``com`` are ported (``dsnt`` needs
``net.model.modules.dsnt`` and is rarely used - default detector is
``argmax``, see ``net/config/default.yaml``).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np
import torch
from torch.nn import functional as F

from net.model.backbone.ResUNet3D import ResUNet3D

# Order matters - must match the order the checkpoint was trained with.
# Default is `net/config/default.yaml` -> target.lmks; override via config if a
# given checkpoint's params.yaml records a different order.
DEFAULT_LANDMARK_NAMES: Sequence[str] = (
    "prn", "enL", "enR", "pg", "n", "sn", "sl", "li", "ls", "acL",
    "acR", "cphL", "cphR", "chR", "chL", "aR", "aL", "exL", "exR",
)


def left_right_permutation(names: Sequence[str]) -> List[int]:
    """Channel permutation that swaps every left/right pair by name.

    A pair is two names that differ only in a trailing ``L``/``R``
    (``enL``/``enR``, ``exL``/``exR``, ...); midline names map to themselves.
    Indexing the model's output channels with it renames each side's point to
    the other side - used because ``best.pt`` names the sides opposite to the
    Casos Mar annotators.
    """
    index = {name: i for i, name in enumerate(names)}
    perm = list(range(len(names)))
    for i, name in enumerate(names):
        partner = name[:-1] + "R"
        if name.endswith("L") and partner in index:
            j = index[partner]
            perm[i], perm[j] = j, i
    return perm


@dataclass(frozen=True)
class ModelBundle:
    """A loaded model plus the metadata used to reconstruct it."""

    model: torch.nn.Module
    device: torch.device
    base_features: int
    n_landmarks: int
    epoch: Optional[int]
    val_loss: Optional[float]


def load_model(
    checkpoint_path,
    device: str = "cpu",
    base_features: Optional[int] = None,
    n_landmarks: Optional[int] = None,
) -> ModelBundle:
    """Build a ``ResUNet3D`` and load a training checkpoint into it.

    Architecture is inferred from the checkpoint's own tensor shapes rather
    than trusted from a config file, since the normal training pipeline does
    not persist architecture params inside the checkpoint (only in a sibling
    ``params.yaml``, which may not travel with the ``.pt`` file) - see
    ``documentation/INFERENCE_PLAN.md`` Stage 2. Pass ``base_features`` /
    ``n_landmarks`` explicitly to override the inferred values.

    Loads the state dict directly rather than via
    ``net.model.utility.checkpoints.load_checkpoint``, which requires an
    optimizer that inference has no use for.
    """
    dev = torch.device(device)
    checkpoint = torch.load(checkpoint_path, map_location=dev, weights_only=False)
    state_dict = checkpoint["model_state_dict"]

    inferred_features = state_dict["initial_block.0.weight"].shape[0]
    inferred_landmarks = state_dict["output_layer.conv.weight"].shape[0]

    model = ResUNet3D(
        input_channels=1,
        output_channels=n_landmarks or inferred_landmarks,
        base_features=base_features or inferred_features,
    )
    model.load_state_dict(state_dict)
    model.to(dev)
    model.eval()

    return ModelBundle(
        model=model,
        device=dev,
        base_features=base_features or inferred_features,
        n_landmarks=n_landmarks or inferred_landmarks,
        epoch=checkpoint.get("epoch"),
        val_loss=checkpoint.get("val_loss"),
    )


def rescale_intensity(volume: np.ndarray) -> np.ndarray:
    """Min-max normalize to ``[0, 1]`` float32.

    Equivalent to ``torchio.RescaleIntensity((0, 1))`` with default (0, 100)
    percentiles, which is what the training pipeline applies before the
    forward pass (``fetusnet.py`` transforms, gated by
    ``training.rescale_inputs`` - default ``True``). Do not skip this: the
    network was never trained on raw-intensity input.
    """
    v = volume.astype(np.float32)
    v_min, v_max = float(v.min()), float(v.max())
    return (v - v_min) / (v_max - v_min + 1e-8)


def predict_heatmaps(bundle: ModelBundle, volume: np.ndarray) -> torch.Tensor:
    """Run the forward pass on one preprocessed volume.

    Args:
        bundle: A loaded model from :func:`load_model`.
        volume: 3D array on the canonical grid (any real dtype); intensity is
            rescaled internally via :func:`rescale_intensity`.

    Returns:
        Raw heatmap logits, shape ``(n_landmarks, D, H, W)``. Sigmoid is
        *not* applied - :func:`get_peak_location` applies it internally, as
        the training/eval loop does (``net/phases/infer.py``).
    """
    rescaled = rescale_intensity(volume)
    tensor = torch.from_numpy(rescaled)[None, None].to(bundle.device)  # (1, 1, D, H, W)
    with torch.inference_mode():
        heatmaps = bundle.model(tensor)[0]  # drop batch dim -> (L, D, H, W)
    return heatmaps


def _peak_argmax(probs: torch.Tensor, mean_multi_peak: bool) -> torch.Tensor:
    max_val = probs.max()
    peaks = torch.nonzero(probs == max_val, as_tuple=False).float()
    if peaks.numel() == 0:
        raise ValueError("No peaks found in the heatmap.")
    if peaks.shape[0] > 1 and mean_multi_peak:
        return peaks.mean(dim=0)
    return peaks[0].float()


def _peak_com(logits: torch.Tensor) -> torch.Tensor:
    """Soft center-of-mass over a *normalised* voxel distribution.

    The upstream ``com`` sums sigmoid values without normalising, which returns
    coordinates in the millions on a 128^3 volume. Softmax over the voxel domain
    is what the ``softmaxce`` training loss normalises with, so use it here.
    """
    d, h, w = logits.shape
    probs = F.softmax(logits.reshape(-1), dim=0).reshape(d, h, w)
    grid = torch.meshgrid(
        torch.arange(d, device=probs.device),
        torch.arange(h, device=probs.device),
        torch.arange(w, device=probs.device),
        indexing="ij",
    )
    grid = torch.stack(grid, dim=-1).float()
    weighted = probs.unsqueeze(-1) * grid
    return weighted.sum(dim=(0, 1, 2))


def get_peak_location(
    heatmaps: torch.Tensor,
    method: str = "argmax",
    mean_multi_peak: bool = True,
) -> torch.Tensor:
    """Extract one voxel-coordinate peak per landmark from raw heatmap logits.

    Applies sigmoid internally (mirrors
    ``net/postprocess/where_is_landmark.py:get_peak_location`` - do not
    pre-activate ``heatmaps``).

    Args:
        heatmaps: Raw logits, shape ``(n_landmarks, D, H, W)``.
        method: ``"argmax"`` (hard peak, ties averaged) or ``"com"`` (soft
            center-of-mass over the sigmoid-probability volume).
        mean_multi_peak: For ``"argmax"``, average tied maxima instead of
            taking the first.

    Returns:
        Tensor of shape ``(n_landmarks, 3)``, voxel coordinates.
    """
    if method not in ("argmax", "com"):
        raise ValueError(f"Unsupported method: {method!r}. Use 'argmax' or 'com'.")

    n_landmarks = heatmaps.shape[0]
    probs = torch.sigmoid(heatmaps)
    coords = torch.zeros((n_landmarks, 3), dtype=torch.float32)
    for i in range(n_landmarks):
        if method == "argmax":
            coords[i] = _peak_argmax(probs[i], mean_multi_peak)
        else:
            coords[i] = _peak_com(heatmaps[i])
    return coords
