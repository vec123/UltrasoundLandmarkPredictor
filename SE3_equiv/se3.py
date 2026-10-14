"""Rigid (SE(3)) transforms of a volume on its own voxel grid, and of points.

Convention: points are in mm in the volume's array frame (voxel index * spacing,
the frame ``infer_landmarks`` writes its .csv/.fcsv in). A transform ``g`` acts as

    g(x) = R (x - c) + c + t

with ``c`` the rotation centre. The augmented volume is ``V'(x') = V(g^-1 x')``,
so anatomy at ``x`` in ``V`` sits at ``g(x)`` in ``V'``: an equivariant model
must predict ``g(f(V))`` on ``V'``.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np
from scipy import ndimage
from scipy.spatial.transform import Rotation

AXES = {"i": 0, "j": 1, "k": 2}  # array axes 0/1/2 of the input volume


@dataclass(frozen=True)
class RigidTransform:
    name: str
    family: str
    rotation: np.ndarray     # (3, 3), det +1
    translation: np.ndarray  # (3,) mm

    @property
    def angle_deg(self) -> float:
        return rotation_angle_deg(self.rotation)

    @property
    def shift_mm(self) -> float:
        return float(np.linalg.norm(self.translation))

    @property
    def rotvec_deg(self) -> np.ndarray:
        return np.degrees(Rotation.from_matrix(self.rotation).as_rotvec())

    def apply_points(self, points_mm: np.ndarray, center_mm: np.ndarray) -> np.ndarray:
        return (points_mm - center_mm) @ self.rotation.T + center_mm + self.translation


def rotation_angle_deg(rotation: np.ndarray) -> float:
    cos = (np.trace(rotation) - 1.0) / 2.0
    return float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))


def _axis_rotation(axis: str, angle_deg: float) -> np.ndarray:
    rotvec = np.zeros(3)
    rotvec[AXES[axis]] = np.radians(angle_deg)
    return Rotation.from_rotvec(rotvec).as_matrix()


# --- samplers ----------------------------------------------------------------

def identity() -> List[RigidTransform]:
    return [RigidTransform("identity", "identity", np.eye(3), np.zeros(3))]


def cube_rotations() -> List[RigidTransform]:
    """The 23 non-trivial rotations mapping a cube onto itself (90/120/180 deg).

    About a voxel-centred point they only permute voxels, so the augmented volume
    is exact - no interpolation blur mixed into the equivariance error.
    """
    out = []
    for m in np.rint(Rotation.create_group("O").as_matrix()):
        angle = rotation_angle_deg(m)
        if angle < 1e-6:
            continue
        axis = Rotation.from_matrix(m).as_rotvec()
        axis = np.rint(axis / np.abs(axis).max()).astype(int)
        label = "".join("{:+d}".format(a) for a in axis)
        out.append((angle, label, m))
    out.sort(key=lambda item: (item[0], item[1]))
    return [RigidTransform("cube_{:.0f}deg_ax{}".format(a, lab), "cube_rotation", m, np.zeros(3))
            for a, lab, m in out]


def axis_sweep(axes: Sequence[str], angles_deg: Sequence[float]) -> List[RigidTransform]:
    return [RigidTransform("rot_{}{:+g}deg".format(ax, a), "axis_sweep", _axis_rotation(ax, a), np.zeros(3))
            for ax in axes for a in angles_deg if a != 0]


def translations(axes: Sequence[str], shifts_mm: Sequence[float]) -> List[RigidTransform]:
    out = []
    for ax in axes:
        for s in shifts_mm:
            t = np.zeros(3)
            t[AXES[ax]] = s
            out.append(RigidTransform("shift_{}{:+g}mm".format(ax, s), "translation", np.eye(3), t))
    return out


def random_se3(n: int, max_angle_deg: float, max_shift_mm: float, seed: int = 0,
               angle_sampling: str = "uniform") -> List[RigidTransform]:
    """Random rotation + translation.

    ``angle_sampling``:

    - ``"uniform"``: uniform axis, angle uniform in ``[0, max_angle_deg]`` -
      small angles as common as large ones (error-vs-angle curves).
    - ``"haar"``: uniform over all rotations (unit quaternions), rejecting
      angles above ``max_angle_deg``. Small angles are rare: under 1% below 30
      degrees.

    Translation: uniform direction, length uniform in ``[0, max_shift_mm]``.
    """
    if angle_sampling not in ("uniform", "haar"):
        raise ValueError("angle_sampling must be 'uniform' or 'haar', got {!r}".format(angle_sampling))
    rng = np.random.default_rng(seed)
    out = []
    for k in range(int(n)):
        if angle_sampling == "haar":
            while True:
                q = rng.normal(size=4)
                rotation = Rotation.from_quat(q / np.linalg.norm(q)).as_matrix()
                if rotation_angle_deg(rotation) <= max_angle_deg:
                    break
        else:
            axis = rng.normal(size=3)
            axis /= np.linalg.norm(axis)
            rotation = Rotation.from_rotvec(axis * np.radians(rng.uniform(0, max_angle_deg))).as_matrix()
        direction = rng.normal(size=3)
        direction /= np.linalg.norm(direction)
        translation = direction * rng.uniform(0, max_shift_mm)
        out.append(RigidTransform("rand{:02d}".format(k), "random_se3", rotation, translation))
    return out


def build_transforms(cfg: dict) -> List[RigidTransform]:
    """Transform list from the ``transforms`` section of the config."""
    out: List[RigidTransform] = []
    if cfg.get("identity", True):
        out += identity()
    if cfg.get("cube_rotations", False):
        out += cube_rotations()
    if cfg.get("axis_sweep"):
        out += axis_sweep(cfg["axis_sweep"].get("axes", list(AXES)), cfg["axis_sweep"]["angles_deg"])
    if cfg.get("translations"):
        out += translations(cfg["translations"].get("axes", list(AXES)), cfg["translations"]["shifts_mm"])
    if cfg.get("random"):
        r = cfg["random"]
        out += random_se3(r.get("n", 10), r.get("max_angle_deg", 180), r.get("max_shift_mm", 0), r.get("seed", 0),
                          r.get("angle_sampling", "uniform"))
    return out


# --- application -------------------------------------------------------------

def resolve_center(center, reference_mm: np.ndarray, shape: Sequence[int], spacing: np.ndarray) -> np.ndarray:
    """Rotation centre in mm, snapped to a voxel centre so 90-degree rotations stay exact.

    ``"landmarks"``: centroid of the reference landmarks (keeps the face in the
    box); ``"volume"``: centre of the grid; or an explicit ``[x, y, z]`` in mm.
    """
    if center == "landmarks":
        c = reference_mm.mean(axis=0)
        return np.rint(c / spacing) * spacing
    if center == "volume":
        return (np.asarray(shape, dtype=float) - 1) / 2.0 * spacing
    return np.asarray(center, dtype=float)


def transform_volume(
    volume: np.ndarray,
    spacing: np.ndarray,
    transform: RigidTransform,
    center_mm: np.ndarray,
    order: int = 1,
) -> Tuple[np.ndarray, bool]:
    """Resample ``volume`` under ``transform`` on the same grid; outside = volume min.

    Returns ``(augmented, exact)``. ``exact`` is True when every output voxel maps
    onto an input voxel (90-degree rotations, whole-voxel shifts): the data is then
    permuted, not interpolated. Integer input keeps its dtype (rounded, clipped).
    """
    spacing = np.asarray(spacing, dtype=float)
    R, t, c = transform.rotation, transform.translation, np.asarray(center_mm, dtype=float)
    # output voxel o -> source index  S^-1 [R^T (S o - c - t) + c]
    matrix = (R.T * spacing[None, :]) / spacing[:, None]
    offset = (c - R.T @ (c + t)) / spacing
    cval = float(volume.min())

    exact = np.allclose(matrix, np.rint(matrix), atol=1e-6) and np.allclose(offset, np.rint(offset), atol=1e-6)
    if exact:
        out = ndimage.affine_transform(volume, np.rint(matrix), offset=np.rint(offset),
                                       order=0, mode="constant", cval=cval)
        return out, True

    out = ndimage.affine_transform(volume.astype(np.float32), matrix, offset=offset,
                                   order=order, mode="constant", cval=cval, prefilter=order > 1)
    if np.issubdtype(volume.dtype, np.integer):
        info = np.iinfo(volume.dtype)
        out = np.clip(np.rint(out), info.min, info.max)
    return out.astype(volume.dtype), False
