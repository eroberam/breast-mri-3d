"""Array indices are always (slice, row, column). Patient coordinates are LPS mm."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import permutations, product

import numpy as np
from scipy.ndimage import affine_transform


@dataclass(frozen=True)
class Grid:
    shape: tuple[int, int, int]
    affine: np.ndarray

    def __post_init__(self) -> None:
        shape = tuple(self.shape)
        affine = np.array(self.affine, dtype=np.float64, copy=True)
        if len(shape) != 3 or any(n < 1 or n != int(n) for n in shape):
            raise ValueError("A grid requires three positive array dimensions.")
        if (affine.shape != (4, 4) or not np.isfinite(affine).all()
                or not np.array_equal(affine[3], [0, 0, 0, 1])
                or abs(np.linalg.det(affine[:3, :3])) < 1e-9):
            raise ValueError("Expected an invertible index-to-LPS affine in millimetres.")
        affine.setflags(write=False)
        object.__setattr__(self, "shape", tuple(int(n) for n in shape))
        object.__setattr__(self, "affine", affine)

    @property
    def voxel_mm3(self) -> float:
        return float(abs(np.linalg.det(self.affine[:3, :3])))

    @property
    def pixel_mm2(self) -> float:
        return float(np.linalg.norm(np.cross(self.affine[:3, 1], self.affine[:3, 2])))

    @property
    def spacing(self) -> np.ndarray:
        return np.linalg.norm(self.affine[:3, :3], axis=0)

    def points(self, indices: np.ndarray) -> np.ndarray:
        return np.asarray(indices) @ self.affine[:3, :3].T + self.affine[:3, 3]

    def resample(self, array: np.ndarray, target: Grid, *, order: int) -> np.ndarray:
        array = np.asarray(array)
        if array.shape != self.shape:
            raise ValueError("Volume dimensions do not agree with their source grid.")
        if order not in (0, 1):
            raise ValueError("Interpolation order must be 0 (labels) or 1 (continuous).")
        if self.shape == target.shape and np.allclose(self.affine, target.affine, atol=1e-6):
            return array.copy()
        pull = np.linalg.solve(self.affine, target.affine)
        axes = tuple(int(n) for n in np.argmax(abs(pull[:3, :3]), axis=0))
        if len(set(axes)) == 3 and tuple(self.shape[a] for a in axes) == target.shape:
            exact = np.eye(4)
            exact[:3, :3] = 0
            flips = []
            for col, row in enumerate(axes):
                reverse = pull[row, col] < 0
                exact[row, col] = -1 if reverse else 1
                if reverse:
                    exact[row, 3] = self.shape[row] - 1
                    flips.append(col)
            if np.allclose(pull, exact, atol=1e-7, rtol=0):
                return np.flip(array.transpose(axes), tuple(flips)).copy()
        return affine_transform(array, pull[:3, :3], offset=pull[:3, 3],
                                output_shape=target.shape, order=order, prefilter=False,
                                mode="constant", cval=0)

    def as_dict(self) -> dict:
        return {"shape_zyx": list(self.shape), "index_zyx_to_lps_mm": self.affine.tolist()}

    @classmethod
    def from_dict(cls, record: dict) -> Grid:
        return cls(tuple(record["shape_zyx"]), record["index_zyx_to_lps_mm"])


def model_grid(source: Grid, minimum_spacing=(1.8, 1.0, 1.0)) -> Grid:
    """Keep the field of view and orientation while bounding downsampled resolution."""
    step = np.maximum(source.spacing, minimum_spacing)
    extent = (np.asarray(source.shape) - 1) * source.spacing
    intervals = np.ceil(extent / step).astype(int)
    actual = np.divide(extent, intervals, out=step.copy(), where=intervals != 0)
    affine = source.affine.copy()
    affine[:3, :3] *= actual / source.spacing
    return Grid(tuple(intervals + 1), affine)


def reorient_lps(array: np.ndarray, source: Grid) -> tuple[np.ndarray, Grid]:
    """Order the last axes along superior, posterior and left without interpolation."""
    if array.shape[-3:] != source.shape:
        raise ValueError("Volume dimensions do not agree with their source grid.")
    direction = source.affine[:3, :3] / source.spacing
    world_axes = (2, 1, 0)
    axes = max(permutations(range(3)),
               key=lambda order: sum(abs(direction[w, i]) for w, i in zip(world_axes, order)))
    transform = np.eye(4)
    transform[:3, :3] = 0
    flips = []
    prefix = array.ndim - 3
    for axis, (world, original) in enumerate(zip(world_axes, axes)):
        reverse = direction[world, original] < 0
        transform[original, axis] = -1 if reverse else 1
        if reverse:
            flips.append(prefix + axis)
            transform[original, 3] = source.shape[original] - 1
    view = array.transpose(*range(prefix), *(prefix + i for i in axes))
    return np.flip(view, tuple(flips)), Grid(view.shape[-3:], source.affine @ transform)


def lps_grid(source: Grid, spacing_mm: float = 1.5) -> Grid:
    """Build an orthogonal patient-oriented grid enclosing the source voxel centres."""
    if not np.isfinite(spacing_mm) or spacing_mm <= 0:
        raise ValueError("Model spacing must be positive and finite.")
    corners = source.points(np.asarray(list(product(*[(0, n - 1) for n in source.shape]))))
    minimum, maximum = corners.min(axis=0), corners.max(axis=0)
    shape = tuple(np.ceil((maximum - minimum)[::-1] / spacing_mm - 1e-9).astype(int) + 1)
    affine = np.eye(4)
    affine[:3, :3] = np.eye(3)[:, ::-1] * spacing_mm
    affine[:3, 3] = minimum
    return Grid(shape, affine)
