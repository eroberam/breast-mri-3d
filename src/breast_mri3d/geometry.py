"""Array indices are always (slice, row, column). Patient coordinates are LPS mm."""

from __future__ import annotations

from dataclasses import dataclass

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
