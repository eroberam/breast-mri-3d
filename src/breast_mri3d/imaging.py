"""Coordinate-safe NIfTI I/O and paired DCE preparation."""

from __future__ import annotations

from pathlib import Path

import nibabel as nib
import numpy as np

from .dicom import MRSeries
from .geometry import Grid


_RAS_TO_LPS = np.diag([-1, -1, 1, 1])
_REVERSE_ARRAY_AXES = np.eye(4)[[2, 1, 0, 3]]


def read_nifti(path: Path, *, time_index: int | None = None,
               source_grid: Grid | None = None) -> tuple[np.ndarray, Grid]:
    image = nib.load(str(path))
    units = image.header.get_xyzt_units()[0]
    if units != "mm" and (units != "unknown" or source_grid is None):
        raise ValueError(f"NIfTI must define a 3D grid in millimetres: {path.name}")
    if time_index is None and len(image.shape) == 3:
        array = np.asanyarray(image.dataobj)
    elif time_index is not None and len(image.shape) == 4 and 0 <= time_index < image.shape[3]:
        array = np.asanyarray(image.dataobj[..., time_index])
    else:
        raise ValueError("Supply a 3D NIfTI or explicitly select a valid phase in a 4D NIfTI.")
    if not any((image.get_qform(coded=True)[1], image.get_sform(coded=True)[1])):
        raise ValueError(f"NIfTI has no declared physical transform: {path.name}")
    data = array.transpose(2, 1, 0)
    affine = _RAS_TO_LPS @ image.affine @ _REVERSE_ARRAY_AXES
    grid = Grid(data.shape, affine)
    if source_grid is not None and (grid.shape != source_grid.shape
                                   or not np.allclose(grid.affine, source_grid.affine, atol=.001, rtol=0)):
        raise ValueError("The NIfTI does not match the physical coordinates of its verified source image.")
    return data, grid


def write_nifti(path: Path, array: np.ndarray, grid: Grid) -> None:
    if array.shape != grid.shape:
        raise ValueError("NIfTI array and grid have different shapes.")
    affine = _RAS_TO_LPS @ grid.affine @ _REVERSE_ARRAY_AXES
    image = nib.Nifti1Image(array.transpose(2, 1, 0), affine)
    image.header.set_xyzt_units("mm")
    image.set_sform(affine, code="scanner")
    nib.save(image, str(path))


def dce_channels(pre: np.ndarray, post: np.ndarray) -> np.ndarray:
    if pre.shape != post.shape or not np.isfinite(pre).all() or not np.isfinite(post).all():
        raise ValueError("The two DCE phases must share a finite, aligned grid.")
    foreground = pre[pre > 0]
    if foreground.size < 16:
        raise ValueError("Precontrast series has too few nonzero voxels to normalize.")
    low, high = np.percentile(foreground, (2, 98))
    if high <= low:
        raise ValueError("Precontrast intensities do not define a useful dynamic range.")
    pre_norm = np.clip((pre - low) / (high - low), -1, 5)
    post_norm = np.clip((post - low) / (high - low), -1, 5)
    return np.stack((pre_norm, post_norm, post_norm - pre_norm)).astype(np.float32)


def prepare_dce(pre: MRSeries, post: MRSeries) -> np.ndarray:
    """Regrid using patient coordinates. The caller confirms registration and phase order."""
    if pre.headers[0].StudyInstanceUID != post.headers[0].StudyInstanceUID:
        raise ValueError("Precontrast and postcontrast must come from the same study.")
    pre_instances = {ds.SOPInstanceUID for ds in pre.headers}
    if (pre.frame_uid != post.frame_uid
            or pre_instances.intersection(ds.SOPInstanceUID for ds in post.headers)):
        raise ValueError("Supply two distinct phases in a shared patient frame of reference.")
    pre_aligned = pre.grid.resample(pre.pixels(), post.grid, order=1)
    return dce_channels(pre_aligned, post.pixels())
