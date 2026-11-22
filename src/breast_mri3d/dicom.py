"""Read explicitly selected classic, regular single-frame MR series."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import hashlib
from pathlib import Path

import numpy as np
import pydicom
from pydicom.dataset import Dataset
from pydicom.errors import InvalidDicomError
from pydicom.pixels import apply_modality_lut
from pydicom.uid import MRImageStorage

from .geometry import Grid


@dataclass(frozen=True)
class MRSeries:
    grid: Grid
    headers: tuple[Dataset, ...]
    files: tuple[Path, ...]

    @property
    def uid(self) -> str:
        return str(self.headers[0].SeriesInstanceUID)

    @property
    def frame_uid(self) -> str:
        return str(self.headers[0].FrameOfReferenceUID)

    @property
    def source_instances_sha256(self) -> str:
        uids = "\n".join(str(ds.SOPInstanceUID) for ds in self.headers)
        return hashlib.sha256(uids.encode("ascii")).hexdigest()

    def pixels(self) -> np.ndarray:
        planes = []
        for path in self.files:
            ds = pydicom.dcmread(path)
            planes.append(apply_modality_lut(ds.pixel_array, ds).astype(np.float32))
        image = np.stack(planes)
        if image.shape != self.grid.shape or not np.isfinite(image).all():
            raise ValueError("Decoded MR pixels have invalid dimensions or values.")
        return image


def scan_series(root: Path) -> dict[str, list[tuple[Path, Dataset]]]:
    root = Path(root)
    if not root.is_dir():
        raise ValueError(f"DICOM directory does not exist: {root}")
    series = defaultdict(list)
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in {".dcm", "", ".ima"}:
            continue
        try:
            header = pydicom.dcmread(path, stop_before_pixels=True)
        except InvalidDicomError:
            continue
        if header.get("SOPClassUID") == MRImageStorage and header.get("SeriesInstanceUID"):
            series[str(header.SeriesInstanceUID)].append((path, header))
    acquisitions = {}
    for uid, entries in series.items():
        phases = split_repeated_positions(entries)
        if len(phases) == 1:
            acquisitions[uid] = phases[0]
        else:
            acquisitions.update({f"{uid}#{i}": items for i, items in enumerate(phases, start=1)})
    return acquisitions


def split_repeated_positions(entries: list[tuple[Path, Dataset]]) -> list[list[tuple[Path, Dataset]]]:
    """Handle classic DCE series storing several timepoints under one series UID.

    The instance order at each repeated spatial position defines a selector, not
    a claim that any particular volume is precontrast or early postcontrast.
    """
    positions = defaultdict(list)
    for entry in entries:
        point = entry[1].get("ImagePositionPatient")
        if point is None:
            return [entries]
        positions[tuple(round(float(x), 4) for x in point)].append(entry)
    counts = {len(group) for group in positions.values()}
    if counts == {1}:
        return [entries]
    if len(positions) < 2 or len(counts) != 1:
        return [entries]
    try:
        order = [int(ds.InstanceNumber) for _, ds in entries]
    except (ValueError, AttributeError):
        return [entries]
    if len(set(order)) != len(order):
        return [entries]
    groups = [sorted(group, key=lambda entry: int(entry[1].InstanceNumber)) for group in positions.values()]
    timepoints = [list(items) for items in zip(*groups, strict=True)]
    # Partial/interleaved acquisitions may repeat positions while changing other plane geometry.
    # Only expose selectors that also satisfy the single-volume geometry checks.
    try:
        for group in timepoints:
            assemble(group)
    except ValueError:
        return [entries]
    return timepoints


def _numeric(ds: Dataset, keyword: str, length: int) -> np.ndarray:
    arr = np.asarray(ds.get(keyword, []), dtype=float)
    if arr.shape != (length,) or not np.isfinite(arr).all():
        raise ValueError(f"Missing or invalid DICOM {keyword}.")
    return arr


def assemble(entries: list[tuple[Path, Dataset]]) -> MRSeries:
    if len(entries) < 2:
        raise ValueError("A regular 3D MR series needs at least two source slices.")
    first = entries[0][1]
    orientation = _numeric(first, "ImageOrientationPatient", 6)
    columns, rows = orientation[:3], orientation[3:]
    normal = np.cross(columns, rows)
    if (not np.allclose([np.linalg.norm(columns), np.linalg.norm(rows)], 1, atol=1e-4)
            or abs(columns @ rows) > 1e-4):
        raise ValueError("The image plane axes are not orthonormal.")
    row_mm, col_mm = _numeric(first, "PixelSpacing", 2)
    if min(row_mm, col_mm) <= 0:
        raise ValueError("DICOM pixel spacing must be positive.")
    dimensions = (int(first.Rows), int(first.Columns))
    positioned = []
    uids = set()
    for path, ds in entries:
        for tag in ("StudyInstanceUID", "SeriesInstanceUID", "FrameOfReferenceUID"):
            if not first.get(tag) or ds.get(tag) != first.get(tag):
                raise ValueError(f"Source images have missing or inconsistent {tag}.")
        if (int(ds.Rows), int(ds.Columns)) != dimensions or not np.allclose(
            _numeric(ds, "ImageOrientationPatient", 6), orientation, atol=1e-4
        ) or not np.allclose(_numeric(ds, "PixelSpacing", 2), [row_mm, col_mm], atol=1e-4):
            raise ValueError("Source slices do not share one regular image plane.")
        uid = ds.get("SOPInstanceUID")
        if not uid or uid in uids:
            raise ValueError("Source instance UIDs must be present and unique.")
        uids.add(uid)
        pos = _numeric(ds, "ImagePositionPatient", 3)
        positioned.append((float(pos @ normal), pos, path, ds))
    positioned.sort(key=lambda item: item[0])
    positions = np.stack([item[1] for item in positioned])
    deltas = np.diff(positions, axis=0)
    interval = float(np.median(deltas @ normal))
    expected = positions[0] + np.arange(len(positions))[:, None] * interval * normal
    if (interval < 1e-3 or np.any(deltas @ normal < 1e-3)
            or not np.allclose(positions, expected, atol=0.025, rtol=0)):
        raise ValueError("Slice locations repeat, drift or have irregular physical spacing.")
    affine = np.eye(4)
    affine[:3, :3] = np.stack((interval * normal, row_mm * rows, col_mm * columns), axis=1)
    affine[:3, 3] = positions[0]
    return MRSeries(Grid((len(positions), *dimensions), affine),
                    tuple(x[3] for x in positioned), tuple(x[2] for x in positioned))


def select(root: Path, uid: str | None = None) -> MRSeries:
    series = scan_series(root)
    if uid is None and len(series) == 1:
        uid = next(iter(series))
    if uid not in series:
        raise ValueError("Select a source acquisition explicitly. Use list-dicom to inspect the case.")
    return assemble(series[uid])
