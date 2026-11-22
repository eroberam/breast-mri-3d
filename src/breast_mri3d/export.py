"""Create a complete case bundle only after its round-trip checks pass."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from scipy.ndimage import label as connected_components
from skimage.measure import find_contours

from . import __version__
from .dicom import MRSeries
from .geometry import Grid
from .imaging import write_nifti
from .labels import REFERENCE_LABELS
from .mesh import export_meshes
from .seg import export_seg


def export_case(destination: Path, source: MRSeries, masks: dict[str, np.ndarray], *,
                mask_grid: Grid, origin: dict) -> Path:
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError(f"Destination already exists: {destination}")
    if not masks or set(masks) - set(REFERENCE_LABELS):
        raise ValueError(f"Provide at least one mask with a known label: {', '.join(REFERENCE_LABELS)}.")
    if origin.get("frame_of_reference_uid") != source.frame_uid:
        raise ValueError("The prediction and reference series must identify the same patient frame.")
    if origin.get("series_instance_uid") != source.uid:
        raise ValueError("The prediction must reference the selected original MR series.")
    if origin.get("source_instances_sha256") != source.source_instances_sha256:
        raise ValueError("The prediction must reference the exact source phase instances.")
    native = {}
    for name, mask in masks.items():
        mask = np.asarray(mask)
        if not np.all((mask == 0) | (mask == 1)):
            raise ValueError(f"Use a thresholded binary mask for {name}.")
        converted = mask_grid.resample(mask.astype(np.uint8), source.grid, order=0).astype(bool)
        if mask.any() and not converted.any():
            raise ValueError(f"The {name} mask is outside the field of view of its DICOM source.")
        native[name] = converted

    destination.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=destination.parent, prefix=".bmri-export-") as tmp:
        root = Path(tmp) / "case"
        root.mkdir()
        label_ids = export_seg(root / "segmentation.dcm", source, native, origin=origin)
        contours = []
        measurements = []
        for name, mask in native.items():
            write_nifti(root / f"{name}.nii.gz", mask.astype(np.uint8), source.grid)
            slice_area = mask.sum(axis=(1, 2)).astype(float) * source.grid.pixel_mm2
            measurements.append({"structure": name, "voxels": int(mask.sum()),
                                 "volume_ml": float(mask.sum() * source.grid.voxel_mm3 / 1000),
                                 "max_slice_area_mm2": float(slice_area.max()),
                                 "components": int(connected_components(mask)[1])})
            for z in np.flatnonzero(slice_area):
                curves = find_contours(np.pad(mask[z], 1), .5)
                for curve in curves:
                    points = np.column_stack((np.full(len(curve), z), curve - 1))
                    contours.append({"structure": name, "slice": int(z),
                                     "source_sop_uid": str(source.headers[z].SOPInstanceUID),
                                     "lps_mm": source.grid.points(points).tolist()})
        geometry = export_meshes(root, native, source.grid)
        for entry in measurements:
            entry["surface_area_mm2"] = geometry["surface_area_mm2"].get(entry["structure"], 0.0)
        _json(root / "measurements.json", measurements)
        _json(root / "contours.json", contours)
        with (root / "measurements.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(measurements[0]))
            writer.writeheader()
            writer.writerows(measurements)
        _json(root / "manifest.json", {
            "schema_version": 1, "package_version": __version__, "purpose": "research",
            "prediction": origin, "native_grid": source.grid.as_dict(), "prediction_grid": mask_grid.as_dict(),
            "dicom_seg_segments": label_ids, "dicom_seg_round_trip_verified": True, **geometry,
        })
        root.rename(destination)
    return destination


def _json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
