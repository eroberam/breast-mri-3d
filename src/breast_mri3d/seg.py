"""Export and verify DICOM segmentation on the original source frames."""

from __future__ import annotations

from pathlib import Path

import highdicom as hd
import numpy as np
from highdicom.sr import CodedConcept
from pydicom.uid import ExplicitVRLittleEndian, generate_uid

from . import __version__
from .dicom import MRSeries
from .labels import DISPLAY, REFERENCE_LABELS


def _algorithm(origin: dict) -> tuple[hd.seg.SegmentAlgorithmTypeValues, hd.AlgorithmIdentificationSequence | None]:
    family = CodedConcept("external", "99BMRI3D", "External Segmentation")
    if origin.get("type") == "research_model":
        kind, name, version = "AUTOMATIC", "Breast MRI 3D residual network", __version__
        family = CodedConcept("123110", "DCM", "Artificial Intelligence")
    elif origin.get("type") == "synthetic_phantom":
        kind, name, version = "AUTOMATIC", "Synthetic phantom generator", __version__
        family = CodedConcept("synthetic", "99BMRI3D", "Synthetic Geometry")
    else:
        kind, name, version = (origin.get(key) for key in ("algorithm_type", "algorithm_name", "algorithm_version"))
    algorithm_type = hd.seg.SegmentAlgorithmTypeValues(kind)
    if algorithm_type == hd.seg.SegmentAlgorithmTypeValues.MANUAL:
        return algorithm_type, None
    if not name or not version:
        raise ValueError("Supply the originating algorithm name and version for non-manual masks.")
    return algorithm_type, hd.AlgorithmIdentificationSequence(name=name, family=family, version=version)


def _description(number: int, label: str, algorithm: tuple) -> hd.seg.SegmentDescription:
    is_lesion = label == "lesion"
    category = CodedConcept("49755003", "SCT", "Morphologically Abnormal Structure") if is_lesion else CodedConcept(
        "91723000", "SCT", "Anatomical Structure"
    )
    return hd.seg.SegmentDescription(
        segment_number=number, segment_label=DISPLAY[label][0], segmented_property_category=category,
        segmented_property_type=CodedConcept(label, "99BMRI3D", DISPLAY[label][0]),
        algorithm_type=algorithm[0], algorithm_identification=algorithm[1],
    )


def export_seg(path: Path, source: MRSeries, masks: dict[str, np.ndarray], *, origin: dict) -> dict[str, int]:
    names = [name for name in REFERENCE_LABELS if name in masks]
    pixels = np.stack([masks[name] for name in names], axis=-1).astype(np.uint8)
    algorithm = _algorithm(origin)
    seg = hd.seg.Segmentation(
        source_images=list(source.headers), pixel_array=pixels,
        segmentation_type=hd.seg.SegmentationTypeValues.BINARY,
        segment_descriptions=[_description(i, name, algorithm) for i, name in enumerate(names, start=1)],
        series_instance_uid=generate_uid(), series_number=700, sop_instance_uid=generate_uid(),
        instance_number=1, manufacturer="Breast MRI 3D research", manufacturer_model_name="breast-mri-3d",
        software_versions=__version__, device_serial_number="research", series_description="Research segmentation",
        transfer_syntax_uid=ExplicitVRLittleEndian, omit_empty_frames=False,
    )
    seg.save_as(path, enforce_file_format=True)
    readback = hd.seg.segread(path)
    uids = [str(ds.SOPInstanceUID) for ds in source.headers]
    reopened = readback.get_pixels_by_source_instance(uids, segment_numbers=list(range(1, len(names) + 1)))
    if not np.array_equal(pixels, reopened):
        raise ValueError("The reloaded DICOM segmentation differs on its source images.")
    if str(readback.FrameOfReferenceUID) != source.frame_uid:
        raise ValueError("The reloaded DICOM segmentation lost the patient frame of reference.")
    return {name: i for i, name in enumerate(names, start=1)}
