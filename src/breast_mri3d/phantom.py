"""Deterministic demonstration with synthetic images and known physical masks."""

from pathlib import Path

import numpy as np
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, MRImageStorage, generate_uid

from .dicom import select
from .export import export_case


def write_phantom(folder: Path, *, oblique=True) -> tuple[str, dict[str, np.ndarray]]:
    folder.mkdir(parents=True, exist_ok=False)
    study, series, frame = (generate_uid() for _ in range(3))
    depth, rows, columns = 24, 48, 64
    angle = np.pi / 7 if oblique else 0
    row_dir = np.array([0, 1., 0])
    col_dir = np.array([np.cos(angle), 0, np.sin(angle)])
    normal = np.cross(col_dir, row_dir)
    z, y, x = np.indices((depth, rows, columns))
    left = ((x - 45) / 13) ** 2 + ((y - 23) / 17) ** 2 + ((z - 12) / 10) ** 2 < 1
    right = ((x - 17) / 13) ** 2 + ((y - 23) / 17) ** 2 + ((z - 12) / 10) ** 2 < 1
    fgt = ((x - 44) / 8) ** 2 + ((y - 23) / 8) ** 2 + ((z - 12) / 5) ** 2 < 1
    lesion = ((x - 45) / 3) ** 2 + ((y - 23) / 4) ** 2 + ((z - 12) / 3) ** 2 < 1
    for i in range(depth):
        uid = generate_uid()
        path = folder / f"slice-{depth - i:03d}.dcm"
        file_meta = FileMetaDataset()
        file_meta.MediaStorageSOPClassUID = MRImageStorage
        file_meta.MediaStorageSOPInstanceUID = uid
        file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
        ds = FileDataset(path, {}, file_meta=file_meta, preamble=bytes(128))
        ds.SOPClassUID, ds.SOPInstanceUID = MRImageStorage, uid
        ds.StudyInstanceUID, ds.SeriesInstanceUID, ds.FrameOfReferenceUID = study, series, frame
        ds.PatientName, ds.PatientID = "Synthetic^Phantom", "SYNTHETIC-001"
        ds.PatientBirthDate, ds.PatientSex = "", ""
        ds.StudyDate, ds.StudyTime, ds.StudyID, ds.AccessionNumber = "20000101", "000000", "1", ""
        ds.ReferringPhysicianName = ""
        ds.Modality, ds.SeriesNumber, ds.InstanceNumber = "MR", 1, i + 1
        ds.SeriesDescription = "Synthetic oblique MR"
        ds.ImageOrientationPatient = [float(round(x, 9)) for x in (*col_dir, *row_dir)]
        ds.ImagePositionPatient = [float(round(x, 9)) for x in (np.array([40, -20, 12]) + normal * 2.1 * i)]
        ds.PixelSpacing, ds.SliceThickness = [0.8, 0.65], 1.5
        ds.Rows, ds.Columns = rows, columns
        ds.SamplesPerPixel, ds.PhotometricInterpretation = 1, "MONOCHROME2"
        ds.BitsAllocated, ds.BitsStored, ds.HighBit, ds.PixelRepresentation = 16, 16, 15, 0
        ds.PixelData = (20 + (left[i] | right[i]).astype("<u2") * 60 + lesion[i].astype("<u2") * 80).tobytes()
        ds.save_as(path, enforce_file_format=True)
    return series, {"left_breast": left, "right_breast": right, "fibroglandular": fgt, "lesion": lesion}


def demo(root: Path) -> Path:
    root = Path(root)
    root.mkdir(parents=True, exist_ok=False)
    uid, masks = write_phantom(root / "source")
    source = select(root / "source", uid)
    return export_case(root / "review", source, masks, mask_grid=source.grid, origin={
        "type": "synthetic_phantom", "series_instance_uid": uid, "frame_of_reference_uid": source.frame_uid,
        "source_instances_sha256": source.source_instances_sha256,
    })
