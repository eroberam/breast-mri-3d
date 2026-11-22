"""Prepare the same development cases for the official nnU-Net reference."""

from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from .geometry import Grid
from .imaging import read_nifti, write_nifti
from .labels import LABELS
from .research import Case, lesion_instances, mask_metrics, match_lesions, read_manifest, summarize_metrics


def _write_images(directory: Path, key: str, case: Case, grid: Grid) -> None:
    channels = np.load(case.path / "image.npy", mmap_mode="r", allow_pickle=False)
    if channels.shape != (3, *grid.shape):
        raise ValueError("Prepared phase arrays do not match the nnU-Net geometry.")
    for channel, image in enumerate(channels):
        write_nifti(directory / f"{key}_{channel:04d}.nii.gz", image.astype(np.float32), grid)


def prepare_nnunet(manifest: Path, raw: Path, preprocessed: Path, dataset_id: int = 73) -> Path:
    if not 1 <= dataset_id <= 999:
        raise ValueError("nnU-Net dataset ID must lie in [1, 999].")
    name = f"Dataset{dataset_id:03d}_BreastMRI3D"
    destination = raw / name
    splits_path = preprocessed / name / "splits_final.json"
    if destination.exists() or splits_path.exists():
        raise FileExistsError(f"nnU-Net dataset or fixed split for {name} exists already.")
    cases = [case for case in read_manifest(manifest) if case.split in {"train", "validation"}]
    if {case.split for case in cases} != {"train", "validation"}:
        raise ValueError("nnU-Net preparation requires a fixed training and validation split.")
    if any("lesion" not in case.metadata["labels"] for case in cases):
        raise ValueError("Each case in this lesion-only comparison needs a lesion reference or confirmed absence.")
    raw.mkdir(parents=True, exist_ok=True)
    splits = {"train": [], "val": []}
    with TemporaryDirectory(dir=raw, prefix=".nnunet-input-") as tmp:
        staged = Path(tmp) / name
        images = staged / "imagesTr"
        masks = staged / "labelsTr"
        images.mkdir(parents=True)
        masks.mkdir()
        for index, case in enumerate(cases, start=1):
            key = f"BreastMRI_{index:04d}"
            grid = case.grid
            y = np.load(case.path / "labels.npy", mmap_mode="r", allow_pickle=False)[LABELS.index("lesion")]
            if y.shape != grid.shape or not np.isin(y, (0, 1)).all():
                raise ValueError("Prepared case arrays do not match the nnU-Net geometry and class contract.")
            _write_images(images, key, case, grid)
            write_nifti(masks / f"{key}.nii.gz", y.astype(np.uint8), grid)
            splits["train" if case.split == "train" else "val"].append(key)
        dataset = {"channel_names": {"0": "precontrast", "1": "early_postcontrast", "2": "enhancement"},
                   "labels": {"background": 0, "lesion": 1}, "numTraining": len(cases), "file_ending": ".nii.gz"}
        (staged / "dataset.json").write_text(json.dumps(dataset, indent=2) + "\n", encoding="utf-8")
        staged.rename(destination)
    splits_path.parent.mkdir(parents=True, exist_ok=True)
    with splits_path.open("x", encoding="utf-8") as stream:
        json.dump([splits], stream, indent=2)
        stream.write("\n")
    return destination


def prepare_nnunet_prediction(manifest: Path, destination: Path, *, split: str) -> Path:
    cases = [case for case in read_manifest(manifest) if case.split == split]
    if split not in {"validation", "test"} or not cases or destination.exists():
        raise ValueError("Prediction preparation needs an unused destination and a nonempty held-out split.")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=destination.parent, prefix=".nnunet-predict-") as tmp:
        staged = Path(tmp) / "images"
        staged.mkdir()
        for i, case in enumerate(cases, 1):
            _write_images(staged, f"BreastMRI_{i:04d}", case, case.grid)
        staged.rename(destination)
    return destination


def score_nnunet(manifest: Path, predictions: Path, output: Path, *, split: str = "validation",
                 prediction_input: bool = False) -> dict:
    if split not in {"validation", "test"} or (split == "test" and not prediction_input):
        raise ValueError("A test score requires images prepared explicitly for prediction.")
    cases = [case for case in read_manifest(manifest)
             if (case.split == split if prediction_input else case.split in {"train", "validation"})]
    records = []
    for index, case in enumerate(cases, start=1):
        if case.split != split:
            continue
        metadata = case.metadata
        if "native_grid" not in metadata or "lesion" not in metadata["labels"]:
            raise ValueError("The nnU-Net native comparison needs original lesion masks for all validation cases.")
        native = Grid.from_dict(metadata["native_grid"])
        model = Grid.from_dict(metadata["grid"])
        segmentation, declared_model = read_nifti(predictions / f"BreastMRI_{index:04d}.nii.gz")
        target, declared_native = read_nifti(case.path / "native_labels" / "lesion.nii.gz")
        for values, stated, expected in ((segmentation, declared_model, model), (target, declared_native, native)):
            if (stated.shape != expected.shape or not np.allclose(stated.affine, expected.affine, atol=.001, rtol=0)
                    or not np.isin(values, (0, 1)).all()):
                raise ValueError("The nnU-Net prediction or reference changed its expected geometry or label encoding.")
        aligned = model.resample(segmentation.astype(np.uint8), native, order=0)
        provenance = metadata["labels"]["lesion"]
        record = {"patient": case.patient, "cohort": case.cohort, "reference": provenance,
                  "grid": "native_" + metadata["reference"].get("type", "dicom"),
                  **mask_metrics(aligned, target, native)}
        if provenance in {"manual", "confirmed_empty"} or metadata["reference"].get("assisted_manual"):
            instances = lesion_instances(case, target, native, metadata["reference"])
            lesions = match_lesions(aligned, target, grid=native, instances=instances)
            if provenance != "confirmed_empty" and not metadata.get("lesion_annotations_exhaustive", False):
                lesions.pop("false_alarms")
            record.update(lesions)
        records.append(record)
    if not records:
        raise ValueError("The manifest has no cases in the requested scoring split.")
    report = {"split": split, "model": "nnUNetv2", "structure": "lesion",
              "metrics": summarize_metrics(records), "records": records}
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    return report
