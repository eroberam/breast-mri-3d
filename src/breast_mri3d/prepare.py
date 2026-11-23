"""Prepare a case without copying DICOM headers or inventing NIfTI geometry."""

from __future__ import annotations

from itertools import permutations, product
import hashlib
import json
from pathlib import Path
import re

import nibabel as nib
import numpy as np

from .dicom import MRSeries, assemble, scan_series
from .geometry import Grid, model_grid
from .imaging import dce_channels, prepare_dce, read_nifti, write_nifti
from .labels import ENVELOPE, LABELS, REFERENCE_LABELS


def prepare_case(directory: Path, pre: MRSeries, post: MRSeries,
                 references: dict[str, tuple[np.ndarray, Grid, str]], *,
                 patient: str, cohort: str, phase_protocol: str | None = None) -> None:
    """Write memory-mappable arrays and distinguish missing labels from negatives."""
    source = {"type": "dicom", "series_instance_uid": post.uid, "frame_of_reference_uid": post.frame_uid,
              "source_instances_sha256": post.source_instances_sha256}
    _write_case(directory, prepare_dce(pre, post), post.grid, references, source, patient, cohort, phase_protocol)


def prepare_volumes(directory: Path, pre: Path, post: Path, references: dict[str, tuple[np.ndarray, Grid, str]], *,
                    pre_index: int | None, post_index: int | None, release: str,
                    patient: str, cohort: str, phase_protocol: str | None = None,
                    confirmed_negative: bool = False) -> None:
    """Ingest a released NIfTI study for research with explicitly documented phase order."""
    if not release.strip() or (pre.resolve() == post.resolve() and pre_index == post_index):
        raise ValueError("Supply a source release and two distinct DCE phases.")
    pre_image, pre_grid = read_nifti(pre, time_index=pre_index)
    post_image, post_grid = read_nifti(post, time_index=post_index)
    aligned = pre_grid.resample(pre_image.astype(np.float32), post_grid, order=1)
    if confirmed_negative:
        if "lesion" in references:
            raise ValueError("A confirmed negative exam cannot have a lesion annotation.")
        references = {**references, "lesion": (np.zeros(post_grid.shape, dtype=np.uint8), post_grid, "confirmed_empty")}
    source = {"type": "nifti", "dataset_release": release, "pre_index": pre_index,
              "post_index": post_index, "pre_file": pre.name, "post_file": post.name}
    _write_case(directory, dce_channels(aligned, post_image.astype(np.float32)), post_grid, references,
                source, patient, cohort, phase_protocol)


def _write_case(directory: Path, channels: np.ndarray, native_grid: Grid,
                references: dict[str, tuple[np.ndarray, Grid, str]], source: dict,
                patient: str, cohort: str, phase_protocol: str | None, *,
                target_grid: Grid | None = None) -> None:
    if not patient.strip() or not cohort.strip():
        raise ValueError("A prepared case needs canonical patient and cohort identifiers.")
    target = target_grid if target_grid is not None else model_grid(native_grid)
    image = np.empty((len(channels), *target.shape), dtype=np.float16)
    for i, channel in enumerate(channels):
        image[i] = native_grid.resample(np.asarray(channel, dtype=np.float32), target, order=1)
    labels = np.zeros((len(LABELS), *target.shape), dtype=np.uint8)
    provenance = {}
    native_labels = {}
    minimum_support = []
    envelope = None
    for name, (mask, grid, kind) in references.items():
        if name not in REFERENCE_LABELS or kind not in {"manual", "derived", "confirmed_empty"}:
            raise ValueError(f"Unsupported label or provenance: {name}, {kind}")
        mask = np.asarray(mask)
        if not np.isin(mask, (0, 1)).all() or (kind == "confirmed_empty" and mask.any()):
            raise ValueError(f"Invalid binary reference for {name}.")
        native = grid.resample(mask.astype(np.uint8), native_grid, order=0)
        train_mask = native_grid.resample(native, target, order=0)
        if mask.any() and not train_mask.any():
            if target_grid is None or name != "lesion" or kind != "manual" or not native.any():
                raise ValueError(f"Reference {name} does not intersect the training grid.")
            centre = native_grid.points(np.argwhere(native).mean(axis=0))
            index = np.rint(np.linalg.solve(target.affine[:3, :3], centre - target.affine[:3, 3]))
            train_mask[tuple(np.clip(index.astype(int), 0, np.asarray(target.shape) - 1))] = 1
            minimum_support.append(name)
        if name == ENVELOPE:
            envelope = train_mask
        else:
            labels[LABELS.index(name)] = train_mask
        provenance[name] = kind
        native_labels[name] = native
    if not provenance:
        raise ValueError("Supply at least one annotated structure or verified absence.")
    directory.mkdir(parents=True, exist_ok=False)
    np.save(directory / "image.npy", image, allow_pickle=False)
    np.save(directory / "labels.npy", labels, allow_pickle=False)
    if envelope is not None:
        np.save(directory / f"{ENVELOPE}.npy", envelope, allow_pickle=False)
    native_dir = directory / "native_labels"
    native_dir.mkdir()
    for name, mask in native_labels.items():
        write_nifti(native_dir / f"{name}.nii.gz", mask, native_grid)
    meta = {"patient": patient, "cohort": cohort, "labels": provenance, "channel_order": list(LABELS),
            "grid": target.as_dict(), "native_grid": native_grid.as_dict(), "phase_protocol": phase_protocol,
            "reference": source}
    if minimum_support:
        meta["minimum_model_grid_support"] = minimum_support
    (directory / "case.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")


def exact_reorientation(moving: np.ndarray, fixed: np.ndarray) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Identify an axis permutation/flip only after complete voxel equality."""
    if moving.ndim != 3 or fixed.ndim != 3 or sorted(moving.shape) != sorted(fixed.shape):
        raise ValueError("Acquisition dimensions do not match under an axis permutation.")
    rng = np.random.default_rng(1729)
    coords = tuple(rng.integers(n, size=511) for n in fixed.shape)
    reference = fixed[coords]
    matches = []
    for axes in permutations(range(3)):
        if tuple(moving.shape[i] for i in axes) != fixed.shape:
            continue
        for flags in product((False, True), repeat=3):
            flips = tuple(i for i, flag in enumerate(flags) if flag)
            candidate = np.flip(moving.transpose(axes), axis=flips)
            if np.array_equal(candidate[coords], reference) and np.array_equal(candidate, fixed):
                matches.append((axes, flips))
    if len(matches) != 1:
        raise ValueError("Could not verify a unique exact match to the source DICOM intensity volume.")
    return matches[0]


def _load_index_array(path: Path) -> tuple[np.ndarray, np.ndarray]:
    image = nib.load(str(path))
    if len(image.shape) != 3:
        raise ValueError("Expected a single 3D acquisition in the published archive.")
    return np.asanyarray(image.dataobj), image.affine


def _match(moving: np.ndarray, candidates: list[tuple[MRSeries, np.ndarray]]) -> tuple[MRSeries, tuple, tuple]:
    found = []
    for series, values in candidates:
        try:
            axes, flips = exact_reorientation(moving, values)
        except ValueError:
            continue
        found.append((series, axes, flips))
    if len(found) != 1:
        raise ValueError("The published phase could not be linked to exactly one native MR series.")
    return found[0]


def prepare_ispy1(raw: Path, published: Path, destination: Path, *, expert: Path | None = None) -> Path:
    """Link derived I-SPY1 labels to raw DICOM by matching both phase volumes.

    BreastDCEDL's unitless NIfTI headers are used only to check its image/label array
    correspondence. Geometry comes from the verified identical DICOM acquisitions.
    This builds a development-only pilot, with no benchmark test split populated.
    """
    destination.mkdir(parents=True, exist_ok=False)
    index = defaultdict_paths(published)
    records, skipped = [], []
    for case in sorted(Path(raw).iterdir()):
        token = re.fullmatch(r"ISPY1_(\d+)", case.name, re.IGNORECASE)
        if not case.is_dir() or token is None:
            continue
        case_id = f"ISPY1_{token[1]}"
        try:
            files = index[case_id]
            pre_public, _ = _load_index_array(files["acq0"])
            post_public, post_affine = _load_index_array(files["acq1"])
            ftv, label_affine = _load_index_array(files["mask"])
            if ftv.shape != post_public.shape or not np.allclose(label_affine, post_affine, atol=1e-5):
                raise ValueError("Derived lesion mask and released early postcontrast do not share a grid.")
            if not np.isin(ftv, (0, 1, 255)).all() or not np.any(ftv):
                raise ValueError("The released lesion mask is empty or has an undocumented encoding.")
            candidates = []
            for entries in scan_series(case).values():
                try:
                    series = assemble(entries)
                    if sorted(series.grid.shape) in (sorted(pre_public.shape), sorted(post_public.shape)):
                        candidates.append((series, series.pixels()))
                except ValueError:
                    continue
            pre, _, _ = _match(pre_public, candidates)
            post, axes, flips = _match(post_public, candidates)
            mask = np.flip(ftv.transpose(axes), axis=flips) > 0
            refs = {"lesion": (mask, post.grid, "derived")}
            annotation_state = "not_available"
            if expert is not None:
                path = Path(expert) / f"{case_id}_0000.nii"
                if path.is_file():
                    try:
                        envelope, geometry = read_nifti(path)
                        if (geometry.shape != pre.grid.shape or not np.allclose(
                                geometry.affine, pre.grid.affine, atol=.1, rtol=0)):
                            raise ValueError("Manual mask and precontrast do not have matching native grids.")
                        # A whole-breast mask has no independent left/right annotations.
                        # Keep it for anatomical QA without turning it into two targets.
                        annotation_state = "manual_envelope_verified"
                    except ValueError:
                        annotation_state = "manual_envelope_geometry_unverified"
            case_dir = destination / case_id
            prepare_case(case_dir, pre, post, refs, patient=case_id, cohort="ISPY1")
            if annotation_state == "manual_envelope_verified":
                write_nifti(case_dir / "manual_breast_envelope.nii.gz", (envelope > 0).astype(np.uint8), geometry)
            records.append({"path": case_id, "patient": case_id, "cohort": "ISPY1",
                            "anatomy_reference": annotation_state})
        except (KeyError, ValueError, OSError) as exc:
            skipped.append({"patient": case_id, "reason": str(exc)})
    if len(records) < 2:
        raise ValueError(f"Fewer than two complete DICOM-linked cases. Skipped: {skipped}")
    records.sort(key=lambda entry: hashlib.sha256(entry["patient"].encode()).hexdigest())
    n_valid = max(1, round(len(records) * .2))
    for i, record in enumerate(records):
        record["split"] = "validation" if i < n_valid else "train"
    manifest = destination / "manifest.json"
    manifest.write_text(json.dumps({"purpose": "development_pilot_derived_references",
                                    "cases": records, "skipped": skipped}, indent=2) + "\n", encoding="utf-8")
    return manifest


def defaultdict_paths(root: Path) -> dict[str, dict[str, Path]]:
    index: dict[str, dict[str, Path]] = {}
    for file in Path(root).rglob("*.nii.gz"):
        match = re.fullmatch(r"(ISPY1_\d+)_spy1_vis1_(acq[01]|mask)\.nii\.gz", file.name, re.IGNORECASE)
        if match:
            target = index.setdefault(match[1].upper(), {})
            key = match[2].lower()
            if key in target:
                raise ValueError(f"Duplicate phase or mask in the supplied release: {match[1]}, {key}")
            target[key] = file
    return index
