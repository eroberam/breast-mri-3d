"""Prepare the reserved AMBL release on geometry confirmed against the original MR."""

from __future__ import annotations

import csv
from collections import defaultdict
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import shutil
import tarfile
from tempfile import NamedTemporaryFile

import numpy as np

from .acquire import AMBL_ASSETS, AMBL_DCE, checksum, write_json
from .dicom import assemble, scan_series
from .imaging import prepare_dce
from .iteration2 import _atomic_case
from .prepare import _load_index_array, _match


RELEASE = "zenodo:17253223/v3"
COLLECTION = "advanced_mri_breast_lesions"
PHASE_MEMBER = re.compile(r"(\d+)sh_dce_acq([01])\.nii\.gz")
MASK_MEMBER = re.compile(r"(sh_\d+)_sus\.nii\.gz")


def inventory(root: Path) -> dict[str, str]:
    table = root / "raw/AMBL/BreastDCEDL_AMBL_paitents.csv"
    if checksum(table, "md5") != AMBL_ASSETS[table.name]:
        raise ValueError("The AMBL patient mapping fails its publisher checksum.")
    with table.open(encoding="utf-8-sig", newline="") as stream:
        pairs = [(row["pid"], row["tcpid"]) for row in csv.DictReader(stream)]
    if (len(pairs) != 88 or len(dict(pairs)) != 88 or len({p for _, p in pairs}) != 88
            or any(not re.fullmatch(r"sh_\d+", alias) or not re.fullmatch(r"AMBL-\d+", patient)
                   for alias, patient in pairs)):
        raise ValueError("The external patient mapping is incomplete or ambiguous.")
    return dict(pairs)


def acquire_original_mr(root: Path, *, limit: int = 88, registered: bool = False) -> int:
    from idc_index import IDCClient

    cases = inventory(root)
    if not 1 <= limit <= len(cases):
        raise ValueError("Choose between one and 88 AMBL subjects.")
    patients = sorted(cases.values(), key=lambda p: hashlib.sha256(f"19:{p}".encode()).digest())[:limit]
    client = IDCClient()
    data = client.sql_query("""
        SELECT PatientID, SeriesInstanceUID, SeriesDescription, instanceCount FROM index
        WHERE collection_id = 'advanced_mri_breast_lesions' AND Modality = 'MR'
        AND SeriesDescription IN ('AX Sen Vibrant MultiPhase', 'Registered AX Sen Vibrant MultiPhase')
    """)
    description = ("Registered " if registered else "") + "AX Sen Vibrant MultiPhase"
    selected = data[(data.SeriesDescription == description) & data.PatientID.isin(patients)]
    selected = selected[~selected.PatientID.duplicated(keep=False)]
    if selected.empty:
        raise ValueError("No unique native AMBL multiphase series was found.")
    destination = root / "raw/AMBL/original-mr"
    destination.mkdir(parents=True, exist_ok=True)
    for start in range(0, len(selected), 8):
        batch = selected.iloc[start:start + 8]
        client.download_from_selection(downloadDir=str(destination), seriesInstanceUID=batch.SeriesInstanceUID.tolist(),
                                       analysis_results="exclude",
                                       show_progress_bar=False, quiet=True)
        print(f"Original AMBL MR series {min(start + len(batch), len(selected))}/{len(selected)}", flush=True)
    return len(selected)


def _write_member(tar: tarfile.TarFile, member: tarfile.TarInfo, output: Path) -> None:
    if output.is_file() and output.stat().st_size == member.size:
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=output.parent, prefix=".ambl-slice-", delete=False) as temporary:
        staged = Path(temporary.name)
        try:
            with tar.extractfile(member) as source:
                shutil.copyfileobj(source, temporary, 4 * 1024 * 1024)
            if temporary.tell() != member.size:
                raise ValueError("The source archive member was truncated.")
            temporary.close()
            staged.replace(output)
        finally:
            temporary.close()
            staged.unlink(missing_ok=True)


def _unpack(root: Path, subjects: set[str]) -> Path:
    directory = root / "raw/AMBL"
    destination = directory / "paired-release"
    inventory_path = destination / "selection.json"
    if inventory_path.exists():
        record = json.loads(inventory_path.read_text(encoding="utf-8"))
        if set(record["subjects"]) == subjects:
            return destination
    for filename, parent, pattern in ((AMBL_DCE, "dce", PHASE_MEMBER), ("sus_tum.tar.gz", "sus_tum", MASK_MEMBER)):
        seen = set()
        with tarfile.open(directory / filename, "r|gz") as archive:
            for member in archive:
                path = PurePosixPath(member.name)
                match = pattern.fullmatch(path.name)
                if not member.isfile() or path.parts != (parent, path.name) or not match:
                    continue
                subject = f"sh_{match[1]}" if parent == "dce" else match[1]
                if subject not in subjects:
                    continue
                key = f"phase-{match[2]}" if parent == "dce" else "lesion"
                if (subject, key) in seen:
                    raise ValueError("The published archive contains duplicate phase or label entries.")
                seen.add((subject, key))
                _write_member(archive, member, destination / subject / f"{key}.nii.gz")
    write_json(inventory_path, {"subjects": sorted(subjects)})
    return destination


def _matched_dce(visit: Path, pre: np.ndarray, post: np.ndarray):
    grouped = defaultdict(list)
    for entries in scan_series(visit).values():
        try:
            series = assemble(entries)
        except ValueError:
            continue
        if sorted(series.grid.shape) == sorted(post.shape):
            grouped[series.uid].append(series)
    # Prefer the original acquisitions when both original and registered versions
    # contain the same phases. Match both volumes inside one series family.
    ordered = sorted(grouped.values(), key=lambda volumes: str(volumes[0].headers[0].get("SeriesDescription", ""))
                     .lower().startswith("registered"))
    for volumes in ordered:
        candidates = [(series, series.pixels()) for series in volumes]
        try:
            first, _, _ = _match(pre, candidates)
            early, axes, flips = _match(post, candidates)
        except ValueError:
            continue
        return first, early, axes, flips
    raise ValueError("Neither native multiphase series uniquely matched both released DCE phases.")


def prepare_ambl(root: Path, model_lock: Path) -> Path:
    from .experiment import verify_external_lock

    verify_external_lock(root, model_lock)
    patients = inventory(root)
    directory = root / "raw/AMBL/original-mr" / COLLECTION
    eligible = {subject: patient for subject, patient in patients.items() if (directory / patient).is_dir()}
    if not eligible:
        raise ValueError("Native source DICOM must be acquired before preparing a physical AMBL evaluation.")
    prepared = root / "prepared/AMBL"
    phases = _unpack(root, set(eligible))
    records = []
    excluded = [{"patient": patient, "reason": "original DICOM unavailable"} for patient in patients.values()
                if patient not in eligible.values()]
    for subject, patient in eligible.items():
        visit = directory / patient
        data = phases / subject
        try:
            pre, pre_affine = _load_index_array(data / "phase-0.nii.gz")
            post, post_affine = _load_index_array(data / "phase-1.nii.gz")
            labels, label_affine = _load_index_array(data / "lesion.nii.gz")
            if (pre.shape != post.shape or labels.shape != post.shape
                    or not np.allclose(pre_affine, post_affine, rtol=0, atol=.001)
                    or not np.allclose(label_affine, post_affine, rtol=0, atol=.001)
                    or not np.isfinite(labels).all() or np.any(labels < 0)
                    or not np.array_equal(labels, np.floor(labels)) or not np.any(labels)):
                raise ValueError("The released DCE phases and lesion reference lack a common index grid.")
            u0, u1, axes, flips = _matched_dce(visit, pre, post)
            tumor = np.flip((labels > 0).transpose(axes), axis=flips)
            reference = {
                "type": "dicom", "dataset_release": RELEASE, "end_point": "all_reported_lesions",
                "assisted_manual": True, "phase_match": "full_voxel_equality_after_reorientation",
                "series_instance_uid": u1.uid, "frame_of_reference_uid": u1.frame_uid,
                "source_instances_sha256": u1.source_instances_sha256,
                "precontrast_series_uid": u0.uid, "precontrast_instances_sha256": u0.source_instances_sha256,
            }
            _atomic_case(prepared / patient, prepare_dce(u0, u1), u1.grid,
                         {"lesion": (tumor, u1.grid, "derived")}, reference, patient, "AMBL", "ambl_dce_early")
        except (OSError, ValueError) as exc:
            excluded.append({"patient": patient, "reason": str(exc)})
        else:
            records.append({"path": patient, "patient": patient, "cohort": "AMBL", "split": "test"})
        if (len(records) + len(excluded)) % 10 == 0:
            print(f"AMBL physical validation eligibility assessed for {len(records) + len(excluded)}/88", flush=True)
    output = prepared / "manifest.json"
    write_json(output, {"purpose": "reserved_external_all_reported_lesions", "model_lock_sha256": checksum(model_lock),
                        "cases": records, "excluded": excluded})
    return output
