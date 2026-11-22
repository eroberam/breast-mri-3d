"""Link verified EA1141 screen-negative exams to original DCE acquisitions."""

from __future__ import annotations

import csv
from datetime import date
import hashlib
import io
import json
from pathlib import Path
import zipfile

from .acquire import checksum, download, write_json


CLINICAL = ("https://wiki.cancerimagingarchive.net/download/attachments/157286463/"
            "EA1141%20Reviewed%20Clinical%20Data%20%20and%20Data%20Dictionaries.zip?api=v2")
_COMBINED = {"dyn fs 1 pre 1 post", "ax t1 3d fs 1 pre + 1 post"}
_PAIRS = (
    ("axial t1 fs pre asset", "axial t1 fs post asset"),
    ("ax 3d fs pre dynamic", "ax 3d fs post dynamic"),
    ("ax 3d t1 fs pre", "ax 3d t1 fs post"),
    ("abs vibrant fs pre", "abs vibrant fs post"),
    ("ax pre - vibrant", "ph1/axpost-vibrant"),
    ("axial t1 vibe fs pre max fov 380", "axial t1 vibe fs post"),
)


def _table(archive: zipfile.ZipFile, suffix: str) -> list[dict]:
    name = next(n for n in archive.namelist() if not n.startswith("__MACOSX/") and n.endswith("/" + suffix))
    with archive.open(name) as data:
        return list(csv.DictReader(io.TextIOWrapper(data, encoding="utf-8-sig")))


def clinical_intervals(archive: Path) -> dict[str, int]:
    with zipfile.ZipFile(archive) as release:
        follow = {r["SUBJECT_DE"]: r for r in _table(release, "ea1141_12month_fup.csv")}
        positive = {r["SUBJECT_DE"] for r in _table(release, "ea1141_year0_mrilesions_outcome.csv")}
        result = {}
        for initial in _table(release, "ea1141_year0_screening_derived.csv"):
            patient = initial["SUBJECT_DE"]
            late = follow.get(patient, {})
            if (patient in positive or initial["MRI_BIRADS_YR0"] != "1" or initial["MRI_BIRADS_NUM35_YR0"] != "0"
                    or late.get("FUP_12MO_CANCER") != "0" or late.get("FUP_12MO_330DAYSYR0") != "1"
                    or late.get("YEAR0_SENSSPEC_REFSTD") != "0" or late.get("MRI_YR1") != "1"):
                continue
            try:
                days = int(late["MRI_YR1_DATE_DAYS"]) - int(initial["MRI_YR0_DATE_DAYS"])
            except ValueError:
                continue
            if days > 0:
                result[f"EA1141-{patient}"] = days
    return result


def _dated_baseline(dates: set[date], interval: int) -> date | None:
    pairs = [(early, late) for early in dates for late in dates if (late - early).days == interval]
    return pairs[0][0] if len(pairs) == 1 else None


def select_ea1141(index, clinical: dict[str, int], limit: int = 80) -> list[dict]:
    chosen = []
    for patient, series in index.groupby("PatientID"):
        if patient not in clinical:
            continue
        dates = {date.fromisoformat(str(value)[:10]) for value in series.StudyDate.dropna().unique()}
        baseline = _dated_baseline(dates, clinical[patient])
        if baseline is None:
            continue
        rows = series[series.StudyDate.astype(str).str[:10] == baseline.isoformat()]
        if rows.StudyInstanceUID.nunique() != 1 or not any("BIL" in str(name).upper() for name in rows.StudyDescription):
            continue
        descriptions: dict[str, list[str]] = {}
        for row in rows.itertuples():
            if row.instanceCount >= 30:
                name = " ".join(str(row.SeriesDescription).lower().split())
                descriptions.setdefault(name, []).append(row.SeriesInstanceUID)
        matches = []
        for description in sorted(_COMBINED):
            if len(descriptions.get(description, [])) == 1:
                matches.append(descriptions[description])
        for pre, post in _PAIRS:
            if len(descriptions.get(pre, [])) == len(descriptions.get(post, [])) == 1:
                matches.append(descriptions[pre] + descriptions[post])
        if len(matches) == 1:
            chosen.append({"patient": patient, "study": rows.StudyInstanceUID.iloc[0],
                           "series": matches[0], "clinical_mri_interval_days": clinical[patient]})
    chosen.sort(key=lambda r: hashlib.sha256(f"19:{r['patient']}".encode()).digest())
    chosen = chosen[:limit]
    training = round(.6 * len(chosen))
    validation = round(.8 * len(chosen))
    for i, row in enumerate(chosen):
        row["split"] = "train" if i < training else "validation" if i < validation else "test"
    return chosen


def prepare_ea1141(root: Path, *, limit: int = 80) -> Path:
    import numpy as np
    from idc_index import IDCClient

    from .dicom import assemble, scan_series
    from .geometry import model_grid
    from .prepare import prepare_case

    source = root / "raw" / "EA1141"
    archive = download(CLINICAL, source / "clinical.zip")
    clinical = clinical_intervals(archive)
    idc = IDCClient()
    index = idc.sql_query("SELECT * FROM index WHERE collection_id='ea1141' AND Modality='MR'")
    cases = select_ea1141(index, clinical, limit=limit)
    if not cases:
        raise ValueError("No EA1141 exams passed negative status, date linkage and explicit DCE phase selection.")
    ledger = root / "acquisition/ea1141-selection.json"
    selection = {"release": "TCIA:10.7937/2BAS-HR33", "clinical_sha256": checksum(archive), "cases": cases}
    if ledger.exists() and json.loads(ledger.read_text(encoding="utf-8")) != selection:
        raise ValueError("This EA1141 selection would change an existing locked partition.")
    write_json(ledger, selection)
    print(f"EA1141 clinically linked DCE exams selected {len(cases)}", flush=True)
    native = source / "selected-original"
    native.mkdir(exist_ok=True)
    idc.download_from_selection(downloadDir=str(native), seriesInstanceUID=sorted({s for c in cases for s in c["series"]}),
                                quiet=True, show_progress_bar=False)
    prepared = root / "prepared/EA1141"
    verified, excluded = [], []
    for row in cases:
        try:
            matches = scan_series(native / "ea1141" / row["patient"] / row["study"])
            if len(row["series"]) == 2:
                ordered = [name for name in row["series"] if name in matches]
            else:
                ordered = sorted((name for name in matches if name.startswith(row["series"][0] + "#")),
                                 key=lambda key: int(key.rpartition("#")[2]))
            if len(ordered) != 2 or len(matches) != 2:
                raise ValueError("The selected MRI does not decode to exactly two documented DCE acquisitions.")
            pre, post = (assemble(matches[name]) for name in ordered)
            grid = post.grid
            # The confirmed-empty target applies to the complete bilateral screening field.
            if pre.grid.shape != grid.shape or not np.allclose(pre.grid.affine, grid.affine, atol=.01, rtol=0):
                raise ValueError("Screening precontrast and early postcontrast grids differ.")
            _ = model_grid(grid)
            destination = prepared / row["patient"]
            if not destination.exists():
                prepare_case(destination, pre, post, {"lesion": (np.zeros(grid.shape, np.uint8), grid, "confirmed_empty")},
                             patient=row["patient"], cohort="EA1141", phase_protocol="ea1141_abbreviated_dce")
            metadata = json.loads((destination / "case.json").read_text(encoding="utf-8"))
            if metadata["reference"]["source_instances_sha256"] != post.source_instances_sha256:
                raise ValueError("The prepared screening case references a different early phase.")
            verified.append({"path": row["patient"], "patient": row["patient"], "cohort": "EA1141", "split": row["split"]})
        except (ValueError, OSError) as exc:
            excluded.append({"patient": row["patient"], "reason": str(exc)})
    result = prepared / "manifest.json"
    write_json(result, {"purpose": "verified_bilateral_mri_screen_negative", "selection_sha256": checksum(ledger),
                        "cases": verified, "excluded": excluded})
    return result
