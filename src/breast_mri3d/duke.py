"""Prepare original Duke 3D envelope, fibroglandular and vessel references."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re

from idc_index import IDCClient
import nrrd
import numpy as np

from .acquire import checksum, write_json
from .dicom import assemble, scan_series
from .geometry import Grid
from .imaging import prepare_dce
from .iteration2 import _atomic_case, require_coverage, same_grid
from .labels import ENVELOPE
from .research import read_manifest


COLLECTION = "duke_breast_cancer_mri"
EARLY_PHASE = {
    "ax dyn pre": "ax dyn 1st pass",
    "ax 3d dyn pre": "ax 3d dyn 1st pass",
    "ax 3d pre": "1st ax dyn",
    "ax 3d dyn": "ph1/ax 3d dyn",
    "ax dynamic": "ph1/ax dynamic",
    "ax vibrant multiphase": "ph1/ax vibrant multiphase",
    "ax 3d dyn mp": "ph1/ax 3d dyn mp",
    "ax dyn": "ph1/ax dyn",
}


def _nrrd_axes(header: dict) -> tuple[tuple[int, ...], int | None]:
    dimensions = header["dimension"]
    directions = np.asarray(header["space directions"], dtype=float)
    spatial = tuple(int(axis) for axis in np.flatnonzero(np.isfinite(directions).all(axis=1)))
    layers = [axis for axis in range(dimensions) if axis not in spatial]
    if (header["space"] != "left-posterior-superior" or dimensions not in {3, 4} or len(spatial) != 3
            or (layers and header["kinds"][layers[0]] != "list")):
        raise ValueError("Expected three spatial Slicer axes in LPS, with an optional label-layer axis.")
    return spatial, layers[0] if layers else None


def nrrd_grid(header: dict) -> Grid:
    spatial, _ = _nrrd_axes(header)
    source_axes = list(reversed(spatial))
    transform = np.eye(4)
    transform[:3, :3] = np.asarray(header["space directions"], dtype=float)[source_axes].T
    transform[:3, 3] = header["space origin"]
    return Grid(tuple(int(header["sizes"][axis]) for axis in source_axes), transform)


def reference_masks(path: Path, expected: dict[str, str]) -> tuple[dict[str, np.ndarray], Grid]:
    data, header = nrrd.read(str(path), index_order="F")
    grid = nrrd_grid(header)
    _, layer_axis = _nrrd_axes(header)
    labels = {}
    for key, name in header.items():
        match = re.fullmatch(r"Segment(\d+)_Name", key)
        if match and name in expected:
            prefix = f"Segment{match[1]}"
            labels[expected[name]] = int(header[f"{prefix}_Layer"]), int(header[f"{prefix}_LabelValue"])
    layer_count = data.shape[layer_axis] if layer_axis is not None else 1
    if (labels.keys() != set(expected.values())
            or {layer for layer, _ in labels.values()} != set(range(layer_count))):
        raise ValueError("The Duke segmentation does not describe the expected classes and layers.")
    masks = {}
    for layer in range(layer_count):
        values = {label: value for label, (number, value) in labels.items() if number == layer}
        volume = data if layer_axis is None else np.take(data, layer, axis=layer_axis)
        if not np.isin(volume, [0, *values.values()]).all():
            raise ValueError("The Duke segmentation layer contains an undocumented label value.")
        masks.update({label: volume.transpose(2, 1, 0) == value for label, value in values.items()})
    return masks, grid


def inventory(raw: Path) -> dict[str, tuple[Path, Path]]:
    pairs = {}
    for breast in sorted((raw / "supplement").rglob("*_Breast.seg.nrrd")):
        match = re.fullmatch(r"Segmentation_Breast_MRI_(\d{3})_Breast\.seg\.nrrd", breast.name)
        if not match or match[1] in pairs:
            raise ValueError("The Duke supplement has missing or ambiguous canonical subject identifiers.")
        dense = breast.with_name(f"Segmentation_Breast_MRI_{match[1]}_Dense_and_Vessels.seg.nrrd")
        if not dense.is_file():
            raise ValueError("The Duke supplement is missing a dense tissue and vessel companion.")
        pairs[match[1]] = (breast, dense)
    if len(pairs) != 100:
        raise ValueError("The downloaded Duke 3D supplement must contain all 100 published paired annotations.")
    return pairs


def select_series(root: Path, pairs: dict[str, tuple[Path, Path]], client: IDCClient, limit: int) -> tuple[list, list]:
    rows = client.sql_query("""
        SELECT PatientID, StudyInstanceUID, SeriesInstanceUID, SeriesDescription, instanceCount
        FROM index WHERE collection_id = 'duke_breast_cancer_mri' AND Modality = 'MR'
    """)
    rows["SeriesDescription"] = rows["SeriesDescription"].fillna("")
    roles = {r.patient: r.split for r in read_manifest(root / "prepared/v02-anatomy-dev.json")}
    plan, out = [], []
    ids = sorted(pairs, key=lambda p: hashlib.sha256(f"19:DUKE_{p}".encode()).digest())
    for code in ids:
        official = f"Breast_MRI_{code}"
        patient = f"DUKE_{code}"
        grid = nrrd_grid(nrrd.read_header(str(pairs[code][0])))
        possible = rows[(rows.PatientID == official) & (rows.instanceCount == grid.shape[0])]
        names = possible.SeriesDescription.str.strip().str.casefold()
        matches = []
        for pre in possible.itertuples():
            early = EARLY_PHASE.get(pre.SeriesDescription.strip().casefold())
            if early is not None:
                post = possible[(possible.StudyInstanceUID == pre.StudyInstanceUID) & (names == early)]
                if len(post) == 1:
                    matches.append([pre.SeriesInstanceUID, str(post.iloc[0].SeriesInstanceUID)])
        if len(matches) != 1:
            out.append({"patient": patient, "reason": "no unique documented DCE pair in the indexed source study"})
            continue
        t = int.from_bytes(hashlib.sha256(f"19:{patient}".encode()).digest()[:4], "big") / 2**32
        plan.append({"code": code, "patient": patient, "source_patient": official, "series": matches[0],
                     "split": roles.get(patient.casefold(), "validation" if t < .2 else "train")})
    return plan[:limit], out


def compatible_lattice(mask: Grid, source: Grid) -> bool:
    relative = np.linalg.solve(source.affine, mask.affine)
    linear = np.rint(relative[:3, :3])
    return (np.all(np.abs(linear).sum(axis=0) == 1) and np.all(np.abs(linear).sum(axis=1) == 1)
            and np.allclose(relative[:3, :3], linear, atol=1e-3, rtol=0)
            and np.allclose(relative[:3, 3], np.rint(relative[:3, 3]), atol=.05, rtol=0))


def prepare_duke(root: Path, *, limit: int = 100) -> Path:
    raw = root / "raw/Duke3D"
    paths = inventory(raw)
    client = IDCClient()
    selected, excluded = select_series(root, paths, client, limit)
    if not selected:
        raise ValueError("No Duke case can be linked to a unique early DCE pair.")
    raw_images = raw / "paired-mr"
    raw_images.mkdir(exist_ok=True)
    case_root = root / "prepared/Duke3D"
    cases = []
    plan = root / "acquisition/duke3d-selection.json"
    new_selection = {"provider": "TCIA Duke 3D supplemental v3", "seed": 19, "cases": selected}
    if plan.is_file() and json.loads(plan.read_text(encoding="utf-8")) != new_selection:
        raise ValueError("The Duke anatomy selection has already been frozen.")
    write_json(plan, new_selection)
    print(f"Duke 3D pairs selected: {len(selected)}; indexed pair ambiguities: {len(excluded)}", flush=True)
    for start in range(0, len(selected), 8):
        batch = selected[start:start + 8]
        pending = [c for c in batch if not (case_root / c["patient"] / "case.json").is_file()]
        if pending:
            uids = [uid for c in pending for uid in c["series"]]
            client.download_from_selection(downloadDir=str(raw_images), seriesInstanceUID=uids,
                                           analysis_results="exclude", use_s5cmd_sync=True,
                                           show_progress_bar=False, quiet=True)
        for item in batch:
            patient = item["patient"]
            target = case_root / patient
            if not (target / "case.json").exists():
                try:
                    studies = raw_images / COLLECTION / item["source_patient"]
                    entries = scan_series(studies)
                    pre, early = [assemble(entries[uid]) for uid in item["series"]]
                    breast, dense = paths[item["code"]]
                    first, breast_grid = reference_masks(breast, {"Breast": ENVELOPE})
                    details, tissue_grid = reference_masks(dense, {"Dense": "fibroglandular", "Vessels": "vessels"})
                    if not same_grid(breast_grid, tissue_grid) or not compatible_lattice(breast_grid, pre.grid):
                        raise ValueError("The breast and tissue masks do not share the source MR lattice.")
                    supports = details["fibroglandular"] | details["vessels"]
                    if not supports.any() or (supports & ~first[ENVELOPE]).sum() / supports.sum() > .02:
                        raise ValueError("The tissue and vascular references do not fit the documented breast envelope.")
                    require_coverage(first[ENVELOPE], breast_grid, early.grid)
                    for mask in details.values():
                        if mask.any():
                            require_coverage(mask, tissue_grid, early.grid)
                    reference = {"type": "dicom", "dataset_release": "TCIA Duke 3D supplemental v3",
                                 "series_instance_uid": early.uid, "frame_of_reference_uid": early.frame_uid,
                                 "source_instances_sha256": early.source_instances_sha256,
                                 "precontrast_instances_sha256": pre.source_instances_sha256,
                                 "annotation_sha256": {"whole_breast": checksum(breast), "tissues": checksum(dense)}}
                    labels = {k: (v, breast_grid, "manual") for k, v in {**first, **details}.items()}
                    _atomic_case(target, prepare_dce(pre, early), early.grid, labels, reference, patient,
                                 "Duke3D", "duke3d_native_early_dce")
                except (ValueError, OSError, KeyError) as exc:
                    excluded.append({"patient": patient, "reason": str(exc)})
                    continue
            cases.append({"patient": patient, "cohort": "Duke3D", "path": patient, "split": item["split"]})
        print(f"Duke 3D manual cases physically prepared {len(cases)}", flush=True)
    output = case_root / "manifest.json"
    write_json(output, {"purpose": "manual_breast_fgt_vessel_supervision", "selection_sha256": checksum(plan),
                        "cases": sorted(cases, key=lambda c: c["patient"]), "excluded": excluded})
    return output


def combine_anatomy(root: Path, duke: Path) -> Path:
    previous = root / "prepared/v02-anatomy-dev.json"
    original = read_manifest(previous)
    tissue = read_manifest(duke)
    if not tissue or not any(row.split == "validation" for row in tissue):
        raise ValueError("Duke anatomical training requires a separate validation partition.")
    replacement = {case.patient for case in tissue}
    patient_roles = {row.patient: row.split for row in original}
    if any(row.patient in patient_roles and row.split != patient_roles[row.patient] for row in tissue):
        raise ValueError("The Duke annotation would change an existing patient partition.")
    chosen = [row for row in original if row.cohort != "MAMA-MIAAI" or row.patient not in replacement]
    cases = [{"patient": row.patient, "path": str(row.path), "cohort": row.cohort, "split": row.split}
             for row in [*chosen, *tissue]]
    output = root / "prepared/v02-anatomy-dev-with-duke.json"
    write_json(output, {"purpose": "fixed_development_with_duke_anatomy", "primary_manifest_sha256": checksum(previous),
                        "duke_manifest_sha256": checksum(duke), "cases": cases})
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--limit", type=int, default=100)
    args = parser.parse_args()
    manifest = prepare_duke(args.root, limit=args.limit)
    print(combine_anatomy(args.root, manifest), flush=True)


if __name__ == "__main__":
    main()
