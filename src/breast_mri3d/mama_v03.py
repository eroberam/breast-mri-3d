"""Acquire pinned Synapse phases and expert masks for the v0.3 study."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import re
from tempfile import TemporaryDirectory
from threading import Semaphore, get_ident, local
from time import sleep

import numpy as np

from .acquire import checksum, write_json
from .cohorts_v03 import exposure, freeze_partitions, patient_key
from .geometry import lps_grid
from .imaging import dce_channels, read_nifti
from .iteration2 import _atomic_case, require_coverage, same_grid
from .mama_mia import EXPERT_FOLDER, IMAGE_FOLDER, _client, read_synapse_token


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def catalog(previous: Path, root: Path, files: SynapseFiles) -> list[dict]:
    acquisition = previous / "acquisition"
    images_index = acquisition / "mama-mia-synapse-index.json"
    masks_index = acquisition / "mama-mia-expert-index.json"
    fresh_index = root / "acquisition/mama-synapse-catalog.json"
    if images_index.is_file() and masks_index.is_file():
        images = read_json(images_index)[IMAGE_FOLDER]["children"]
        masks = read_json(masks_index)["expert_children"]
    elif fresh_index.is_file():
        index = read_json(fresh_index)
        images, masks = index["images"], index["masks"]
    else:
        client = files.connection()
        images = list(client.getChildren(IMAGE_FOLDER))
        masks = list(client.getChildren(EXPERT_FOLDER))
        for entry in masks:
            if entry["name"].endswith(".nii.gz") and "versionNumber" not in entry:
                entry["versionNumber"] = int(client.get(entry["id"], downloadFile=False).versionNumber)
        write_json(fresh_index, {"images": images, "masks": masks})
    masks = {item["name"].removesuffix(".nii.gz"): item for item in masks if item["name"].endswith(".nii.gz")}
    rows = []
    for image in images:
        patient = image["name"]
        if patient not in masks:
            continue
        if not re.fullmatch(r"(?:DUKE|ISPY1|ISPY2|NACT)_[A-Za-z0-9_-]+", patient):
            raise ValueError("Unexpected Synapse subject identifier.")
        mask = masks[patient]
        rows.append({"patient": patient, "collection": patient.split("_", 1)[0], "folder": image["id"],
                     "mask": {"id": mask["id"], "version": int(mask["versionNumber"])}})
    if not rows or len({row["patient"] for row in rows}) != len(rows):
        raise ValueError("The paired Synapse catalog must contain one row per subject.")
    return sorted(rows, key=lambda row: row["patient"])


class SynapseFiles:
    def __init__(self, cache: Path, token_file: Path):
        self.cache = cache
        self.token = read_synapse_token(token_file)
        self.session = local()

    def connection(self):
        if not hasattr(self.session, "client"):
            cache = self.cache / str(get_ident())
            cache.mkdir(parents=True, exist_ok=True)
            self.session.client = _client(cache, self.token)
        return self.session.client

    def fetch(self, entity: dict, path: Path) -> str:
        receipt = path.with_name(path.name + ".source.json")
        if receipt.is_file() and path.is_file():
            source = read_json(receipt)
            if source["entity"] != entity or checksum(path) != source["sha256"]:
                raise ValueError("A cached object differs from its pinned Synapse version.")
            return source["sha256"]
        path.parent.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(dir=path.parent, prefix=".synapse-") as tmp:
            fetched = self.connection().get(entity["id"], version=entity["version"], downloadLocation=tmp)
            saved = Path(fetched.path)
            digest = checksum(saved)
            saved.replace(path)
            write_json(receipt, {"entity": entity, "sha256": digest})
        return digest


def _checked_mask(root: Path, row: dict, files: SynapseFiles) -> dict:
    patient = row["patient"]
    path = root / "raw/MAMA-MIA" / patient / f"{patient}.nii.gz"
    sha = files.fetch(row["mask"], path)
    mask, grid = read_nifti(path)
    if np.count_nonzero(mask) == 0 or not np.isin(mask, (0, 1)).all():
        raise ValueError("A reference contour must be a nonempty binary mask.")
    return {**row, "mask_sha256": sha, "volume_ml": float(np.count_nonzero(mask) * grid.voxel_mm3 / 1000)}


def _status(error: Exception) -> int | None:
    return getattr(getattr(error, "response", None), "status_code", None)


def retry(operation, *args):
    for delay in (0, 3, 10):
        if delay:
            sleep(delay)
        try:
            return operation(*args)
        except (ValueError, FileNotFoundError):
            raise
        except Exception as error:
            if _status(error) in (400, 401, 403, 404) or delay == 10:
                raise


def acquire_masks(previous: Path, root: Path, token_file: Path, workers: int = 8) -> Path:
    output = root / "acquisition/mama-mask-inventory.json"
    if output.is_file():
        return output
    files = SynapseFiles(root / "acquisition/.synapse-cache", token_file)
    rows = catalog(previous, root, files)
    try:
        retry(_checked_mask, root, rows[0], files)
    except Exception as error:
        if _status(error) in (401, 403):
            raise RuntimeError(f"Synapse rejected the supplied authorization (HTTP {_status(error)}).") from None
        print(f"First mask returned {type(error).__name__}; continuing with the catalog.", flush=True)
    print(f"Synapse expert contour inventory has {len(rows)} candidate cases.", flush=True)
    records, failures = [], []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        waiting = {pool.submit(retry, _checked_mask, root, row, files): row for row in rows}
        for index, future in enumerate(as_completed(waiting), 1):
            try:
                records.append(future.result())
            except Exception as error:
                failures.append({"patient": waiting[future]["patient"], "error": type(error).__name__,
                                 "http": _status(error)})
            if index % 25 == 0 or index == len(rows):
                print(f"Checked contours {index}/{len(rows)}; verified {len(records)}.", flush=True)
    if not records:
        raise RuntimeError("No expert contours could be acquired from Synapse.")
    write_json(output, {"release": "syn60868042", "cases": sorted(records, key=lambda row: row["patient"]),
                        "excluded": failures})
    return output


def _get_phases(directory: Path, row: dict, files: SynapseFiles) -> list[dict]:
    manifest = directory / "phase-versions.json"
    if manifest.is_file():
        return read_json(manifest)
    indexed = {item["name"]: item for item in files.connection().getChildren(row["folder"])}
    phases = []
    for phase in (0, 1):
        name = f"{row['patient']}_{phase:04d}.nii.gz"
        entry = indexed.get(name)
        if entry is None:
            raise ValueError("The release has no documented precontrast and first postcontrast pair.")
        version = entry.get("versionNumber")
        if version is None:
            version = files.connection().get(entry["id"], downloadFile=False).versionNumber
        phases.append({"name": name, "entity": {"id": entry["id"], "version": int(version)}})
    write_json(manifest, phases)
    return phases


def _prepare_subject(root: Path, row: dict, files: SynapseFiles, cpu_slots: Semaphore) -> dict:
    patient = row["patient"]
    cohort = f"MAMA-MIA-{row['collection']}"
    directory = root / "raw/MAMA-MIA" / patient
    destination = root / "prepared/MAMA-MIA" / patient
    outcome = {"patient": patient, "cohort": cohort, "collection": row["collection"],
               "path": destination.relative_to(root / "prepared").as_posix(), "volume_ml": row["volume_ml"]}
    phase_rows = _get_phases(directory, row, files)
    provenance = [{**phase, "sha256": files.fetch(phase["entity"], directory / phase["name"])}
                  for phase in phase_rows]
    mask_path = directory / f"{patient}.nii.gz"
    if checksum(mask_path) != row["mask_sha256"]:
        raise ValueError("An expert contour changed after the mask inventory was written.")
    source = {"type": "nifti", "dataset_release": "syn60868042",
              "end_point": "expert_reviewed_primary_tumor", "model_geometry": "lps_1.5mm",
              "phases": provenance, "label": {**row["mask"], "sha256": row["mask_sha256"]}}
    with cpu_slots:
        if destination.exists():
            present = read_json(destination / "case.json")
            if present["reference"] != source or present["patient"] != patient:
                raise ValueError("The cached canonical exam differs from the pinned source files.")
            return outcome
        pre, pre_grid = read_nifti(directory / phase_rows[0]["name"])
        post, post_grid = read_nifti(directory / phase_rows[1]["name"])
        mask, mask_grid = read_nifti(mask_path)
        if not same_grid(pre_grid, post_grid):
            raise ValueError("The two released DCE phases do not share the documented lattice.")
        require_coverage(mask, mask_grid, post_grid)
        canonical = lps_grid(post_grid)
        _atomic_case(destination, dce_channels(pre.astype(np.float32), post.astype(np.float32)), post_grid,
                     {"lesion": (mask, mask_grid, "manual")}, source, patient, cohort,
                     f"mama_mia_{row['collection'].lower()}_early_dce", target_grid=canonical)
    return outcome


def acquire_selected_phases(previous: Path, root: Path, token_file: Path, workers: int = 8) -> dict:
    inventory = root / "acquisition/mama-mask-inventory.json"
    seen, historical = exposure(previous)
    output = root / "acquisition/mama-verified-phases.json"
    if output.is_file():
        receipt = read_json(output)
        if receipt["mask_inventory_sha256"] != checksum(inventory):
            raise ValueError("The phase receipt no longer matches the mask inventory.")
        return freeze_partitions(previous, root, receipt["cases"])
    selected = [row for row in read_json(inventory)["cases"]
                if patient_key(row["patient"]) not in historical
                and (row["collection"] != "DUKE" or patient_key(row["patient"]) not in seen)]
    selected.sort(key=lambda row: (row["collection"] == "DUKE", row["patient"]))
    files = SynapseFiles(root / "acquisition/.synapse-cache", token_file)
    cpu_slots = Semaphore(2)
    print(f"Preparing the documented phase pairs for {len(selected)} eligible subject candidates.", flush=True)
    records, failures = [], []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        waiting = {pool.submit(retry, _prepare_subject, root, row, files, cpu_slots): row for row in selected}
        for i, future in enumerate(as_completed(waiting), 1):
            try:
                records.append(future.result())
            except Exception as error:
                failures.append({"patient": waiting[future]["patient"], "error": type(error).__name__,
                                 "http": _status(error)})
            if i % 15 == 0 or i == len(waiting):
                print(f"Processed paired exams {i}/{len(waiting)}; verified {len(records)}.", flush=True)
    if not records:
        raise RuntimeError("No valid paired images could be prepared.")
    write_json(output, {"mask_inventory_sha256": checksum(inventory),
                        "cases": sorted(records, key=lambda row: row["patient"]), "excluded": failures})
    return freeze_partitions(previous, root, records)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("previous", type=Path)
    parser.add_argument("root", type=Path)
    parser.add_argument("--token-file", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--phases", action="store_true", help="prepare phase pairs after verifying contour inventory")
    args = parser.parse_args()
    previous, root = args.previous.resolve(), args.root.resolve()
    print(acquire_masks(previous, root, args.token_file, args.workers), flush=True)
    if args.phases:
        receipt = acquire_selected_phases(previous, root, args.token_file, args.workers)
        print(json.dumps(receipt["counts"], indent=2), flush=True)


if __name__ == "__main__":
    main()
