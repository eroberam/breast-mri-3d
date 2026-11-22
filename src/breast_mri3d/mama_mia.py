"""Selectively fetch expert MAMA-MIA masks and their two DCE phases from Synapse."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import logging
from pathlib import Path
import re

import numpy as np

from .acquire import checksum, write_json
from .imaging import dce_channels, read_nifti
from .iteration2 import _atomic_case, require_coverage, same_grid


EXPERT_FOLDER = "syn64871175"
IMAGE_FOLDER = "syn64871114"


def _client(root: Path, token: str):
    import synapseclient

    logging.getLogger("synapseclient").setLevel(logging.CRITICAL)
    client = synapseclient.Synapse(cache_root_dir=str(root / ".cache"),
                                   configPath=str(root / ".client-config"), cache_client=False, silent=True)
    client.login(authToken=token, silent=True)
    return client


def prepare_mama(root: Path, *, token_file: Path, limit: int = 32, workers: int = 3) -> Path:
    token = token_file.read_text(encoding="utf-8-sig").strip()
    token = re.sub(r"^(?:Bearer\s+|(?:SYNAPSE_AUTH_TOKEN|authToken)\s*=\s*)", "", token, flags=re.I).strip('"\'')
    if not re.fullmatch(r"[A-Za-z0-9_.-]{20,}", token):
        raise ValueError("The supplied file does not contain a supported Synapse token format.")
    if limit < 10:
        raise ValueError("Select at least ten cases for both source strata.")
    raw = root / "raw/MAMA-MIA"
    prepared = root / "prepared/MAMA-MIA"
    raw.mkdir(parents=True, exist_ok=True)
    meta = root / "acquisition"
    client = _client(raw, token)
    previous = meta / "mama-mia-synapse-index.json"
    mapping = meta / "mama-mia-expert-index.json"
    if previous.exists() and mapping.exists():
        folders = json.loads(previous.read_text(encoding="utf-8"))[IMAGE_FOLDER]["children"]
        masks = json.loads(mapping.read_text(encoding="utf-8"))["expert_children"]
    else:
        folders = list(client.getChildren(IMAGE_FOLDER))
        masks = list(client.getChildren(EXPERT_FOLDER))
    mask_ids = {row["name"].removesuffix(".nii.gz"): row["id"] for row in masks if row["name"].endswith(".nii.gz")}
    candidates = []
    for prefix, cap in [("DUKE_", limit // 2), ("ISPY2_", limit - limit // 2)]:
        group = [row for row in folders if row["name"].startswith(prefix) and row["name"] in mask_ids]
        group.sort(key=lambda row: hashlib.sha256(f"19:{row['name']}".encode()).hexdigest())
        group = group[:cap]
        split_at = max(1, round(.2 * len(group)))
        for i, row in enumerate(group):
            candidates.append({"patient": row["name"], "folder": row["id"], "mask": mask_ids[row["name"]],
                               "split": "validation" if i < split_at else "train"})
    selection = meta / "mama-mia-selection.json"
    choice = {"release": "syn60868042", "seed": 19, "cases": candidates}
    if selection.exists() and json.loads(selection.read_text(encoding="utf-8")) != choice:
        raise ValueError("The selected MAMA-MIA cohort is already locked.")
    write_json(selection, choice)

    def process(case):
        patient = case["patient"]
        directory = raw / patient
        directory.mkdir(exist_ok=True)
        phases = {row["name"]: row["id"] for row in client.getChildren(case["folder"])}
        ids = [phases[f"{patient}_{i:04d}.nii.gz"] for i in range(2)] + [case["mask"]]
        sources, files = [], []
        for entity in ids:
            stored = client.get(entity, downloadFile=False)
            current = client.get(entity, version=stored.versionNumber, downloadLocation=str(directory))
            path = Path(current.path)
            sources.append({"entity": entity, "version": stored.versionNumber, "sha256": checksum(path)})
            files.append(path)
        pre, pre_grid = read_nifti(files[0])
        post, post_grid = read_nifti(files[1])
        mask, mask_grid = read_nifti(files[2])
        if not same_grid(pre_grid, post_grid) or not np.isin(mask, (0, 1)).all():
            raise ValueError("MAMA-MIA paired phases and expert labels lack the required geometry or encoding.")
        coverage = require_coverage(mask, mask_grid, post_grid)
        channels = dce_channels(pre.astype(np.float32), post.astype(np.float32))
        provenance = {"type": "nifti", "dataset_release": "syn60868042", "files": sources,
                      "end_point": "expert_reviewed_primary_tumor", "label_fov_coverage": coverage}
        _atomic_case(prepared / patient, channels, post_grid, {"lesion": (mask, mask_grid, "manual")},
                     provenance, patient, "MAMA-MIA", f"mama_mia_{patient.split('_')[0].lower()}_early_dce")
        return {"path": patient, "patient": patient, "cohort": "MAMA-MIA", "split": case["split"]}

    verified, failures = [], []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(process, patient): patient for patient in candidates}
        for future in as_completed(futures):
            try:
                verified.append(future.result())
            except Exception as exc:
                # Provider errors can include signed URLs; retain the class and status only.
                status = getattr(getattr(exc, "response", None), "status_code", None)
                failures.append({"patient": futures[future]["patient"], "error": type(exc).__name__, "http_status": status})
            if (len(verified) + len(failures)) % 4 == 0:
                print(f"MAMA-MIA expert cases verified {len(verified)}, excluded {len(failures)}", flush=True)
    manifest = prepared / "manifest.json"
    write_json(manifest, {"purpose": "expert_reviewed_primary_tumor", "selection_sha256": checksum(selection),
                          "cases": sorted(verified, key=lambda row: row["patient"]), "excluded": failures})
    return manifest
