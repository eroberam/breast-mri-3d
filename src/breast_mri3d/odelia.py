"""Read approved ODELIA shards by column and prepare bilateral negative exams."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import re

import numpy as np

from .acquire import checksum, write_json
from .geometry import Grid
from .imaging import dce_channels
from .iteration2 import _atomic_case, same_grid


REPOSITORY = "ODELIA-AI/ODELIA-Challenge-2025"
META_COLUMNS = ["PatientID", "Institution", "Split", "Lesion_Left", "Lesion_Right"]
DCE_COLUMNS = ["Image_Pre", "Affine_Pre", "Image_Post_1", "Affine_Post_1"]


def _volume(value, affine) -> tuple[np.ndarray, Grid]:
    import pyarrow as pa

    if not value.is_valid or not affine.is_valid:
        raise ValueError("The requested DCE phase is unavailable.")
    data = value.values
    shape = [len(data)]
    while pa.types.is_list(data.type) or pa.types.is_large_list(data.type) or pa.types.is_fixed_size_list(data.type):
        if pa.types.is_fixed_size_list(data.type):
            size = data.type.list_size
        else:
            lengths = data.value_lengths().to_numpy(zero_copy_only=False)
            if len(lengths) == 0 or not np.all(lengths == lengths[0]):
                raise ValueError("A DCE phase contains ragged voxel dimensions.")
            size = int(lengths[0])
        shape.append(size)
        data = data.flatten()
    if len(shape) != 4 or shape[0] != 1 or min(shape) < 1:
        raise ValueError("The DCE payload does not contain one three-dimensional MR channel.")
    volume = data.to_numpy(zero_copy_only=False).reshape(shape)[0].transpose(2, 1, 0)
    ras = np.asarray(affine.as_py(), dtype=float)
    if ras.shape != (4, 4):
        raise ValueError("Expected the provider's RAS affine in millimetres.")
    reverse_axes = np.eye(4)[[2, 1, 0, 3]]
    lps = np.diag([-1, -1, 1, 1]) @ ras @ reverse_axes
    return volume, Grid(volume.shape, lps)


def prepare_odelia(root: Path, *, token_file: Path, limit: int = 80, workers: int = 5) -> Path:
    from huggingface_hub import HfApi, HfFileSystem
    import pyarrow.parquet as pq

    match = re.search(r"\bhf_[A-Za-z0-9]+\b", token_file.read_text(encoding="utf-8-sig"))
    if not match:
        raise ValueError("The supplied file does not contain a Hugging Face token.")
    fs = HfFileSystem(token=match[0])
    meta_path = root / "acquisition/odelia-release.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    else:
        repo = HfApi(token=match[0]).dataset_info(REPOSITORY, files_metadata=True)
        meta = {"revision": repo.sha, "files": [{"path": f.rfilename, "bytes": f.size} for f in repo.siblings]}
        write_json(meta_path, meta)
    base = f"datasets/{REPOSITORY}@{meta['revision']}"
    shards = [row["path"] for row in meta["files"] if re.fullmatch(r"data/(?:train|val)-[^/]+\.parquet", row["path"])]
    if not shards:
        raise ValueError("Bilateral development shards are unavailable.")

    def caselist(shard):
        records = []
        with fs.open(f"{base}/{shard}", "rb", block_size=262144) as stream:
            reader = pq.ParquetFile(stream)
            for group in range(reader.num_row_groups):
                table = reader.read_row_group(group, columns=META_COLUMNS)
                for row, item in enumerate(table.to_pylist()):
                    if not item["PatientID"] or not item["Institution"] or str(item["Institution"]).upper() == "RSH":
                        continue
                    split = {"train": "train", "val": "validation", "validation": "validation"}.get(str(item["Split"]).lower())
                    if split is None:
                        continue
                    identity = f"{str(item['Institution']).upper()}:{item['PatientID']}"
                    patient = f"ODELIA-{hashlib.sha256(identity.encode()).hexdigest()[:20]}"
                    records.append({"patient": patient, "cohort_site": item["Institution"], "split": split,
                                    "negative": item["Lesion_Left"] == item["Lesion_Right"] == 0,
                                    "shard": shard, "group": group, "row": row})
        return records

    acquisition = root / "acquisition/odelia-negative-selection.json"
    if acquisition.exists():
        chosen = json.loads(acquisition.read_text(encoding="utf-8"))
        if chosen["revision"] != meta["revision"] or chosen["limit"] != limit:
            raise ValueError("The negative ODELIA selection is already locked to a release and size.")
        selection = chosen["cases"]
    else:
        roles: dict[str, set[str]] = {}
        negatives = []
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(caselist, shard) for shard in shards]
            for i, future in enumerate(as_completed(futures), 1):
                for row in future.result():
                    roles.setdefault(row["patient"], set()).add(row["split"])
                    if row["negative"]:
                        negatives.append(row)
                if i % 50 == 0:
                    print(f"ODELIA metadata shards scanned {i}/{len(shards)}", flush=True)
        selection, seen = [], set()
        negatives.sort(key=lambda row: (hashlib.sha256(f"19:{row['patient']}".encode()).digest(), row["shard"], row["row"]))
        for split, quota in (("train", round(.75 * limit)), ("validation", limit - round(.75 * limit))):
            count = 0
            for row in negatives:
                patient = row["patient"]
                if row["split"] != split or len(roles[patient]) != 1 or patient in seen:
                    continue
                selection.append(row)
                seen.add(patient)
                count += 1
                if count >= quota:
                    break
        write_json(acquisition, {"revision": meta["revision"], "limit": limit, "cases": selection})

    prepared = root / "prepared/ODELIA"

    def process(candidate):
        patient = candidate["patient"]
        destination = prepared / patient
        if not (destination / "case.json").exists():
            with fs.open(f"{base}/{candidate['shard']}", "rb", block_size=1024 ** 2) as stream:
                table = pq.ParquetFile(stream).read_row_group(candidate["group"], columns=DCE_COLUMNS)
            row = candidate["row"]
            pre, pre_grid = _volume(table["Image_Pre"][row], table["Affine_Pre"][row])
            post, post_grid = _volume(table["Image_Post_1"][row], table["Affine_Post_1"][row])
            if not same_grid(pre_grid, post_grid):
                raise ValueError("The two documented ODELIA phases have different physical grids.")
            source = {"type": "nifti", "dataset_release": f"{REPOSITORY}@{meta['revision']}",
                      "shard": candidate["shard"], "row_group": candidate["group"], "row": row,
                      "label": "bilateral_lesion_code_zero", "native_footprint": post_grid.as_dict()}
            _atomic_case(destination, dce_channels(pre.astype(np.float32), post.astype(np.float32)), post_grid,
                         {"lesion": (np.zeros(post_grid.shape, dtype=np.uint8), post_grid, "confirmed_empty")},
                         source, patient, "ODELIA", f"odelia_{str(candidate['cohort_site']).lower()}_early_post")
        return {"patient": patient, "path": patient, "cohort": "ODELIA", "split": candidate["split"]}

    correct, skipped = [], []
    with ThreadPoolExecutor(max_workers=min(workers, 3)) as pool:
        futures = {pool.submit(process, row): row for row in selection}
        for future in as_completed(futures):
            try:
                correct.append(future.result())
            except Exception as exc:
                # Leave authenticated URL details out of preparation logs and reports.
                skipped.append({"patient": futures[future]["patient"], "error": type(exc).__name__})
            if (len(correct) + len(skipped)) % 5 == 0:
                print(f"ODELIA paired negative exams {len(correct)}, excluded {len(skipped)}", flush=True)
    result = prepared / "manifest.json"
    write_json(result, {"purpose": "bilateral_lesion_negative", "selection_sha256": checksum(acquisition),
                        "cases": sorted(correct, key=lambda row: row["patient"]), "excluded": skipped})
    return result
