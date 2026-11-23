"""Freeze disjoint patient partitions after phase and geometry checks."""

from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re

from .acquire import checksum, write_json
from .research import read_manifest


HISTORICAL_INPUT = "prepared/v02-lesion-dev.json"
EXPOSURE_FILES = (
    HISTORICAL_INPUT, "prepared/v02-anatomy-dev.json", "prepared/v02-anatomy-dev-with-duke.json",
    "prepared/v02-calibration-dev.json", "prepared/Duke3D/manifest.json", "prepared/MAMA-MIAAI/manifest.json",
    "prepared/ISPY1-STV/manifest.json", "acquisition/v02-pilot-history.json",
    "acquisition/duke3d-selection.json", "acquisition/ispy1-stv-selection.json",
    "acquisition/mama-mia-selection.json",
)
MANIFESTS = {"development": "development.json", "internal": "internal-test.json", "duke": "duke-test.json"}


def patient_key(value: str) -> str:
    text = value.strip().upper().replace("BREAST_MRI_", "DUKE_")
    text = re.sub(r"^ACRIN[-_]6657[-_]", "ISPY1_", text)
    match = re.fullmatch(r"(DUKE|NACT|ISPY1|ISPY2)[_-]0*([0-9]+)", text)
    if match:
        return f"{match[1]}_{int(match[2])}"
    return re.sub(r"[^A-Z0-9]", "", text)


def exposure(previous: Path) -> tuple[set[str], set[str]]:
    paths = [previous / name for name in EXPOSURE_FILES if (previous / name).is_file()]
    seen = set()
    for path in paths:
        rows = json.loads(path.read_text(encoding="utf-8-sig"))["cases"]
        for row in rows:
            seen.update(patient_key(row[field]) for field in ("patient", "source_patient") if row.get(field))
    historical = {patient_key(case.patient) for case in read_manifest(previous / HISTORICAL_INPUT)
                  if case.split == "validation"}
    return seen, historical


def volume_band(volume_ml: float) -> int:
    return sum(volume_ml > edge for edge in (3.141592653589793 / 6, 5., 20.))


def stratified(rows: list[dict], count: int, seed: int) -> set[str]:
    if not 0 <= count <= len(rows):
        raise ValueError("The holdout size exceeds the number of eligible unique patients.")
    groups = defaultdict(list)
    for row in rows:
        groups[row["collection"], volume_band(row["volume_ml"])].append(row)
    if not count:
        return set()
    quotas = {key: divmod(len(group) * count, len(rows)) for key, group in groups.items()}
    remaining = count - sum(whole for whole, _ in quotas.values())
    extras = set(sorted(groups, key=lambda key: (-quotas[key][1], key))[:remaining])
    chosen = set()
    for key, group in groups.items():
        order = sorted(group, key=lambda row: hashlib.sha256(f"{seed}:{row['patient']}".encode()).digest())
        chosen.update(patient_key(row["patient"]) for row in order[:quotas[key][0] + (key in extras)])
    return chosen


def assign(rows: list[dict], seen: set[str], historical: set[str], *,
           internal: int = 320, adjustment: int = 120, seed: int = 19) -> dict[str, list[dict]]:
    keys = [patient_key(row["patient"]) for row in rows]
    if len(set(keys)) != len(keys):
        raise ValueError("Duplicate original patients must be resolved before locking the partitions.")
    fresh = [row for row, key in zip(rows, keys) if row["collection"] != "DUKE" and key not in seen | historical]
    internal_ids = stratified(fresh, internal, seed)
    adjustment_ids = stratified([r for r in fresh if patient_key(r["patient"]) not in internal_ids],
                                adjustment, seed + 1)
    result = {k: [] for k in MANIFESTS}
    for row, key in zip(rows, keys):
        if key in historical or row["collection"] == "DUKE" and key in seen:
            continue
        if row["collection"] == "DUKE":
            name, split = "duke", "test"
        elif key in internal_ids:
            name, split = "internal", "test"
        else:
            name, split = "development", "validation" if key in adjustment_ids else "train"
        result[name].append({**row, "patient_key": key, "split": split})
    for values in result.values():
        values.sort(key=lambda row: row["patient_key"])
    return result


def verify_partitions(root: Path) -> dict:
    directory = root / "prepared"
    record = json.loads((directory / "partition-lock.json").read_text(encoding="utf-8"))
    observed = set()
    for name, path in MANIFESTS.items():
        if checksum(directory / path) != record["manifests"][name]:
            raise ValueError("The frozen patient manifest changed.")
        cases = read_manifest(directory / path)
        if dict(Counter(c.split for c in cases)) != record["counts"][name]:
            raise ValueError("The locked partition counts have changed.")
        for case in cases:
            key = patient_key(case.patient)
            valid_splits = {"train", "validation"} if name == "development" else {"test"}
            if key in observed or (key.startswith("DUKE_") != (name == "duke")) or case.split not in valid_splits:
                raise ValueError("A patient is duplicated or assigned to an incompatible locked cohort.")
            observed.add(key)
            if checksum(case.path / "case.json") != record["case_receipts"][key]:
                raise ValueError("A frozen case provenance record changed.")
    return record


def freeze_partitions(previous: Path, root: Path, eligible: list[dict], *,
                      internal: int = 320, adjustment: int = 120) -> dict:
    directory = root / "prepared"
    lock = directory / "partition-lock.json"
    if lock.exists():
        return verify_partitions(root)
    seen, historical = exposure(previous)
    partitions = assign(eligible, seen, historical, internal=internal, adjustment=adjustment)
    manifests, receipts, counts = {}, {}, {}
    for name, path in MANIFESTS.items():
        cases = partitions[name]
        if not cases:
            raise ValueError(f"No verified cases available for {name}.")
        for row in cases:
            receipts[row["patient_key"]] = checksum(directory / row["path"] / "case.json")
        manifest = directory / path
        write_json(manifest, {"purpose": f"v03_{name}", "cases": cases})
        manifests[name] = checksum(manifest)
        counts[name] = dict(Counter(row["split"] for row in cases))
    record = {"seed": 19, "manifests": manifests, "case_receipts": receipts, "counts": counts,
              "historical_controls": len(historical), "previous_exposure_count": len(seen)}
    write_json(lock, record)
    return verify_partitions(root)
