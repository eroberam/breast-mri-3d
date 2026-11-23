"""Run the 45-minute v0.2 replay with an exact orientation change."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import shutil
import subprocess
import sys
from tempfile import TemporaryDirectory

import numpy as np

from breast_mri3d.acquire import checksum, write_json
from breast_mri3d.geometry import reorient_lps
from breast_mri3d.research import Case, read_manifest


def prepare_case(case: Case, destination: Path, parent_digest: str) -> dict:
    path = destination / case.patient
    image = np.load(case.path / "image.npy", mmap_mode="r", allow_pickle=False)
    image, grid = reorient_lps(image, case.grid)
    meta = {**case.metadata, "grid": grid.as_dict(), "source_model_grid": case.grid.as_dict(),
            "orientation": "lps_voxel_permutation", "parent_manifest_sha256": parent_digest}
    if path.exists():
        prior = json.loads((path / "case.json").read_text(encoding="utf-8"))
        if prior != meta:
            raise ValueError("The existing pilot case belongs to a different input release.")
    else:
        with TemporaryDirectory(dir=destination, prefix=".axis-") as tmp:
            stage = Path(tmp) / "case"
            stage.mkdir()
            np.save(stage / "image.npy", image, allow_pickle=False)
            labels = np.load(case.path / "labels.npy", mmap_mode="r", allow_pickle=False)
            labels, _ = reorient_lps(labels, case.grid)
            np.save(stage / "labels.npy", labels, allow_pickle=False)
            shutil.copytree(case.path / "native_labels", stage / "native_labels")
            write_json(stage / "case.json", meta)
            del image, labels
            stage.rename(path)
    return {"patient": case.patient, "cohort": case.cohort, "split": case.split,
            "path": path.relative_to(destination.parent).as_posix()}


def run_module(log: Path, module: str, *args) -> None:
    with log.open("w", encoding="utf-8") as stream:
        subprocess.run([sys.executable, "-u", "-m", module, *map(str, args)], stdout=stream,
                       stderr=subprocess.STDOUT, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("previous", type=Path)
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    previous, root = args.previous.resolve(), args.root.resolve()
    source = previous / "prepared/v02-lesion-dev.json"
    original_fit = previous / "runs/v02/c0-supervised.training.json"
    fit = json.loads(original_fit.read_text(encoding="utf-8"))
    threshold_lock = json.loads((previous / "runs/v02/external-lock.json").read_text(encoding="utf-8"))["compact"]
    digest = checksum(source)
    if fit["manifest_sha256"] != digest or fit["training_cases"] != 121 or fit["budget_seconds"] != 2700:
        raise ValueError("The historical fit does not match the expected 121-case, 45-minute development split.")
    cases = read_manifest(source)
    folder = root / "prepared/pilot-lps"
    folder.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=3) as pool:
        rows = list(pool.map(lambda case: prepare_case(case, folder, digest), cases))
    manifest = folder.parent / "pilot-lps.json"
    write_json(manifest, {"purpose": "v02_orientation_control", "source_manifest_sha256": digest, "cases": rows})
    print(f"Prepared {len(rows)} historical cases with an exact voxel permutation.", flush=True)
    run = root / "runs/pilot"
    run.mkdir(parents=True, exist_ok=True)
    checkpoint = run / "canonical.pt"
    if not checkpoint.exists():
        run_module(run / "train.log", "breast_mri3d.cli", "train", manifest, "--out", checkpoint,
                   "--steps", 1000000, "--budget-seconds", 2700, "--width", 12,
                   "--seed", fit["seed"], "--patch", *fit["patch_zyx"], "--device", "cuda")
    else:
        prior = json.loads(checkpoint.with_suffix(".training.json").read_text(encoding="utf-8"))
        if prior["manifest_sha256"] != checksum(manifest):
            raise ValueError("The saved orientation fit was trained on a different manifest.")
    prediction = run / "validation.json"
    if not prediction.exists():
        run_module(run / "evaluate.log", "breast_mri3d.cli", "evaluate", manifest, checkpoint,
                   "--split", "validation", "--threshold", threshold_lock["threshold"], "--out", prediction)
    comparison = root / "reports/orientation-pilot.json"
    if not comparison.exists():
        run_module(run / "paired.log", "breast_mri3d.experiment", "compare",
                   previous / "runs/v02/compact-validation-instances.json", prediction, "--out", comparison)
    print("Completed the historical orientation comparison.", flush=True)


if __name__ == "__main__":
    main()
