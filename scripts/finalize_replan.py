"""Freeze the adjustment-selected model and evaluate the three original holdouts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from filelock import FileLock

from breast_mri3d.acquire import checksum, write_json
from breast_mri3d.cohorts_v03 import MANIFESTS, verify_partitions
from breast_mri3d.reference_v03 import CONFIGURATION, DATASET_NAME, TRAINER
from breast_mri3d.research import read_manifest
from breast_mri3d.study_v03 import Study
from breast_mri3d.validation_v03 import paired_patients
from package_mamamia import build
from replan_adjustment import read


COMPONENT_LIMIT = {"raw": None, "largest": 1, "two_largest": 2}


def freeze_candidate(study: Study, lock: Path) -> dict:
    if lock.is_file():
        return read(lock)
    partition = study.prepared / "partition-lock.json"
    review = study.reports / "replan-context-matched.json"
    diagnostic = read(review)
    origin = study.runs / "replan-distillation"
    context = study.runs / "replan-context"
    base_patch = read(origin / "recipe.json")["patch_zyx"]
    large_patch = diagnostic["large_patch_zyx"]
    arms = {
        "control": (origin / "control.pt", origin / "control/inputs.json", base_patch),
        "inference_context": (origin / "control.pt", context / "inference_context/inputs.json", large_patch),
        "training_context": (context / "context.pt", context / "training_context/inputs.json", large_patch),
    }
    candidates = {}
    for name, (path, prediction_inputs, patch) in arms.items():
        scored = read(prediction_inputs)
        artifact = study.relative_receipt(path)
        if scored["checkpoint_sha256"] != artifact["sha256"] or scored["manifest_sha256"] != checksum(study.development):
            raise ValueError("An adjustment prediction was made from a different checkpoint or cohort.")
        chosen = diagnostic["chosen"][name]
        physical = diagnostic["physical_comparison"][name]["all"]
        candidates[name] = {"name": name, "model": artifact, "patch_zyx": patch,
                            "mirror_left_right": chosen["view"] == "mirrored", "threshold": chosen["threshold"],
                            "postprocessing": chosen["postprocessing"], "locator": None,
                            "adjustment_mean_dice": chosen["mean_dice"], "adjustment_zero_overlap": chosen["zero_overlap"],
                            "adjustment_matched": physical["proposed"]["matched"]}
    reference_result = diagnostic["physical_comparison"]["control"]["all"]["reference"]
    reference_choice = diagnostic["chosen"]["reference"]
    eligible = [c for c in candidates.values() if c["adjustment_matched"] >= reference_result["matched"]
                and c["adjustment_zero_overlap"] <= reference_choice["zero_overlap"]]
    selected = max(eligible, key=lambda candidate: candidate["adjustment_mean_dice"]) if eligible else max(
        candidates.values(), key=lambda candidate: (candidate["adjustment_matched"], candidate["adjustment_mean_dice"]))
    plans = read(study.runs / "reference-plans.json")["plans"]
    reference = (study.reference_results / DATASET_NAME / f"{TRAINER}__{plans}__{CONFIGURATION}" /
                 "fold_0/checkpoint_6h.pth")
    manifests = {name: study.relative_receipt(study.prepared / MANIFESTS[name]) for name in ("internal", "duke")}
    negatives = study.prepared / "ODELIA/manifest.json"
    negative_cases = read_manifest(negatives)
    if (not negative_cases or any(case.split != "test" or case.metadata["labels"].get("lesion") != "confirmed_empty"
                                  for case in negative_cases)):
        raise ValueError("The ODELIA evaluation needs only documented bilateral negative test exams.")
    manifests["odelia"] = study.relative_receipt(negatives)
    frozen = {"protocol": "replanned_v03", "partition": study.relative_receipt(partition),
              "adjustment": study.relative_receipt(review), "cohorts": manifests, "proposed": selected,
              "candidate_selection": "native_mean_dice_with_reference_detection_constraint",
              "selection_constraint_met": bool(eligible), "candidates": list(candidates.values()),
              "reference": {"model": study.relative_receipt(reference), "threshold": .5,
                            "postprocessing": reference_choice["postprocessing"], "mirror": True,
                            "adjustment_mean_dice": reference_choice["mean_dice"],
                            "adjustment_matched": reference_result["matched"]}}
    write_json(lock, frozen)
    return frozen


def run(root: Path, trainer_dir: Path) -> Path:
    study = Study(root, trainer_dir)
    verify_partitions(study.root)
    directory = study.runs / "replan-final"
    directory.mkdir(exist_ok=True)
    lock = directory / "evaluation-lock.json"
    frozen = freeze_candidate(study, lock)
    study.checked_path(frozen["partition"])
    study.checked_path(frozen["adjustment"])
    proposed, reference = frozen["proposed"], frozen["reference"]
    weights, comparator = study.checked_path(proposed["model"]), study.checked_path(reference["model"])
    lock_sha = checksum(lock)
    print(json.dumps({"selected": proposed["name"], "patch": proposed["patch_zyx"],
                      "adjustment_dice": proposed["adjustment_mean_dice"], "evaluation_lock_sha256": lock_sha}), flush=True)
    archive = study.root / "benchmark/MAMA-MIA-task1-replanned.zip"
    if not archive.is_file():
        build(study.root, lock, archive)
    if checksum(archive) != read(archive.with_suffix(".json"))["sha256"]:
        raise ValueError("The local blind archive differs from its packaged receipt.")
    for name, item in frozen["cohorts"].items():
        manifest = study.checked_path(item)
        output = directory / name
        output.mkdir(exist_ok=True)
        marker = output / "inputs.json"
        identity = {"evaluation_lock_sha256": lock_sha, "manifest_sha256": item["sha256"]}
        if marker.is_file() and read(marker) != identity:
            raise ValueError("The held-out predictions belong to a different frozen evaluation.")
        write_json(marker, identity)
        own, base = output / "compact.json", output / "reference.json"
        if not own.is_file():
            options = ["--patch", *proposed["patch_zyx"], "--threshold", proposed["threshold"]]
            if proposed["mirror_left_right"]:
                options.append("--mirror-left-right")
            limit = COMPONENT_LIMIT[proposed["postprocessing"]]
            if limit is not None:
                options.extend(("--component-limit", limit))
            study.module(f"replan-final-{name}-compact", "validation_v03", manifest, weights, "--split", "test",
                         "--out", own, *options)
        if not base.is_file():
            limit = COMPONENT_LIMIT[reference["postprocessing"]]
            options = ["--component-limit", limit] if limit is not None else []
            study.module(f"replan-final-{name}-reference", "reference_v03", "evaluate", manifest, comparator,
                         output / "nnunet", base, "--split", "test", *options)
        paired = paired_patients(read(base), read(own))
        write_json(study.reports / f"replan-final-{name}-paired.json", {"evaluation_lock_sha256": lock_sha, **paired})
    receipt = study.reports / "replan-final-completed.json"
    write_json(receipt, {"evaluation_lock_sha256": lock_sha, "cohorts": list(frozen["cohorts"]),
                         "benchmark_archive": study.relative_receipt(archive), "submission": "local"})
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--trainer-dir", type=Path, required=True)
    parser.add_argument("--gpu-wait-seconds", type=float, default=1800.)
    args = parser.parse_args()
    with FileLock(args.root / "nnunet/gpu-training.lock", timeout=args.gpu_wait_seconds):
        print(run(args.root, args.trainer_dir), flush=True)


if __name__ == "__main__":
    main()
