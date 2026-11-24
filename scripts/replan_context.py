"""Separate larger inference context from larger training context at matched updates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from filelock import FileLock

from breast_mri3d.acquire import checksum, write_json
from breast_mri3d.research import read_manifest
from breast_mri3d.study_v03 import Study
from breast_mri3d.validation_v03 import paired_patients
from replan_adjustment import compare_cached, make_probabilities, read


PATCHES = ((80, 160, 160), (64, 144, 144))


def run(root: Path, trainer_dir: Path) -> Path:
    study = Study(root, trainer_dir)
    previous = study.runs / "replan-distillation"
    source_report = study.reports / "replan-distillation-matched.json"
    first = read(source_report)
    if first["eligible_to_extend"]:
        raise ValueError("The narrower study already identified an intervention eligible for extension.")
    recipe = read(previous / "recipe.json")
    source = (study.runs / "compact-subtraction-6h.pt" if recipe["initial"] == "subtraction" else
              study.runs / "adjustment-diagnostic/dce-intermediate.pt")
    if checksum(source) != recipe["initial_sha256"] or checksum(study.development) != recipe["development_sha256"]:
        raise ValueError("The larger-patch trial must use the same training data and initial weights.")
    control = previous / "control.pt"
    control_fit = read(control.with_suffix(".training.json"))
    if control_fit["updates"] != recipe["updates"] or control_fit["initial_checkpoint_sha256"] != recipe["initial_sha256"]:
        raise ValueError("The existing supervised checkpoint lacks the matched fitting recipe.")
    report = study.reports / "replan-context-matched.json"
    if report.is_file():
        return report
    folder = study.runs / "replan-context"
    folder.mkdir(parents=True, exist_ok=True)
    options = ["--seed", recipe["seed"], "--width", 12, "--input-mode", control_fit["input_mode"],
               "--model-spacing-mm", 1.5, "--learning-rate", recipe["learning_rate"], "--initial-checkpoint", source]
    selected_patch = folder / "patch.json"
    if selected_patch.is_file():
        patch = tuple(read(selected_patch)["patch_zyx"])
    else:
        for patch in PATCHES:
            name = f"context-probe-{patch[0]}"
            try:
                study.module(name, "cli", "train", study.development, "--out", folder / f"{name}.pt",
                             "--steps", 8, "--patch", *patch, *options)
                break
            except RuntimeError:
                log = (study.logs / f"{name}.log").read_text(encoding="utf-8", errors="replace")
                if "out of memory" not in log.lower() or patch == PATCHES[-1]:
                    raise
        write_json(selected_patch, {"patch_zyx": list(patch)})
    candidate = folder / "context.pt"
    if not candidate.is_file():
        study.module("replan-context-large", "cli", "train", study.development, "--out", candidate,
                     "--steps", recipe["updates"], "--patch", *patch, *options, "--resumable")
    fit = read(candidate.with_suffix(".training.json"))
    if fit["updates"] != control_fit["updates"] or fit["initial_checkpoint_sha256"] != recipe["initial_sha256"]:
        raise ValueError("The larger context did not preserve the matched update and initialization.")
    cases = [case for case in read_manifest(study.development) if case.split == "validation"]
    make_probabilities(cases, control, folder / "inference_context", study.development, patch_override=patch)
    make_probabilities(cases, candidate, folder / "training_context", study.development)
    names = ("control", "inference_context", "training_context")
    comparisons = compare_cached(cases, folder, names, preserve_detection=True, model_dirs={"control": previous / "control"})
    samples = comparisons.pop("selected_records")
    effects = {name: paired_patients(samples["control"], samples[name]) for name in names[1:]}
    write_json(report, {"update_control": control_fit["updates"], "large_patch_zyx": list(patch),
                       "controlled_training_seconds": {"control": control_fit["elapsed_seconds"],
                                                       "larger_context": fit["elapsed_seconds"]},
                       "matched_intervention_report_sha256": checksum(source_report), **comparisons,
                       "paired_control_comparisons": effects})
    print(json.dumps(comparisons["chosen"], indent=2), flush=True)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--trainer-dir", type=Path, required=True)
    args = parser.parse_args()
    with FileLock(args.root / "nnunet/gpu-training.lock", timeout=1800):
        print(run(args.root, args.trainer_dir), flush=True)


if __name__ == "__main__":
    main()
