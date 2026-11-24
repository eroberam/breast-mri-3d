"""Fit one matched-update prototype pair using only development data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from filelock import FileLock

from breast_mri3d.acquire import checksum, write_json
from breast_mri3d.reference_v03 import DATASET_NAME
from breast_mri3d.research import read_manifest
from breast_mri3d.study_v03 import Study
from breast_mri3d.validation_v03 import paired_patients
from replan_adjustment import compare_cached, make_probabilities, read


def run(root: Path, trainer_dir: Path, intervention: str, updates: int) -> Path:
    study = Study(root, trainer_dir)
    report = study.reports / f"replan-{intervention}-matched.json"
    if report.is_file():
        return report
    diagnostic = study.reports / "replan-adjustment.json"
    results = read(diagnostic)["chosen"]
    initial = max(("subtraction", "dce_intermediate"), key=lambda name: results[name]["mean_dice"])
    source = (study.runs / "compact-subtraction-6h.pt" if initial == "subtraction" else
              study.runs / "adjustment-diagnostic/dce-intermediate.pt")
    source_digest = checksum(source)
    prediction_inputs = read(study.runs / "adjustment-diagnostic" / initial / "inputs.json")
    if source_digest != prediction_inputs["checkpoint_sha256"]:
        raise ValueError("The fitted initialization differs from the scored adjustment checkpoint.")
    input_mode = "subtraction" if initial == "subtraction" else "dce"
    patch = read(study.runs / "patch-selection.json")["patch_zyx"]
    identity = {"initial": initial, "initial_sha256": source_digest,
                "development_sha256": checksum(study.development), "diagnostic_sha256": checksum(diagnostic),
                "updates": updates, "seed": 19, "learning_rate": .0001, "patch_zyx": patch, "intervention": intervention}
    folder = study.runs / f"replan-{intervention}"
    folder.mkdir(parents=True, exist_ok=True)
    spec = folder / "recipe.json"
    if spec.is_file() and read(spec) != identity:
        raise ValueError("The fitted prototypes must share one frozen recipe.")
    write_json(spec, identity)
    extra = []
    teacher_summary = None
    if intervention == "distillation":
        plans = read(study.runs / "reference-plans.json")["plans"]
        reference = study.reference_checkpoint(plans, 6)
        targets = folder / "teacher"
        study.module("replan-teacher-prepare", "teacher_v03", "prepare", study.development, reference,
                     study.raw / DATASET_NAME / "imagesTr", targets)
        extra = ["--lesion-teacher", targets / "teacher.json"]
        teacher = read(targets / "teacher.json")
        teacher_summary = {key: teacher[key] for key in ("elapsed_seconds", "temperature", "calibration_exams",
                                                        "teacher_checkpoint_sha256")}
    elif intervention == "boundary":
        extra = ["--boundary-weight", .25]
    else:
        raise ValueError("Unrecognized matched intervention.")
    cases = [case for case in read_manifest(study.development) if case.split == "validation"]
    checkpoints = {}
    details = {}
    for name, options in (("control", ()), (intervention, extra)):
        checkpoint = folder / f"{name}.pt"
        if not checkpoint.is_file():
            study.module(f"replan-{intervention}-{name}", "cli", "train", study.development, "--out", checkpoint,
                         "--steps", updates, "--seed", 19, "--width", 12, "--patch", *patch,
                         "--input-mode", input_mode, "--model-spacing-mm", 1.5, "--learning-rate", .0001,
                         "--initial-checkpoint", source, "--resumable", *options)
        summary = read(checkpoint.with_suffix(".training.json"))
        if summary["updates"] != updates or summary["initial_checkpoint_sha256"] != identity["initial_sha256"]:
            raise ValueError("The prototype did not use the agreed update count and initialization.")
        checkpoints[name] = checkpoint
        details[name] = summary
    for name, checkpoint in checkpoints.items():
        make_probabilities(cases, checkpoint, folder / name, study.development)

    comparisons = compare_cached(cases, folder, tuple(checkpoints), preserve_detection=True)
    samples = comparisons.pop("selected_records")
    paired = paired_patients(samples["control"], samples[intervention])
    difference = paired["all"]["proposed_minus_reference"]["dice"]["mean_difference"]
    control, proposed = paired["all"]["reference"], paired["all"]["proposed"]
    qualifies = difference >= .01 and proposed["missed"] <= control["missed"]
    write_json(report, {"recipe": identity, "training": details, "teacher": teacher_summary, **comparisons,
                        "paired_control_comparison": paired, "eligible_to_extend": qualifies})
    print(json.dumps({"eligible_to_extend": qualifies, "paired_mean_dice_difference": difference,
                      "chosen": comparisons["chosen"]}, indent=2), flush=True)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--trainer-dir", type=Path, required=True)
    parser.add_argument("--intervention", choices=("distillation", "boundary"), required=True)
    parser.add_argument("--updates", type=int, default=15000)
    parser.add_argument("--gpu-wait-seconds", type=float, default=1800.)
    args = parser.parse_args()
    with FileLock(args.root / "nnunet/gpu-training.lock", timeout=args.gpu_wait_seconds):
        print(run(args.root, args.trainer_dir, args.intervention, args.updates), flush=True)


if __name__ == "__main__":
    main()
