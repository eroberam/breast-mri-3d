"""Calibrate a development-only ResEncM teacher for one optional lesion ablation."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import expit, logit

from .acquire import checksum, write_json
from .imaging import read_nifti
from .research import Case, read_manifest


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_teacher(receipt: Path, manifest: Path, training: list[Case]) -> dict[Path, Path]:
    identity = read(receipt)
    rows = identity["training"]
    targets = {}
    if (identity["development_sha256"] != checksum(manifest)
            or [row["patient"] for row in rows] != [case.patient for case in training]):
        raise ValueError("Teacher probabilities must cover the exact development training patients.")
    parent = receipt.parent.resolve()
    for case, row in zip(training, rows):
        path = (parent / row["probability"]).resolve()
        if not path.is_relative_to(parent) or checksum(path) != row["sha256"]:
            raise ValueError("A training-only teacher probability file failed its integrity check.")
        targets[case.path] = path
    return targets


def _samples(probability: np.ndarray, mask: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    positive = np.flatnonzero(mask)
    negative = np.flatnonzero((~mask) & (probability >= .05))
    if not len(negative):
        negative = np.flatnonzero(~mask)
    count = min(2048, len(positive), len(negative))
    indices = np.concatenate([rng.choice(items, size=count, replace=False) for items in (positive, negative)])
    return logit(np.clip(probability.flat[indices], 1e-4, 1 - 1e-4)), mask.flat[indices]


def _temperature(logits: np.ndarray, truth: np.ndarray) -> float:
    def loss(value: float) -> float:
        scaled = logits / value
        return float(np.mean(np.logaddexp(0., scaled) - scaled * truth))

    result = minimize_scalar(loss, bounds=(.5, 4.), method="bounded")
    if not result.success or not np.isfinite(result.x):
        raise ValueError("Teacher calibration failed on the manual adjustment contours.")
    return float(result.x)


def prepare_teacher(manifest: Path, checkpoint: Path, raw_inputs: Path, output: Path) -> Path:
    import torch
    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

    start = time.monotonic()
    candidates = [(i, case) for i, case in enumerate(read_manifest(manifest), 1)
                  if case.split in {"train", "validation"}]
    teacher_sha, manifest_sha = checksum(checkpoint), checksum(manifest)
    identity = {"teacher_checkpoint_sha256": teacher_sha, "development_sha256": manifest_sha,
                "calibration_method": "temperature_on_balanced_manual_adjustment_voxels"}
    output.mkdir(parents=True, exist_ok=True)
    receipt = output / "teacher.json"
    if receipt.is_file():
        if any(read(receipt)[k] != v for k, v in identity.items()):
            raise ValueError("Cached teacher targets were produced by another checkpoint or split.")
        return receipt
    origin = output / "inputs.json"
    if origin.is_file() and read(origin) != identity:
        raise ValueError("The partial teacher cache was created by another fit.")
    write_json(origin, identity)
    predictor = nnUNetPredictor(tile_step_size=.5, use_gaussian=True, use_mirroring=True,
                                perform_everything_on_device=True, device=torch.device("cuda"), allow_tqdm=False)
    predictor.initialize_from_trained_model_folder(str(checkpoint.parent.parent), use_folds=(0,),
                                                   checkpoint_name=checkpoint.name)
    reader = predictor.plans_manager.image_reader_writer_class()

    def inference(index: int, case: Case) -> np.ndarray:
        source = raw_inputs / f"BreastMRI_{index:04d}_0000.nii"
        _, grid = read_nifti(source)
        if grid.shape != case.grid.shape or not np.allclose(grid.affine, case.grid.affine, atol=.001, rtol=0):
            raise ValueError("Teacher input does not share the training array's physical lattice.")
        array, properties = reader.read_images([str(source)])
        _, probabilities = predictor.predict_single_npy_array(array, properties, save_or_return_probabilities=True)
        foreground = probabilities[1]
        if (foreground.shape != grid.shape or not np.isfinite(foreground).all()
                or foreground.min() < 0 or foreground.max() > 1):
            raise ValueError("ResEncM returned invalid lesion probabilities.")
        return foreground

    calibration = output / "calibration.json"
    if calibration.is_file():
        details = read(calibration)
    else:
        adjustment = [(i, c) for i, c in candidates if c.split == "validation"]
        rng = np.random.default_rng(19)
        samples = []
        for count, (index, case) in enumerate(adjustment, 1):
            target = np.load(case.path / "labels.npy", mmap_mode="r", allow_pickle=False)[-1].astype(bool)
            samples.append(_samples(inference(index, case), target, rng))
            if count % 20 == 0:
                print(f"Calibrated teacher on {count}/{len(adjustment)} adjustment exams.", flush=True)
        logits, truth = (np.concatenate(group) for group in zip(*samples))
        if len(logits) == 0:
            raise ValueError("Teacher calibration requires sampled manual contour and background voxels.")
        details = {"temperature": _temperature(logits, truth), "calibration_exams": len(adjustment),
                   "calibration_voxels": len(logits)}
        write_json(calibration, details)
    records = []
    training = [(i, c) for i, c in candidates if c.split == "train"]
    for count, (index, case) in enumerate(training, 1):
        name = f"teacher-{index:04d}.npy"
        target = output / name
        if not target.is_file():
            prob = np.clip(inference(index, case), 1e-4, 1 - 1e-4)
            scaled = expit(logit(prob) / details["temperature"]).astype(np.float16)
            temporary = target.with_suffix(".partial.npy")
            np.save(temporary, scaled, allow_pickle=False)
            os.replace(temporary, target)
        records.append({"patient": case.patient, "probability": name, "sha256": checksum(target)})
        if count % 25 == 0 or count == len(training):
            print(f"Prepared {count}/{len(training)} training-only teacher targets.", flush=True)
    write_json(receipt, {**identity, **details, "training": records,
                         "elapsed_seconds": round(time.monotonic() - start, 3)})
    return receipt


def run(root: Path, trainer_dir: Path) -> Path:
    from .study_v03 import Study

    report = root / "reports/v03-distillation-ablation.json"
    if report.is_file():
        return report
    study = Study(root, trainer_dir)
    if not (study.reports / "v03-completed.json").is_file():
        raise ValueError("Finish the primary locked comparison before the adjustment-only ablation.")
    lock = read(study.lock)
    if read(study.reports / "v03-completed.json")["lock_sha256"] != checksum(study.lock):
        raise ValueError("The completed comparison belongs to a different evaluation lock.")
    checkpoint = study.checked_path(lock["reference"]["model"])
    baseline_scores = {name: read(study.checked_path(lock["trials"][name])) for name in ("subtraction", "dce")}
    mode = max(baseline_scores, key=lambda key: baseline_scores[key]["mean_dice"])
    from .reference_v03 import DATASET_NAME

    count = 6 * 3600
    cache = study.runs / "teacher"
    study.module("distillation-prepare", "teacher_v03", "prepare", study.development, checkpoint,
                 study.raw / DATASET_NAME / "imagesTr", cache)
    weight = study.runs / "compact-distilled-6h.pt"
    if not weight.is_file():
        study.module("compact-distilled-6h", "cli", "train", study.development, "--out", weight,
                     "--steps", 2_000_000, "--budget-seconds", count, "--seed", 19, "--width", 12,
                     "--patch", *lock["patch_zyx"], "--input-mode", mode, "--model-spacing-mm", 1.5,
                     "--resumable", "--lesion-teacher", cache / "teacher.json")
    fit = read(weight.with_suffix(".training.json"))
    adjusted = read(study.native_compact("distilled-6h", study.development, weight, "validation"))
    control = baseline_scores[mode]
    teacher = read(cache / "teacher.json")
    write_json(report, {"role": "secondary_adjustment_only", "input_mode": mode,
                       "reference_mean_dice": control["mean_dice"], "distilled_mean_dice": adjusted["mean_dice"],
                       "reference_threshold": control["threshold"], "distilled_threshold": adjusted["threshold"],
                       "teacher_sha256": teacher["teacher_checkpoint_sha256"],
                       "teacher_calibration_temperature": teacher["temperature"],
                       "teacher_preparation_seconds": teacher["elapsed_seconds"], "student_fit_seconds": fit["elapsed_seconds"],
                       "distilled_updates": fit["distilled_updates"], "lock_sha256": checksum(study.lock)})
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_subparsers(dest="action", required=True)
    prepare = actions.add_parser("prepare")
    for name in ("manifest", "checkpoint", "raw", "output"):
        prepare.add_argument(name, type=Path)
    ablation = actions.add_parser("ablate")
    ablation.add_argument("root", type=Path)
    ablation.add_argument("--trainer-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.action == "prepare":
        print(prepare_teacher(args.manifest, args.checkpoint, args.raw, args.output))
    else:
        guard = args.root / "runs/v03/gpu-queue.pid"
        with guard.open("x", encoding="utf-8") as stream:
            stream.write(f"{os.getpid()}\n")
        try:
            print(run(args.root, args.trainer_dir))
        finally:
            guard.unlink()


if __name__ == "__main__":
    main()
