"""Calibrate an internally trained envelope teacher and materialize trusted soft labels."""

from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from scipy.special import expit, logit

from .acquire import checksum, write_json
from .labels import ENVELOPE, LABELS
from .research import read_manifest


def cache_envelope_teacher(manifest: Path, checkpoint: Path, output: Path, *, device_name: str = "cuda",
                           minimum_dice: float = .8, patch: tuple[int, int, int] = (32, 128, 128)) -> dict:
    import torch

    from .inference import open_model, predict, structure_probability

    cases = read_manifest(manifest)
    development = [case for case in cases if case.split == "train"]
    calibration = [case for case in cases if case.split == "validation" and ENVELOPE in case.metadata["labels"]]
    if not calibration:
        raise ValueError("Envelope distillation requires validation references outside the teacher's training cases.")
    if output.exists():
        raise FileExistsError("The envelope teacher has already been calibrated for this run.")
    device = torch.device(device_name)
    model, state = open_model(checkpoint, device)
    if state["manifest_sha256"] != checksum(manifest) or ENVELOPE not in state["supervised_labels"]:
        raise ValueError("The teacher must have learned manual envelopes on this exact development partition.")
    temperatures = (.6, .8, 1., 1.2, 1.5, 2.)
    losses, scores = np.zeros(len(temperatures)), []
    generator = np.random.default_rng(19)

    def probability(case):
        image = np.load(case.path / "image.npy", mmap_mode="r", allow_pickle=False)
        channels, _ = predict(model, image, patch, device)
        return structure_probability(channels, ENVELOPE)

    for case in calibration:
        p = probability(case)
        target = np.load(case.path / f"{ENVELOPE}.npy", allow_pickle=False).astype(bool)
        predicted = p >= .5
        union_count = int(predicted.sum()) + int(target.sum())
        scores.append(float(2 * (target & predicted).sum() / union_count) if union_count else 0.)
        for value in (False, True):
            eligible = np.flatnonzero(target.ravel() == value)
            if not len(eligible):
                continue
            selected = generator.choice(eligible, size=min(4096, len(eligible)), replace=False)
            sample = logit(p.ravel()[selected].clip(1e-5, 1 - 1e-5))
            for index, t in enumerate(temperatures):
                losses[index] += float(np.logaddexp(0., -(1 if value else -1) * sample / t).mean())
    median_dice = float(np.median(scores))
    temperature = temperatures[int(losses.argmin())]
    details = {"manual_validation_cases": len(scores), "median_envelope_dice": median_dice,
               "temperature": temperature, "teacher_sha256": checksum(checkpoint),
               "trust_low": .1, "trust_high": .9, "eligible": median_dice >= minimum_dice}
    emitted, trusted_voxels = 0, 0
    if details["eligible"]:
        patients = sorted({case.patient for case in development})
        for case in development:
            if ENVELOPE in case.metadata["labels"]:
                continue
            p = expit(logit(probability(case).clip(1e-5, 1 - 1e-5)) / temperature)
            trusted = (p >= details["trust_high"]) | (p <= details["trust_low"])
            if not np.any(trusted & (p >= details["trust_high"])):
                continue
            provenance = {"patient": case.patient, "grid": case.metadata["grid"], "channel_order": list(LABELS),
                          "teacher_sha256": details["teacher_sha256"], "training_patients": patients,
                          "calibration_record": details}
            record = case.path / "teacher.json"
            if record.exists() and json.loads(record.read_text(encoding="utf-8")) != provenance:
                raise ValueError("A different teacher has already been associated with a prepared case.")
            with TemporaryDirectory(dir=case.path, prefix=".teacher-") as staging:
                stage = Path(staging)
                for name, values in [("teacher-envelope-probability.npy", p.astype(np.float16)),
                                     ("teacher-envelope-trusted.npy", trusted)]:
                    np.save(stage / name, values, allow_pickle=False)
                    (stage / name).replace(case.path / name)
                write_json(record, provenance)
            emitted += 1
            trusted_voxels += int(trusted.sum())
            if emitted % 10 == 0:
                print(f"Calibrated envelope probabilities cached for {emitted} development cases", flush=True)
    result = {**details, "emitted_training_cases": emitted, "trusted_voxels": trusted_voxels}
    write_json(output, result)
    return result
