"""Select operating thresholds on development and score locked native predictions."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import time

import numpy as np

from .acquire import checksum, write_json
from .cohorts_v03 import patient_key
from .geometry import Grid
from .imaging import read_nifti
from .research import (Case, keep_largest_components, lesion_instances, mask_metrics, match_lesions,
                       read_manifest, summarize_metrics)


THRESHOLDS = (.30, .45, .60, .75, .90)


def native_reference(case: Case) -> tuple[np.ndarray, Grid]:
    meta = case.metadata
    grid = Grid.from_dict(meta["native_grid"])
    target, actual = read_nifti(case.path / "native_labels" / "lesion.nii.gz")
    if (actual.shape != grid.shape or not np.allclose(actual.affine, grid.affine, atol=.001, rtol=0)
            or not np.isin(target, (0, 1)).all()):
        raise ValueError("The original reference mask does not match the verified native coordinates.")
    return target.astype(bool), grid


def native_score(case: Case, predicted: np.ndarray, target: np.ndarray, grid: Grid) -> dict:
    meta = case.metadata
    kind = meta["labels"]["lesion"]
    row = {"patient": case.patient, "cohort": case.cohort, "reference": kind,
           "grid": "native_" + meta["reference"].get("type", "dicom"), **mask_metrics(predicted, target, grid)}
    instances = lesion_instances(case, target, grid, meta["reference"])
    detections = match_lesions(predicted, target, grid=grid, instances=instances)
    if kind != "confirmed_empty" and not meta.get("lesion_annotations_exhaustive", False):
        detections.pop("false_alarms")
    return {**row, **detections}


def score_compact(manifest: Path, checkpoint: Path, output: Path, *, split: str,
                  threshold: float | None, locator: Path | None = None, device_name: str = "cuda",
                  patch_override: tuple[int, ...] | None = None, mirror_left_right: bool = False,
                  component_limit: int | None = None) -> dict:
    import torch

    from .cascade import COARSE_SPACING_MM, lesion_probability
    from .inference import open_model

    if threshold is None and split != "validation":
        raise ValueError("Threshold selection is restricted to the adjustment split.")
    if threshold is None and component_limit is not None:
        raise ValueError("Threshold selection with component filtering requires an adjustment policy sweep.")
    if threshold is not None and not 0 < threshold < 1:
        raise ValueError("Provide a probability threshold strictly inside (0, 1).")
    identity = {"manifest_sha256": checksum(manifest), "checkpoint_sha256": checksum(checkpoint),
                "locator_sha256": checksum(locator) if locator else None, "split": split, "threshold": threshold}
    if patch_override or mirror_left_right or component_limit is not None:
        identity.update(patch_override=list(patch_override) if patch_override is not None else None,
                        mirror_left_right=mirror_left_right, component_limit=component_limit)
    cases = [case for case in read_manifest(manifest) if case.split == split]
    if not cases:
        raise ValueError("No eligible patients in the requested split.")
    if output.is_file():
        result = json.loads(output.read_text(encoding="utf-8"))
        if result["inputs"] != identity:
            raise ValueError("The existing evaluation used a different input lock.")
        return result
    partial = output.with_suffix(".progress.json")
    rows = []
    elapsed_before = 0.
    if partial.is_file():
        progress = json.loads(partial.read_text(encoding="utf-8"))
        rows, elapsed_before = progress["records"], progress["elapsed_seconds"]
        if progress["inputs"] != identity or [r["patient"] for r in rows] != [c.patient for c in cases[:len(rows)]]:
            raise ValueError("The stored predictions do not follow the frozen manifest.")
    started = time.monotonic()
    device = torch.device(device_name)
    model, summary = open_model(checkpoint, device)
    coarse, details = open_model(locator, device) if locator else (None, {})
    if coarse is not None and details["model_spacing_mm"] != COARSE_SPACING_MM:
        raise ValueError("The locator was fitted on a different physical grid.")
    patch = patch_override if patch_override is not None else tuple(summary["patch_zyx"])
    for index, case in enumerate(cases[len(rows):], len(rows) + 1):
        target, native = native_reference(case)
        image = np.load(case.path / "image.npy", mmap_mode="r", allow_pickle=False)
        model_grid = case.grid
        if not np.allclose(model_grid.spacing, summary["model_spacing_mm"]):
            raise ValueError("Inference input spacing differs from the frozen training spacing.")
        prob = lesion_probability(model, image, model_grid, patch, device, locator=coarse,
                                  locator_patch=tuple(details["patch_zyx"]) if coarse is not None else None,
                                  mirror_left_right=mirror_left_right)
        if threshold is None:
            aligned = model_grid.resample(prob, native, order=0)
            truth_volume = np.count_nonzero(target)
            scores = []
            for cutoff in THRESHOLDS:
                prediction = aligned >= cutoff
                denom = np.count_nonzero(prediction) + truth_volume
                scores.append(float(2 * np.count_nonzero(prediction & target) / denom) if denom else 1.)
            row = {"patient": case.patient, "cohort": case.cohort, "dice_by_threshold": scores}
        else:
            mask = prob >= threshold
            if component_limit is not None:
                mask = keep_largest_components(mask, component_limit)
            aligned = model_grid.resample(mask.astype(np.uint8), native, order=0).astype(bool)
            row = native_score(case, aligned, target, native)
        rows.append(row)
        if index % 8 == 0 or index == len(cases):
            write_json(partial, {"inputs": identity, "records": rows,
                                 "elapsed_seconds": elapsed_before + time.monotonic() - started})
            print(f"Scored {index}/{len(cases)} {split} exams.", flush=True)
    result = {"inputs": identity, "cases": len(rows), "records": rows,
              "elapsed_seconds": elapsed_before + time.monotonic() - started}
    if threshold is None:
        scores = np.asarray([row["dice_by_threshold"] for row in rows])
        means = scores.mean(axis=0)
        index = max(range(len(means)), key=lambda i: (means[i], THRESHOLDS[i]))
        result.update(threshold=THRESHOLDS[index], mean_dice=float(means[index]),
                      candidate_thresholds=[{"threshold": cut, "mean_dice": float(mean)}
                                            for cut, mean in zip(THRESHOLDS, means)])
    else:
        result.update(threshold=threshold, metrics=summarize_metrics(rows))
    write_json(output, result)
    partial.unlink(missing_ok=True)
    return result


def _paired(reference: list[dict], proposed: list[dict], *, seed: int, replicates: int) -> dict:
    strata = defaultdict(list)
    for index, row in enumerate(reference):
        strata[row["cohort"]].append(index)
    rng = np.random.default_rng(seed)
    draws = np.concatenate([rng.choice(i, size=(replicates, len(i)), replace=True) for i in strata.values()], axis=1)
    a = summarize_metrics(reference)
    b = summarize_metrics(proposed)
    effect = {}
    for metric in ("dice", "abs_volume_error_ml"):
        delta = np.array([y[metric] - x[metric] for x, y in zip(reference, proposed)])
        effect[metric] = {"mean_difference": float(delta.mean()),
                          "ci95": [float(n) for n in np.percentile(delta[draws].mean(axis=1), (2.5, 97.5))]}
    common = np.array([y["hd95_mm"] - x["hd95_mm"] if x["hd95_mm"] is not None and y["hd95_mm"] is not None
                       else np.nan for x, y in zip(reference, proposed)])
    finite = np.isfinite(common)
    resamples = common[draws]
    complete = resamples[np.isfinite(resamples).any(axis=1)]
    effect["hd95_mm"] = {"n_complete_pairs": int(finite.sum()),
                         "median_paired_difference": float(np.median(common[finite])) if finite.any() else None,
                         "ci95": np.percentile(np.nanmedian(complete, axis=1), (2.5, 97.5)).tolist()
                         if len(complete) else None}
    for model, rows in ((a, reference), (b, proposed)):
        for field in ("matched", "missed", "small_matched", "small_lesions", "false_alarms"):
            if all(field in row for row in rows):
                model[field] = sum(row[field] for row in rows)
        if model.get("small_lesions"):
            model["small_lesion_sensitivity"] = model["small_matched"] / model["small_lesions"]
        if "false_alarms" in model:
            model["false_alarms_per_exam"] = model["false_alarms"] / len(rows)
    if a.get("small_lesions") and a["small_lesions"] == b.get("small_lesions"):
        count = np.array([row["small_lesions"] for row in reference])
        changes = np.array([y["small_matched"] - x["small_matched"] for x, y in zip(reference, proposed)])
        denominators = count[draws].sum(axis=1)
        estimates = changes[draws].sum(axis=1)[denominators > 0] / denominators[denominators > 0]
        effect["small_lesion_sensitivity"] = {"difference": float(changes.sum() / count.sum()),
                                              "ci95": np.percentile(estimates, (2.5, 97.5)).tolist()}
    if all(row["reference"] == "confirmed_empty" for row in reference):
        delta = np.array([y["false_alarms"] - x["false_alarms"] for x, y in zip(reference, proposed)])
        effect["false_alarms_per_exam"] = {
            "mean_difference": float(delta.mean()),
            "ci95": [float(n) for n in np.percentile(delta[draws].mean(axis=1), (2.5, 97.5))]}
    return {"patients": len(reference), "reference": a, "proposed": b, "proposed_minus_reference": effect}


def paired_patients(reference: dict, proposed: dict, *, seed: int = 19, replicates: int = 10000) -> dict:
    def keyed(records: list[dict]) -> dict:
        keys = [(row["cohort"], patient_key(row["patient"])) for row in records]
        if not keys or len({key[1] for key in keys}) != len(keys):
            raise ValueError("The paired cohort must contain one record per original patient.")
        return dict(zip(keys, records))

    a, b = keyed(reference["records"]), keyed(proposed["records"])
    if a.keys() != b.keys():
        raise ValueError("The reference and proposed model do not cover the same frozen patients.")
    ids = sorted(a)
    if any(a[k]["grid"] != b[k]["grid"] or a[k]["reference"] != b[k]["reference"] for k in ids):
        raise ValueError("A paired patient must have the same annotation endpoint and native coordinate system.")
    return {"bootstrap_replicates": replicates, "bootstrap_seed": seed,
            "all": _paired([a[k] for k in ids], [b[k] for k in ids], seed=seed, replicates=replicates),
            "by_collection": {group: _paired([a[k] for k in ids if k[0] == group], [b[k] for k in ids if k[0] == group],
                                             seed=seed, replicates=replicates) for group in sorted({k[0] for k in ids})}}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--split", choices=("validation", "test"), required=True)
    parser.add_argument("--threshold", type=float)
    parser.add_argument("--locator", type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--patch", type=int, nargs=3)
    parser.add_argument("--mirror-left-right", action="store_true")
    parser.add_argument("--component-limit", type=int)
    args = parser.parse_args()
    score_compact(args.manifest, args.checkpoint, args.out, split=args.split, threshold=args.threshold,
                  locator=args.locator, device_name=args.device,
                  patch_override=tuple(args.patch) if args.patch else None, mirror_left_right=args.mirror_left_right,
                  component_limit=args.component_limit)


if __name__ == "__main__":
    main()
