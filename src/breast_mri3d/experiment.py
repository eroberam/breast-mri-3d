"""Validation threshold selection and paired, patient-level reporting."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy import ndimage

from .acquire import checksum, write_json


THRESHOLDS = (.2, .3, .4, .5, .6, .7, .8, .9)


def tune_threshold(manifest: Path, checkpoint: Path, output: Path, *, device_name: str,
                   patch: tuple[int, int, int] = (32, 128, 128)) -> dict:
    import torch

    from .geometry import Grid
    from .imaging import read_nifti
    from .inference import binary_masks, open_model, predict, structure_probability
    from .research import read_manifest

    cases = [case for case in read_manifest(manifest) if case.split == "validation"
             and case.metadata["labels"].get("lesion") in {"manual", "confirmed_empty"}]
    if not any(case.metadata["labels"]["lesion"] == "manual" for case in cases):
        raise ValueError("Threshold selection requires expert lesion references from validation patients.")
    if output.exists():
        raise FileExistsError("The validation threshold report already exists.")
    device = torch.device(device_name)
    model, summary = open_model(checkpoint, device)
    if "lesion" not in summary["supervised_labels"]:
        raise ValueError("This checkpoint does not predict a supervised lesion channel.")
    use_ranking = bool(summary["verified_ranking_pairs"])
    settings = [(p, r) for p in THRESHOLDS for r in ((None, -1., 0., 1.) if use_ranking else (None,))]
    values = {setting: {"positive": [], "negative": []} for setting in settings}
    for i, case in enumerate(cases, start=1):
        prepared = np.load(case.path / "image.npy", mmap_mode="r", allow_pickle=False)
        probability, ranking = predict(model, prepared, patch, device, ranking=use_ranking)
        native = Grid.from_dict(case.metadata["native_grid"])
        target, coordinates = read_nifti(case.path / "native_labels/lesion.nii.gz")
        if (coordinates.shape != native.shape or not np.allclose(coordinates.affine, native.affine, atol=.001, rtol=0)
                or not np.isin(target, (0, 1)).all()):
            raise ValueError("Threshold selection requires a binary reference on its recorded physical grid.")
        target = target.astype(bool)
        target_count = int(target.sum())
        positive = case.metadata["labels"]["lesion"] == "manual"
        model_grid = case.grid
        # Nearest-neighbour projection selects the same voxel before or after
        # thresholding. Outside the source, zero is background at every threshold.
        native_probability = model_grid.resample(structure_probability(probability, "lesion"), native, order=0)
        for prob_threshold, rank_threshold in settings:
            if rank_threshold is None:
                aligned = native_probability >= prob_threshold
            else:
                prediction = binary_masks(probability, ["lesion"], prob_threshold, ranking=ranking,
                                          rank_threshold=rank_threshold)["lesion"]
                aligned = model_grid.resample(prediction.astype(np.uint8), native, order=0).astype(bool)
            current = values[prob_threshold, rank_threshold]
            if positive:
                denominator = int(aligned.sum()) + target_count
                current["positive"].append(float(2 * (aligned & target).sum() / denominator) if denominator else 0.)
            else:
                current["negative"].append(int(ndimage.label(aligned)[1]))
        if i % 5 == 0 or i == len(cases):
            print(f"Validation threshold cases {i}/{len(cases)}", flush=True)
    candidates = []
    for (probability, ranking), scores in values.items():
        negatives = scores["negative"]
        candidates.append({"threshold": probability, "rank_threshold": ranking,
                           "median_positive_dice": float(np.median(scores["positive"])),
                           "mean_negative_false_alarms": float(np.mean(negatives)) if negatives else None})
    selected = max(candidates, key=lambda c: (c["median_positive_dice"],
                                                -(c["mean_negative_false_alarms"] or 0.),
                                                -abs(c["threshold"] - .5), c["rank_threshold"] is None))
    report = {"basis": "validation_only", "manifest_sha256": checksum(manifest),
              "checkpoint_sha256": checksum(checkpoint), "selected": selected,
              "cases": len(cases), "candidates": candidates}
    write_json(output, report)
    return selected


def _archive_integrity(root: Path) -> Path:
    from .acquire import AMBL_ASSETS

    archive_record = root / "raw/AMBL/archive-integrity.json"
    archive = json.loads(archive_record.read_text(encoding="utf-8-sig"))
    if (archive.get("record") != "zenodo:17253223/v3"
            or {f["name"]: f["md5"] for f in archive.get("files", [])} != AMBL_ASSETS):
        raise ValueError("The external release does not have the expected provider integrity record.")
    if any((archive_record.parent / item["name"]).stat().st_size != item["bytes"] for item in archive["files"]):
        raise ValueError("The reserved AMBL archive differs from its integrity record.")
    return archive_record


def lock_external(root: Path, compact: Path, calibration: Path, reference: Path, output: Path) -> dict:
    if output.exists():
        raise FileExistsError("An external model and threshold lock already exists.")
    archive_record = _archive_integrity(root)
    compact_hash = checksum(compact)
    calibration_record = json.loads(calibration.read_text(encoding="utf-8"))
    if calibration_record["basis"] != "validation_only" or calibration_record["checkpoint_sha256"] != compact_hash:
        raise ValueError("The selected compact checkpoint must match its validation-only calibration.")
    lock = {"archive": "zenodo:17253223/v3", "archive_integrity_sha256": checksum(archive_record),
            "compact": {"path": str(compact.resolve()), "sha256": compact_hash,
                        "calibration_sha256": checksum(calibration), **calibration_record["selected"]},
            "reference": {"path": str(reference.resolve()), "sha256": checksum(reference), "threshold": .5}}
    write_json(output, lock)
    return lock


def verify_external_lock(root: Path, path: Path) -> dict:
    lock = json.loads(path.read_text(encoding="utf-8"))
    if lock["archive_integrity_sha256"] != checksum(_archive_integrity(root)):
        raise ValueError("The downloaded AMBL files differ from the external lock.")
    for arm in ("compact", "reference"):
        if checksum(Path(lock[arm]["path"])) != lock[arm]["sha256"]:
            raise ValueError("The frozen checkpoint differs from its recorded content.")
    return lock


def paired_intervals(reference: dict, proposed: dict, *, replicates: int = 2000, seed: int = 19) -> dict:
    """Bootstrap paired median differences by canonical patient and comparable endpoint."""
    keys = ("patient", "cohort", "reference", "grid")

    def indexed(report: dict) -> dict:
        records = [row for row in report["records"] if row.get("structure", "lesion") == "lesion"]
        result = {tuple(row[key] for key in keys): row for row in records}
        if len(result) != len(records):
            raise ValueError("Paired comparison expects one lesion record per patient and endpoint.")
        return result

    a, b = indexed(reference), indexed(proposed)
    if not a or set(a) != set(b):
        raise ValueError("Comparison inputs must cover identical patients, provenance and measurement grids.")
    generator = np.random.default_rng(seed)
    result = {}
    for metric in ("dice", "abs_volume_error_ml", "hd95_mm", "false_alarms"):
        valid_keys = sorted(a) if metric == "false_alarms" else sorted(key for key in a if key[2] != "confirmed_empty")
        values = [(a[key].get(metric), b[key].get(metric)) for key in valid_keys]
        values = np.asarray([(left, right) for left, right in values if left is not None and right is not None])
        if not len(values):
            continue
        difference = values[:, 1] - values[:, 0]
        draws = generator.integers(0, len(values), size=(replicates, len(values)))
        low, high = np.quantile(np.median(difference[draws], axis=1), (.025, .975))
        result[metric] = {"n": len(values), "reference_median": float(np.median(values[:, 0])),
                          "proposed_median": float(np.median(values[:, 1])),
                          "paired_median_difference": float(np.median(difference)), "ci95": [float(low), float(high)]}
    counts = [(a[key]["small_matched"], b[key]["small_matched"], a[key]["small_lesions"])
              for key in sorted(a) if a[key].get("small_lesions") and b[key].get("small_lesions")]
    counts = np.asarray(counts, dtype=int).reshape(-1, 3)
    if len(counts) and counts[:, 2].sum():
        totals = counts.sum(axis=0)
        draws = generator.integers(0, len(counts), size=(replicates, len(counts)))
        drawn = counts[draws].sum(axis=1)
        low, high = np.quantile((drawn[:, 1] - drawn[:, 0]) / drawn[:, 2], (.025, .975))
        result["small_lesion_sensitivity"] = {"n_lesions": int(totals[2]), "n_patients": int(sum(counts[:, 2] > 0)),
                                              "reference": float(totals[0] / totals[2]),
                                              "proposed": float(totals[1] / totals[2]),
                                              "ci95_paired_difference": [float(low), float(high)]}
    return {"seed": seed, "bootstrap_replicates": replicates, "metrics": result}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("calibrate", "envelope-teacher"):
        task = commands.add_parser(name)
        task.add_argument("manifest", type=Path)
        task.add_argument("checkpoint", type=Path)
        task.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
        task.add_argument("--patch", type=int, nargs=3, default=(32, 128, 128))
        task.add_argument("--out", type=Path, required=True)
    lock = commands.add_parser("lock-external")
    for path in ("root", "compact", "calibration", "reference"):
        lock.add_argument(path, type=Path)
    lock.add_argument("--out", type=Path, required=True)
    compare = commands.add_parser("compare")
    compare.add_argument("reference", type=Path)
    compare.add_argument("proposed", type=Path)
    compare.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "calibrate":
        result = tune_threshold(args.manifest, args.checkpoint, args.out, device_name=args.device, patch=tuple(args.patch))
    elif args.command == "envelope-teacher":
        from .teacher import cache_envelope_teacher

        result = cache_envelope_teacher(args.manifest, args.checkpoint, args.out, device_name=args.device,
                                        patch=tuple(args.patch))
    elif args.command == "lock-external":
        lock_external(args.root, args.compact, args.calibration, args.reference, args.out)
        result = {"lock": str(args.out)}
    else:
        if args.out.exists():
            raise FileExistsError("The paired comparison report already exists.")
        a, b = [json.loads(path.read_text(encoding="utf-8")) for path in (args.reference, args.proposed)]
        result = paired_intervals(a, b)
        write_json(args.out, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
