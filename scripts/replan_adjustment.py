"""Compare localization and contour errors on the locked adjustment split."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import os
from pathlib import Path
import time

from filelock import FileLock
import numpy as np
from scipy import ndimage
import torch

from breast_mri3d.acquire import checksum, write_json
from breast_mri3d.cascade import lesion_probability
from breast_mri3d.imaging import read_nifti
from breast_mri3d.inference import open_model
from breast_mri3d.labels import LABELS
from breast_mri3d.research import read_manifest
from breast_mri3d.validation_v03 import THRESHOLDS, native_reference, native_score, paired_patients


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def export_intermediate(resume: Path, checkpoint: Path) -> dict:
    state = torch.load(resume, map_location="cpu", weights_only=True)
    config = state["config"]
    summary = {"elapsed_seconds": state["elapsed_seconds"], "updates": len(state["history"]),
               "model_spacing_mm": config["model_spacing_mm"], "patch_zyx": config["patch"],
               "manifest_sha256": config["manifest_sha256"], "input_mode": config["input_mode"],
               "source_resume_sha256": checksum(resume)}
    if not checkpoint.is_file():
        temporary = checkpoint.with_suffix(".partial.pt")
        torch.save({"state_dict": state["model"], "width": config["width"], "dual_decoder": config["dual_decoder"],
                    "input_mode": config["input_mode"], "labels": list(LABELS), "summary": summary}, temporary)
        os.replace(temporary, checkpoint)
    elif torch.load(checkpoint, map_location="cpu", weights_only=True)["summary"] != summary:
        raise ValueError("The intermediate checkpoint is from a different continuation.")
    return summary


def largest_components(mask: np.ndarray, limit: int) -> np.ndarray:
    components, count = ndimage.label(mask, structure=np.ones((3, 3, 3), dtype=bool))
    if count <= limit:
        return mask
    volumes = np.bincount(components.ravel())[1:]
    largest = 1 + np.argsort(-volumes, kind="stable")[:limit]
    return np.isin(components, largest)


def native_sampling(model, native, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Count the native voxels and reference positives represented by each model voxel."""
    total = int(np.prod(model.shape))
    indices = np.arange(1, total + 1, dtype=np.int32).reshape(model.shape)
    mapped = model.resample(indices, native, order=0)
    covered = np.bincount(mapped.ravel(), minlength=total + 1)[1:]
    covered_target = np.bincount(mapped[target], minlength=total + 1)[1:]
    return covered, covered_target


def counts(prediction: np.ndarray, covered: np.ndarray, covered_target: np.ndarray,
           true: int, voxel_ml: float) -> dict:
    selected = prediction.ravel().astype(bool, copy=False)
    overlap = int(covered_target[selected].sum())
    predicted = int(covered[selected].sum())
    return {"dice": 2 * overlap / (predicted + true) if predicted + true else 1.,
            "extra_ml": (predicted - overlap) * voxel_ml, "omitted_ml": (true - overlap) * voxel_ml}


def aggregate(rows: list[dict]) -> dict:
    return {"n": len(rows), "mean_dice": float(np.mean([r["dice"] for r in rows])),
            "median_dice": float(np.median([r["dice"] for r in rows])),
            "median_extra_ml": float(np.median([r["extra_ml"] for r in rows])),
            "median_omitted_ml": float(np.median([r["omitted_ml"] for r in rows])),
            "zero_overlap": sum(bool(r["dice"] == 0) for r in rows)}


def make_probabilities(cases, checkpoint: Path, folder: Path, manifest: Path, *,
                       patch_override: tuple[int, ...] | None = None) -> None:
    checkpoint_sha = checksum(checkpoint)
    identity = {"checkpoint_sha256": checkpoint_sha, "manifest_sha256": checksum(manifest),
                "view": "single_and_left_right_mean", "split": "validation"}
    if patch_override is not None:
        identity["patch_zyx"] = list(patch_override)
    folder.mkdir(parents=True, exist_ok=True)
    receipt = folder / "inputs.json"
    if receipt.is_file() and read(receipt) != identity:
        raise ValueError("Previously scored probabilities belong to another adjustment input.")
    write_json(receipt, identity)
    if all((folder / f"{index:04d}.npy").is_file() for index in range(1, len(cases) + 1)):
        return
    device = torch.device("cuda")
    model, info = open_model(checkpoint, device)
    if info["manifest_sha256"] != identity["manifest_sha256"]:
        raise ValueError("The model was fitted on a different development manifest.")
    patch = patch_override if patch_override is not None else tuple(info["patch_zyx"])
    for index, case in enumerate(cases, 1):
        path = folder / f"{index:04d}.npy"
        if path.is_file():
            continue
        images = np.load(case.path / "image.npy", mmap_mode="r", allow_pickle=False)
        grid = case.grid
        single = lesion_probability(model, images, grid, patch, device)
        mirrored = lesion_probability(model, images[..., ::-1], grid, patch, device)[..., ::-1]
        both = np.stack((single, .5 * (single + mirrored)))
        temp = path.with_suffix(".partial.npy")
        np.save(temp, both, allow_pickle=False)
        os.replace(temp, path)
        if index % 20 == 0 or index == len(cases):
            print(f"{folder.name}: {index}/{len(cases)} adjustment prediction pairs.", flush=True)
    del model
    torch.cuda.empty_cache()


def candidate_scores(case, index: int, root: Path, models: dict[str, Path], target, native,
                     components: dict[str, int]) -> dict:
    grid = case.grid
    voxel_ml = native.voxel_mm3 / 1000
    covered, covered_target = native_sampling(grid, native, target)
    target_count = int(np.count_nonzero(target))
    result = {}
    base = root.parent / "reference-6h-adjustment-nnunet"
    _, source_grid = read_nifti(base / f"inputs/BreastMRI_{index:04d}_0000.nii")
    reference, _ = read_nifti(base / f"outputs/BreastMRI_{index:04d}.nii", source_grid=source_grid)
    if source_grid.shape != grid.shape or not np.allclose(source_grid.affine, grid.affine, atol=.001, rtol=0):
        raise ValueError("The comparator does not share the original model lattice.")
    def score(name, view, threshold, mask):
        result[(name, view, threshold, "raw")] = counts(mask, covered, covered_target, target_count, voxel_ml)
        for rule, limit in components.items():
            retained = largest_components(mask, limit)
            result[(name, view, threshold, rule)] = counts(retained, covered, covered_target, target_count, voxel_ml)

    score("reference", "mirrored", .5, reference.astype(bool, copy=False))
    for name, path in models.items():
        probabilities = np.load(path / f"{index:04d}.npy", mmap_mode="r", allow_pickle=False)
        for view, volume in zip(("single", "mirrored"), probabilities):
            for cut in THRESHOLDS:
                score(name, view, cut, volume >= cut)
    return result


def compare_cached(cases, folder: Path, models: tuple[str, ...], *, preserve_detection: bool = False,
                   model_dirs: dict[str, Path] | None = None) -> dict:
    runs = folder.parent
    locations = {name: (model_dirs or {}).get(name, folder / name) for name in models}
    components = {"largest": 1, "two_largest": 2} if preserve_detection else {"largest": 1}
    profiles = defaultdict(list)
    for index, case in enumerate(cases, 1):
        target, native = native_reference(case)
        for key, row in candidate_scores(case, index, folder, locations, target, native, components).items():
            profiles[key].append(row)
        if index % 10 == 0 or index == len(cases):
            print(f"Compared overlap and volume errors for {index}/{len(cases)} exams.", flush=True)
    summaries = {key: aggregate(rows) for key, rows in profiles.items()}
    def operating_point(name: str) -> tuple:
        choices = [key for key in summaries if key[0] == name]
        if preserve_detection:
            ceiling = summaries[("reference", "mirrored", .5, "raw")]["zero_overlap"]
            feasible = [key for key in choices if summaries[key]["zero_overlap"] <= ceiling]
            if not feasible:
                return min(choices, key=lambda key: (summaries[key]["zero_overlap"], -summaries[key]["mean_dice"]))
            choices = feasible
        return max(choices, key=lambda key: summaries[key]["mean_dice"])

    selected = {name: operating_point(name) for name in ("reference", *models)}

    def describe(key: tuple) -> dict:
        name, view, cut, rule = key
        return {"model": name, "view": view, "threshold": cut, "postprocessing": rule, **summaries[key]}

    metrics = {name: describe(key) for name, key in selected.items()}
    write_json(folder / "threshold-sweep.json", {"chosen": metrics, "recipes": [describe(k) for k in summaries]})
    selected_geometry = defaultdict(list)
    for index, case in enumerate(cases, 1):
        target, native = native_reference(case)
        grid = case.grid
        for name, (_, view, cut, rule) in selected.items():
            if name == "reference":
                output = runs / f"reference-6h-adjustment-nnunet/outputs/BreastMRI_{index:04d}.nii"
                _, source = read_nifti(runs / f"reference-6h-adjustment-nnunet/inputs/BreastMRI_{index:04d}_0000.nii")
                mask = read_nifti(output, source_grid=source)[0].astype(bool)
            else:
                scores = np.load(locations[name] / f"{index:04d}.npy", mmap_mode="r", allow_pickle=False)
                view_index = ("single", "mirrored").index(view)
                mask = scores[view_index] >= cut
            if rule in components:
                mask = largest_components(mask, components[rule])
            aligned = grid.resample(mask.astype(np.uint8), native, order=0).astype(bool)
            selected_geometry[name].append(native_score(case, aligned, target, native))
        if index % 20 == 0 or index == len(cases):
            print(f"Verified native geometry metrics for {index}/{len(cases)} exams.", flush=True)
    for name, rows in selected_geometry.items():
        if not np.isclose(metrics[name]["mean_dice"], np.mean([r["dice"] for r in rows]), atol=1e-7, rtol=0):
            raise ValueError("The selected geometry comparison differs from the threshold sweep.")
    records = {name: {"records": rows} for name, rows in selected_geometry.items()}
    return {"cases": len(cases), "chosen": metrics, "recipes": [describe(k) for k in summaries],
            "physical_comparison": {name: paired_patients(records["reference"], records[name]) for name in models},
            "selected_records": records}


def run(root: Path) -> Path:
    started = time.monotonic()
    runs = root / "runs/v03"
    manifest = root / "prepared/development.json"
    cases = [case for case in read_manifest(manifest) if case.split == "validation"]
    if len(cases) != 120:
        raise ValueError("The diagnostic must use the locked 120-patient adjustment split.")
    folder = runs / "adjustment-diagnostic"
    folder.mkdir(parents=True, exist_ok=True)
    resume = runs / "compact-dce-6h.resume.pt"
    intermediate = folder / "dce-intermediate.pt"
    details = export_intermediate(resume, intermediate)
    models = {"subtraction": runs / "compact-subtraction-6h.pt", "dce_intermediate": intermediate}
    for name, checkpoint in models.items():
        make_probabilities(cases, checkpoint, folder / name, manifest)
    comparable = compare_cached(cases, folder, tuple(models))
    report = root / "reports/replan-adjustment.json"
    write_json(report, {"activity": "adjustment_diagnostic", "dce_intermediate": details, **comparable,
                        "elapsed_seconds": round(time.monotonic() - started, 3)})
    print(json.dumps(comparable["chosen"], indent=2), flush=True)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    root = parser.parse_args().root
    with FileLock(root / "nnunet/gpu-training.lock", timeout=0):
        print(run(root), flush=True)


if __name__ == "__main__":
    main()
