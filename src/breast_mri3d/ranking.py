"""Match manual lesions to cross-fitted false candidates in training cases."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy import ndimage
from scipy.optimize import linear_sum_assignment

from .labels import LABELS
from .research import Case, read_manifest


@dataclass(frozen=True)
class Candidate:
    case: Case
    center: tuple[int, int, int]
    volume_ml: float
    enhancement: float
    protocol: str


def components(case: Case, mask: np.ndarray, enhancement: np.ndarray) -> list[Candidate]:
    protocol = case.metadata.get("phase_protocol")
    if not isinstance(protocol, str) or not protocol.strip():
        raise ValueError("Ranking cases require a documented phase protocol group.")
    if mask.shape != case.grid.shape or enhancement.shape != mask.shape:
        raise ValueError("Ranking proposals and enhancement must share the prepared physical grid.")
    voxels_ml = case.grid.voxel_mm3 / 1000
    labels, _ = ndimage.label(mask)
    candidates = []
    for i, extent in enumerate(ndimage.find_objects(labels), start=1):
        region = labels[extent] == i
        coordinates = np.argwhere(region)
        center = coordinates[np.square(coordinates - coordinates.mean(axis=0)).sum(axis=1).argmin()]
        center = tuple(int(index + axis.start) for index, axis in zip(center, extent))
        candidates.append(Candidate(case, center, len(coordinates) * voxels_ml,
                                    float(enhancement[extent][region].mean(dtype=np.float64)), protocol))
    return candidates


def pair_candidates(positives: list[Candidate], negatives: list[Candidate], *,
                    maximum_volume_ratio: float, maximum_enhancement_gap: float) -> list[tuple[Candidate, Candidate]]:
    if maximum_volume_ratio <= 1 or maximum_enhancement_gap <= 0:
        raise ValueError("Volume-ratio and enhancement calipers must permit a nonzero difference.")
    matched = []
    volume_caliper = np.log(maximum_volume_ratio)
    for protocol in sorted({row.protocol for row in positives} & {row.protocol for row in negatives}):
        manual = [row for row in positives if row.protocol == protocol]
        cross_fitted = [row for row in negatives if row.protocol == protocol]
        volumes = np.log([x.volume_ml for x in manual])[:, None] - np.log([x.volume_ml for x in cross_fitted])
        enhancement = np.array([x.enhancement for x in manual])[:, None] - [x.enhancement for x in cross_fitted]
        eligible = (np.abs(volumes) <= volume_caliper) & (np.abs(enhancement) <= maximum_enhancement_gap)
        cost = np.abs(volumes) / volume_caliper + np.abs(enhancement) / maximum_enhancement_gap
        cost[~eligible] = 2 * (min(cost.shape) + 1)
        rows, columns = linear_sum_assignment(cost)
        matched.extend((manual[i], cross_fitted[j]) for i, j in zip(rows, columns) if eligible[i, j])
    return matched


def build_pairs(manifest: Path, folds: Path, destination: Path, *, threshold: float,
                patch: tuple[int, int, int], device_name: str, maximum_volume_ratio: float,
                maximum_enhancement_gap: float) -> int:
    import torch

    from .inference import binary_masks, open_model, predict

    if destination.exists():
        raise FileExistsError(f"Pair manifest already exists: {destination}")
    development = {case.path: case for case in read_manifest(manifest) if case.split == "train"}
    positive_cases = [c for c in development.values() if c.metadata["labels"].get("lesion") == "manual"]
    negative_cases = {c.path for c in development.values() if c.metadata["labels"].get("lesion") == "confirmed_empty"}
    if not positive_cases or not negative_cases:
        raise ValueError("Pair mining requires manual lesions and documented negative training exams.")
    lesion_index = LABELS.index("lesion")
    positives = []
    for case in positive_cases:
        mask = np.load(case.path / "labels.npy", mmap_mode="r", allow_pickle=False)[lesion_index]
        x = np.load(case.path / "image.npy", mmap_mode="r", allow_pickle=False)
        positives.extend(components(case, mask, x[2]))
    device = torch.device(device_name)
    fold_rows = json.loads(folds.read_text(encoding="utf-8"))["folds"]
    negative_info = {}
    negatives = []
    for row in fold_rows:
        fold_manifest = folds.parent / row["manifest"]
        checkpoint = folds.parent / row["checkpoint"]
        fold_cases = read_manifest(fold_manifest)
        if any(c.path not in development for c in fold_cases):
            raise ValueError("Cross-fitting cannot use locked model-selection or test cases.")
        training_patients = sorted({c.patient for c in fold_cases if c.split == "train"})
        target = [c for c in fold_cases if c.split == "validation" and c.path in negative_cases]
        if not target:
            continue
        model, state = open_model(checkpoint, device)
        if (state["manifest_sha256"] != hashlib.sha256(fold_manifest.read_bytes()).hexdigest()
                or "lesion" not in state["supervised_labels"] or state["distilled_updates"]
                or state["verified_ranking_pairs"]):
            raise ValueError("Use supervised lesion checkpoints trained on the exact recorded cross-fitting manifest.")
        provenance = {"proposal_model_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                      "proposal_training_patients": training_patients}
        for case in target:
            if case.path in negative_info or case.patient in training_patients:
                raise ValueError("Each negative case must be scored exactly once by a model that excluded its patient.")
            x = np.load(case.path / "image.npy", mmap_mode="r", allow_pickle=False)
            probs, _ = predict(model, x, patch, device)
            mask = binary_masks(probs, ["lesion"], threshold)["lesion"]
            negatives.extend(components(case, mask, x[2]))
            negative_info[case.path] = provenance
        del model
    if set(negative_info) != negative_cases:
        raise ValueError("The folds do not give exactly one cross-fitted prediction for every confirmed negative.")
    matched = pair_candidates(positives, negatives, maximum_volume_ratio=maximum_volume_ratio,
                              maximum_enhancement_gap=maximum_enhancement_gap)
    if not matched:
        raise ValueError("No cross-fitted candidates met the prespecified matching calipers.")
    pairs = [{"positive": {"case": str(p.case.path), "center_zyx": p.center},
              "negative": {"case": str(n.case.path), "center_zyx": n.center, **negative_info[n.case.path]},
              "matching_record": {"protocol": p.protocol, "positive_volume_ml": p.volume_ml,
                                  "negative_volume_ml": n.volume_ml, "positive_enhancement": p.enhancement,
                                  "negative_enhancement": n.enhancement}}
             for p, n in matched]
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("x", encoding="utf-8") as stream:
        json.dump({"pairs": pairs, "threshold": threshold, "maximum_volume_ratio": maximum_volume_ratio,
                   "maximum_enhancement_gap": maximum_enhancement_gap}, stream, indent=2, allow_nan=False)
        stream.write("\n")
    return len(pairs)
