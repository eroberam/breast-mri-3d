"""Reproducible tiled inference and split-aware exploratory evaluation."""

from __future__ import annotations

from itertools import product
import json
from pathlib import Path
import time

import numpy as np
from scipy import ndimage
import torch

from .geometry import Grid
from .imaging import read_nifti
from .labels import ENVELOPE, LABELS, REFERENCE_LABELS
from .network import BreastMRI3D
from .research import mask_metrics, match_lesions, read_manifest, summarize_metrics


def open_model(checkpoint: Path, device: torch.device) -> tuple[BreastMRI3D, dict]:
    record = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if record.get("labels") != list(LABELS):
        raise ValueError("Checkpoint has an incompatible label order.")
    model = BreastMRI3D(record["width"], dual_decoder=record["dual_decoder"])
    model.load_state_dict(record["state_dict"], strict=True)
    if device.type == "cpu":
        torch.set_num_threads(min(torch.get_num_threads(), 6))
    return model.to(device).eval(), record["summary"]


def _starts(n: int, width: int) -> list[int]:
    last = n - width
    starts = list(range(0, last + 1, max(1, width // 2)))
    if starts[-1] != last:
        starts.append(last)
    return starts


@torch.inference_mode()
def predict(model: BreastMRI3D, volume: np.ndarray, patch: tuple[int, int, int],
            device: torch.device, *, ranking: bool = False) -> tuple[np.ndarray, np.ndarray | None]:
    if (volume.ndim != 4 or volume.shape[0] != 3 or len(patch) != 3
            or any(n < 8 or n % 8 for n in patch)):
        raise ValueError("Expected three DCE channels and a patch size divisible by eight.")
    original = volume.shape[1:]
    src = np.pad(volume, [(0, 0)] + [(0, max(0, p - n)) for p, n in zip(patch, original)])
    shape = src.shape[1:]
    total = np.zeros((len(LABELS), *shape), dtype=np.float32)
    scores = np.zeros(shape, dtype=np.float32) if ranking else None
    covered = np.zeros(shape, dtype=np.float32)
    z, y, x = (np.linspace(-1, 1, n, dtype=np.float32) for n in patch)
    importance = np.exp(-2 * (z[:, None, None] ** 2 + y[None, :, None] ** 2 + x[None, None, :] ** 2))
    for starts in product(*[_starts(n, p) for n, p in zip(shape, patch)]):
        sl = tuple(slice(s, s + p) for s, p in zip(starts, patch))
        sample = np.asarray(src[(..., *sl)], dtype=np.float32, order="C")
        batch = torch.as_tensor(sample[None], device=device)
        with torch.autocast(device.type, enabled=device.type == "cuda"):
            logits, candidate = model(batch)
        total[(..., *sl)] += logits[0].float().sigmoid().cpu().numpy() * importance
        covered[sl] += importance
        if scores is not None:
            scores[sl] += candidate[0, 0].float().cpu().numpy() * importance
    valid = tuple(slice(0, n) for n in original)
    total /= covered
    if scores is not None:
        scores /= covered
    return total[(..., *valid)], scores[valid] if scores is not None else None


def structure_probability(probs: np.ndarray, label: str) -> np.ndarray:
    if label == ENVELOPE:
        return 1 - (1 - probs[0]) * (1 - probs[1])
    return probs[LABELS.index(label)]


def binary_masks(probs: np.ndarray, labels: list[str], threshold: float, *,
                 ranking: np.ndarray | None = None, rank_threshold: float | None = None) -> dict[str, np.ndarray]:
    if not 0 < threshold < 1:
        raise ValueError("A probability threshold must lie in (0, 1).")
    result = {name: structure_probability(probs, name) >= threshold for name in labels}
    if rank_threshold is not None:
        if ranking is None or "lesion" not in result or not np.isfinite(rank_threshold):
            raise ValueError("Ranking requires trained lesion scores and a finite validation-selected threshold.")
        for proposal, value in proposal_scores(result["lesion"], ranking):
            if value < rank_threshold:
                result["lesion"][proposal] = False
    return result


def proposal_scores(mask: np.ndarray, ranking: np.ndarray) -> list[tuple]:
    if mask.shape != ranking.shape:
        raise ValueError("Rank scores and lesion probabilities must share the same grid.")
    components, _ = ndimage.label(mask)
    scored = []
    for i, box in enumerate(ndimage.find_objects(components), start=1):
        coords = np.argwhere(components[box] == i)
        offset = np.array([interval.start for interval in box])
        distances = np.square(coords - coords.mean(axis=0)).sum(axis=1)
        peak = tuple(coords[distances.argmin()] + offset)
        global_indices = tuple((coords + offset).T)
        scored.append((global_indices, float(ranking[peak])))
    return scored


def evaluate(manifest: Path, checkpoint: Path, output: Path, *, split: str, device_name: str,
             patch: tuple[int, int, int], threshold: float, rank_threshold: float | None = None) -> dict:
    cases = [case for case in read_manifest(manifest) if case.split == split]
    if not cases:
        raise ValueError(f"No {split} cases are in the manifest.")
    if not 0 < threshold < 1:
        raise ValueError("A probability threshold must lie in (0, 1).")
    if output.exists():
        raise FileExistsError(f"Evaluation already exists: {output}")
    device = torch.device(device_name)
    model, summary = open_model(checkpoint, device)
    if rank_threshold is not None and not summary["verified_ranking_pairs"]:
        raise ValueError("The requested checkpoint was not trained with verified ranking pairs.")
    started = time.monotonic()
    results = []
    for i, case in enumerate(cases, start=1):
        x = np.load(case.path / "image.npy", mmap_mode="r", allow_pickle=False)
        reference = np.load(case.path / "labels.npy", mmap_mode="r", allow_pickle=False)
        prob, rank = predict(model, x, patch, device, ranking=rank_threshold is not None)
        pred = binary_masks(prob, summary["supervised_labels"], threshold,
                            ranking=rank, rank_threshold=rank_threshold)
        model_space = case.grid
        metadata = case.metadata
        provenance = metadata["labels"]
        for label in summary["supervised_labels"]:
            if label not in provenance:
                continue
            native = metadata.get("native_grid")
            if native is None:
                grid = model_space
                loaded = (np.load(case.path / f"{ENVELOPE}.npy", allow_pickle=False) if label == ENVELOPE
                          else reference[LABELS.index(label)])
                a, b = pred[label], loaded.astype(bool)
            else:
                grid = Grid.from_dict(native)
                target, declared_grid = read_nifti(case.path / "native_labels" / f"{label}.nii.gz")
                if (declared_grid.shape != grid.shape or not np.allclose(declared_grid.affine, grid.affine, atol=.001, rtol=0)
                        or not np.isin(target, (0, 1)).all()):
                    raise ValueError("Native validation masks must match the recorded physical DICOM grid.")
                a = model_space.resample(pred[label].astype(np.uint8), grid, order=0).astype(bool)
                b = target.astype(bool)
            record = {"patient": case.patient, "cohort": case.cohort, "structure": label,
                      "reference": provenance[label], "grid": "native_" + metadata["reference"].get("type", "dicom") if native else "model",
                      **mask_metrics(a, b, grid)}
            if label == "lesion" and (provenance[label] in {"manual", "confirmed_empty"}
                                      or metadata["reference"].get("assisted_manual")):
                detections = match_lesions(a, b, grid=grid)
                if provenance[label] != "confirmed_empty" and not metadata.get("lesion_annotations_exhaustive", False):
                    detections.pop("false_alarms")
                record.update(detections)
            results.append(record)
        print(f"evaluated case {i}/{len(cases)}", flush=True)
    by_label = {}
    strata = []
    for label in REFERENCE_LABELS:
        items = [x for x in results if x["structure"] == label]
        if items:
            by_label[label] = summarize_metrics(items)
        for cohort, reference, space in sorted({(x["cohort"], x["reference"], x["grid"]) for x in items}):
            cohort_rows = [x for x in items if (x["cohort"], x["reference"], x["grid"]) == (cohort, reference, space)]
            strata.append({"structure": label, "cohort": cohort, "reference": reference, "grid": space,
                           **summarize_metrics(cohort_rows)})
    report = {"split": split, "scope": json.loads(manifest.read_text(encoding="utf-8")).get("purpose"),
              "cases": len(cases), "threshold": threshold, "rank_threshold": rank_threshold,
              "structure_metrics": by_label, "stratified_metrics": strata,
              "elapsed_seconds": round(time.monotonic() - started, 3), "records": results}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return report
