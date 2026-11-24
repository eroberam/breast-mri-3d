"""Strict split handling, partial-label losses and verified training-only pairs."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
from scipy import ndimage
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree

from .geometry import Grid
from .labels import ENVELOPE, LABELS, REFERENCE_LABELS


@dataclass(frozen=True)
class Case:
    path: Path
    patient: str
    cohort: str
    split: str

    @property
    def metadata(self) -> dict:
        return json.loads((self.path / "case.json").read_text(encoding="utf-8"))

    @property
    def grid(self) -> Grid:
        return Grid.from_dict(self.metadata["grid"])


def read_manifest(path: Path) -> list[Case]:
    path = Path(path).resolve()
    contents = json.loads(path.read_text(encoding="utf-8"))
    role: dict[str, str] = {}
    seen: set[Path] = set()
    cases = []
    for row in contents["cases"]:
        patient = str(row["patient"]).strip().casefold()
        split = row["split"]
        if not patient or split not in {"train", "validation", "test"}:
            raise ValueError("Manifest requires patient identifiers and a recognized split.")
        if patient in role and role[patient] != split:
            raise ValueError(f"Patient appears in more than one split: {patient}")
        role[patient] = split
        case_path = (path.parent / row["path"]).resolve()
        if case_path in seen:
            raise ValueError("A prepared case occurs twice in the manifest.")
        seen.add(case_path)
        case = Case(case_path, patient, row["cohort"], split)
        meta = case.metadata
        if str(meta["patient"]).strip().casefold() != patient or meta["cohort"] != row["cohort"]:
            raise ValueError("Prepared metadata does not match the manifest patient or cohort.")
        if meta["channel_order"] != list(LABELS) or not set(meta["labels"]).issubset(REFERENCE_LABELS):
            raise ValueError("Prepared label vocabulary does not match the package.")
        if ENVELOPE in meta["labels"] and not (case.path / f"{ENVELOPE}.npy").is_file():
            raise ValueError("The declared breast envelope is missing its supervised array.")
        cases.append(case)
    return cases


def keep_largest_components(mask: np.ndarray, limit: int) -> np.ndarray:
    """Retain the largest 26-connected components on a common physical grid."""
    if type(limit) is not int or limit < 1:
        raise ValueError("The connected component limit must be a positive integer.")
    components, count = ndimage.label(mask, structure=np.ones((3, 3, 3), dtype=bool))
    if count <= limit:
        return np.asarray(mask, dtype=bool)
    volumes = np.bincount(components.ravel())[1:]
    largest = 1 + np.argsort(-volumes, kind="stable")[:limit]
    return np.isin(components, largest)


def match_lesions(prediction: np.ndarray, reference: np.ndarray, *, iou_threshold: float = .1,
                  grid: Grid | None = None, instances: np.ndarray | None = None) -> dict:
    """Match components one-to-one. False alarms require exhaustive truth or known negatives."""
    if not 0 < iou_threshold <= 1:
        raise ValueError("IoU threshold must lie in (0, 1].")
    p, count_p = ndimage.label(prediction)
    if instances is None:
        t, count_t = ndimage.label(reference)
    else:
        foreground = np.asarray(reference, dtype=bool)
        instances = np.asarray(instances)
        if (instances.shape != foreground.shape or not np.issubdtype(instances.dtype, np.integer)
                or np.any(instances < 0) or not np.array_equal(instances > 0, foreground)):
            raise ValueError("Lesion identifiers must cover exactly the reference foreground.")
        values = instances[foreground]
        identifiers = np.unique(values)
        t = np.zeros(foreground.shape, dtype=np.int32)
        t[foreground] = np.searchsorted(identifiers, values) + 1
        count_t = len(identifiers)
    hits = np.zeros(count_t, dtype=bool)
    t_size = np.bincount(t.flat)[1:]
    if count_p and count_t:
        p_size = np.bincount(p.flat)[1:]
        common = (p > 0) & (t > 0)
        joint = np.bincount((p[common] - 1) * count_t + t[common] - 1, minlength=count_p * count_t)
        intersection = joint.reshape(count_p, count_t)
        iou = intersection / (p_size[:, None] + t_size[None, :] - intersection)
        valid = iou >= iou_threshold
        # Maximize the number of valid matches before comparing overlap quality.
        row, col = linear_sum_assignment(valid * (1 + iou / (1 + min(count_p, count_t))), maximize=True)
        hits[col] = valid[row, col]
    matched = int(hits.sum())
    result = {"matched": matched, "missed": int(count_t - matched), "false_alarms": int(count_p - matched)}
    if grid is not None:
        small = (t_size * grid.voxel_mm3) <= np.pi * 10 ** 3 / 6
        result.update(small_lesions=int(small.sum()), small_matched=int((small & hits).sum()))
    return result


def lesion_instances(case: Case, reference: np.ndarray, grid: Grid, source: dict) -> np.ndarray | None:
    endpoint = source.get("end_point")
    if endpoint in {"manual_structural_primary_tumor", "expert_reviewed_primary_tumor"}:
        return np.asarray(reference, dtype=np.int32)
    if endpoint == "all_reported_lesions" and case.cohort == "AMBL":
        from .ambl import native_instances

        return native_instances(case, reference, grid)
    return None


def mask_metrics(a: np.ndarray, b: np.ndarray, grid: Grid) -> dict:
    if a.shape != grid.shape or b.shape != grid.shape:
        raise ValueError("Metrics require two masks on the evaluation grid.")
    a, b = np.asarray(a, dtype=bool), np.asarray(b, dtype=bool)
    a_count, b_count = int(a.sum()), int(b.sum())
    hd95 = None
    if a_count and b_count:
        points = [grid.points(np.argwhere(x ^ ndimage.binary_erosion(x))) for x in (a, b)]
        distances = [cKDTree(dst).query(src)[0] for src, dst in (points, points[::-1])]
        hd95 = max(float(np.percentile(distance, 95)) for distance in distances)
    return {"dice": float(2 * (a & b).sum() / (a_count + b_count)) if a_count + b_count else 1.0,
            "hd95_mm": hd95, "abs_volume_error_ml": float(abs(a_count - b_count) * grid.voxel_mm3 / 1000)}


def summarize_metrics(rows: list[dict]) -> dict:
    surfaces = [row["hd95_mm"] for row in rows if row["hd95_mm"] is not None]
    return {"n": len(rows), "median_dice": float(np.median([row["dice"] for row in rows])),
            "mean_dice": float(np.mean([row["dice"] for row in rows])),
            "median_hd95_mm": float(np.median(surfaces)) if surfaces else None, "n_hd95": len(surfaces),
            "median_abs_volume_error_ml": float(np.median([row["abs_volume_error_ml"] for row in rows]))}
