"""Low-resolution proposals followed by high-resolution lesion prediction."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from scipy import ndimage

from .acquire import checksum, write_json
from .geometry import Grid, lps_grid
from .imaging import read_nifti
from .prepare import _write_case
from .research import read_manifest


COARSE_SPACING_MM = 2.5
PROPOSAL_THRESHOLD = .35
PROPOSAL_MARGIN_MM = 20.
PROPOSAL_LIMIT = 3


def prepare_coarse(manifest: Path, destination: Path, workers: int = 2) -> Path:
    if destination.exists():
        if json.loads(destination.read_text(encoding="utf-8"))["fine_manifest_sha256"] != checksum(manifest):
            raise ValueError("Coarse inputs belong to another development split.")
        read_manifest(destination)
        return destination
    cases = [case for case in read_manifest(manifest) if case.split == "train"]
    base = destination.parent / "coarse-cases"
    base.mkdir(parents=True, exist_ok=True)

    def prepare(case) -> dict:
        suffix = case.path.name
        target = base / suffix
        meta = case.metadata
        if not target.is_dir():
            image = np.load(case.path / "image.npy", mmap_mode="r", allow_pickle=False)
            fine = case.grid
            native = Grid.from_dict(meta["native_grid"])
            label, label_grid = read_nifti(case.path / "native_labels" / "lesion.nii.gz")
            if label_grid.shape != native.shape or not np.allclose(label_grid.affine, native.affine, atol=.001):
                raise ValueError("Coarse targets require the original lesion geometry.")
            references = {"lesion": (label, native, meta["labels"]["lesion"])}
            with TemporaryDirectory(dir=base, prefix=".coarse-") as directory:
                staged = Path(directory) / "case"
                _write_case(staged, image, fine, references, meta["reference"], case.patient,
                            case.cohort, meta["phase_protocol"], target_grid=lps_grid(fine, COARSE_SPACING_MM))
                staged.rename(target)
        return {"patient": case.patient, "cohort": case.cohort, "split": "train",
                "path": target.relative_to(destination.parent).as_posix()}

    with ThreadPoolExecutor(max_workers=workers) as pool:
        entries = list(pool.map(prepare, cases))
    write_json(destination, {"fine_manifest_sha256": checksum(manifest), "cases": entries})
    read_manifest(destination)
    return destination


def proposal_boxes(probability: np.ndarray, coarse: Grid, fine: Grid) -> list[tuple[slice, ...]]:
    components, count = ndimage.label(probability >= PROPOSAL_THRESHOLD)
    if not count:
        return []
    confidence = ndimage.maximum(probability, labels=components, index=np.arange(1, count + 1))
    order = np.argsort(-confidence, kind="stable")[:PROPOSAL_LIMIT]
    boxes = ndimage.find_objects(components)
    result = []
    margin = np.ceil(PROPOSAL_MARGIN_MM / fine.spacing).astype(int)
    for index in order:
        intervals = boxes[index]
        world = coarse.points(np.array([[s.start for s in intervals], [s.stop - 1 for s in intervals]]))
        coords = np.linalg.solve(fine.affine[:3, :3], (world - fine.affine[:3, 3]).T).T
        starts = np.maximum(0, np.floor(coords.min(axis=0)).astype(int) - margin)
        stops = np.minimum(fine.shape, np.ceil(coords.max(axis=0)).astype(int) + margin + 1)
        if np.all(starts < stops):
            result.append(tuple(slice(int(a), int(b)) for a, b in zip(starts, stops)))
    return result


def lesion_probability(model, image: np.ndarray, grid: Grid, patch: tuple[int, ...], device, *,
                       locator=None, locator_patch: tuple[int, ...] | None = None,
                       mirror_left_right: bool = False) -> np.ndarray:
    from .inference import predict
    from .labels import LABELS

    if mirror_left_right:
        options = {"locator": locator, "locator_patch": locator_patch}
        original = lesion_probability(model, image, grid, patch, device, **options)
        reflected = lesion_probability(model, image[..., ::-1], grid, patch, device, **options)
        return .5 * (original + reflected[..., ::-1])
    label_indices = (LABELS.index("lesion"),)
    if locator is None:
        return predict(model, image, patch, device, label_indices=label_indices)[0][0]
    coarse = lps_grid(grid, COARSE_SPACING_MM)
    inputs = np.stack([grid.resample(np.asarray(channel, dtype=np.float32), coarse, order=1) for channel in image])
    candidates = predict(locator, inputs, locator_patch or patch, device, label_indices=label_indices)[0][0]
    output = np.zeros(grid.shape, dtype=np.float32)
    for box in proposal_boxes(candidates, coarse, grid):
        cropped = predict(model, image[(..., *box)], patch, device, label_indices=label_indices)[0][0]
        np.maximum(output[box], cropped, out=output[box])
    return output
