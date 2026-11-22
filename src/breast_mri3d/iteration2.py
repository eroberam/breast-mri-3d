"""Reproducible preparation entry points for the expert-contour experiment."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import re
from tempfile import TemporaryDirectory
from urllib.parse import quote

import nibabel as nib
import numpy as np

from .acquire import acquire_ambl, acquire_ispy, checksum, download, ispy_id, safe_relative, write_json
from .geometry import Grid
from .imaging import dce_channels, read_nifti
from .labels import ENVELOPE
from .prepare import _write_case
from .research import read_manifest


PHYSIONET_RELEASE = "https://physionet.org/files/mama-miaai-herewegoagain/1.0.0"


def same_grid(a: Grid, b: Grid) -> bool:
    return a.shape == b.shape and np.allclose(a.affine, b.affine, atol=1e-3, rtol=0)


def require_coverage(mask: np.ndarray, source: Grid, target: Grid, minimum: float = .995) -> float:
    """Check support in patient coordinates before projecting a reference mask."""
    positive = np.argwhere(mask)
    if not len(positive):
        raise ValueError("The expert reference is empty.")
    transform = np.linalg.solve(target.affine, source.affine)
    sampled = positive[::max(1, len(positive) // 20000)]
    mapped = sampled @ transform[:3, :3].T + transform[:3, 3]
    recovered = np.all((mapped >= -.5) & (mapped < np.asarray(target.shape) - .5), axis=1)
    fraction = float(recovered.mean())
    if fraction < minimum:
        raise ValueError(f"Only {fraction:.2%} of the reference lies within the documented image field of view.")
    return fraction


def _atomic_case(destination: Path, channels: np.ndarray, grid: Grid, references: dict,
                 reference: dict, patient: str, cohort: str, protocol: str | None = None) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        case = json.loads((destination / "case.json").read_text(encoding="utf-8"))
        if (case["patient"] != patient or case["reference"] != reference
                or case["labels"] != {name: value[2] for name, value in references.items()}):
            raise ValueError("The prepared case belongs to a different source or label release.")
        return
    with TemporaryDirectory(dir=destination.parent, prefix=".case-") as temporary:
        staged = Path(temporary) / "case"
        _write_case(staged, channels, grid, references, reference, patient, cohort, protocol)
        staged.rename(destination)


def prepare_stv(root: Path, *, workers: int = 2) -> Path:
    ledger = root / "acquisition" / "ispy1-stv-acquired.json"
    data = json.loads(ledger.read_text(encoding="utf-8"))
    raw = root / "raw" / "ISPY1-STV"
    prepared = root / "prepared" / "ISPY1-STV"

    def prepare_one(case: dict) -> dict:
        files = {key: raw.joinpath(*safe_relative(name).parts) for key, name in case["files"].items()}
        if any(checksum(files[key], "md5") != case["md5"][name] for key, name in case["files"].items()):
            raise ValueError("A local phase or expert mask failed its TCIA checksum.")
        pre, pre_grid = read_nifti(files["pre"])
        post, post_grid = read_nifti(files["post"])
        if not same_grid(pre_grid, post_grid):
            raise ValueError("The documented precontrast and early postcontrast do not share the release geometry.")
        mask, mask_grid = read_nifti(files["lesion"])
        if not np.isin(mask, (0, 1)).all():
            raise ValueError("The expert STV reference does not use the documented binary encoding.")
        coverage = require_coverage(mask, mask_grid, post_grid)
        channels = dce_channels(pre.astype(np.float32), post.astype(np.float32))
        patient = case["patient"]
        reference = {"type": "nifti", "dataset_release": f"{data['dataset']}@{data['revision']}",
                     "end_point": "manual_structural_primary_tumor", "precontrast": files["pre"].name,
                     "early_postcontrast": files["post"].name, "label_fov_coverage": coverage,
                     "file_md5": case["md5"]}
        _atomic_case(prepared / patient, channels, post_grid, {"lesion": (mask, mask_grid, "manual")},
                     reference, patient, "ISPY1-STV", "ispy1_native_dce_early_post")
        return {"path": patient, "patient": patient, "cohort": "ISPY1-STV", "split": case["split"]}

    records, failures = [], []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(prepare_one, case): case for case in data["cases"]}
        for future in as_completed(futures):
            try:
                records.append(future.result())
            except (ValueError, OSError) as exc:
                failures.append({"patient": futures[future]["patient"], "reason": str(exc)})
            if (len(records) + len(failures)) % 10 == 0:
                print(f"I-SPY1 physically prepared {len(records)}, excluded {len(failures)}", flush=True)
    if {case["split"] for case in records} != {"train", "validation"}:
        raise RuntimeError("The verified cohort needs both training and validation cases.")
    manifest = prepared / "manifest.json"
    write_json(manifest, {"purpose": "expert_structural_primary_tumor_development", "seed": data["seed"],
                         "acquisition_ledger_sha256": checksum(ledger),
                         "cases": sorted(records, key=lambda c: c["patient"]), "excluded": failures})
    return manifest


def _subject_split(records: list[dict], subject: str) -> str:
    by_patient = {row["patient"].casefold(): row["split"] for row in records}
    if subject.casefold() in by_patient:
        return by_patient[subject.casefold()]
    fraction = int.from_bytes(hashlib.sha256(f"19:{subject}".encode()).digest()[:4], "big") / 2 ** 32
    return "validation" if fraction < .2 else "train"


def _manual_envelope(path: Path, pre_file: Path, reference: Grid) -> tuple[np.ndarray, Grid]:
    image = nib.load(str(path))
    if image.header.get_xyzt_units()[0] == "unknown":
        # The documented mask is on the companion precontrast acquisition.
        pre = nib.load(str(pre_file))
        if image.shape != pre.shape or not np.allclose(image.affine, pre.affine, rtol=0, atol=1e-3):
            raise ValueError("The unitless envelope does not exactly match its physical companion image.")
        return np.asanyarray(image.dataobj).transpose(2, 1, 0), reference
    return read_nifti(path)


def prepare_envelopes(root: Path, *, limit: int = 60, workers: int = 4) -> Path:
    raw = root / "raw" / "MAMA-MIAAI"
    prepared = root / "prepared" / "MAMA-MIAAI"
    stv = json.loads((root / "prepared/ISPY1-STV/manifest.json").read_text(encoding="utf-8"))["cases"]
    sums = download(f"{PHYSIONET_RELEASE}/SHA256SUMS.txt", raw / "SHA256SUMS.txt")
    hashes = {}
    by_name: dict[str, list[str]] = {}
    for line in sums.read_text(encoding="utf-8-sig").splitlines():
        pieces = re.fullmatch(r"([0-9a-f]{64})\s+\*?(.+)", line.strip())
        if pieces is None:
            continue
        name = safe_relative(pieces[2]).as_posix()
        hashes[name] = pieces[1]
        if name.startswith("BiasCorrection/"):
            by_name.setdefault(safe_relative(name).name, []).append(name)
    selection = []
    for path in hashes:
        if not path.startswith("BreastSegmentation/expert/"):
            continue
        matched = re.fullmatch(r"((?:ISPY1|ISPY2|NACT|DUKE)_\d+)_0000\.nii(?:\.gz)?", safe_relative(path).name)
        if not matched:
            continue
        subject = matched[1]
        pre = by_name.get(f"{subject}_0000.nii.gz", [])
        post = by_name.get(f"{subject}_0001.nii.gz", [])
        if len(pre) == 1 and len(post) == 1:
            patient = ispy_id(subject) if subject.startswith("ISPY1_") else subject
            selection.append({"patient": patient, "files": [*pre, *post, path],
                              "split": _subject_split(stv, patient)})
    selection.sort(key=lambda c: (not c["patient"].startswith("ISPY1_"),
                                  hashlib.sha256(f"19:{c['patient']}".encode()).digest()))

    def prepare_one(case: dict) -> dict:
        pre_path, post_path, mask_path = [download(f"{PHYSIONET_RELEASE}/{quote(name, safe='/')}",
                                                  raw.joinpath(*safe_relative(name).parts),
                                                  expected=hashes[name], algorithm="sha256") for name in case["files"]]
        pre, pre_grid = read_nifti(pre_path)
        post, post_grid = read_nifti(post_path)
        if not same_grid(pre_grid, post_grid):
            raise ValueError("Manual envelope companion phases differ in physical geometry.")
        mask, mask_grid = _manual_envelope(mask_path, pre_path, pre_grid)
        if not same_grid(mask_grid, pre_grid) or not np.isin(mask, (0, 1)).all():
            raise ValueError("The manual envelope and its companion precontrast do not share a binary physical grid.")
        require_coverage(mask, mask_grid, post_grid)
        patient = case["patient"]
        reference = {"type": "nifti", "dataset_release": "physionet:10.13026/w6wc-jh81/1.0.0",
                     "end_point": "manual_whole_breast_envelope", "files": {name: hashes[name] for name in case["files"]}}
        channels = dce_channels(pre.astype(np.float32), post.astype(np.float32))
        _atomic_case(prepared / patient, channels, post_grid, {ENVELOPE: (mask, mask_grid, "manual")},
                     reference, patient, "MAMA-MIAAI")
        return {"path": patient, "patient": patient, "cohort": "MAMA-MIAAI", "split": case["split"]}

    verified, excluded = [], []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(prepare_one, case): case for case in selection[:limit]}
        for future in as_completed(futures):
            try:
                verified.append(future.result())
            except (ValueError, OSError) as exc:
                excluded.append({"patient": futures[future]["patient"], "reason": str(exc)})
            if (len(verified) + len(excluded)) % 5 == 0:
                print(f"Manual envelopes physically verified {len(verified)}, excluded {len(excluded)}", flush=True)
    manifest = prepared / "manifest.json"
    write_json(manifest, {"purpose": "manual_whole_breast_envelope", "source_sha256": checksum(sums),
                         "cases": sorted(verified, key=lambda row: row["patient"]), "excluded": excluded})
    return manifest


def combine_development(output: Path, primary: list[Path], auxiliary: list[Path]) -> Path:
    """Freeze the expert tumor/cohort split and align overlapping anatomy records to it."""
    if output.exists():
        raise FileExistsError("The combined manifest already exists.")
    roles = {}
    inputs = []
    records = []
    groups = [(manifest, is_auxiliary, read_manifest(manifest)) for manifests, is_auxiliary
              in [(primary, False), (auxiliary, True)] for manifest in manifests]
    reserved = {case.patient for _, _, cases in groups for case in cases if case.split == "test"}
    for manifest, is_auxiliary, cases in groups:
        inputs.append({"file": str(manifest.resolve()), "sha256": checksum(manifest),
                       "role": "auxiliary" if is_auxiliary else "primary"})
        for case in cases:
            if case.split == "test":
                continue
            if case.patient in reserved:
                raise ValueError("A reserved patient also appears in development.")
            assigned = roles.setdefault(case.patient, case.split)
            if case.split != assigned and not (is_auxiliary and ENVELOPE in case.metadata["labels"]):
                raise ValueError("A canonical patient disagrees with the fixed tumor split.")
            records.append({"patient": case.patient, "path": str(case.path), "cohort": case.cohort,
                            "split": assigned})
    if len({row["path"] for row in records}) != len(records):
        raise ValueError("The same prepared case cannot appear more than once.")
    if {row["split"] for row in records} != {"train", "validation"}:
        raise ValueError("Combined development needs training and validation patients.")
    write_json(output, {"purpose": "fixed_development", "sources": inputs, "cases": records})
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path, help="Local storage for study files and run artifacts")
    commands = parser.add_subparsers(dest="command", required=True)
    fetch = commands.add_parser("acquire-ispy1")
    fetch.add_argument("--legacy", type=Path, help="optional local manifest of previous pilot patients")
    fetch.add_argument("--limit", type=int, default=120)
    fetch.add_argument("--seed", type=int, default=19)
    fetch.add_argument("--workers", type=int, default=8)
    ambl = commands.add_parser("acquire-ambl")
    ambl.add_argument("--ranges", action="store_true", help="download the DCE archive in resumable HTTP ranges")
    ambl.add_argument("--local-dce", type=Path, help="verified copy of an already downloaded Zenodo archive")
    original = commands.add_parser("acquire-ambl-mr")
    original.add_argument("--limit", type=int, default=88)
    original.add_argument("--registered", action="store_true")
    external = commands.add_parser("prepare-ambl")
    external.add_argument("--model-lock", type=Path, required=True)
    prep = commands.add_parser("prepare-stv")
    prep.add_argument("--workers", type=int, default=2)
    envelope = commands.add_parser("prepare-envelopes")
    envelope.add_argument("--limit", type=int, default=60)
    envelope.add_argument("--workers", type=int, default=4)
    negative = commands.add_parser("prepare-ea1141")
    negative.add_argument("--limit", type=int, default=80)
    mama = commands.add_parser("prepare-mama")
    mama.add_argument("--token-file", required=True, type=Path)
    mama.add_argument("--limit", type=int, default=32)
    mama.add_argument("--workers", type=int, default=3)
    odelia = commands.add_parser("prepare-odelia")
    odelia.add_argument("--token-file", required=True, type=Path)
    odelia.add_argument("--limit", type=int, default=80)
    odelia.add_argument("--workers", type=int, default=5)
    merge = commands.add_parser("combine-development")
    merge.add_argument("--out", type=Path, required=True)
    merge.add_argument("--primary", type=Path, nargs="+", required=True)
    merge.add_argument("--auxiliary", type=Path, nargs="*", default=[])
    args = parser.parse_args()
    if args.command == "acquire-ispy1":
        print(acquire_ispy(args.root, args.legacy, limit=args.limit, seed=args.seed, workers=args.workers))
    elif args.command == "acquire-ambl":
        print(acquire_ambl(args.root, ranged=args.ranges, local_dce=args.local_dce))
    elif args.command == "acquire-ambl-mr":
        from .ambl import acquire_original_mr

        print(acquire_original_mr(args.root, limit=args.limit, registered=args.registered))
    elif args.command == "prepare-ambl":
        from .ambl import prepare_ambl

        print(prepare_ambl(args.root, args.model_lock))
    elif args.command == "prepare-envelopes":
        print(prepare_envelopes(args.root, limit=args.limit, workers=args.workers))
    elif args.command == "prepare-ea1141":
        from .ea1141 import prepare_ea1141

        print(prepare_ea1141(args.root, limit=args.limit))
    elif args.command == "prepare-mama":
        from .mama_mia import prepare_mama

        print(prepare_mama(args.root, token_file=args.token_file, limit=args.limit, workers=args.workers))
    elif args.command == "prepare-odelia":
        from .odelia import prepare_odelia

        print(prepare_odelia(args.root, token_file=args.token_file, limit=args.limit, workers=args.workers))
    elif args.command == "combine-development":
        print(combine_development(args.out, args.primary, args.auxiliary))
    else:
        print(prepare_stv(args.root, workers=args.workers))


if __name__ == "__main__":
    main()
