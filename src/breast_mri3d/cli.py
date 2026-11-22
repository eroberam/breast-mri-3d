"""Independent command-line entry points for the research workflow."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="breast-mri-3d")
    commands = p.add_subparsers(dest="command", required=True)
    inspect = commands.add_parser("list-dicom", help="list MR series for explicit phase selection")
    inspect.add_argument("dicom", type=Path)
    prep = commands.add_parser("prepare-ispy1", help="link local derived masks to original DICOM")
    prep.add_argument("--dicom", required=True, type=Path)
    prep.add_argument("--release", required=True, type=Path)
    prep.add_argument("--expert-envelopes", type=Path)
    prep.add_argument("--out", required=True, type=Path)
    case = commands.add_parser("prepare-case", help="prepare explicitly paired acquisitions and reference masks")
    _export_args(case)
    case.add_argument("--pre", required=True)
    _label_args(case)
    case.add_argument("--mask-source-sha256", required=True, help="digest of the annotation's ordered source SOP instances")
    volumes = commands.add_parser("prepare-volumes", help="prepare phase-verified released NIfTI for training")
    volumes.add_argument("--pre", type=Path, required=True)
    volumes.add_argument("--post", type=Path, required=True)
    volumes.add_argument("--pre-index", type=int)
    volumes.add_argument("--post-index", type=int)
    volumes.add_argument("--release", required=True, help="immutable data-provider release identifier")
    volumes.add_argument("--out", required=True, type=Path)
    _label_args(volumes)
    baseline = commands.add_parser("prepare-nnunet", help="write lesion labels and the fixed development split")
    baseline.add_argument("manifest", type=Path)
    baseline.add_argument("--raw", required=True, type=Path)
    baseline.add_argument("--preprocessed", required=True, type=Path)
    baseline.add_argument("--dataset-id", type=int, default=73)
    baseline_score = commands.add_parser("score-nnunet", help="score nnU-Net validation masks on their native grid")
    baseline_score.add_argument("manifest", type=Path)
    baseline_score.add_argument("predictions", type=Path)
    baseline_score.add_argument("--out", required=True, type=Path)
    baseline_score.add_argument("--split", choices=("validation", "test"), default="validation")
    baseline_score.add_argument("--prediction-input", action="store_true")
    baseline_predict = commands.add_parser("prepare-nnunet-prediction", help="write held-out phase channels for nnU-Net")
    baseline_predict.add_argument("manifest", type=Path)
    baseline_predict.add_argument("--split", choices=("validation", "test"), required=True)
    baseline_predict.add_argument("--out", required=True, type=Path)
    pairs = commands.add_parser("match-proposals", help="mine and match cross-fitted negative training candidates")
    pairs.add_argument("manifest", type=Path)
    pairs.add_argument("folds", type=Path, help="JSON with fold manifest and supervised checkpoint paths")
    pairs.add_argument("--threshold", type=float, required=True)
    pairs.add_argument("--max-volume-ratio", type=float, required=True)
    pairs.add_argument("--max-enhancement-gap", type=float, required=True)
    pairs.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    pairs.add_argument("--patch", type=int, nargs=3, default=(32, 128, 128))
    pairs.add_argument("--out", required=True, type=Path)
    fit = commands.add_parser("train", help="train the compact residual network from scratch")
    fit.add_argument("manifest", type=Path)
    fit.add_argument("--out", required=True, type=Path)
    fit.add_argument("--steps", type=int, default=100)
    fit.add_argument("--budget-seconds", type=float, help="stop after this training time or --steps, whichever comes first")
    fit.add_argument("--width", type=int, default=12)
    fit.add_argument("--seed", type=int, default=19)
    fit.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    fit.add_argument("--patch", type=int, nargs=3, default=(24, 64, 64), metavar=("Z", "Y", "X"))
    fit.add_argument("--teachers", action="store_true")
    fit.add_argument("--pairs", type=Path)
    fit.add_argument("--shared-decoder", action="store_true", help="controlled local single-decoder reference")
    score = commands.add_parser("evaluate", help="evaluate a trained checkpoint on a requested split")
    score.add_argument("manifest", type=Path)
    score.add_argument("checkpoint", type=Path)
    score.add_argument("--split", choices=("validation", "test"), required=True)
    score.add_argument("--threshold", type=float, required=True)
    score.add_argument("--rank-threshold", type=float)
    score.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    score.add_argument("--patch", type=int, nargs=3, default=(32, 128, 128))
    score.add_argument("--out", type=Path, required=True)
    export = commands.add_parser("export", help="standardize binary NIfTI masks on a source MR series")
    _export_args(export)
    export.add_argument("--mask", action="append", required=True, metavar="LABEL=FILE")
    export.add_argument("--mask-series-uid", required=True, help="declared MR source UID of the supplied masks")
    export.add_argument("--mask-frame-uid", required=True, help="declared patient frame of the masks")
    export.add_argument("--mask-source-sha256", required=True, help="digest of the mask's ordered source SOP instances")
    export.add_argument("--algorithm-type", required=True, choices=("MANUAL", "SEMIAUTOMATIC", "AUTOMATIC"))
    export.add_argument("--algorithm-name", help="algorithm that produced the supplied masks")
    export.add_argument("--algorithm-version", help="version of the supplied-mask algorithm")
    segment = commands.add_parser("segment", help="run paired DCE inference and export a review bundle")
    _export_args(segment)
    segment.add_argument("--pre", required=True, help="precontrast acquisition selector from list-dicom")
    segment.add_argument("--checkpoint", required=True, type=Path)
    segment.add_argument("--threshold", type=float, required=True)
    segment.add_argument("--rank-threshold", type=float)
    segment.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    segment.add_argument("--patch", type=int, nargs=3, default=(32, 128, 128))
    demo = commands.add_parser("demo", help="generate a synthetic verified DICOM-to-print example")
    demo.add_argument("--out", required=True, type=Path)
    return p


def _export_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("dicom", type=Path)
    p.add_argument("--series", required=True, help="source or early postcontrast selector from list-dicom")
    p.add_argument("--out", required=True, type=Path)


def _label_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--patient", required=True, help="canonical identifier shared across data providers")
    p.add_argument("--cohort", required=True)
    p.add_argument("--phase-protocol", help="documented protocol group for proposal matching")
    p.add_argument("--mask", action="append", default=[], metavar="LABEL=FILE", help="manual reference NIfTI")
    p.add_argument("--confirmed-negative", action="store_true", help="source documents no lesions in either breast")


def _read_masks(specs: list[str]) -> dict:
    from .imaging import read_nifti

    masks = {}
    for spec in specs:
        label, delimiter, value = spec.partition("=")
        if not delimiter or label in masks:
            raise ValueError("Pass each label once as --mask LABEL=FILE.")
        mask, grid = read_nifti(Path(value))
        if not ((mask == 0) | (mask == 1)).all():
            raise ValueError(f"Mask for {label} must be binary.")
        masks[label] = (mask, grid)
    return masks


def _prepare_case(args: argparse.Namespace) -> None:
    import numpy as np

    from .dicom import assemble, scan_series
    from .prepare import prepare_case

    series = scan_series(args.dicom)
    if args.pre not in series or args.series not in series:
        raise ValueError("The specified precontrast or postcontrast source series was not found.")
    pre, post = assemble(series[args.pre]), assemble(series[args.series])
    if post.source_instances_sha256 != args.mask_source_sha256:
        raise ValueError("Annotation provenance does not match the original postcontrast slices.")
    references = {name: (mask, grid, "manual") for name, (mask, grid) in _read_masks(args.mask).items()}
    if args.confirmed_negative:
        if "lesion" in references:
            raise ValueError("A confirmed negative exam cannot have a lesion annotation.")
        references["lesion"] = (np.zeros(post.grid.shape, dtype=np.uint8), post.grid, "confirmed_empty")
    prepare_case(args.out, pre, post, references, patient=args.patient, cohort=args.cohort,
                 phase_protocol=args.phase_protocol)
    print(args.out)


def _export(args: argparse.Namespace) -> None:
    from .dicom import select
    from .export import export_case

    source = select(args.dicom, args.series)
    if (source.uid != args.mask_series_uid or source.frame_uid != args.mask_frame_uid
            or source.source_instances_sha256 != args.mask_source_sha256):
        raise ValueError("Declared mask DICOM provenance does not match the source images.")
    masks = {}
    for label, (mask, grid) in _read_masks(args.mask).items():
        mapped = grid.resample(mask, source.grid, order=0)
        if mask.any() and not mapped.any():
            raise ValueError(f"Mask for {label} does not intersect its declared source.")
        masks[label] = mapped
    path = export_case(args.out, source, masks, mask_grid=source.grid,
                       origin={"type": "supplied_masks", "series_instance_uid": args.mask_series_uid,
                               "frame_of_reference_uid": args.mask_frame_uid,
                               "source_instances_sha256": args.mask_source_sha256,
                               "algorithm_type": args.algorithm_type, "algorithm_name": args.algorithm_name,
                               "algorithm_version": args.algorithm_version})
    print(path)


def _segment(args: argparse.Namespace) -> None:
    import torch

    from .dicom import assemble, scan_series
    from .export import export_case
    from .geometry import model_grid
    from .imaging import prepare_dce
    from .inference import binary_masks, open_model, predict

    import numpy as np

    series = scan_series(args.dicom)
    if args.pre not in series or args.series not in series:
        raise ValueError("The specified precontrast or postcontrast source series was not found.")
    pre, post = assemble(series[args.pre]), assemble(series[args.series])
    grid = model_grid(post.grid)
    aligned = prepare_dce(pre, post)
    inputs = np.stack([post.grid.resample(x, grid, order=1) for x in aligned])
    device = torch.device(args.device)
    model, summary = open_model(args.checkpoint, device)
    if args.rank_threshold is not None and not summary["verified_ranking_pairs"]:
        raise ValueError("The requested checkpoint was not trained with verified ranking pairs.")
    probs, rank = predict(model, inputs, tuple(args.patch), device, ranking=args.rank_threshold is not None)
    labels = summary["supervised_labels"]
    mask = binary_masks(probs, labels, args.threshold, ranking=rank, rank_threshold=args.rank_threshold)
    path = export_case(args.out, post, mask, mask_grid=grid, origin={
        "type": "research_model", "checkpoint_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        "threshold": args.threshold, "rank_threshold": args.rank_threshold,
        "series_instance_uid": post.uid, "frame_of_reference_uid": post.frame_uid,
        "precontrast_series_uid": pre.uid, "source_instances_sha256": post.source_instances_sha256,
        "precontrast_instances_sha256": pre.source_instances_sha256,
    })
    print(path)


def main() -> None:
    args = _parser().parse_args()
    if args.command == "list-dicom":
        from .dicom import assemble, scan_series

        rows = []
        for key, entries in scan_series(args.dicom).items():
            row = {"selector": key, "slices": len(entries),
                   "description": str(entries[0][1].get("SeriesDescription", ""))}
            try:
                series = assemble(entries)
                row.update(series_instance_uid=series.uid, frame_of_reference_uid=series.frame_uid,
                           source_instances_sha256=series.source_instances_sha256)
            except ValueError as exc:
                row["geometry_error"] = str(exc)
            rows.append(row)
        print(json.dumps(rows, indent=2))
    elif args.command == "prepare-ispy1":
        from .prepare import prepare_ispy1

        print(prepare_ispy1(args.dicom, args.release, args.out, expert=args.expert_envelopes))
    elif args.command == "prepare-case":
        _prepare_case(args)
    elif args.command == "prepare-volumes":
        from .prepare import prepare_volumes

        references = {name: (mask, grid, "manual") for name, (mask, grid) in _read_masks(args.mask).items()}
        prepare_volumes(args.out, args.pre, args.post, references, pre_index=args.pre_index,
                        post_index=args.post_index, release=args.release, patient=args.patient,
                        cohort=args.cohort, phase_protocol=args.phase_protocol,
                        confirmed_negative=args.confirmed_negative)
        print(args.out)
    elif args.command == "prepare-nnunet":
        from .baseline import prepare_nnunet

        print(prepare_nnunet(args.manifest, args.raw, args.preprocessed, args.dataset_id))
    elif args.command == "score-nnunet":
        from .baseline import score_nnunet

        report = score_nnunet(args.manifest, args.predictions, args.out, split=args.split,
                              prediction_input=args.prediction_input)
        print(json.dumps(report["metrics"], indent=2))
    elif args.command == "prepare-nnunet-prediction":
        from .baseline import prepare_nnunet_prediction

        print(prepare_nnunet_prediction(args.manifest, args.out, split=args.split))
    elif args.command == "match-proposals":
        from .ranking import build_pairs

        print(build_pairs(args.manifest, args.folds, args.out, threshold=args.threshold,
                          patch=tuple(args.patch), device_name=args.device,
                          maximum_volume_ratio=args.max_volume_ratio,
                          maximum_enhancement_gap=args.max_enhancement_gap))
    elif args.command == "train":
        from .train import train

        print(json.dumps(train(args.manifest, args.out, steps=args.steps, width=args.width,
                               patch=tuple(args.patch), seed=args.seed, device_name=args.device,
                               teachers=args.teachers, pairs_path=args.pairs, budget_seconds=args.budget_seconds,
                               dual_decoder=not args.shared_decoder), indent=2))
    elif args.command == "evaluate":
        from .inference import evaluate

        report = evaluate(args.manifest, args.checkpoint, args.out, split=args.split, device_name=args.device,
                          patch=tuple(args.patch), threshold=args.threshold, rank_threshold=args.rank_threshold)
        print(json.dumps(report["structure_metrics"], indent=2))
    elif args.command == "export":
        _export(args)
    elif args.command == "segment":
        _segment(args)
    elif args.command == "demo":
        from .phantom import demo

        print(demo(args.out))


if __name__ == "__main__":
    main()
