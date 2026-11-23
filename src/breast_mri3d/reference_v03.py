"""Run the official v2 ResEncM model with one CUDA process and native-grid evaluation."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil

from .baseline import prepare_nnunet, prepare_nnunet_prediction, score_nnunet
from .research import read_manifest


DATASET_ID = 76
DATASET_NAME = "Dataset076_BreastMRI3D"
TRAINER = "V03BudgetTrainer"
CONFIGURATION = "3d_fullres"
FILE_ENDING = ".nii"


def prepare(manifest: Path) -> None:
    prepare_nnunet(manifest, Path(os.environ["nnUNet_raw"]), Path(os.environ["nnUNet_preprocessed"]),
                   DATASET_ID, channel_indices=(2,), ending=FILE_ENDING, workers=4)


def train(plans: str, hours: float) -> None:
    import torch
    from nnunetv2.run.run_training import get_trainer_from_args

    os.environ["BMRI_REFERENCE_BUDGET_SECONDS"] = str(hours * 3600)
    trainer = get_trainer_from_args(DATASET_NAME, CONFIGURATION, 0, TRAINER, plans, device=torch.device("cuda"))
    folder = Path(trainer.output_folder)
    target = folder / f"checkpoint_{hours:g}h.pth"
    if target.is_file():
        return
    candidates = [p for name in ("final", "latest", "best") if (p := folder / f"checkpoint_{name}.pth").is_file()]
    if candidates:
        trainer.load_checkpoint(str(max(candidates, key=lambda path: path.stat().st_mtime_ns)))
    trainer.run_training()
    elapsed = json.loads((folder / "budget.json").read_text(encoding="utf-8"))["elapsed_seconds"]
    if elapsed < hours * 3600:
        raise RuntimeError("nnU-Net stopped without reaching the target budget.")
    partial = target.with_suffix(".partial.pth")
    shutil.copyfile(folder / "checkpoint_final.pth", partial)
    os.replace(partial, target)
    shutil.copyfile(folder / "budget.json", target.with_suffix(".budget.json"))


def evaluate(manifest: Path, checkpoint: Path, directory: Path, output: Path, split: str) -> None:
    import torch
    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

    cases = [c for c in read_manifest(manifest) if c.split == split]
    if not cases:
        raise ValueError("No cases are eligible for the requested reference evaluation.")
    inputs, outputs = directory / "inputs", directory / "outputs"
    prefixes = {f"BreastMRI_{i:04d}" for i in range(1, len(cases) + 1)}
    if not inputs.is_dir():
        prepare_nnunet_prediction(manifest, inputs, split=split, channel_indices=(2,),
                                 ending=FILE_ENDING, workers=4)
    if {p.name for p in inputs.glob(f"*{FILE_ENDING}")} != {f"{x}_0000{FILE_ENDING}" for x in prefixes}:
        raise ValueError("Reference prediction inputs differ from the fixed cohort.")
    outputs.mkdir(parents=True, exist_ok=True)
    expected = {f"{key}{FILE_ENDING}" for key in prefixes}
    if expected != {p.name for p in outputs.glob(f"*{FILE_ENDING}")}:
        predictor = nnUNetPredictor(tile_step_size=.5, use_gaussian=True, use_mirroring=True,
                                    perform_everything_on_device=True, device=torch.device("cuda"), allow_tqdm=False)
        predictor.initialize_from_trained_model_folder(str(checkpoint.parent.parent), use_folds=(0,),
                                                       checkpoint_name=checkpoint.name)
        predictor.predict_from_files_sequential(str(inputs), str(outputs), overwrite=False)
    if expected != {p.name for p in outputs.glob(f"*{FILE_ENDING}")}:
        raise ValueError("Reference prediction outputs differ from the fixed cohort.")
    if not output.is_file():
        score_nnunet(manifest, outputs, output, split=split, prediction_input=True, ending=FILE_ENDING)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="action", required=True)
    commands.add_parser("prepare").add_argument("manifest", type=Path)
    fit = commands.add_parser("train")
    fit.add_argument("plans")
    fit.add_argument("--hours", required=True, type=float)
    test = commands.add_parser("evaluate")
    test.add_argument("manifest", type=Path)
    test.add_argument("checkpoint", type=Path)
    test.add_argument("directory", type=Path)
    test.add_argument("output", type=Path)
    test.add_argument("--split", choices=("validation", "test"), required=True)
    args = parser.parse_args()
    if args.action == "prepare":
        prepare(args.manifest)
    elif args.action == "train":
        train(args.plans, args.hours)
    else:
        evaluate(args.manifest, args.checkpoint, args.directory, args.output, args.split)


if __name__ == "__main__":
    main()
