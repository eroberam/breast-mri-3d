# Expert-contour experiment

The compact network is trained from a random initialization. The comparison uses an official nnU-Net v2 3D full-resolution network and a shared, fixed patient partition. The expert development endpoints and the locked external cohort have separate manifests.

## Data preparation

Use a private data root outside the checkout. The optional data dependencies provide the Hugging Face, Synapse and Imaging Data Commons clients.

```bash
python -m pip install -e '.[data,dev]'
DATA=/path/to/private-data
python -m breast_mri3d.iteration2 "$DATA" acquire-ispy1 --limit 120 --seed 19
python -m breast_mri3d.iteration2 "$DATA" prepare-stv
python -m breast_mri3d.iteration2 "$DATA" prepare-envelopes --limit 60
python -m breast_mri3d.iteration2 "$DATA" prepare-mama --token-file /path/to/local-synapse-token
python -m breast_mri3d.iteration2 "$DATA" prepare-odelia --token-file /path/to/local-huggingface-token
python -m breast_mri3d.iteration2 "$DATA" prepare-ea1141 --limit 80
python -m breast_mri3d.iteration2 "$DATA" acquire-ambl --local-dce /path/to/BreastDCEDL_AMBL_dce.tar.gz
python -m breast_mri3d.iteration2 "$DATA" acquire-ambl-mr
```

- I-SPY1 uses baseline structural tumor volume (STV) references matched to the documented phase pair and checked against the TCIA checksums. Use `--legacy` with the local manifest of any earlier pilot to keep those patients in training. The original pilot manifest is local to the research laboratory, so an independent split can differ when that option is omitted. [TCIA expert contours](https://www.cancerimagingarchive.net/analysis-result/ispy1-tumor-seg-radiomics/)
- The MAMA-MIA subset includes Duke and I-SPY2 expert-reviewed primary tumor contours. MAMA-MIAAI contributes whole-breast envelopes after an exact companion-image grid check. The combined manifest aligns overlapping patients to their established split. [MAMA-MIA](https://doi.org/10.7303/syn60868042) · [MAMA-MIAAI](https://physionet.org/content/mama-miaai-herewegoagain/1.0.0/)
- The Duke 3D supplemental release contains whole-breast, fibroglandular and vascular annotations. Download the TCIA **3D Breast and FGT MRI Segmentations** NRRD package under `raw/Duke3D/supplement`. The adapter retrieves the documented original phase pairs, verifies that the Slicer segmentations occupy the same physical lattice as precontrast, and checks tissue coverage in early postcontrast. [Duke segmentation study](https://doi.org/10.1038/s41598-024-54048-2) · [TCIA supplemental downloads](https://www.cancerimagingarchive.net/collection/duke-breast-cancer-mri/)
- ODELIA uses lesion code zero on both sides in provider training and validation. It downloads only precontrast and the first documented postcontrast volume. EA1141 has separate patient splits. It requires baseline MRI BI-RADS 1, the recorded absence of lesions, no cancer during follow-up and explicitly identified DCE phases. [ODELIA methods](https://arxiv.org/abs/2506.00474) · [EA1141](https://www.cancerimagingarchive.net/collection/ea1141/)
- AMBL v3 is hashed before fitting. Its unitless NIfTI files establish shared array indices. Scanner-space assessment requires full voxel equality of both releases after a unique permutation and flip to a documented DICOM pair. Its lesion endpoint includes all reported lesions and the contours combine manual annotation with enhancement thresholding. [AMBL release](https://zenodo.org/records/17253223) · [AMBL methods](https://arxiv.org/html/2509.26440v2)

```bash
P="$DATA/prepared"
python -m breast_mri3d.iteration2 "$DATA" combine-development --out "$P/v02-lesion-dev.json" \
  --primary "$P/ISPY1-STV/manifest.json" "$P/MAMA-MIA/manifest.json"
python -m breast_mri3d.iteration2 "$DATA" combine-development --out "$P/v02-anatomy-dev.json" \
  --primary "$P/ISPY1-STV/manifest.json" "$P/MAMA-MIA/manifest.json" --auxiliary "$P/MAMA-MIAAI/manifest.json"
python -m breast_mri3d.iteration2 "$DATA" combine-development --out "$P/v02-calibration-dev.json" \
  --primary "$P/ISPY1-STV/manifest.json" "$P/MAMA-MIA/manifest.json" \
  --auxiliary "$P/ODELIA/manifest.json" "$P/EA1141/manifest.json"
```

## Fitting and ablations

To add Duke 3D anatomy, run `python -m breast_mri3d.duke "$DATA" --limit 100` once the NRRD files and the combined anatomy manifest are available. The adapter's `v02-anatomy-dev-with-duke.json` can replace the anatomy manifest in C1 and teacher calibration. It preserves the fixed tumor split and uses the Duke annotations for overlapping envelope patients.

The initial ceiling is 2,700 seconds per GPU arm after setup. The custom model stops after the first update that hits the budget. The nnU-Net trainer concludes its current validation hooks and writes the final checkpoint. Both log elapsed time, updates and peak allocated CUDA memory. Planning, threshold calibration and teacher inference are timed separately. The comparison uses one GPU process at a time, the same lesion-training cases and seed 19.

```bash
export nnUNet_raw="$DATA/nnunet/raw"
export nnUNet_preprocessed="$DATA/nnunet/preprocessed"
export nnUNet_results="$DATA/nnunet/results"
export nnUNet_extTrainer="$PWD/configs/nnunet"
export nnUNet_compile=false
export nnUNet_n_proc_DA=2
export OMP_NUM_THREADS=2
export BMRI_REFERENCE_BUDGET_SECONDS=2700
export BMRI_REFERENCE_SEED=19
RUN="$DATA/runs/v02"
mkdir -p "$RUN"
breast-mri-3d prepare-nnunet "$P/v02-lesion-dev.json" --raw "$nnUNet_raw" --preprocessed "$nnUNet_preprocessed" --dataset-id 75
nnUNetv2_plan_and_preprocess -d 75 -gpu_memory_target 5.3 -c 3d_fullres -np 2 -npfp 2 --verify_dataset_integrity
breast-mri-3d train "$P/v02-lesion-dev.json" --seed 19 --steps 1000000 --budget-seconds 2700 \
  --width 12 --patch 24 96 96 --out "$RUN/c0-supervised.pt"
nnUNetv2_train 75 3d_fullres 0 -tr V02BudgetTrainer -device cuda
breast-mri-3d train "$P/v02-anatomy-dev.json" --seed 19 --steps 1000000 --budget-seconds 2700 \
  --width 12 --patch 24 96 96 --out "$RUN/c1-anatomy.pt"
python -m breast_mri3d.experiment envelope-teacher "$P/v02-anatomy-dev.json" "$RUN/c1-anatomy.pt" \
  --out "$RUN/envelope-calibration.json"
```

C1 adds manual whole-breast supervision with a partial-label loss. If the teacher eligibility report passes, C2 repeats C1 with the same initialization protocol and adds `--teachers`. The locally trained teacher's envelope probabilities are temperature-calibrated on manual validation references. Its cache records the training partition, checkpoint digest, confidence boundaries and physical grid. Published teachers require an independently verified training-patient list and structure-specific calibration before use.

For an exploratory C2 evaluation after the primary model lock, use the anatomical partition employed for C1. Setting `--minimum-dice 0` evaluates the effect of confidence-filtered targets with the usual temperature calibration and probability tails of 0.1 and 0.9. Use development validation for this ablation.

```bash
ANATOMY_MANIFEST="$P/v02-anatomy-dev-with-duke.json"
python -m breast_mri3d.experiment envelope-teacher "$ANATOMY_MANIFEST" "$RUN/c1-anatomy.pt" \
  --minimum-dice 0 --out "$RUN/c2-exploratory-envelope-calibration.json"
breast-mri-3d train "$ANATOMY_MANIFEST" --seed 19 --steps 1000000 --budget-seconds 2700 \
  --width 12 --patch 24 96 96 --teachers --out "$RUN/c2-exploratory.pt"
```

C3 depends on verified cross-fitted proposal pairs. Generate negatives with the other two training folds, then call `match-proposals` with `--max-volume-ratio 2 --max-enhancement-gap 0.5`. Use the same documented protocol on both sides of each pair. Train C3 by adding `--pairs` to C2 when the candidate ledger contains approved pairs. See [candidate and teacher rules](validation.md).

## Validation, lock and external scoring

Calibrate each compact checkpoint on validation with the calibration manifest. Select the arm by the predeclared median positive Dice, with mean negative false alarms as the tie-breaker. Preserve the validation reports. Freeze both checkpoints and the selected threshold before preparing AMBL.

```bash
python -m breast_mri3d.experiment calibrate "$P/v02-calibration-dev.json" "$RUN/c0-supervised.pt" \
  --out "$RUN/c0-threshold.json"
REF="$nnUNet_results/Dataset075_BreastMRI3D/V02BudgetTrainer__nnUNetPlans__3d_fullres/fold_0"
python -m breast_mri3d.experiment lock-external "$DATA" "$RUN/c0-supervised.pt" \
  "$RUN/c0-threshold.json" "$REF/checkpoint_final.pth" --out "$RUN/external-lock.json"
python -m breast_mri3d.iteration2 "$DATA" prepare-ambl --model-lock "$RUN/external-lock.json"
breast-mri-3d prepare-nnunet-prediction "$P/AMBL/manifest.json" --split test --out "$RUN/ambl-input"
nnUNetv2_predict -i "$RUN/ambl-input" -o "$RUN/nnunet-ambl" -d 75 -tr V02BudgetTrainer \
  -c 3d_fullres -f 0 -chk checkpoint_final.pth -npp 2 -nps 2
breast-mri-3d score-nnunet "$P/AMBL/manifest.json" "$RUN/nnunet-ambl" --split test --prediction-input \
  --out "$RUN/nnunet-ambl.json"
```

If multiprocessing is unavailable, use the official sequential predictor with the same checkpoint, input and output paths. Run this command before scoring.

```bash
python - "$RUN/ambl-input" "$RUN/nnunet-ambl" "$REF/checkpoint_final.pth" <<'PY'
from pathlib import Path
import sys
import torch
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

inputs, outputs, checkpoint = map(Path, sys.argv[1:])
outputs.mkdir(parents=True, exist_ok=True)
predictor = nnUNetPredictor(tile_step_size=0.5, use_gaussian=True, use_mirroring=True,
                            perform_everything_on_device=True, device=torch.device("cuda"))
predictor.initialize_from_trained_model_folder(str(checkpoint.parent.parent),
                                               use_folds=(0,), checkpoint_name=checkpoint.name)
predictor.predict_from_files_sequential(str(inputs), str(outputs), overwrite=False)
PY
```

Run `breast-mri-3d evaluate` on the same manifest and locked compact checkpoint with `--split test`, the locked `--threshold` and, if selected, `--rank-threshold`. Then use `python -m breast_mri3d.experiment compare REFERENCE_JSON PROPOSED_JSON --out PAIRED_JSON`. The report pairs patients before bootstrapping. Surface distance is reported only when both rows have a measurable HD95. Small lesions use a 10 mm equivalent-sphere diameter. Primary-tumor references count each annotated tumor once, including when its mask is discontinuous. AMBL lesion matching retains the release's individual identifiers on the verified DICOM grid. Negative false alarms are counted only where absence or exhaustive annotation is documented.

Report official benchmark performance separately under its frozen submission protocol. [MAMA-MIA challenge](https://mama-research.github.io/index.html)
