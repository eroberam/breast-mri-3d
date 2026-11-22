# Breast MRI 3D

Independent research software for paired breast DCE-MRI, training of a compact 3D segmenter, and physically registered DICOM-to-print export. The public package, experiment configurations, tests and validation notes are organized separately from local studies, training artifacts and checkpoints.

The initial I-SPY1 experiment measures a supervised development run against **derived functional references**. Expert anatomy, distillation, matched-negative training and external evaluation have separate provenance and acceptance criteria. This software supports retrospective research and anatomical review.

The [expert-contour experiment](docs/iteration2.md) documents selective data acquisition, patient-level partitions, the time-matched comparison and the frozen external evaluation. A [synthetic 3MF example](examples/synthetic/assembly.3mf) accompanies the reproducible DICOM export demo.

## Install

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
python -m pip install 'torch==2.8.0' --index-url https://download.pytorch.org/whl/cu128
pytest -q
```

Use the PyTorch wheel matching the accelerator and OS. The geometry/export commands work without PyTorch. Windows uses `.venv\Scripts\python.exe` in place of activation if necessary.

## Export from DICOM

```bash
breast-mri-3d demo --out outputs/synthetic-demo
breast-mri-3d list-dicom /path/to/study
breast-mri-3d segment /path/to/study --pre 'PRE_SELECTOR' --series 'EARLY_POST_SELECTOR' \
  --checkpoint models/model.pt --threshold 0.5 --out outputs/research-case
```

Select phases explicitly and confirm they are suitable for coordinate regridding. When a classic MR series contains repeated spatial positions, `list-dicom` exposes distinct acquisition selectors with `#` suffixes. A selector is an acquisition identifier, not a classification of contrast timing. Each export refers to the exact ordered SOP instances, including when multiple phases have the same SeriesInstanceUID.

Alternatively, `export` accepts externally produced binary NIfTI masks with declared millimetre affine, original series UID, frame UID and ordered source-instance digest. Supply the segmentation algorithm type and the algorithm name and version for automatic or semiautomatic masks. Use `breast-mri-3d export --help` for its arguments. Import accepts regular classic single-frame MR. Paired DCE preparation regrids in the shared patient frame and assumes no significant patient motion between the selected acquisitions.

The bundle contains original-frame DICOM SEG, NIfTI masks in scanner coordinates, contours in patient LPS mm, a CSV/JSON measurement report, GLB in metres and registered STL meshes plus a multipart 3MF in mm. An atomic export completes only after DICOM SEG voxel equality and 3MF mesh/placement round-trip verification. A single common translation is applied to all printable pieces. The print partition prioritizes lesion, fibroglandular tissue and breast envelopes. The quantitative anatomy masks remain independent and may overlap. A valid file still requires preparation appropriate to the printer and material.

The manifest records each surface's marching-cubes isovalue. For diagonal contacts that make the level-0.5 surface open, the method retries at 0.50001 to obtain a closed volume while preserving the binary masks and their voxel-derived measurements.

## First supervised run

The two-phase I-SPY1 pilot links the original MR arrays to the published BreastDCEDL volumes by exact voxel equality after one permutation/flip. Both precontrast and early postcontrast must match uniquely. The archived label is placed on the verified native postcontrast grid. The weak functional label remains identified as derived throughout evaluation. See the [pilot results](docs/pilot.md).

```bash
MRI=/mnt/d/Backup-MyData/Databases_Clean/Medical/Breast/MRI
breast-mri-3d prepare-ispy1 --dicom "$MRI/TCIA_ISPY1/ispy1" \
  --release "$MRI/BreastDCEDL/extracted" --out data/ispy1-pilot
breast-mri-3d train data/ispy1-pilot/manifest.json --seed 19 --steps 480 \
  --width 12 --patch 24 96 96 --out models/ispy1-pilot.pt
breast-mri-3d evaluate data/ispy1-pilot/manifest.json models/ispy1-pilot.pt \
  --split validation --threshold 0.5 --out outputs/ispy1-validation.json
```

Repeat the training command with `--steps 3200` and a new output path to reproduce the longer, exploratory time-budget run.

Available phases in a reference dataset, checkpoint supervision and project evaluation are distinct. `segment` exposes the structures supervised by the checkpoint. The development pilot only learns the lesion channel. The software supports independent anatomical/vessel targets, offline teacher supervision on missing labels and documented training-only ranking pairs. See [research protocol](docs/validation.md) and the [source registry](configs/sources.json).

Use `prepare-case` for new DICOM-linked training targets. Use `prepare-volumes` for a released NIfTI cohort whose physical frame and phase order are documented. An explicit index selects a phase from a 4D NIfTI. A NIfTI-only cohort can contribute to research metrics on its native grid. DICOM SEG export requires the original MR instances.

## nnU-Net reference

The adapter exports the exact development cases and the fixed patient split for a lesion-only comparison. It excludes any reserved test cases. Configure paths for the official nnU-Net CLI before planning and training.

```bash
python -m pip install 'torchvision==0.23.0' --index-url https://download.pytorch.org/whl/cu128
python -m pip install 'nnunetv2==2.8.1'
export nnUNet_raw="$PWD/data/nnunet/raw"
export nnUNet_preprocessed="$PWD/data/nnunet/preprocessed"
export nnUNet_results="$PWD/runs/nnunet"
export nnUNet_compile=false
export nnUNet_n_proc_DA=2
export nnUNet_extTrainer="$PWD/configs/nnunet"
breast-mri-3d prepare-nnunet data/ispy1-pilot/manifest.json --raw "$nnUNet_raw" \
  --preprocessed "$nnUNet_preprocessed"
nnUNetv2_plan_and_preprocess -d 73 -gpu_memory_target 6 -c 3d_fullres -np 2 -npfp 2 --verify_dataset_integrity
nnUNetv2_train 73 3d_fullres 0 -tr PilotTrainer -device cuda
breast-mri-3d score-nnunet data/ispy1-pilot/manifest.json \
  "$nnUNet_results/Dataset073_BreastMRI3D/PilotTrainer__nnUNetPlans__3d_fullres/fold_0/validation" \
  --out outputs/ispy1-nnunet-validation.json
```

The pilot trainer keeps the default nnU-Net architecture and optimization with a fixed budget of 480 updates. Its patch size and batch size differ from the compact model. Use a prespecified compute-matched budget and a fully trained official reference for manuscript comparisons. [nnU-Net data format and split documentation](https://github.com/MIC-DKFZ/nnUNet/tree/master/documentation)

Git tracks source and configuration. Follow each dataset's access and redistribution terms when distributing scans, annotations or checkpoints.
