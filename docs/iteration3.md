# Patient-oriented primary-tumor segmentation

This experiment fits models from scratch for primary-tumor segmentation in breast DCE-MRI. It compares a single time-matched nnU-Net v2 ResEncM with a compact residual model, then tests the selected candidate on reserved collections in the source physical coordinates. The framework and synthetic examples are distributed under MIT.

## Data and historical replay

Set `PREVIOUS` and `DATA` to private local directories. The earlier experiment prepares `PREVIOUS/prepared/v02-lesion-dev.json` and records patient exposure across related collections. The exact 45-minute replay uses the 121/30 split, checkpoint metadata, operating point, and reference predictions produced by the laboratory's v0.2 procedure. [Previous experiment](iteration2.md)

```bash
python -m pip install -e '.[data,dev,train,reference]'
python -m breast_mri3d.mama_v03 "$PREVIOUS" "$DATA" --phases --token-file /path/to/synapse-token
python -m breast_mri3d.iteration2 "$DATA" prepare-odelia --limit 60 \
  --unseen-test-of "$PREVIOUS/acquisition/odelia-negative-selection.json" \
  --token-file /path/to/huggingface-token
python scripts/v03_pilot.py "$PREVIOUS" "$DATA"
```

The MAMA-MIA adapter downloads both documented DCE phases and a versioned expert primary-tumor mask. It validates geometry before allocating 320 fresh patients to internal test and 120 to adjustment with seed 19. Allocation is stratified by source and native contour volume. The original 30 controls and previously exposed Duke cases are excluded. Eligible Duke patients form a separate test cohort. ODELIA reserves new exams with absence of lesions documented on both sides. [MAMA-MIA release](https://doi.org/10.7303/syn60868042) · [ODELIA release](https://huggingface.co/datasets/ODELIA-AI/ODELIA-Challenge-2025)

The historical control preserves the exact voxel sampling through a permutation and flips. The larger experiment uses an orthogonal 1.5 mm grid with increasing superior, posterior and left coordinates along the array axes. A small lesion that disappears on downsampling receives one model-grid training voxel. Scoring uses its original native-grid contour.

## Sequential CUDA experiment

```bash
python -m breast_mri3d.study_v03 "$DATA" --trainer-dir configs/nnunet
```

For independent replication with a locally prepared previous cohort, use `--skip-historical-replay` to start from the verified partitions without the laboratory's historical checkpoints.

Run one GPU fit at a time. The program waits for the split receipts and replay, then trains a single-fold ResEncM on the subtraction, with the official augmentation and mirrored inference. The reference uses a 5.3 GB planner target and can replan at 4.2 GB if training exhausts GPU memory. The compact arms share a patch of 48 × 128 × 128 voxels, with 32 × 112 × 112 as the common fallback. They compare three paired DCE channels with suppression of precontrast and postcontrast in the input tensor. The cascade uses a 2.5 mm locator and a 1.5 mm refiner with a distance-weighted loss in millimetres. The detector supplies the evaluation regions.

The initial limit is six hours per arm. Each stage of the cascade receives three. The leading compact arm and the reference continue to a cumulative twelve hours. Adjustment scores select the checkpoint. Compact threshold selection maximizes native mean Dice on the 120 adjustment patients over 0.30, 0.45, 0.60, 0.75 and 0.90. The reference uses its binary argmax. Candidate weights, operating points and input manifests are hashed before held-out scoring.

Native scoring retains zero Dice when a tumor is missed. It reports paired mean Dice and patient bootstrap confidence intervals, median Dice, HD95 on cases with two surfaces, and absolute volume error. When nnU-Net omits the NIfTI unit, the scorer transfers millimetres from the original model input after matching the shape and affine. Small-lesion sensitivity uses one-to-one matching at IoU ≥ 0.10 and diameter-equivalent volume ≤ 10 mm. Negative-exam false alarms use bilateral ODELIA controls. Local scores and the official blind Task 1 leaderboard estimate different endpoints and populations. [Official benchmark methods and results](https://doi.org/10.1109/TMI.2026.3733343) · [Long-term benchmark](https://www.codabench.org/competitions/7425/)

## DICOM export

Confirm the phase selectors using the acquisition documents. Threshold and patch size follow the selected checkpoint recipe. To run the cascade, also pass the matching low-resolution checkpoint.

```bash
breast-mri-3d segment /path/to/dicom --pre PRE_SELECTOR --series POST_SELECTOR \
  --checkpoint compact.pt --threshold 0.60 --out /path/to/review
```

The synthetic demo verifies DICOM SEG and the reopened 3MF. The source volume, measurement, and print transform remain in their documented physical units. [Geometry validation](validation.md)
