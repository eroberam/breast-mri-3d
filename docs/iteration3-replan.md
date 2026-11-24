# Matched experiments for detection and contour agreement

Use the v0.3 preparation recipe to obtain the versioned source material and frozen manifests. Store the private data root outside this checkout. The revised study uses the same adjustment and held-out cohorts. [Data and partitions](iteration3.md)

## Initial checkpoints

The commands assume that the installed CUDA PyTorch can see the accelerator. Use the project's equivalent environment variables and commands on Windows.

```bash
P="$DATA/prepared"
R="$DATA/runs/v03"
mkdir -p "$R"
export nnUNet_raw="$DATA/nnunet/raw"
export nnUNet_preprocessed="$DATA/nnunet/preprocessed"
export nnUNet_results="$DATA/nnunet/results"
export nnUNet_extTrainer="$PWD/configs/nnunet"
export nnUNet_compile=false
export nnUNet_n_proc_DA=2
export OMP_NUM_THREADS=2
python -m breast_mri3d.reference_v03 prepare "$P/development.json"
nnUNetv2_plan_and_preprocess -d 76 -pl nnUNetPlannerResEncM -gpu_memory_target 5.3 \
  -overwrite_plans_name BreastMRI3DResEncM -c 3d_fullres -np 2 -npfp 2
python -m breast_mri3d.reference_v03 train BreastMRI3DResEncM --hours 6
REF="$nnUNet_results/Dataset076_BreastMRI3D/V03BudgetTrainer__BreastMRI3DResEncM__3d_fullres/fold_0/checkpoint_6h.pth"
python -m breast_mri3d.reference_v03 evaluate "$P/development.json" "$REF" \
  "$R/reference-6h-adjustment-nnunet" "$R/reference-6h-adjustment-reference.json" --split validation
```

Run the compact arms sequentially on the same device. The early DCE fit reproduces the recorded update count from the paused run. The regular six-hour training report records its actual update count.

```bash
python -m breast_mri3d.cli train "$P/development.json" --out "$R/compact-subtraction-6h.pt" \
  --steps 2000000 --budget-seconds 21600 --patch 48 128 128 --width 12 --seed 19 \
  --input-mode subtraction --model-spacing-mm 1.5 --resumable
python -m breast_mri3d.cli train "$P/development.json" --out "$R/compact-dce-6h.pt" \
  --steps 23205 --patch 48 128 128 --width 12 --seed 19 --input-mode dce --model-spacing-mm 1.5 --resumable
printf '%s\n' '{"patch_zyx": [48, 128, 128]}' > "$R/patch-selection.json"
printf '%s\n' '{"plans": "BreastMRI3DResEncM"}' > "$R/reference-plans.json"
```

When resuming an existing experiment, use its original patch and reference receipts.

## Adjustment and locked tests

```bash
python scripts/replan_adjustment.py "$DATA"
python scripts/replan_fit.py "$DATA" --trainer-dir configs/nnunet --intervention distillation --updates 15000
python scripts/replan_context.py "$DATA" --trainer-dir configs/nnunet
python scripts/finalize_replan.py "$DATA" --trainer-dir configs/nnunet
```

The first command distinguishes omitted and added volume, evaluates left–right mirroring and checks the effect of retaining one component. The second compares the supervised continuation with a teacher-trained continuation at the same update count. The contextual experiment tests a larger inference window on the supervised weights and a separate fit with that window. Retaining up to two components is an adjustment option in both comparisons.

The final command selects a checkpoint and a policy from adjustment, verifies the patient manifest hashes, and scores the locked internal, Duke and ODELIA tests. It also creates a private Task 1 archive containing inference code and the selected weights. The operating point records the patch, mirroring, threshold and component limit. Run the benchmark archive under the organisers' current protocol. [MAMA-MIA benchmark](https://www.codabench.org/competitions/7425/)

For native DICOM export, pass the frozen settings to `breast-mri-3d segment`. The manifest records the physical source series, fitted weights and inference parameters. [DICOM exchange checks](validation.md)
