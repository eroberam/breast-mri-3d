# DICOM-to-print and training pilot

This development experiment checked the pipeline with 15 I-SPY1 studies and the derived functional masks distributed by BreastDCEDL. The original and released intensities matched exactly after a unique axis permutation and flip for both DCE phases in each case. A fixed hash-based patient split assigned 12 studies to training and three to validation. Training used the lesion channel only.

| Measure | Observed |
| --- | ---: |
| Parameters | 1,061,526 |
| Training updates | 480 |
| Training time in the local run | 24.3 s |
| Peak CUDA memory allocated by PyTorch | 351.6 MiB |
| Validation median Dice on native grid | 0.6810 |
| Validation median absolute volume error | 21.1835 mL |
| Validation median HD95 | 79.10 mm |

The model used randomly initialized weights, mixed-precision AdamW, 24 × 96 × 96 training patches and a 0.5 probability threshold for evaluation. The environment was Python 3.12 on Windows with PyTorch 2.8.0+cu128 and an RTX 3060 Ti. The memory number measures PyTorch allocations during training. It excludes CUDA context and application overhead.

The synthetic oblique reference verified DICOM SEG voxel equality on the original instances and multipart 3MF placement and units. The print assembly verifies vertex coordinates per axis to 0.002 mm and verifies triangle connectivity. A second run took a held-out DCE study from original MR through the checkpoint, exported a lesion bundle and passed both internal round-trip checks. Its patient-level evidence stays with the local study outputs.

## nnU-Net on the same 12/3 split

The plain nnU-Net 3D reference used the same input volumes and 480 optimizer updates. It used its planned 56 × 160 × 160 patches, batch size two and standard augmentation. Its two training epochs took 165.8 s. Scored on the same three native-grid validation masks, it obtained median Dice 0.8541, median absolute volume error 3.2397 mL and median HD95 4.46 mm. Batch size, augmentation, patch size and training time differed between architectures. This comparison identifies a stronger reference for the pilot and leaves the planned comparison on expert labels open.

The compact model was retrained from the same initialization for 3,200 updates after the first comparison, taking 155.9 s. Its median Dice reached 0.8934, with 7.7466 mL median absolute volume error and 6.09 mm median HD95. The longer run improved the compact model on all three reported metrics. nnU-Net retained lower volume error and HD95. All three estimates use the same three development patients and derived labels. The time-budget run is exploratory and its analysis should be replicated on independent patients.

The distillation cache, cross-fitted proposal mining, matching by physical volume and enhancement, and the joint training entry point passed a synthetic integration test. Real teacher training and proposal ranking require the documented development data and teacher exposure records.

## Acquisition state

Two supplementary files were retrieved. The AMBL label archive passed its published MD5 check and remains reserved. One PhysioNet manual envelope was retained pending validation of its physical correspondence and contributes zero training labels. Synapse returned HTTP 403 for the MAMA-MIA files and Hugging Face returned HTTP 401 for ODELIA. Follow the providers' access and attribution steps using your own account. [MAMA-MIA download instructions](https://github.com/LidiaGarrucho/MAMA-MIA#downloading-the-dataset) · [ODELIA data-use terms](https://huggingface.co/datasets/ODELIA-AI/ODELIA-Challenge-2025) · [AMBL labels](https://zenodo.org/records/17253223)

## Manuscript analysis to perform

The pilot establishes pipeline feasibility. Its three cases and derived labels leave accuracy against expert contours and failure rates unresolved. The clinical hypothesis is that anatomy supervision and matched hard negatives recover small lesions at a prespecified false-alarm operating point. Use patient-level paired estimates and confidence intervals, independent annotations and the same matching policy for all model arms. Include the full operating-point curve on exhaustive or explicitly negative exams and stratify by lesion size and external cohort. Report native volume error and export fidelity as separate measurements.

Anatomy training, external teachers, cross-fitted matched negatives and the blind challenge require their respective datasets. The comparison must control training budget and partition. Report access terms and the provenance of any distributed weights. [MAMA-MIA dataset and baseline](https://github.com/LidiaGarrucho/MAMA-MIA) · [Official blind benchmark](https://mama-research.github.io/index.html) · [Breast lesion detection benchmark](https://proceedings.mlr.press/v301/geissler26a.html)
