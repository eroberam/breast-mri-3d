# Validation and provenance

## Separate evaluation questions

- **Image geometry** uses decoded acquisition positions, image-plane axes, ordered source SOP instances and pixel spacing. Slice spacing comes from the scalar projection of successive positions, independently of nominal slice thickness. For a regular grid, a voxel volume is the absolute determinant of its linear affine in mm³. Surface area uses triangles transformed into patient space.
- **Model accuracy** compares predictions with actual annotated anatomy on the labelled cases. Tumor Dice includes misses, and HD95 is undefined when either surface is absent. Lesion matching and false-alarm counts require exhaustive targets or specifically documented negative studies. A primary tumor label alone does not adjudicate every other finding in the exam.
- **Exchange fidelity** compares the same native mask with its re-imported DICOM SEG on the ordered reference instances. Printable STLs share the 3MF assembly transform. The export checks the geometry and relative position of all parts on reload.

## Training data contract

Each prepared case contains `image.npy` in channel–slice–row–column order, `labels.npy` in the label vocabulary order, and a `case.json` with canonical patient identifier, dataset, class-wise provenance, grid and original reference identifiers. `native_labels/` preserves binary reference masks on the original postcontrast grid for measurement. Missing classes do not contribute to supervised loss. `confirmed_empty` requires explicit annotation of absence in both breasts. Report `derived` and `manual` references separately in manuscript analyses.

Assign every appearance of a patient to one split using the original provider identifiers; map aliases before combining datasets. Public MAMA-MIA, Duke, I-SPY and MAMA-MIAAI versions can contain the same people. The `test` split is reserved for the locked experiment. Dataset preparation, threshold choice, model selection and teacher calibration use development cases.

### Offline teacher cache

For `train --teachers`, provide `teacher.npz` next to eligible training cases. It contains `probabilities` and `trusted` arrays with the same shape and geometry as `labels.npy`. Evaluate teacher reliability on development annotations and use only supported labels. Its sibling `teacher.json` declares `patient`, `grid`, `channel_order`, `teacher_sha256`, `training_patients` using the canonical identifiers and a `calibration_record`. A model with unverifiable exposure can be a descriptive comparator. The training entry point rejects teachers with known exposure to reserved patients and avoids overwriting labelled voxels with teacher targets.

### Matched proposals

Make cross-fitting folds exclusively from the training cases of the master experiment. Mark each fold's held-out patients as validation in its own manifest, and train the supervised fold models. Give `match-proposals` a JSON document with `folds`, a list of objects containing a fold `manifest` and its `checkpoint`. The command checks the checkpoint's manifest hash and applies the fold model to documented negative patients that it excluded from training. It matches connected components to manual lesions using volume in mL, mean normalized enhancement and a shared `phase_protocol` group. Set the prespecified maximum volume ratio and enhancement gap on the command line.

The resulting ranking manifest contains `pairs`. Each entry records `positive` and `negative` case directories and their `center_zyx` on the training grid, as well as the volume, enhancement and protocol match. The negative endpoint carries the proposal checkpoint hash and canonical training patient IDs. To supply an independently adjudicated negative region, provide its `adjudication_record` and the same provenance. Select a ranking threshold on validation and pass it into inference and evaluation using `--rank-threshold`.

## Comparison and claim

Use the shared-decoder variant as an architectural ablation and report its parameter count and compute. Compare with official nnU-Net 3D on the same cases. Use matched compute for the supervised, distilled and paired-negative arms, counting the additional forwards in the ranking arm. Obtain independent tissue and lesion annotations and report volume errors in native geometry. Freeze the checkpoint and thresholds before requesting official blind test evaluation through the MAMA-MIA benchmark.

- [MAMA-MIA benchmark](https://mama-research.github.io/index.html)
- [DICOM image plane module](https://dicom.nema.org/medical/dicom/current/output/chtml/part03/sect_C.7.6.2.html)
- [DICOM segmentation storage](https://dicom.nema.org/medical/dicom/current/output/chtml/part03/sect_A.51.html)
- [3MF Core Specification](https://3mf.io/spec/core-v1-3-0/)
