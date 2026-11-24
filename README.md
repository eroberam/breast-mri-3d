# Breast MRI 3D

Research software for compact 3D segmentation of paired breast DCE-MRI and export in verified patient coordinates.

## Install and verify

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
pytest -q
breast-mri-3d demo --out outputs/synthetic
```

The geometry and export commands run without PyTorch. For training and inference, install a PyTorch build compatible with your operating system and accelerator. On Windows, use the commands from `.venv\Scripts` or an activated environment.

## Use original DICOM geometry

Use `breast-mri-3d list-dicom /path/to/study` to identify acquisition selectors. Confirm precontrast and early postcontrast from the acquisition documentation before running the segmenter. Multiple phases in one MR series can have separate selectors.

```bash
breast-mri-3d segment /path/to/study --pre PRE_SELECTOR --series POST_SELECTOR \
  --checkpoint model.pt --threshold 0.5 --out outputs/case
```

Use `breast-mri-3d export --help` to supply binary masks with documented scanner geometry and exact DICOM source references. The export produces DICOM SEG, NIfTI masks, quantitative CSV and JSON, contours in patient LPS millimetres, a review GLB, aligned STL parts and a 3MF assembly. The manifest records verification of DICOM SEG voxel equality and 3MF placement on reopening.

Volume measurements use the native mask and physical voxel size. The printable assembly partitions overlaps in order of lesion, vessels, fibroglandular tissue and breast envelope. The review and print surfaces record their marching-cubes levels. A diagonal contact that prevents a closed surface at level 0.5 triggers a retry at 0.50001 while preserving the binary mask.

## Reproduce the experiment

The v0.3 recipe compares subtraction and paired DCE inputs on a common patient grid, with a low-resolution locator and an optional high-resolution refiner. To fit the models and run the locked comparisons, use the [v0.3 experiment](docs/iteration3.md). Prior cohort preparation and partial-label supervision are documented separately. [v0.2 experiment](docs/iteration2.md) · [Validation](docs/validation.md) · [Data sources](configs/sources.json)

The adjustment-driven revision uses matched updates to test distillation and larger physical windows. Its procedures include the held-out selection rule, the inference settings and a local benchmark archive. [Revised experiment](docs/iteration3-replan.md)

The [synthetic multipart assembly](examples/synthetic/assembly.3mf) can be opened directly in a 3MF viewer. The demo generates self-contained synthetic DICOM input and verifies a fresh export.

Source code is distributed under the [MIT license](LICENSE).
