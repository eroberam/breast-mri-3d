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

Volume measurements use the native mask and physical voxel size. The review and print surfaces record their marching-cubes levels. A diagonal contact that prevents a closed surface at level 0.5 triggers a retry at 0.50001 while preserving the binary mask.

## Reproduce the experiment

Use the [expert-contour recipe](docs/iteration2.md) to acquire the source cohorts, train the compact model and its time-matched reference, and score a frozen external evaluation. The [validation specification](docs/validation.md) describes partial labels and geometric checks. Provider conditions and source links are in the [source registry](configs/sources.json).

The [synthetic multipart assembly](examples/synthetic/assembly.3mf) can be opened directly in a 3MF viewer. The demo generates self-contained synthetic DICOM input and verifies a fresh export.

Source code is distributed under the [MIT license](LICENSE).
