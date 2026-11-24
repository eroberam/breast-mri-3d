"""Codabench Task 1 adapter for the MAMA-MIA restricted dataset interface."""

from __future__ import annotations

import json
import os
from pathlib import Path
import re
import sys

import numpy as np
import torch


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from breast_mri3d.acquire import checksum
from breast_mri3d.cascade import COARSE_SPACING_MM, lesion_probability
from breast_mri3d.geometry import lps_grid
from breast_mri3d.imaging import dce_channels, read_nifti, write_nifti
from breast_mri3d.inference import open_model
from breast_mri3d.research import keep_largest_components


class Model:
    def __init__(self, dataset):
        self.dataset = dataset
        self.device = torch.device(os.environ.get("BMRI_BENCHMARK_DEVICE", "cuda"))
        self.config = json.loads((ROOT / "operating-point.json").read_text(encoding="utf-8"))
        self.model, self.summary = self._checked_open("compact.pt", self.config["checkpoint_sha256"])
        self.locator = self.locator_summary = None
        if self.config["locator_sha256"]:
            self.locator, self.locator_summary = self._checked_open("locator.pt", self.config["locator_sha256"])
            if self.locator_summary["model_spacing_mm"] != COARSE_SPACING_MM:
                raise ValueError("The packaged locator has incompatible spacing.")

    def _checked_open(self, filename, digest):
        path = ROOT / filename
        if checksum(path) != digest:
            raise ValueError("A packaged checkpoint failed its checksum.")
        return open_model(path, self.device)

    def predict_segmentation(self, output_dir):
        output = Path(output_dir) / "pred_segmentations"
        output.mkdir(parents=True, exist_ok=True)
        for patient in self.dataset.get_patient_id_list():
            if not re.fullmatch(r"[A-Za-z0-9_-]+", patient):
                raise ValueError("The benchmark returned an invalid patient identifier.")
            paths = self.dataset.get_dce_mri_path_list(patient)
            if len(paths) < 2:
                raise ValueError("The benchmark case must include precontrast and first postcontrast.")
            pre, pre_grid = read_nifti(Path(paths[0]))
            post, native = read_nifti(Path(paths[1]))
            if pre_grid.shape != native.shape or not np.allclose(pre_grid.affine, native.affine, atol=.001, rtol=0):
                raise ValueError("Benchmark phases do not share the documented patient lattice.")
            grid = lps_grid(native, self.summary["model_spacing_mm"])
            channels = dce_channels(pre.astype(np.float32), post.astype(np.float32))
            image = np.stack([native.resample(x, grid, order=1) for x in channels]).astype(np.float16)
            probability = lesion_probability(
                self.model, image, grid, tuple(self.config.get("patch_zyx") or self.summary["patch_zyx"]), self.device,
                locator=self.locator, locator_patch=tuple(self.locator_summary["patch_zyx"]) if self.locator_summary else None,
                mirror_left_right=self.config.get("mirror_left_right", False))
            predicted = probability >= self.config["threshold"]
            if self.config.get("component_limit") is not None:
                predicted = keep_largest_components(predicted, self.config["component_limit"])
            original = grid.resample(predicted.astype(np.uint8), native, order=0)
            write_nifti(output / f"{patient}.nii.gz", original, native)
        return str(output)
