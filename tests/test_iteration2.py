import numpy as np
import pytest

from breast_mri3d.acquire import allocate_ispy, safe_relative
from breast_mri3d.experiment import paired_intervals
from breast_mri3d.geometry import Grid
from breast_mri3d.iteration2 import require_coverage
from breast_mri3d.mesh import surface


def test_expert_allocation_keeps_pilot_patients_in_training():
    rows, hashes = [], {}
    for index in range(10):
        patient = f"ISPY1_{index + 1000}"
        phases = [f"images/{patient}_DCE_000{i}_N3.nii.gz" for i in range(2)]
        mask = f"NIfTI-Files/masks_stv_manual/{patient}.nii.gz"
        rows.append({"patient_id": patient, "images_native": phases, "mask": mask})
        hashes.update(dict.fromkeys(phases + [mask], "a" * 32))
    cases = allocate_ispy(rows, hashes, {"ISPY_1_1000", "ISPY1_1002"}, 10, 19)
    assert sum(c["split"] == "validation" for c in cases) == 2
    assert all(c["split"] == "train" for c in cases if c["patient"] in {"ISPY1_1000", "ISPY1_1002"})
    with pytest.raises(ValueError, match="Unsafe"):
        safe_relative("images/../../foreign.nii")


def test_reference_fov_requires_patient_coordinates():
    grid = Grid((8, 8, 8), np.eye(4))
    values = np.zeros(grid.shape, np.uint8)
    values[2:6, 2:6, 2:6] = 1
    assert require_coverage(values, grid, grid) == 1
    shifted = np.eye(4)
    shifted[0, 3] = 75
    with pytest.raises(ValueError, match="field of view"):
        require_coverage(values, grid, Grid(grid.shape, shifted))


def test_envelope_probability_has_its_own_output_label():
    pytest.importorskip("torch")
    from breast_mri3d.inference import binary_masks
    from breast_mri3d.labels import ENVELOPE, LABELS

    probs = np.full((len(LABELS), 2, 2, 2), .35, dtype=np.float32)
    masks = binary_masks(probs, [ENVELOPE], .5)
    assert set(masks) == {ENVELOPE}
    assert masks[ENVELOPE].all()


def test_paired_intervals_keep_missing_surface_observations_explicit():
    shared = {"patient": "phantom", "cohort": "fixture", "reference": "manual", "grid": "native_dicom"}
    reference = {"records": [{**shared, "dice": .6, "hd95_mm": None, "abs_volume_error_ml": 1.2}]}
    proposed = {"records": [{**shared, "dice": .8, "hd95_mm": 2.0, "abs_volume_error_ml": .3}]}
    compared = paired_intervals(reference, proposed, replicates=40)["metrics"]
    assert compared["dice"]["ci95"] == pytest.approx([.2, .2])
    assert compared["abs_volume_error_ml"]["paired_median_difference"] == pytest.approx(-.9)
    assert "hd95_mm" not in compared
    proposed["records"][0]["grid"] = "model"
    with pytest.raises(ValueError, match="identical patients"):
        paired_intervals(reference, proposed)


def test_diagonal_voxel_contacts_produce_a_closed_print_surface():
    mask = np.array([
        [[1, 0, 1], [0, 1, 1], [1, 0, 0]],
        [[0, 0, 1], [1, 0, 0], [0, 0, 0]],
        [[1, 0, 0], [0, 1, 0], [0, 1, 1]],
    ], dtype=bool)
    original = mask.copy()
    geometry = Grid(mask.shape, np.diag([3., .7, 1.2, 1.]))
    mesh = surface(mask, geometry)
    assert mesh.is_volume
    assert mesh.metadata["isovalue"] == .50001
    assert np.array_equal(mask, original)
