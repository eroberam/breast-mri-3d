import json

import numpy as np
import pytest

from breast_mri3d.cascade import proposal_boxes
from breast_mri3d.cohorts_v03 import assign, patient_key
from breast_mri3d.dicom import select
from breast_mri3d.export import export_case
from breast_mri3d.geometry import Grid, lps_grid, reorient_lps
from breast_mri3d.imaging import read_nifti
from breast_mri3d.phantom import write_phantom
from breast_mri3d.validation_v03 import paired_patients


def test_reoriented_phantom_exports_identical_source_mask_and_volume(tmp_path):
    uid, masks = write_phantom(tmp_path / "source")
    mr = select(tmp_path / "source", uid)
    permute = np.eye(4)
    permute[:3, :3] = np.eye(3)[:, [2, 1, 0]]
    permute[2, 0] = -1
    permute[2, 3] = mr.grid.shape[2] - 1
    reordered = {key: mask.transpose(2, 1, 0)[::-1] for key, mask in masks.items()}
    source = Grid(next(iter(reordered.values())).shape, mr.grid.affine @ permute)
    pairs = {key: reorient_lps(mask, source) for key, mask in reordered.items()}
    grid = pairs["lesion"][1]
    assert grid.voxel_mm3 == pytest.approx(mr.grid.voxel_mm3)
    assert np.array_equal(pairs["lesion"][0], masks["lesion"])
    assert np.allclose(grid.affine, mr.grid.affine)
    out = export_case(tmp_path / "review", mr, {key: pair[0] for key, pair in pairs.items()}, mask_grid=grid,
                      origin={"type": "synthetic_phantom", "series_instance_uid": mr.uid,
                              "frame_of_reference_uid": mr.frame_uid, "source_instances_sha256": mr.source_instances_sha256})
    reference, _ = read_nifti(out / "lesion.nii.gz")
    receipt = json.loads((out / "manifest.json").read_text())
    assert np.array_equal(reference, masks["lesion"])
    assert receipt["dicom_seg_round_trip_verified"] and receipt["assembly_3mf_verified"]


def test_physical_proposal_margins_and_empty_case():
    source = Grid((76, 87, 90), np.diag([2., 2., 2., 1.]))
    fine, coarse = lps_grid(source, 1.5), lps_grid(source, 2.5)
    prediction = np.zeros(coarse.shape, dtype=np.float32)
    assert not proposal_boxes(prediction, coarse, fine)
    index = np.array([30, 24, 21])
    prediction[tuple(index)] = .9
    (box,) = proposal_boxes(prediction, coarse, fine)
    center = coarse.points(index)
    start = fine.points([s.start for s in box])
    stop = fine.points([s.stop - 1 for s in box])
    assert np.all(center - start >= 20.)
    assert np.all(stop - center >= 20.)


def test_component_retention_preserves_the_two_dominant_candidates():
    from breast_mri3d.research import keep_largest_components

    model_mask = np.zeros((18, 22, 19), dtype=bool)
    model_mask[1:5, 1:5, 1:5] = True
    model_mask[10:13, 10:13, 10:13] = True
    model_mask[16, 19, 17] = True
    assert keep_largest_components(model_mask, 1).sum() == 64
    retained = keep_largest_components(model_mask, 2)
    assert retained.sum() == 91
    assert not retained[16, 19, 17]


def test_partitions_exclude_cross_release_aliases_and_historical_controls():
    rows = [{"patient": f"ISPY1_{i}", "collection": "ISPY1", "volume_ml": i / 3}
            for i in range(1, 21)]
    rows += [{"patient": f"DUKE_{i:03d}", "collection": "DUKE", "volume_ml": 3.} for i in range(1, 5)]
    prior = {patient_key("Breast_MRI_002"), patient_key("ACRIN-6657-005")}
    historical = {patient_key("ISPY1_3")}
    partitions = assign(rows, prior, historical, internal=6, adjustment=4)
    assert all(row["patient_key"] not in prior | historical for row in partitions["internal"])
    assert len(partitions["internal"]) == 6
    assert len([row for row in partitions["development"] if row["split"] == "validation"]) == 4
    assert all(row["collection"] != "DUKE" for row in partitions["development"])
    assert {row["patient_key"] for row in partitions["duke"]} == {"DUKE_1", "DUKE_3", "DUKE_4"}
    assert assign(rows[::-1], prior, historical, internal=6, adjustment=4) == partitions


def test_paired_means_keep_misses_and_require_matching_patients():
    def scored(patient, dice, surface):
        return {"patient": patient, "cohort": "ISPY1", "reference": "manual", "grid": "native_nifti",
                "dice": dice, "hd95_mm": surface, "abs_volume_error_ml": 1.}

    before = {"records": [scored("ISPY1_001", .6, 8.), scored("ISPY1_002", .8, 1.)]}
    after = {"records": [scored("ACRIN-6657-001", .9, 3.), scored("ACRIN-6657-002", 0., None)]}
    comparison = paired_patients(before, after, replicates=256)["all"]
    assert comparison["proposed_minus_reference"]["dice"]["mean_difference"] == pytest.approx(-.25)
    assert comparison["proposed_minus_reference"]["hd95_mm"]["n_complete_pairs"] == 1
    with pytest.raises(ValueError, match="same frozen patients"):
        paired_patients(before, {"records": after["records"][:1]})


def test_reference_uses_only_the_subtraction_channel(tmp_path):
    from breast_mri3d.baseline import prepare_nnunet
    from breast_mri3d.prepare import _write_case

    grid = lps_grid(Grid((8, 8, 8), np.eye(4)))
    channels = np.stack([np.full(grid.shape, value, dtype=np.float32) for value in (1, 2, 3)])
    refs = {"lesion": (np.ones(grid.shape, dtype=np.uint8), grid, "manual")}
    rows = []
    for value, split in enumerate(("train", "validation")):
        patient = f"p{value}"
        _write_case(tmp_path / patient, channels, grid, refs, {"type": "synthetic"}, patient, "phantom", None,
                    target_grid=grid)
        rows.append({"patient": patient, "cohort": "phantom", "split": split, "path": patient})
    manifest = tmp_path / "development.json"
    manifest.write_text(json.dumps({"cases": rows}))
    dataset = prepare_nnunet(manifest, tmp_path / "raw", tmp_path / "pre", 76, channel_indices=(2,), ending=".nii")
    assert json.loads((dataset / "dataset.json").read_text())["channel_names"] == {"0": "enhancement"}
    assert len(list((dataset / "imagesTr").glob("*.nii"))) == 2
    sampled, actual = read_nifti(dataset / "imagesTr/BreastMRI_0001_0000.nii")
    assert np.all(sampled == 3)
    assert np.array_equal(actual.affine, grid.affine)


def test_reference_inherits_missing_units_only_from_an_identical_source_grid(tmp_path):
    import nibabel as nib

    path = tmp_path / "reference.nii"
    nib.save(nib.Nifti1Image(np.ones((5, 4, 3), dtype=np.uint8), np.eye(4)), path)
    with pytest.raises(ValueError, match="millimetres"):
        read_nifti(path)
    affine = np.diag([-1., -1., 1., 1.])
    affine[:3, :3] = affine[:3, (2, 1, 0)]
    source = Grid((3, 4, 5), affine)
    data, back = read_nifti(path, source_grid=source)
    assert data.shape == source.shape and np.array_equal(back.affine, source.affine)
    shifted = affine.copy()
    shifted[0, 3] = 2
    with pytest.raises(ValueError, match="verified source"):
        read_nifti(path, source_grid=Grid(source.shape, shifted))


def test_resume_restores_optimizer_and_foreground_sampling(tmp_path):
    torch = pytest.importorskip("torch")
    from breast_mri3d.acquire import checksum, write_json
    from breast_mri3d.prepare import _write_case
    from breast_mri3d.train import train

    grid = Grid((8, 8, 8), np.eye(4) * [1.5, 1.5, 1.5, 1.])
    image = np.ones((3, *grid.shape), dtype=np.float32)
    lesion = np.zeros(grid.shape, dtype=np.uint8)
    lesion[2:5, 2:5, 2:5] = 1
    _write_case(tmp_path / "case", image, grid, {"lesion": (lesion, grid, "manual")}, {"type": "synthetic"},
                "p1", "phantom", None, target_grid=grid)
    manifest = tmp_path / "development.json"
    manifest.write_text(json.dumps({"cases": [{"patient": "p1", "cohort": "phantom", "path": "case",
                                               "split": "train"}]}))
    recipe = {"width": 4, "patch": (8, 8, 8), "seed": 19, "device_name": "cpu", "input_mode": "subtraction",
              "model_spacing_mm": 1.5, "boundary_weight": .15, "resumable": True}
    train(manifest, tmp_path / "first.pt", steps=1, **recipe)
    result = train(manifest, tmp_path / "continued.pt", steps=2, resume_from=tmp_path / "first.resume.pt", **recipe)
    train(manifest, tmp_path / "continuous.pt", steps=2, **recipe)
    a, b = (torch.load(tmp_path / name, weights_only=True)["state_dict"]
            for name in ("continued.pt", "continuous.pt"))
    assert result["updates"] == 2
    assert all(torch.equal(a[name], b[name]) for name in a)
    probabilities = tmp_path / "teacher.npy"
    np.save(probabilities, np.where(lesion, .75, .01).astype(np.float16))
    teacher = tmp_path / "teacher.json"
    write_json(teacher, {"development_sha256": checksum(manifest), "training": [
        {"patient": "p1", "probability": probabilities.name, "sha256": checksum(probabilities)}]})
    distilled = train(manifest, tmp_path / "distilled.pt", steps=1, lesion_teacher=teacher, **recipe)
    assert distilled["distilled_updates"] == 1
    assert distilled["lesion_teacher_sha256"] == checksum(teacher)
    initial = tmp_path / "first.pt"
    warm = train(manifest, tmp_path / "warm.pt", steps=1, initial_checkpoint=initial, learning_rate=.0001, **recipe)
    assert warm["updates"] == 1 and warm["initial_checkpoint_sha256"] == checksum(initial)
    assert warm["learning_rate"] == .0001
