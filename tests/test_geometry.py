from copy import deepcopy
import json

import numpy as np
import pytest

from breast_mri3d.dicom import assemble, select, split_repeated_positions
from breast_mri3d.export import export_case
from breast_mri3d.geometry import Grid, model_grid
from breast_mri3d.imaging import read_nifti, write_nifti
from breast_mri3d.phantom import demo, write_phantom
from breast_mri3d.prepare import exact_reorientation


def test_synthetic_round_trip(tmp_path):
    path = demo(tmp_path / "demo")
    manifest = json.loads((path / "manifest.json").read_text())
    assert manifest["dicom_seg_round_trip_verified"]
    assert manifest["assembly_3mf_verified"]
    assert (path / "review.glb").is_file()
    assert len(manifest["print_parts"]) == 4
    mask, grid = read_nifti(path / "lesion.nii.gz")
    assert grid.voxel_mm3 == pytest.approx(.8 * .65 * 2.1)
    assert grid.pixel_mm2 == pytest.approx(.8 * .65)
    record = next(row for row in json.loads((path / "measurements.json").read_text()) if row["structure"] == "lesion")
    assert record["volume_ml"] == pytest.approx(mask.sum() * grid.voxel_mm3 / 1000)


def test_repeated_series_and_phase_digest(tmp_path):
    uid, masks = write_phantom(tmp_path / "mr")
    series = select(tmp_path / "mr", uid)
    second = []
    from pydicom.uid import generate_uid

    for path, ds in zip(series.files, series.headers):
        item = deepcopy(ds)
        item.SOPInstanceUID = generate_uid()
        item.InstanceNumber = int(ds.InstanceNumber) + len(series.headers)
        second.append((path, item))
    acquisitions = split_repeated_positions(second + list(zip(series.files, series.headers)))
    assert len(acquisitions) == 2
    first, other = [assemble(group) for group in acquisitions]
    assert first.uid == other.uid
    assert first.source_instances_sha256 != other.source_instances_sha256
    with pytest.raises(ValueError, match="phase instances"):
        export_case(tmp_path / "wrong", other, masks, mask_grid=first.grid, origin={
            "frame_of_reference_uid": first.frame_uid, "series_instance_uid": first.uid,
            "source_instances_sha256": first.source_instances_sha256,
        })
    assert not (tmp_path / "wrong").exists()


def test_mask_refuses_nifti_with_unknown_units(tmp_path):
    import nibabel as nib

    path = tmp_path / "unscaled.nii.gz"
    nib.save(nib.Nifti1Image(np.ones((3, 4, 5), dtype=np.uint8), np.eye(4)), path)
    with pytest.raises(ValueError, match="millimetres"):
        read_nifti(path)
    grid = Grid((3, 4, 5), np.diag([2, 3, 4, 1]))
    array = np.arange(60, dtype=np.int16).reshape(grid.shape)
    write_nifti(path, array, grid)
    back, coordinates = read_nifti(path)
    assert np.array_equal(array, back)
    assert np.array_equal(grid.affine, coordinates.affine)


def test_index_conversion_requires_all_pixels_to_match():
    left = np.arange(840).reshape(6, 10, 14)
    right = left.transpose(2, 0, 1)[::-1]
    axes, flips = exact_reorientation(left, right)
    assert np.array_equal(np.flip(left.transpose(axes), flips), right)
    with pytest.raises(ValueError, match="unique exact match"):
        exact_reorientation(left, right + 1)
    grid = Grid(left.shape, np.diag([2, .5, .5, 1]))
    planned = model_grid(grid)
    assert np.allclose(grid.points(np.array(grid.shape) - 1), planned.points(np.array(planned.shape) - 1))


def test_prepared_nifti_verifies_phase_and_preserves_native_reference(tmp_path):
    from breast_mri3d.prepare import prepare_volumes

    grid = Grid((8, 16, 16), np.diag([2, .5, .5, 1]))
    pre = tmp_path / "early.nii.gz"
    post = tmp_path / "late.nii.gz"
    output = tmp_path / "prepared"
    values = np.arange(1, 2049, dtype=np.float32).reshape(grid.shape)
    mask = (values > 1000) & (values < 1500)
    write_nifti(pre, values, grid)
    write_nifti(post, values + 20, grid)
    prepare_volumes(output, pre, post, {"lesion": (mask, grid, "manual")}, pre_index=None, post_index=None,
                    release="synthetic", patient="p", cohort="demo")
    native, native_grid = read_nifti(output / "native_labels" / "lesion.nii.gz")
    assert np.array_equal(native, mask)
    assert np.array_equal(native_grid.affine, grid.affine)
    record = json.loads((output / "case.json").read_text())
    assert record["reference"]["type"] == "nifti"
