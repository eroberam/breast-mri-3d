import json

import numpy as np
import pytest

from breast_mri3d.geometry import Grid
from breast_mri3d.labels import LABELS
from breast_mri3d.ranking import Candidate, pair_candidates
from breast_mri3d.research import Case, match_lesions, read_manifest


def test_patient_in_derived_cohort_cannot_cross_splits(tmp_path):
    (tmp_path / "first").mkdir()
    (tmp_path / "first" / "case.json").write_text(json.dumps({
        "patient": "ISPY1_1", "cohort": "ISPY1", "channel_order": LABELS, "labels": {"lesion": "manual"}
    }))
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"cases": [
        {"patient": "ISPY1_1", "split": "train", "path": "first", "cohort": "ISPY1"},
        {"patient": "ispy1_1", "split": "test", "path": "derived", "cohort": "MAMA-MIA"},
    ]}))
    with pytest.raises(ValueError, match="more than one split"):
        read_manifest(manifest)


def test_component_matching_counts_miss_and_false_alarm():
    truth = np.zeros((10, 10, 10), dtype=bool)
    truth[1:3, 1:3, 1:3] = True
    truth[7:9, 7:9, 7:9] = True
    predicted = np.zeros_like(truth)
    predicted[1:3, 1:3, 1:3] = True
    predicted[1:3, 7:9, 1:3] = True
    assert match_lesions(predicted, truth) == {"matched": 1, "missed": 1, "false_alarms": 1}


def test_candidate_matching_respects_physical_and_protocol_calipers(tmp_path):
    case = Case(tmp_path, "patient", "cohort", "train")
    positive = Candidate(case, (1, 1, 1), 2.0, .5, "early-A")
    matching = Candidate(case, (1, 1, 5), 1.9, .45, "early-A")
    negatives = [matching, Candidate(case, (1, 1, 6), 2.0, .5, "early-B"),
                 Candidate(case, (1, 1, 7), 7.0, .5, "early-A"),
                 Candidate(case, (1, 1, 8), 2.0, .9, "early-A")]
    assert pair_candidates([positive], negatives, maximum_volume_ratio=2,
                           maximum_enhancement_gap=.2) == [(positive, matching)]


def test_unknown_labels_are_ignored_unless_teacher_is_trusted():
    torch = pytest.importorskip("torch")
    from breast_mri3d.train import distillation_loss, segmentation_loss

    logits = torch.zeros((1, 5, 3, 3, 3), requires_grad=True)
    truth = torch.zeros_like(logits)
    truth[:, -1, 1, 1, 1] = 1
    has_label = torch.tensor([[False, False, False, False, True]])
    segmentation_loss(logits, truth, has_label).backward()
    assert torch.count_nonzero(logits.grad[:, :4]) == 0
    logits.grad = None
    teacher = torch.ones_like(logits)
    trust = torch.ones_like(logits, dtype=torch.bool)
    distillation_loss(logits, teacher, trust, has_label).backward()
    assert torch.count_nonzero(logits.grad[:, -1]) == 0
    assert torch.count_nonzero(logits.grad[:, :4]) > 0


def test_cross_fitted_fold_manifest_survives_checkpoint_and_supports_joint_training(tmp_path):
    pytest.importorskip("torch")
    from breast_mri3d.ranking import build_pairs
    from breast_mri3d.train import train

    shape = (8, 8, 8)
    grid = Grid(shape, np.diag([2, 1, 1, 1])).as_dict()
    rng = np.random.default_rng(11)
    records = []
    for name, kind in [("positive", "manual"), ("negative", "confirmed_empty")]:
        case = tmp_path / name
        case.mkdir()
        np.save(case / "image.npy", rng.uniform(.1, .9, (3, *shape)).astype(np.float16))
        labels = np.zeros((len(LABELS), *shape), dtype=np.uint8)
        labels[-1] = kind == "manual"
        np.save(case / "labels.npy", labels)
        case_info = {"patient": name, "cohort": "synthetic", "grid": grid, "phase_protocol": "phantom",
                     "labels": {"lesion": kind}, "channel_order": list(LABELS)}
        (case / "case.json").write_text(json.dumps(case_info), encoding="utf-8")
        trusted = np.zeros_like(labels, dtype=bool)
        trusted[2] = True
        np.savez(case / "teacher.npz", probabilities=np.full(labels.shape, .25), trusted=trusted)
        teacher = {"patient": name, "grid": grid, "channel_order": list(LABELS), "teacher_sha256": "a" * 64,
                   "training_patients": ["positive"], "calibration_record": "synthetic"}
        (case / "teacher.json").write_text(json.dumps(teacher), encoding="utf-8")
        records.append({"patient": name, "cohort": "synthetic", "path": name, "split": "train"})

    master, fold = tmp_path / "master.json", tmp_path / "fold.json"
    master.write_text(json.dumps({"cases": records}), encoding="utf-8")
    fold.write_text(json.dumps({"cases": [records[0], {**records[1], "split": "validation"}]}), encoding="utf-8")
    original_manifest = fold.read_bytes()
    checkpoint = tmp_path / "fold.pt"
    train(fold, checkpoint, steps=2, width=4, patch=shape, seed=19, device_name="cpu")
    assert fold.read_bytes() == original_manifest
    folds = tmp_path / "folds.json"
    folds.write_text(json.dumps({"folds": [{"manifest": fold.name, "checkpoint": checkpoint.name}]}), encoding="utf-8")
    pairs = tmp_path / "pairs.json"
    assert build_pairs(master, folds, pairs, threshold=.00001, patch=shape, device_name="cpu",
                       maximum_volume_ratio=2, maximum_enhancement_gap=1) == 1
    result = train(master, tmp_path / "distilled.pt", steps=4, width=4, patch=shape, seed=19, device_name="cpu",
                   teachers=True, pairs_path=pairs)
    assert result["distilled_updates"] == 4
    assert result["verified_ranking_pairs"] == 1
