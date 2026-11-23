"""Train our own weights with partial supervision and optional offline knowledge transfer."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
from tempfile import TemporaryDirectory
import time

import numpy as np
from scipy.ndimage import distance_transform_edt
import torch
import torch.nn.functional as F

from .labels import ENVELOPE, LABELS, REFERENCE_LABELS
from .network import BreastMRI3D
from .research import Case, read_manifest


def segmentation_loss(logit: torch.Tensor, target: torch.Tensor, known: torch.Tensor) -> torch.Tensor:
    weight = known[:, :, None, None, None].to(logit.dtype)
    voxels = target[0, 0].numel()
    bce = (F.binary_cross_entropy_with_logits(logit, target, reduction="none") * weight).sum()
    bce = bce / (weight.sum() * voxels).clamp_min(1)
    prob = logit.sigmoid()
    dims = (2, 3, 4)
    foreground = target.sum(dims)
    has_fg = (foreground > 0).to(logit.dtype) * known.to(logit.dtype)
    dice = (2 * (prob * target).sum(dims) + 1) / (prob.sum(dims) + foreground + 1)
    return bce + ((1 - dice) * has_fg).sum() / has_fg.sum().clamp_min(1)


def distillation_loss(logit: torch.Tensor, soft: torch.Tensor,
                      trusted: torch.Tensor, known: torch.Tensor) -> torch.Tensor:
    eligible = trusted.bool() & ~known[:, :, None, None, None].bool()
    penalty = F.binary_cross_entropy_with_logits(logit, soft, reduction="none")
    return (penalty * eligible).sum() / eligible.sum().clamp_min(1)


def crop(array: np.ndarray, center: tuple[int, ...], size: tuple[int, ...]) -> np.ndarray:
    begins = [min(max(0, c - k // 2), max(0, dim - k)) for c, k, dim in zip(center, size, array.shape[-3:])]
    slices = tuple(slice(start, start + width) for start, width in zip(begins, size))
    patch = np.asarray(array[(..., *slices)])
    padding = [(0, 0)] * (array.ndim - 3) + [(0, k - n) for k, n in zip(size, patch.shape[-3:])]
    return np.pad(patch, padding) if any(n for _, n in padding) else patch


def _teacher_file(case: Case) -> Path:
    return case.path / "teacher.npz"


def _has_teacher(case: Case) -> bool:
    return _teacher_file(case).is_file() or (case.path / "teacher-envelope-probability.npy").is_file()


def envelope_logit(logit: torch.Tensor) -> torch.Tensor:
    outside = logit[:, :2].neg().sigmoid().prod(dim=1, keepdim=True)
    return (1 - outside).clamp(1e-6, 1 - 1e-6).logit()


def _metadata_json(record: Path) -> dict:
    return json.loads(record.read_text(encoding="utf-8"))


def _validate_teacher(case: Case, reserved_patients: set[str]) -> None:
    info = _metadata_json(case.path / "teacher.json")
    if (str(info.get("patient", "")).casefold() != case.patient or info.get("grid") != case.metadata["grid"]
            or info.get("channel_order") != list(LABELS) or not info.get("calibration_record")
            or not re.fullmatch(r"[0-9a-fA-F]{64}", str(info.get("teacher_sha256", "")))):
        raise ValueError(f"Teacher probabilities lack matching geometry, model hash or calibration: {case.path.name}")
    teacher_patients = info.get("training_patients")
    if not isinstance(teacher_patients, list) or not teacher_patients:
        raise ValueError("The teacher training partition must be traceable before distillation.")
    if reserved_patients & {str(patient).strip().casefold() for patient in teacher_patients}:
        raise ValueError("A teacher was trained on a reserved validation or test patient.")


def _load_pairs(path: Path, cases: list[Case]) -> list[tuple]:
    allowed = {c.path.resolve(): c for c in cases if c.split == "train"}
    pairs = []
    for row in _metadata_json(path)["pairs"]:
        sides = []
        for name in ("positive", "negative"):
            entry = row[name]
            folder = (path.parent / entry["case"]).resolve()
            if folder not in allowed:
                raise ValueError("Ranking pairs must reference development training cases only.")
            case = allowed[folder]
            center = tuple(entry["center_zyx"])
            if (len(center) != 3 or any(type(n) is not int or not 0 <= n < limit
                                        for n, limit in zip(center, case.grid.shape))):
                raise ValueError("Ranking centre lies outside its prepared case.")
            if name == "positive":
                labels = np.load(folder / "labels.npy", mmap_mode="r", allow_pickle=False)
                if case.metadata["labels"].get("lesion") != "manual" or not labels[(-1, *center)]:
                    raise ValueError("Positive ranking centres require a manual lesion reference.")
            else:
                provenance = case.metadata["labels"].get("lesion")
                if provenance != "confirmed_empty" and not entry.get("adjudication_record"):
                    raise ValueError("Negative proposals require a lesion-free case or explicit adjudication.")
                if (case.patient in {str(p).strip().casefold() for p in entry.get("proposal_training_patients", [])}
                        or not entry.get("proposal_training_patients")
                        or not re.fullmatch(r"[0-9a-fA-F]{64}", str(entry.get("proposal_model_sha256", "")))):
                    raise ValueError("Negative proposal models must certify out-of-fold provenance.")
                if not row.get("matching_record"):
                    raise ValueError("A matched pair needs a documented size, enhancement and phase comparison.")
            sides.append((case, center))
        pairs.append(tuple(sides))
    if not pairs:
        raise ValueError("Ranking was requested with an empty verified-pair manifest.")
    return pairs


def _sample(case: Case, rng: np.random.Generator, size: tuple[int, int, int], *,
            use_teacher: bool, foreground_rate: float = .7) -> tuple:
    image = np.load(case.path / "image.npy", mmap_mode="r", allow_pickle=False)
    labels = np.load(case.path / "labels.npy", mmap_mode="r", allow_pickle=False)
    facts = case.metadata["labels"]
    envelope_known = ENVELOPE in facts
    envelope = np.load(case.path / f"{ENVELOPE}.npy", mmap_mode="r", allow_pickle=False) if envelope_known else None
    known = np.array([name in facts for name in LABELS], dtype=bool)
    center = tuple(int(rng.integers(n)) for n in image.shape[-3:])
    if rng.random() < foreground_rate:
        positive = np.argwhere(labels[-1])
        if not len(positive):
            non_lesion = [i for i, name in enumerate(LABELS[:-1]) if name in facts]
            if non_lesion:
                chosen = non_lesion[int(rng.integers(len(non_lesion)))]
                positive = np.argwhere(labels[chosen])
            elif envelope is not None:
                positive = np.argwhere(envelope)
        if len(positive):
            center = tuple(int(n) for n in positive[rng.integers(len(positive))])
    x = crop(image, center, size).astype(np.float32, copy=True)
    y = crop(labels, center, size).astype(np.float32, copy=True)
    if use_teacher and _teacher_file(case).is_file():
        with np.load(_teacher_file(case), allow_pickle=False) as aux:
            if aux["probabilities"].shape != labels.shape or aux["trusted"].shape != labels.shape:
                raise ValueError("Teacher arrays have the wrong shape for their prepared case.")
            soft = crop(aux["probabilities"], center, size).astype(np.float32)
            trust = crop(aux["trusted"], center, size).astype(bool)
            if not np.isfinite(soft).all() or not ((soft >= 0) & (soft <= 1)).all():
                raise ValueError("Teacher probabilities are outside [0, 1].")
    else:
        soft, trust = np.zeros_like(y), np.zeros_like(y, dtype=bool)
    envelope_target = (crop(envelope, center, size).astype(np.float32) if envelope is not None
                       else np.zeros(size, dtype=np.float32))
    envelope_soft = np.zeros(size, dtype=np.float32)
    envelope_trusted = np.zeros(size, dtype=bool)
    teacher_envelope = case.path / "teacher-envelope-probability.npy"
    if use_teacher and not envelope_known and teacher_envelope.is_file():
        raw_soft = np.load(teacher_envelope, mmap_mode="r", allow_pickle=False)
        raw_trust = np.load(case.path / "teacher-envelope-trusted.npy", mmap_mode="r", allow_pickle=False)
        if raw_soft.shape != labels.shape[1:] or raw_trust.shape != raw_soft.shape:
            raise ValueError("The envelope teacher and prepared case must share the same grid.")
        envelope_soft = crop(raw_soft, center, size).astype(np.float32)
        envelope_trusted = crop(raw_trust, center, size).astype(bool)
        if not np.isfinite(envelope_soft).all() or not ((envelope_soft >= 0) & (envelope_soft <= 1)).all():
            raise ValueError("Envelope teacher probabilities are outside [0, 1].")
    return x, y, known, soft, trust, envelope_target[None], np.array([envelope_known]), envelope_soft[None], envelope_trusted[None]


def _ranking(model: BreastMRI3D, pair: tuple, patch: tuple[int, ...], device: torch.device) -> torch.Tensor:
    scores = []
    for case, center in pair:
        image = np.load(case.path / "image.npy", mmap_mode="r", allow_pickle=False)
        patch_data = crop(image, center, patch).astype(np.float32)
        _, candidates = model(torch.as_tensor(patch_data[None], device=device))
        coordinates = tuple(c - min(max(0, c - k // 2), max(0, n - k)) for c, k, n in
                            zip(center, patch, image.shape[-3:]))
        scores.append(candidates[(0, 0, *coordinates)])
    return F.softplus(scores[1] - scores[0])


def _lesion_patch(case: Case, rng: np.random.Generator, patch: tuple[int, ...],
                  foreground: dict[Path, np.ndarray], teacher: Path | None = None
                  ) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    image = np.load(case.path / "image.npy", mmap_mode="r", allow_pickle=False)
    label = np.load(case.path / "labels.npy", mmap_mode="r", allow_pickle=False)[-1]
    center = tuple(int(rng.integers(n)) for n in label.shape)
    if rng.random() < .65:
        if case.path not in foreground:
            foreground[case.path] = np.argwhere(label).astype(np.int32)
        positive = foreground[case.path]
        if len(positive):
            jitter = [int(rng.integers(-n // 4, n // 4 + 1)) for n in patch]
            center = tuple(int(c) + j for c, j in zip(positive[rng.integers(len(positive))], jitter))
    x, y = crop(image, center, patch), crop(label, center, patch)
    reflect = rng.random() < .5
    if reflect:
        x, y = x[..., ::-1], y[..., ::-1]
    x = x.astype(np.float32, copy=True, order="C")
    x *= rng.uniform(.9, 1.1)
    hint = None
    if teacher is not None:
        probabilities = np.load(teacher, mmap_mode="r", allow_pickle=False)
        if probabilities.shape != label.shape:
            raise ValueError("The lesion teacher probabilities have a different grid.")
        hint = crop(probabilities, center, patch)
        if reflect:
            hint = hint[..., ::-1]
        hint = hint.astype(np.float32, copy=True, order="C")
        if not np.isfinite(hint).all() or hint.min() < 0 or hint.max() > 1:
            raise ValueError("Invalid lesion teacher probabilities.")
    return x, y.astype(np.float32, copy=True, order="C"), hint


def _distance_penalty(target: np.ndarray, spacing_mm: float) -> np.ndarray:
    positive = target > 0
    inside = distance_transform_edt(positive, sampling=spacing_mm)
    outside = distance_transform_edt(~positive, sampling=spacing_mm)
    # A discrepancy farther from the reference surface costs more, up to 20 mm.
    return np.minimum(inside + outside, 20.).astype(np.float32) / 20.


def _save_resume(path: Path, state: dict) -> None:
    with TemporaryDirectory(dir=path.parent, prefix=".resume-") as tmp:
        staged = Path(tmp) / path.name
        torch.save(state, staged)
        os.replace(staged, path)


def train(manifest: Path, destination: Path, *, steps: int, width: int,
          patch: tuple[int, int, int], seed: int, device_name: str, teachers: bool = False,
          pairs_path: Path | None = None, dual_decoder: bool = True,
          budget_seconds: float | None = None, input_mode: str = "dce",
          model_spacing_mm: float | None = None, boundary_weight: float = 0.,
          resumable: bool = False, resume_from: Path | None = None, lesion_teacher: Path | None = None) -> dict:
    if steps < 1 or len(patch) != 3 or any(n < 8 or n % 8 for n in patch):
        raise ValueError("Training requires positive steps and three patch dimensions divisible by eight.")
    if budget_seconds is not None and (not np.isfinite(budget_seconds) or budget_seconds <= 0):
        raise ValueError("A training time budget must be finite and positive.")
    if destination.suffix.lower() not in {".pt", ".pth"}:
        raise ValueError("Use a .pt or .pth filename for the checkpoint.")
    report = destination.with_suffix(".training.json")
    cases = read_manifest(manifest)
    development = [c for c in cases if c.split == "train"]
    if not development:
        raise ValueError("The manifest has no training cases.")
    lesion_only = model_spacing_mm is not None
    if lesion_only:
        if (not np.isfinite(model_spacing_mm) or model_spacing_mm <= 0 or teachers or pairs_path
                or any("lesion" not in c.metadata["labels"] or not np.allclose(c.grid.spacing, model_spacing_mm)
                       for c in development)):
            raise ValueError("Lesion-only training requires a fixed physical grid and known lesion labels.")
    if not np.isfinite(boundary_weight) or boundary_weight < 0 or boundary_weight and not lesion_only:
        raise ValueError("The distance penalty requires lesion-only training in millimetres.")
    if resume_from and not resumable:
        raise ValueError("A continuation requires resumable training.")
    lesion_targets = {}
    if lesion_teacher is not None:
        if not lesion_only:
            raise ValueError("A lesion teacher requires the matched lesion-only training recipe.")
        from .teacher_v03 import load_teacher

        lesion_targets = load_teacher(lesion_teacher, manifest, development)
    if destination.exists() or report.exists():
        raise FileExistsError(f"Checkpoint or training report already exists: {destination}")
    if teachers:
        target_cases = [c for c in development if _has_teacher(c)]
        if not target_cases:
            raise ValueError("Distillation requested without any training-only teacher probabilities.")
        reserved = {c.patient for c in cases if c.split != "train"}
        for case in target_cases:
            _validate_teacher(case, reserved)
    pairs = _load_pairs(pairs_path, cases) if pairs_path is not None else []
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    device = torch.device(device_name)
    use_amp = device.type == "cuda"
    if use_amp and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but PyTorch cannot see a CUDA device.")
    if device.type == "cpu":
        torch.set_num_threads(min(torch.get_num_threads(), 6))
    model = BreastMRI3D(width, dual_decoder=dual_decoder, input_mode=input_mode).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    resume_file = destination.with_suffix(".resume.pt")
    manifest_hash = hashlib.sha256(manifest.read_bytes()).hexdigest()
    config = {"manifest_sha256": manifest_hash, "width": width, "patch": list(patch), "seed": seed,
              "input_mode": input_mode, "model_spacing_mm": model_spacing_mm, "boundary_weight": boundary_weight,
              "dual_decoder": dual_decoder, "teachers": teachers, "pairs": str(pairs_path) if pairs_path else None}
    if lesion_teacher is not None:
        config["lesion_teacher_sha256"] = hashlib.sha256(lesion_teacher.read_bytes()).hexdigest()
    history = []
    distilled_batches = 0
    supervised = set()
    previous_time = 0.
    if resumable:
        destination.parent.mkdir(parents=True, exist_ok=True)
        source = resume_file if resume_file.is_file() else resume_from
        if source is not None:
            state = torch.load(source, map_location="cpu", weights_only=True)
            if state["config"] != config:
                raise ValueError("The continuation does not match the original training recipe.")
            model.load_state_dict(state["model"])
            optimizer.load_state_dict(state["optimizer"])
            scaler.load_state_dict(state["scaler"])
            rng.bit_generator.state = state["rng"]
            torch.set_rng_state(state["torch_rng"])
            if use_amp and state["cuda_rng"] is not None:
                torch.cuda.set_rng_state(state["cuda_rng"], device)
            history, previous_time = state["history"], state["elapsed_seconds"]
            distilled_batches, supervised = state["distilled_batches"], set(state["supervised"])
    if use_amp:
        torch.cuda.reset_peak_memory_stats(device)
    foreground = {}
    started = time.monotonic()
    last_log = started
    last_save = started

    def save_progress(elapsed: float) -> None:
        _save_resume(resume_file, {"config": config, "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                                   "scaler": scaler.state_dict(), "rng": rng.bit_generator.state,
                                   "torch_rng": torch.get_rng_state(),
                                   "cuda_rng": torch.cuda.get_rng_state(device) if use_amp else None,
                                   "history": history, "elapsed_seconds": elapsed,
                                   "distilled_batches": distilled_batches, "supervised": sorted(supervised)})

    model.train()
    for step in range(len(history), steps):
        if history and budget_seconds is not None and previous_time + time.monotonic() - started >= budget_seconds:
            break
        case = development[int(rng.integers(len(development)))]
        optimizer.zero_grad(set_to_none=True)
        if lesion_only:
            image, lesion, hint = _lesion_patch(case, rng, patch, foreground, lesion_targets.get(case.path))
            x = torch.as_tensor(image[None], device=device)
            y = torch.as_tensor(lesion[None, None], device=device)
            known = torch.ones((1, 1), dtype=torch.bool, device=device)
            with torch.autocast(device.type, enabled=use_amp):
                logit = model.lesion(x)
            logit = logit.float()
            loss = segmentation_loss(logit, y, known)
            if hint is not None:
                # Keep teacher signal near plausible foreground and preserve the manual supervision.
                selected = ((lesion > 0) | (hint >= .1)) & (np.abs(hint - lesion) <= .6)
                if selected.any():
                    soft = torch.as_tensor(hint[None, None], device=device)
                    support = torch.as_tensor(selected[None, None], device=device)
                    soft_loss = F.binary_cross_entropy_with_logits(logit, soft, reduction="none")
                    loss = loss + .25 * ((soft_loss * support).sum() / support.sum())
                    distilled_batches += 1
            if boundary_weight and lesion.any() and not lesion.all():
                distance = torch.as_tensor(_distance_penalty(lesion, model_spacing_mm), device=device)
                loss = loss + boundary_weight * ((logit.sigmoid() - y).abs() * distance).mean()
            supervised.add("lesion")
        else:
            batch = _sample(case, rng, patch, use_teacher=teachers)
            supervised.update(name for name, eligible in zip(LABELS, batch[2]) if eligible)
            envelope_known = bool(batch[6][0])
            envelope_distilled = bool(batch[8].any())
            if envelope_known:
                supervised.add(ENVELOPE)
            x, y, known, soft, trusted = (torch.as_tensor(t[None], device=device) for t in batch[:5])
            if (batch[4] & ~batch[2][:, None, None, None]).any() or envelope_distilled:
                distilled_batches += 1
            with torch.autocast(device.type, enabled=use_amp):
                logit, _ = model(x)
            loss = segmentation_loss(logit.float(), y, known)
            if teachers:
                loss = loss + .25 * distillation_loss(logit.float(), soft, trusted, known)
            if envelope_known or envelope_distilled:
                union = envelope_logit(logit.float())
                envelope_y, envelope_k, envelope_t, envelope_valid = (
                    torch.as_tensor(t[None], device=device) for t in batch[5:])
                if envelope_known:
                    loss = loss + .5 * segmentation_loss(union, envelope_y, envelope_k)
                if envelope_distilled:
                    loss = loss + .25 * distillation_loss(union, envelope_t, envelope_valid, envelope_k)
        if pairs:
            # Backpropagate the segmentation before creating two additional patch graphs.
            scaler.scale(loss).backward()
            with torch.autocast(device.type, enabled=use_amp):
                ranking_loss = _ranking(model, pairs[step % len(pairs)], patch, device)
            scaler.scale(.2 * ranking_loss.float()).backward()
        else:
            scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        value = float(loss.detach())
        if not np.isfinite(value):
            raise ValueError(f"Non-finite loss at update {step + 1}.")
        history.append(value)
        now = time.monotonic()
        elapsed = previous_time + now - started
        reached_budget = budget_seconds is not None and elapsed >= budget_seconds
        if step == 0 or now - last_log >= 30 or step + 1 == steps or reached_budget:
            print(f"update {step + 1}/{steps} seconds={elapsed:.1f} loss={value:.5f}", flush=True)
            last_log = now
        if resumable and now - last_save >= 180:
            save_progress(elapsed)
            last_save = now
        if reached_budget:
            break
    if (teachers or lesion_teacher is not None) and not distilled_batches:
        raise ValueError("Distillation was requested but never sampled an eligible teacher target.")
    destination.parent.mkdir(parents=True, exist_ok=True)
    elapsed = previous_time + time.monotonic() - started
    if resumable:
        save_progress(elapsed)
    summary = {"training_cases": len(development), "updates": len(history), "seed": seed, "device": str(device),
               "elapsed_seconds": round(elapsed, 3), "patch_zyx": list(patch),
               "budget_seconds": budget_seconds,
               "manifest_sha256": manifest_hash, "input_mode": input_mode, "model_spacing_mm": model_spacing_mm,
               "boundary_weight": boundary_weight,
               "loss_first": history[0], "loss_last": history[-1], "parameters": sum(p.numel() for p in model.parameters()),
               "distilled_updates": distilled_batches, "verified_ranking_pairs": len(pairs),
               "supervised_labels": [name for name in REFERENCE_LABELS if name in supervised]}
    if lesion_teacher is not None:
        summary["lesion_teacher_sha256"] = config["lesion_teacher_sha256"]
    if use_amp:
        summary["peak_cuda_allocated_mib"] = torch.cuda.max_memory_allocated(device) / 2**20
        summary["accelerator"] = torch.cuda.get_device_name(device)
    # Complete both writes before publishing the checkpoint or its sidecar.
    with TemporaryDirectory(dir=destination.parent, prefix=".training-") as tmp:
        checkpoint = Path(tmp) / destination.name
        metadata = Path(tmp) / report.name
        torch.save({"state_dict": model.cpu().state_dict(), "width": width, "dual_decoder": dual_decoder,
                    "input_mode": input_mode,
                    "labels": list(LABELS), "summary": summary}, checkpoint)
        metadata.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        if destination.exists() or report.exists():
            raise FileExistsError(f"Checkpoint or training report appeared during training: {destination}")
        checkpoint.rename(destination)
        metadata.rename(report)
    return summary
