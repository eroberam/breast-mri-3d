"""Run the frozen v0.3 experiments with a single sequential GPU queue."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from .acquire import checksum, write_json
from .cascade import COARSE_SPACING_MM, prepare_coarse
from .cohorts_v03 import MANIFESTS, patient_key, verify_partitions
from .reference_v03 import CONFIGURATION, DATASET_ID, DATASET_NAME, TRAINER
from .research import read_manifest
from .validation_v03 import paired_patients


LARGE_PATCH = (48, 128, 128)
SMALL_PATCH = (32, 112, 112)
FINE_SPACING_MM = 1.5
PLANS = "BreastMRI3DResEncM"


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


class Study:
    def __init__(self, root: Path, trainer_directory: Path):
        self.root = root.resolve()
        self.prepared = self.root / "prepared"
        self.development = self.prepared / MANIFESTS["development"]
        self.coarse = self.prepared / "development-coarse.json"
        self.runs = self.root / "runs" / "v03"
        self.reports = self.root / "reports"
        self.logs = self.runs / "logs"
        self.raw = self.root / "nnunet" / "raw"
        self.preprocessed = self.root / "nnunet" / "preprocessed"
        self.reference_results = self.root / "nnunet" / "results"
        self.runs.mkdir(parents=True, exist_ok=True)
        self.logs.mkdir(exist_ok=True)
        self.env = os.environ.copy()
        self.env.update({"nnUNet_raw": str(self.raw), "nnUNet_preprocessed": str(self.preprocessed),
                         "nnUNet_results": str(self.reference_results), "nnUNet_extTrainer": str(trainer_directory.resolve()),
                         "nnUNet_compile": "false", "nnUNet_n_proc_DA": "2", "OMP_NUM_THREADS": "2",
                         "MKL_NUM_THREADS": "2", "PYTHONUNBUFFERED": "1"})
        self.lock = self.runs / "evaluation-lock.json"

    def command(self, name: str, command: list[str]) -> None:
        log = self.logs / f"{name}.log"
        print(f"Starting {name}.", flush=True)
        with log.open("a", encoding="utf-8") as stream:
            finished = subprocess.run(command, env=self.env, stdout=stream, stderr=subprocess.STDOUT, check=False)
        if finished.returncode:
            raise RuntimeError(f"{name} exited with status {finished.returncode}. Check {log}.")
        print(f"Completed {name}.", flush=True)

    def module(self, name: str, module: str, *arguments: object) -> None:
        self.command(name, [sys.executable, "-m", f"breast_mri3d.{module}", *map(str, arguments)])

    def wait_file(self, path: Path, timeout_seconds: float) -> None:
        if not path.is_file():
            print(f"Waiting for {path.name}.", flush=True)
        deadline = time.monotonic() + timeout_seconds
        while not path.is_file() and time.monotonic() < deadline:
            time.sleep(20)
        if not path.is_file():
            raise TimeoutError(f"Timed out waiting for {path}.")

    def reference_plan(self, plans: str, memory_gb: float) -> None:
        receipt = self.runs / f"{plans}.json"
        if receipt.is_file():
            return
        if not (self.raw / DATASET_NAME).is_dir():
            self.module("reference-prepare", "reference_v03", "prepare", self.development)
        exe = shutil.which("nnUNetv2_plan_and_preprocess", path=str(Path(sys.executable).parent))
        if exe is None:
            raise FileNotFoundError("Install the pinned nnUNetv2 command in the training environment.")
        self.command(f"{plans}-preprocess", [exe, "-d", str(DATASET_ID), "-pl", "nnUNetPlannerResEncM",
                                             "-gpu_memory_target", str(memory_gb), "-overwrite_plans_name", plans,
                                             "-c", CONFIGURATION, "-np", "2", "-npfp", "2"])
        write_json(receipt, {"memory_target_gb": memory_gb, "development_sha256": checksum(self.development)})

    def reference_checkpoint(self, plans: str, hours: int) -> Path:
        path = (self.reference_results / DATASET_NAME / f"{TRAINER}__{plans}__{CONFIGURATION}" /
                "fold_0" / f"checkpoint_{hours}h.pth")
        if not path.is_file():
            self.module(f"{plans}-{hours}h", "reference_v03", "train", plans, "--hours", hours)
        return path

    def fit(self, name: str, mode: str, patch: tuple[int, ...], *, hours: float, steps: int = 2_000_000,
            locator: bool = False, boundary: float = 0., earlier: str | None = None) -> Path:
        path = self.runs / f"{name}.pt"
        if not path.is_file():
            args = ["train", self.coarse if locator else self.development, "--out", path, "--steps", steps,
                    "--budget-seconds", hours * 3600, "--seed", 19, "--width", 12, "--patch", *patch,
                    "--input-mode", mode, "--model-spacing-mm", COARSE_SPACING_MM if locator else FINE_SPACING_MM,
                    "--boundary-weight", boundary, "--resumable"]
            if earlier:
                args.extend(["--resume-from", self.runs / f"{earlier}.resume.pt"])
            self.module(name, "cli", *args)
        return path

    def patch_probe(self) -> tuple[int, ...]:
        receipt = self.runs / "patch-selection.json"
        if receipt.is_file():
            return tuple(read(receipt)["patch_zyx"])
        try:
            self.fit("memory-probe-large", "dce", LARGE_PATCH, hours=.08, steps=8, boundary=.15)
            patch = LARGE_PATCH
        except RuntimeError:
            log = (self.logs / "memory-probe-large.log").read_text(encoding="utf-8", errors="replace")
            if "out of memory" not in log.lower():
                raise
            self.fit("memory-probe-small", "dce", SMALL_PATCH, hours=.08, steps=8, boundary=.15)
            patch = SMALL_PATCH
        write_json(receipt, {"patch_zyx": list(patch), "large_patch_requested_zyx": list(LARGE_PATCH)})
        return patch

    def native_reference(self, name: str, manifest: Path, model: Path, split: str) -> Path:
        report = self.runs / f"{name}-reference.json"
        if not report.is_file():
            self.module(f"{name}-reference", "reference_v03", "evaluate", manifest, model,
                        self.runs / f"{name}-nnunet", report, "--split", split)
        return report

    def native_compact(self, name: str, manifest: Path, model: Path, split: str, *,
                       threshold: float | None = None, locator: Path | None = None) -> Path:
        report = self.runs / f"{name}-compact.json"
        if not report.is_file():
            args = [manifest, model, "--out", report, "--split", split]
            if threshold is not None:
                args.extend(("--threshold", threshold))
            if locator:
                args.extend(("--locator", locator))
            self.module(f"{name}-compact", "validation_v03", *args)
        return report

    def relative_receipt(self, path: Path) -> dict:
        return {"path": path.relative_to(self.root).as_posix(), "sha256": checksum(path)}

    def checked_path(self, record: dict) -> Path:
        path = self.root / record["path"]
        if not path.resolve().is_relative_to(self.root) or checksum(path) != record["sha256"]:
            raise ValueError("A frozen artifact is outside the study or has changed.")
        return path

    def fit_and_lock(self, patch: tuple[int, ...], negatives: Path) -> dict:
        plans = read(self.runs / "reference-plans.json")["plans"]
        reference6 = self.reference_checkpoint(plans, 6)
        tuning_reference6 = self.native_reference("reference-6h-adjustment", self.development, reference6, "validation")
        treatments = {}
        for mode in ("subtraction", "dce"):
            name = f"compact-{mode}-6h"
            model = self.fit(name, mode, patch, hours=6)
            tune = self.native_compact(name, self.development, model, "validation")
            treatments[mode] = (model, tune, None)
        mode = max(treatments, key=lambda key: read(treatments[key][1])["mean_dice"])
        prepare_coarse(self.development, self.coarse)
        coarse = self.fit("cascade-locator-3h", mode, patch, hours=3, locator=True)
        fine = self.fit("cascade-refiner-3h", mode, patch, hours=3, boundary=.15)
        tune = self.native_compact("cascade-6h", self.development, fine, "validation", locator=coarse)
        treatments["cascade"] = (fine, tune, coarse)
        winner = max(treatments, key=lambda key: read(treatments[key][1])["mean_dice"])
        if winner == "cascade":
            coarse = self.fit("cascade-locator-6h", mode, patch, hours=6, locator=True, earlier="cascade-locator-3h")
            fine = self.fit("cascade-refiner-6h", mode, patch, hours=6, boundary=.15, earlier="cascade-refiner-3h")
        else:
            fine = self.fit(f"compact-{winner}-12h", winner, patch, hours=12, earlier=f"compact-{winner}-6h")
            coarse = None
        tune = self.native_compact(f"{winner}-12h", self.development, fine, "validation", locator=coarse)
        treatments[f"{winner}-12h"] = (fine, tune, coarse)
        reference12 = self.reference_checkpoint(plans, 12)
        tuning_reference12 = self.native_reference("reference-12h-adjustment", self.development, reference12, "validation")
        selected_reference, selected_ref_tune = max(((reference6, tuning_reference6), (reference12, tuning_reference12)),
                                                     key=lambda pair: read(pair[1])["metrics"]["mean_dice"])
        selected, selected_tune, locator = max((treatments[winner], treatments[f"{winner}-12h"]),
                                             key=lambda trial: read(trial[1])["mean_dice"])
        cohorts = {label: self.relative_receipt(self.prepared / MANIFESTS[label]) for label in ("internal", "duke")}
        cohorts["odelia"] = self.relative_receipt(negatives)
        freeze = {"partition": self.relative_receipt(self.prepared / "partition-lock.json"), "cohorts": cohorts,
                  "proposed": {"model": self.relative_receipt(selected), "threshold": read(selected_tune)["threshold"],
                               "adjustment": self.relative_receipt(selected_tune), "postprocessing": "none",
                               "locator": self.relative_receipt(locator) if locator else None},
                  "reference": {"model": self.relative_receipt(selected_reference), "threshold": .5,
                                "adjustment": self.relative_receipt(selected_ref_tune), "postprocessing": "none",
                                "mirroring": True},
                  "trials": {label: self.relative_receipt(value[1]) for label, value in treatments.items()},
                  "reference_trials": [self.relative_receipt(path) for path in (tuning_reference6, tuning_reference12)],
                  "selection_metric": "native_mean_dice", "patch_zyx": list(patch), "seed": 19}
        write_json(self.lock, freeze)
        return freeze

    def score_holdouts(self, freeze: dict) -> None:
        self.checked_path(freeze["partition"])
        verify_partitions(self.root)
        proposed, reference = freeze["proposed"], freeze["reference"]
        model, comparator = self.checked_path(proposed["model"]), self.checked_path(reference["model"])
        if read(self.checked_path(proposed["adjustment"]))["threshold"] != proposed["threshold"]:
            raise ValueError("The frozen threshold differs from the adjustment receipt.")
        self.checked_path(reference["adjustment"])
        locator = self.checked_path(proposed["locator"]) if proposed["locator"] else None
        for cohort, manifest in freeze["cohorts"].items():
            cases = self.checked_path(manifest)
            compact = self.native_compact(cohort, cases, model, "test", threshold=proposed["threshold"], locator=locator)
            baseline = self.native_reference(cohort, cases, comparator, "test")
            result = paired_patients(read(baseline), read(compact))
            write_json(self.reports / f"v03-{cohort}-paired.json", {"lock_sha256": checksum(self.lock), **result})
        write_json(self.reports / "v03-completed.json", {"lock_sha256": checksum(self.lock),
                   "completed_cohorts": list(freeze["cohorts"]), "benchmark_submission": "local_preparation_only"})

    def run(self, timeout_seconds: float, *, historical_replay: bool = True) -> None:
        self.wait_file(self.prepared / "partition-lock.json", timeout_seconds)
        verify_partitions(self.root)
        if self.lock.is_file():
            self.score_holdouts(read(self.lock))
            return
        negatives = self.prepared / "ODELIA" / "manifest.json"
        self.wait_file(negatives, timeout_seconds)
        cases = read_manifest(negatives)
        keys = {patient_key(case.patient) for case in cases}
        if (not cases or len(keys) != len(cases) or any(case.split != "test" or
                case.metadata["labels"].get("lesion") != "confirmed_empty" for case in cases)):
            raise ValueError("False-positive estimates require a reserved set of confirmed negative exams.")
        with ThreadPoolExecutor(max_workers=1) as pool:
            ready = pool.submit(self.reference_plan, PLANS, 5.3)
            if historical_replay:
                self.wait_file(self.reports / "orientation-pilot.json", timeout_seconds)
            patch = self.patch_probe()
            ready.result()
        plans_receipt = self.runs / "reference-plans.json"
        if not plans_receipt.is_file():
            plans = PLANS
            try:
                self.reference_checkpoint(plans, 6)
            except RuntimeError:
                log = (self.logs / f"{plans}-6h.log").read_text(encoding="utf-8", errors="replace")
                if "out of memory" not in log.lower():
                    raise
                plans = f"{PLANS}LowMemory"
                self.reference_plan(plans, 4.2)
                self.reference_checkpoint(plans, 6)
            write_json(plans_receipt, {"plans": plans})
        self.score_holdouts(self.fit_and_lock(patch, negatives))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--trainer-dir", required=True, type=Path)
    parser.add_argument("--wait-hours", type=float, default=8.)
    parser.add_argument("--skip-historical-replay", action="store_true", help="run an independent replication")
    args = parser.parse_args()
    study = Study(args.root, args.trainer_dir)
    queue = study.runs / "gpu-queue.pid"
    with queue.open("x", encoding="utf-8") as stream:
        stream.write(f"{os.getpid()}\n")
    try:
        study.run(args.wait_hours * 3600, historical_replay=not args.skip_historical_replay)
    finally:
        queue.unlink()


if __name__ == "__main__":
    main()
