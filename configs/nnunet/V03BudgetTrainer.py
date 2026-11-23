"""The nnU-Net trainer and augmentations with a resumable elapsed-time budget."""

import json
import math
import os
from pathlib import Path
import random
import time

import numpy as np
import torch
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer


class V03BudgetTrainer(nnUNetTrainer):
    def __init__(self, plans, configuration, fold, dataset_json, device):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.budget_seconds = float(os.environ.get("BMRI_REFERENCE_BUDGET_SECONDS", "21600"))
        self.schedule_seconds = float(os.environ.get("BMRI_REFERENCE_SCHEDULE_SECONDS", "43200"))
        self.random_seed = int(os.environ.get("BMRI_REFERENCE_SEED", "19"))
        if not math.isfinite(self.schedule_seconds) or not 0 < self.budget_seconds <= self.schedule_seconds:
            raise ValueError("The time budget must be positive and within the full learning-rate schedule.")
        self.num_val_iterations_per_epoch = 12
        self.save_every = 3
        self.elapsed_seconds = 0.
        self.updates = 0

    def save_checkpoint(self, filename):
        if self.disable_checkpointing or self.local_rank != 0:
            return
        self.my_init_kwargs["bmri_budget"] = {"elapsed_seconds": self.elapsed_seconds, "updates": self.updates,
                                             "schedule_seconds": self.schedule_seconds, "seed": self.random_seed}
        path = Path(filename)
        temp = path.with_suffix(".partial.pth")
        super().save_checkpoint(str(temp))
        os.replace(temp, path)

    def run_training(self):
        if self.is_ddp:
            raise ValueError("The time-matched reference uses one GPU process.")
        previous = self.my_init_kwargs.get("bmri_budget", {})
        if previous and (previous["schedule_seconds"] != self.schedule_seconds or previous["seed"] != self.random_seed):
            raise ValueError("The resumed reference must use the original training schedule and seed.")
        elapsed_before = previous.get("elapsed_seconds", 0.)
        self.updates = previous.get("updates", 0)
        self.elapsed_seconds = elapsed_before
        random.seed(self.random_seed + self.current_epoch)
        np.random.seed(self.random_seed + self.current_epoch)
        torch.manual_seed(self.random_seed + self.current_epoch)
        self.on_train_start()
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)
        start = time.monotonic()

        def elapsed():
            return elapsed_before + time.monotonic() - start

        while self.elapsed_seconds < self.budget_seconds:
            self.on_epoch_start()
            self.on_train_epoch_start()
            fraction = min(1., elapsed() / self.schedule_seconds)
            for group in self.optimizer.param_groups:
                group["lr"] = self.initial_lr * max(.001, 1 - fraction) ** .9
            training = []
            for _ in range(self.num_iterations_per_epoch):
                training.append(self.train_step(next(self.dataloader_train)))
                self.updates += 1
                if elapsed() >= self.budget_seconds:
                    break
            self.on_train_epoch_end(training)
            with torch.no_grad():
                self.on_validation_epoch_start()
                validation = [self.validation_step(next(self.dataloader_val))
                              for _ in range(self.num_val_iterations_per_epoch)]
                self.on_validation_epoch_end(validation)
            self.elapsed_seconds = elapsed()
            self.on_epoch_end()
            self.elapsed_seconds = elapsed()
        self.on_train_end()
        report = {"budget_seconds": self.budget_seconds, "elapsed_seconds": round(self.elapsed_seconds, 3),
                  "updates": self.updates, "seed": self.random_seed, "device": str(self.device)}
        if self.device.type == "cuda":
            report["peak_cuda_allocated_mib"] = torch.cuda.max_memory_allocated(self.device) / 2 ** 20
        path = Path(self.output_folder) / "budget.json"
        staged = path.with_suffix(".partial.json")
        staged.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        os.replace(staged, path)
