"""Standard nnU-Net 3D training hooks under a fixed elapsed-time budget."""

import json
import math
import os
from pathlib import Path
import random
import time

import numpy as np
import torch
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer


class V02BudgetTrainer(nnUNetTrainer):
    def __init__(self, plans, configuration, fold, dataset_json, device):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.budget_seconds = float(os.environ.get("BMRI_REFERENCE_BUDGET_SECONDS", "2700"))
        self.random_seed = int(os.environ.get("BMRI_REFERENCE_SEED", "19"))
        if not math.isfinite(self.budget_seconds) or self.budget_seconds <= 0:
            raise ValueError("BMRI_REFERENCE_BUDGET_SECONDS must be positive and finite.")
        self.num_iterations_per_epoch = 180
        self.num_val_iterations_per_epoch = 12
        self.save_every = 3

    def run_training(self):
        if self.is_ddp:
            raise ValueError("The time-matched reference uses one GPU process.")
        random.seed(self.random_seed)
        np.random.seed(self.random_seed)
        torch.manual_seed(self.random_seed)
        self.on_train_start()
        use_cuda = self.device.type == "cuda"
        if use_cuda:
            torch.cuda.reset_peak_memory_stats(self.device)
        start = time.monotonic()
        updates = 0
        while updates == 0 or time.monotonic() - start < self.budget_seconds:
            self.on_epoch_start()
            self.on_train_epoch_start()
            progress = min(1., (time.monotonic() - start) / self.budget_seconds)
            for group in self.optimizer.param_groups:
                group["lr"] = self.initial_lr * max(.001, 1 - progress) ** .9
            training = []
            for _ in range(self.num_iterations_per_epoch):
                training.append(self.train_step(next(self.dataloader_train)))
                updates += 1
                if time.monotonic() - start >= self.budget_seconds:
                    break
            self.on_train_epoch_end(training)
            with torch.no_grad():
                self.on_validation_epoch_start()
                validation = [self.validation_step(next(self.dataloader_val))
                              for _ in range(self.num_val_iterations_per_epoch)]
                self.on_validation_epoch_end(validation)
            self.on_epoch_end()
        elapsed = time.monotonic() - start
        self.on_train_end()
        report = {"budget_seconds": self.budget_seconds, "elapsed_seconds": round(elapsed, 3),
                  "updates": updates, "seed": self.random_seed, "device": str(self.device)}
        if use_cuda:
            report["peak_cuda_allocated_mib"] = torch.cuda.max_memory_allocated(self.device) / 2 ** 20
        path = Path(self.output_folder) / "budget.json"
        path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
