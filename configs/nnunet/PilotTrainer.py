"""Run the standard nnU-Net trainer with a fixed development update budget."""

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer


class PilotTrainer(nnUNetTrainer):
    def __init__(self, plans, configuration, fold, dataset_json, device):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 2
        self.num_iterations_per_epoch = 240
        self.num_val_iterations_per_epoch = 12
