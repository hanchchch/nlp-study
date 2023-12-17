import os

import torch


class Checkpoint:
    def __init__(self, model: torch.nn.Module, checkpoint_path: str = "model.pt"):
        self.model = model
        self.checkpoint_path = checkpoint_path

    def save(self, epoch: int, loss: float, prev_epoch: int = 0, optimizer=None):
        torch.save(
            {
                "epoch": prev_epoch + epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict()
                if optimizer
                else {},
                "loss": loss,
            },
            self.checkpoint_path,
        )

    def load(self) -> tuple[int, bool]:
        prev_epoch = 0
        loaded = False
        if os.path.exists(self.checkpoint_path):
            checkpoint = torch.load(self.checkpoint_path)
            self.model.load_state_dict(checkpoint["model_state_dict"])
            prev_epoch = checkpoint["epoch"]
            loaded = True
        return prev_epoch, loaded
