import os

import torch


class Checkpoint:
    def __init__(self, model: torch.nn.Module, checkpoint_path: str = "model.pt"):
        self.model = model
        self.checkpoint_path = checkpoint_path
        self.data_index = None
        self.loss = None

    def save(self, epoch: int, loss: float, prev_epoch: int = 0, optimizer=None, data_index: int=None):
        self.data_index = data_index
        self.loss = loss
        torch.save(
            {
                "epoch": prev_epoch + epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict()
                if optimizer
                else {},
                "loss": self.loss,
                "data_index": self.data_index,
            },
            self.checkpoint_path,
        )

    def load(self) -> tuple[int, bool]:
        prev_epoch = 0
        loaded = False
        if os.path.exists(self.checkpoint_path):
            checkpoint = torch.load(self.checkpoint_path)
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.data_index = checkpoint.get("data_index", None)
            self.loss = checkpoint.get("loss", None)
            prev_epoch = checkpoint["epoch"]
            loaded = True
        return prev_epoch, loaded
