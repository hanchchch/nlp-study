import logging

import torch
from torch.nn.utils.rnn import pad_sequence

from common.trainer import Trainer

logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)


class RNNTrainer(Trainer):
    def get_loss(self, x: torch.Tensor, y: torch.Tensor):
        output = self.model(x)
        return self.criterion(output, y)

    def collate_fn(self, batch):
        x, y = zip(*batch)
        x = pad_sequence(x, batch_first=True)
        y = torch.stack(y)
        return x, y

    def on_epoch_done(self, epoch: int, loss: float):
        super().on_epoch_done(epoch, loss)
        with open(f"week2/loss_{self.model.name}.csv", "a") as f:
            test_loss = self.test()
            f.write(f"{epoch},{test_loss}\n")
