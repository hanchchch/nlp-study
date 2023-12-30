import logging

import torch
from torch.nn.utils.rnn import pad_sequence

from common.trainer import Trainer

logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)


class TransformerTrainer(Trainer):
    def get_loss(self, x: torch.Tensor, y: torch.Tensor):
        output = self.model(x)
        return self.criterion(output, y)

    def collate_fn(self, batch):
        x, y = zip(*batch)
        x = pad_sequence(x, batch_first=True)
        y = torch.stack(y)
        return x, y
