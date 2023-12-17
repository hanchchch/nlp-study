import logging

import torch

from common.trainer import Trainer

logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)


class RNNTrainer(Trainer):
    def get_loss(self, x: torch.Tensor, y: torch.Tensor):
        output = self.model(x)
        return self.criterion(output, y)
