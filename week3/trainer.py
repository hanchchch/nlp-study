import logging

import torch
from torch.nn.utils.rnn import pad_sequence

from common.trainer import Trainer

logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)


class TransformerTrainer(Trainer):
    pad_token_id = 0

    def make_x_y_even_seq_len(self, x: torch.Tensor, y: torch.Tensor):
        x_len = x.shape[1]
        y_len = y.shape[1]
        if x_len > y_len:
            y = torch.cat(
                [y, torch.zeros((y.shape[0], x_len - y_len), dtype=torch.long)], dim=1
            )
        elif x_len < y_len:
            x = torch.cat(
                [x, torch.zeros((x.shape[0], y_len - x_len), dtype=torch.long)], dim=1
            )
        return x, y

    def get_loss(self, x: torch.Tensor, y: torch.Tensor):
        x, y = self.make_x_y_even_seq_len(x, y)
        output = self.model(x, y)
        return self.criterion(output, y)

    def collate_fn(self, batch):
        x, y = zip(*batch)
        x = pad_sequence(x, batch_first=True, padding_value=self.pad_token_id)
        y = pad_sequence(y, batch_first=True, padding_value=self.pad_token_id)
        return x, y
