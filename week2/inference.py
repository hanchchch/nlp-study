import logging

import torch

from common.checkpoint import Checkpoint
from common.vocab import Vocab

logger = logging.getLogger(__name__)


class Inference:
    def __init__(
        self,
        model: callable,
        vocab: Vocab,
        checkpoint_path="model.pt",
        device=torch.device("cpu"),
    ):
        self.model = model
        self.device = device
        self.vocab = vocab
        self.checkpoint = Checkpoint(self.model, checkpoint_path)
        self.prev_epoch, self.checkpoint_loaded = self.checkpoint.load()

    def infer(self, sentence: str):
        self.model.eval()

        with torch.no_grad():
            token_ids = self.vocab(sentence)
            output = self.model(
                torch.tensor([token_ids], dtype=torch.long).to(
                    self.device
                )  # batch size 1
            )
            return torch.argmax(output)
