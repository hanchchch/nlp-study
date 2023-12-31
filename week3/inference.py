import logging

import torch

from common.checkpoint import Checkpoint
from week3.core.dataset import WMTEnDeDataset

logger = logging.getLogger(__name__)


class Inference:
    def __init__(
        self,
        model: callable,
        dataset: WMTEnDeDataset,
        checkpoint_path="model.pt",
    ):
        self.model = model
        self.dataset = dataset
        self.checkpoint = Checkpoint(self.model, checkpoint_path)
        self.prev_epoch, self.checkpoint_loaded = self.checkpoint.load()

    def infer(self, sentence: str, max_length: int = 50):
        self.model.eval()

        with torch.no_grad():
            tokens = torch.tensor(self.dataset.vocab_en(self.dataset.wrap(sentence)), dtype=torch.long)
            tokens = torch.cat([tokens, torch.zeros(max_length - tokens.shape[0], dtype=torch.long)])
            output = torch.tensor(self.dataset.vocab_de(self.dataset.START_TOKEN), dtype=torch.long)


            for i in range(max_length):
                predictions = self.model(
                    tokens[None,:],
                    torch.cat([output, torch.zeros(max_length - output.shape[0], dtype=torch.long)])[None,:],
                )

                predictions = predictions[:, -1:, :]
                predicted_id = torch.argmax(predictions, dim=-1)[0]

                if self.dataset.vocab_de(self.dataset.START_TOKEN)[0] == predicted_id[0]:
                    break
                
                output = torch.cat([output, predicted_id], dim=-1)

            return output
