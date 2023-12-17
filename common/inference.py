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

    def search(self, query: torch.Tensor, top_k: int = 10):
        similarity = torch.cosine_similarity(
            query, self.model.input_to_projection.weight, dim=1
        )
        top_k_similarities, top_k_indices = torch.topk(similarity, top_k)
        return [
            (self.vocab[index], similarity.item())
            for similarity, index in zip(top_k_similarities, top_k_indices)
        ]

    def embed(self, sentence: str) -> torch.Tensor:
        token_ids = self.vocab(sentence)
        return self.model.input_to_projection(
            torch.tensor(token_ids, dtype=torch.long).to(self.device)
        )

    def infer(self, sentence: str, top_k: int = 10):
        self.model.eval()

        with torch.no_grad():
            output = self.embed(sentence)
            return self.search(output, top_k)
