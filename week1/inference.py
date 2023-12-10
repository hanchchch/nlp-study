import logging

import torch
from word2vec import Checkpoint, Word2Vec

logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)


class Inference:
    def __init__(
        self,
        model: Word2Vec,
        sentence_to_token_ids: callable,
        token_id_to_word: callable,
        checkpoint_path="model.pt",
        device=torch.device("cpu"),
    ):
        self.model = model
        self.device = device
        self.sentence_to_token_ids = sentence_to_token_ids
        self.token_id_to_word = token_id_to_word
        self.checkpoint = Checkpoint(self.model, checkpoint_path)
        self.prev_epoch, self.checkpoint_loaded = self.checkpoint.load()

    def search(self, query: torch.Tensor, top_k: int = 10):
        similarity = torch.cosine_similarity(
            query, self.model.input_to_projection.weight, dim=1
        )
        return torch.topk(similarity, top_k)

    def embed(self, sentence: str):
        token_ids = self.sentence_to_token_ids(sentence)
        return self.model.input_to_projection(
            torch.tensor(token_ids, dtype=torch.long).to(self.device)
        )

    def infer(self, sentence: str, top_k: int = 10):
        self.model.eval()

        with torch.no_grad():
            output = self.embed(sentence)
            top_k_similarities, top_k_indices = self.search(output, top_k)

            return [
                (self.token_id_to_word(index), similarity.item())
                for similarity, index in zip(top_k_similarities, top_k_indices)
            ]
