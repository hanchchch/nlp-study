import logging
import os

import torch
from torchtext.vocab import build_vocab_from_iterator

logger = logging.getLogger(__name__)


class Vocab:
    def __init__(
        self,
        sentences: list[str],
        tokenizer: callable,
        vocab_cache_path: str = None,
    ):
        self.tokenizer = tokenizer
        self.vocab_cache_path = vocab_cache_path

        self.vocab = self._load_cache()
        if self.vocab is None:
            self.vocab = self._create_vocab(sentences)
            logger.info("building vocab")
            self._save_cache()
        else:
            logger.info(f"loaded vocab from {self.vocab_cache_path}")

    def _load_cache(self):
        if self.vocab_cache_path and os.path.exists(self.vocab_cache_path):
            return torch.load(self.vocab_cache_path)
        return None

    def _save_cache(self):
        if self.vocab_cache_path:
            torch.save(self.vocab, self.vocab_cache_path)

    def _create_vocab(self, sentences: list[str]):
        vocab = build_vocab_from_iterator(
            self.yield_tokens(sentences),
            specials=["<unk>"],
            min_freq=50,
        )
        vocab.set_default_index(vocab["<unk>"])
        return vocab

    def sentence_to_token_ids(self, sentence: str) -> list[int]:
        return self.vocab(self.tokenizer(sentence))

    def token_id_to_word(self, token_id: int) -> str:
        return self.vocab.get_itos()[token_id]

    def yield_tokens(self, sentences: list[str]):
        for sentence in sentences:
            yield self.tokenizer(sentence)

    def vocab_count(self):
        return len(self.vocab)

    def __call__(self, sentence: str) -> list[int]:
        return self.sentence_to_token_ids(sentence)

    def __getitem__(self, token_id: int) -> str:
        return self.token_id_to_word(token_id)
