import logging
import os

import torch
from torch.utils.data import IterableDataset
from torchtext.datasets import WikiText2
from torchtext.vocab import build_vocab_from_iterator

logger = logging.getLogger(__name__)


class ContextWordsDataset(IterableDataset):
    TOTAL_SENTENCES = 1_801_350
    TOTAL_LENGTH = 96_928_391

    def __init__(
        self,
        root: str,
        split: str,
        tokenizer: callable,
        window_size: int = 2,
        vocab_cache_path: str = None,
    ):
        self.split = split
        self.tokenizer = tokenizer
        self.window_size = window_size
        self.vocab_cache_path = vocab_cache_path

        train, valid, test = WikiText2(root=root, split=("train", "valid", "test"))
        self.train = train
        self.valid = valid
        self.test = test

        self.vocab = None
        self.vocab = self.create_vocab()

    def __len__(self):
        return self.TOTAL_LENGTH
    
    def __iter__(self):
        for sentence in self.sentences():
            tokens = self.sentence_to_token_ids(sentence)
            if len(tokens) < self.window_size * 2 + 1:
                continue

            for idx in range(len(tokens) - self.window_size * 2):
                token_id_sequence = tokens[idx : (idx + self.window_size * 2 + 1)]
                output = token_id_sequence.pop(self.window_size)
                inputs = token_id_sequence
                yield (
                    torch.tensor(inputs, dtype=torch.long),
                    torch.tensor(output, dtype=torch.long),
                )

    def sentence_to_token_ids(self, sentence: str):
        return self.vocab(self.tokenizer(sentence))
    
    def token_id_to_word(self, token_id: int):
        return self.vocab.get_itos()[token_id]

    def sentences(self):
        if self.split == "train":
            return self.train
        elif self.split == "valid":
            return self.valid
        elif self.split == "test":
            return self.test
        else:
            raise KeyError(f"wrong split: {self.split}")

    def yield_tokens(self, sentences: list[str]):
        for sentence in sentences:
            yield self.tokenizer(sentence)

    def create_vocab(self):
        if self.vocab:
            return self.vocab

        if self.vocab_cache_path and os.path.exists(self.vocab_cache_path):
            logger.info(f"loading vocab from {self.vocab_cache_path}")
            return torch.load(self.vocab_cache_path)

        logger.info("building vocab")
        vocab = build_vocab_from_iterator(
            self.yield_tokens(self.sentences()),
            specials=["<unk>"],
            min_freq=50,
        )
        vocab.set_default_index(vocab["<unk>"])

        if self.vocab_cache_path:
            torch.save(vocab, self.vocab_cache_path)

        return vocab

    def get_vocab_count(self):
        return len(self.vocab)