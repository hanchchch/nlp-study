import logging
import os

import torch
from torch.utils.data import IterableDataset
from torchtext.datasets import WikiText103
from torchtext.vocab import build_vocab_from_iterator

logger = logging.getLogger(__name__)

class ContextWordsDataset(IterableDataset):
    TOTAL_SENTENCES = 1_801_350

    def __init__(self, root: str, split: str, tokenizer: callable, window_size: int = 2, vocab_cache_path: str = None):
        self.split = split
        self.tokenizer = tokenizer
        self.window_size = window_size
        self.vocab_cache_path = vocab_cache_path

        train, valid, test = WikiText103(
            root=root, split=("train", "valid", "test")
        )
        self.train = train
        self.valid = valid
        self.test = test
        
        self.vocab = None
        self.vocab = self.get_vocab()

    def __iter__(self):
        for sentence in self.get_sentences():
            words = self.tokenizer(sentence)
            if len(words) == 0:
                continue

            for word_index, word in enumerate(words):
                context_words = self.get_context_words(words, word_index)
                if len(context_words) != 4:
                    continue

                yield (torch.cat([self.one_hot_encode(w) for w in context_words]), self.one_hot_encode(word))

    def get_sentences(self):
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

    def get_vocab(self):
        if self.vocab:
            return self.vocab
        
        if self.vocab_cache_path and os.path.exists(self.vocab_cache_path):
            logger.info(f"loading vocab from {self.vocab_cache_path}")
            return torch.load(self.vocab_cache_path)

        logger.info("building vocab")
        vocab = build_vocab_from_iterator(
            self.yield_tokens(self.get_sentences()),
            specials=["<unk>"],
            min_freq=2,
            max_tokens=30000,
        )
        vocab.set_default_index(vocab["<unk>"])
        
        if self.vocab_cache_path:
            torch.save(self.vocab, self.vocab_cache_path)

        return vocab
    
    def get_vocab_count(self) -> int:
        return len(self.get_vocab())

    def get_context_words(self, words: list[str], word_index: int) -> list[str]:
        return [
            words[i]
            for i in range(
                max(word_index - self.window_size, 0),
                min(word_index + self.window_size + 1, len(words)),
            )
            if i != word_index
        ]

    def one_hot_encode(self, word: str) -> torch.Tensor:
        index = self.vocab[word]
        one_hot = torch.zeros(1, len(self.vocab), dtype=torch.int)
        one_hot[0][index] = 1
        return torch.Tensor([index]).long()

