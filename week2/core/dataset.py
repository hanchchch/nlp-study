import logging

import torch
from torch.utils.data import IterableDataset
from torchtext.datasets import AG_NEWS

from common.vocab import Vocab

logger = logging.getLogger(__name__)


class AGNewsDataset(IterableDataset):
    LABELS = ["World", "Sports", "Business", "Sci/Tech"]

    def __init__(self, root: str, split: str):
        self.split = split
        train, test = AG_NEWS(root=root, split=("train", "test"))
        self.train = train
        self.test = test
        self.vocab = None
        self.max_x_size = None

    def __len__(self):
        if self.split == "train":
            return 120_000
        elif self.split == "test":
            return 7_600
        else:
            raise KeyError(f"wrong split: {self.split}")
        

    def __iter__(self):
        if self.vocab is None or self.max_x_size is None:
            raise ValueError("vocab must be set")

        for label, sentence in self.rows():
            x = torch.tensor(self.vocab(sentence), dtype=torch.long)
            y = self._label_encode(label)
            yield x, y

    def _label_encode(self, label: int) -> torch.Tensor:
        one_hot = torch.zeros(len(self.LABELS), dtype=torch.float)
        one_hot[label - 1] = 1
        return one_hot

    def set_vocab(self, vocab: Vocab):
        self.vocab = vocab
        self.max_x_size = 0
        for _, sentence in self.rows():
            self.max_x_size = max(self.max_x_size, len(self.vocab(sentence)))

    def sentences(self):
        for _, sentence in self.rows():
            yield sentence

    def rows(self):
        if self.split == "train":
            return self.train
        elif self.split == "test":
            return self.test
        else:
            raise KeyError(f"wrong split: {self.split}")
