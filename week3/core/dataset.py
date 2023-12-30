import os.path as osp

from torch.utils.data import IterableDataset

from common.vocab import Vocab


class WMTEnDeDataset(IterableDataset):
    DIRNAME = "wmt14"
    VOCAB_FILENAME = "vocab.50K"
    VOCAB_CACHENAME = "vocab.pt"
    TEST_FILENAMES = ["newstest2012", "newstest2013", "newstest2014", "newstest2015"]


    def __init__(self, root: str, split: str = "train"):
        self.root = root
        self.split = split
        self.vocab_en = self.get_vocab("en")
        self.vocab_de = self.get_vocab("de")

    def get_filename(self, purpose: str, lang: str) -> str:
        return osp.join(self.root, WMTEnDeDataset.DIRNAME, f"{purpose}.{lang}")

    def get_vocab(self, lang: str) -> Vocab:
        with open(self.get_filename(WMTEnDeDataset.VOCAB_FILENAME, lang)) as f:
            v = f.readlines()
            return Vocab(v, vocab_cache_path=self.get_filename(WMTEnDeDataset.VOCAB_CACHENAME, lang), min_freq=1)
    
    def iter(self, f_en, f_de):
        eof = False
        while not eof:
            en = f_en.readline()
            de = f_de.readline()
            if not en or not de:
                eof = True
                continue
            yield (en, de)

    def __iter__(self):
        if self.split == "train":
            f_en = open(self.get_filename("train", "en"))
            f_de = open(self.get_filename("train", "de"))
            yield from self.iter(f_en, f_de)
        elif self.split == "test":
            for test in WMTEnDeDataset.TEST_FILENAMES:
                f_en = open(self.get_filename(test, "en"))
                f_de = open(self.get_filename(test, "de"))
                yield from self.iter(f_en, f_de)
        else:
            raise ValueError(f"unknown split: {self.split}")
    