import argparse
import logging
import os
import sys

import torch
from torchtext.data.utils import get_tokenizer

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common.vocab import Vocab
from week2.core.dataset import AGNewsDataset
from week2.core.model import RNN
from week2.inference import Inference
from week2.trainer import RNNTrainer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

tokenizer_name = "basic_english"
vocab_cache_path = f"week2/vocab_{tokenizer_name}.pth"
tokenizer = get_tokenizer(tokenizer_name)

checkpoint_path = "week2/model.pt"

hidden_size = 100
learning_rate = 0.001


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["train", "infer"])
    parser.add_argument("-s", "--sentence", type=str, default="")
    parser.add_argument("-g", "--gpu", type=bool, default=torch.cuda.is_available())
    args = parser.parse_args()

    if args.gpu and not torch.cuda.is_available():
        logger.warning("gpu is not available, use cpu instead")
        args.gpu = False

    device = torch.device("cuda" if args.gpu else "cpu")
    logger.info(f"device: {device}")
    dataset = AGNewsDataset(root=".data", split="train")
    vocab = Vocab(
        dataset.sentences(),
        tokenizer=tokenizer,
        vocab_cache_path=vocab_cache_path,
    )
    dataset.set_vocab(vocab)
    logger.info(f"dataset loaded, {vocab.vocab_count()} vocabs")

    model = RNN(
        input_size=vocab.vocab_count(),
        hidden_size=hidden_size,
        output_size=dataset.TOTAL_LABELS,
    )

    if args.mode == "train":
        trainer = RNNTrainer(
            dataset=dataset,
            model=model,
            criterion=torch.nn.CrossEntropyLoss(),
            optimizer=torch.optim.AdamW(model.parameters(), lr=learning_rate),
            checkpoint_path=checkpoint_path,
        )
        if trainer.checkpoint_loaded:
            logger.info(f"checkpoint loaded, epoch: {trainer.prev_epoch}")
        trainer.train()

    elif args.mode == "infer":
        if not args.sentence:
            print("usage: python main.py infer <sentence>")
            exit(1)

        inference = Inference(
            model,
            vocab,
            device=device,
            checkpoint_path=checkpoint_path,
        )
        if not inference.checkpoint_loaded:
            logger.error("no checkpoint loaded")
            exit(1)

        index = inference.infer(args.sentence)
        logger.info(f"{index+1}: {dataset.LABELS[index]}")

    else:
        raise ValueError("invalid mode")
