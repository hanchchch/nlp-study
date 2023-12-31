import argparse
import logging
import os
import sys

import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from week3.core.dataset import WMTEnDeDataset
from week3.core.model import Transformer
from week3.trainer import TransformerTrainer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

learning_rate = 0.0001
epochs = 100


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["train", "infer", "test"])
    parser.add_argument("-g", "--gpu", type=bool, default=torch.cuda.is_available())
    args = parser.parse_args()

    if args.gpu and not torch.cuda.is_available():
        logger.warning("gpu is not available, use cpu instead")
        args.gpu = False

    device = torch.device("cuda" if args.gpu else "cpu")
    torch.set_default_device(device)
    logger.info(f"device: {device}")

    trainset = WMTEnDeDataset(root=".data", split="train")
    testset = WMTEnDeDataset(root=".data", split="test")
    logger.info(f"dataset loaded, {trainset.vocab_en.vocab_count()} vocabs")

    model = Transformer()
    checkpoint_path = f"week3/model.pt"

    trainer = TransformerTrainer(
        trainset=trainset,
        testset=testset,
        model=model,
        criterion=torch.nn.CrossEntropyLoss(),
        optimizer=torch.optim.AdamW(model.parameters(), lr=learning_rate),
        checkpoint_path=checkpoint_path,
        batch_size=64,
        epoch=epochs,
    )

    if args.mode == "train":
        if trainer.checkpoint_loaded:
            logger.info(f"checkpoint loaded, epoch: {trainer.prev_epoch}")

        trainer.train()
