import logging
import math

import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from common.checkpoint import Checkpoint

logger = logging.getLogger(__name__)


class Trainer:
    collate_fn = None

    def __init__(
        self,
        trainset: Dataset,
        model: callable,
        testset=None,
        trainset_total=None,
        checkpoint_path="model.pt",
        device=torch.device("cpu"),
        criterion=None,
        optimizer=None,
        batch_size=512,
        epoch=10,
    ):
        if criterion is None:
            raise ValueError("criterion must be specified")
        if optimizer is None:
            raise ValueError("optimizer must be specified")

        self.trainset = trainset
        self.testset = testset
        self.model = model
        self.shuffle = True
        self.device = device
        self.criterion = criterion
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.epoch = epoch
        self.trainset_total = trainset_total or len(trainset)
        self.checkpoint = Checkpoint(self.model, checkpoint_path)
        self.prev_epoch, self.checkpoint_loaded = self.checkpoint.load()

    def get_loss(self, x: torch.Tensor, y: torch.Tensor):
        raise NotImplementedError()

    def train(self):
        dataloader = DataLoader(
            self.trainset,
            batch_size=self.batch_size,
            collate_fn=self.collate_fn,
        )
        for epoch in range(self.epoch):
            loss_total = 0
            i = 0

            with tqdm(
                dataloader,
                unit="batch",
                total=math.ceil(self.trainset_total / self.batch_size),
            ) as tepoch:
                for x, y in tepoch:
                    tepoch.set_description(f"Epoch {epoch+1}")

                    x = x.to(self.device)
                    y = y.to(self.device)

                    self.optimizer.zero_grad()
                    loss = self.get_loss(x, y)

                    loss.backward()
                    self.optimizer.step()

                    loss_total += loss.item()
                    i += 1

                    tepoch.set_postfix(loss=f"{loss_total / i:.3f}")

            self.checkpoint.save(epoch + 1, loss_total / i)

    def test(self):
        dataloader = DataLoader(
            self.testset,
            batch_size=self.batch_size,
            collate_fn=self.collate_fn,
        )
        loss_total = 0
        i = 0

        with tqdm(
            dataloader,
            unit="batch",
            total=math.ceil(len(self.testset) / self.batch_size),
        ) as tepoch:
            for x, y in tepoch:
                tepoch.set_description(f"Test")

                x = x.to(self.device)
                y = y.to(self.device)

                loss = self.get_loss(x, y)

                loss_total += loss.item()
                i += 1

                tepoch.set_postfix(loss=f"{loss_total / i:.3f}")

        return loss_total / i