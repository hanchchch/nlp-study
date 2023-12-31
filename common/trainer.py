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
        self.criterion = criterion
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.epoch = epoch
        self.trainset_total = trainset_total or len(trainset)
        self.checkpoint = Checkpoint(self.model, checkpoint_path)
        self.prev_epoch, self.checkpoint_loaded = self.checkpoint.load()

    def get_loss(self, x: torch.Tensor, y: torch.Tensor):
        raise NotImplementedError()

    def on_epoch_done(self, epoch: int, loss: float):
        self.checkpoint.save(epoch, loss, self.prev_epoch, self.optimizer)

    def train(self):
        dataloader = DataLoader(
            self.trainset,
            batch_size=self.batch_size,
            collate_fn=self.collate_fn,
        )
        loss_total = 0
        i = 0
        skipped = False
        try:
            for epoch in range(self.prev_epoch, self.epoch):
                loss_total = 0
                i = 0

                with tqdm(
                    dataloader,
                    unit="batch",
                    total=math.ceil(self.trainset_total / self.batch_size),
                ) as tepoch:
                    for x, y in tepoch:
                        if not skipped and self.checkpoint.data_index is not None:
                            if i < self.checkpoint.data_index:
                                i += 1
                                continue
                            else:
                                loss_total = self.checkpoint.loss * i
                                skipped = True

                        tepoch.set_description(f"Epoch {epoch+1}")

                        self.optimizer.zero_grad()
                        loss = self.get_loss(x, y)

                        loss.backward()
                        self.optimizer.step()

                        loss_total += loss.item()
                        i += 1

                        tepoch.set_postfix(loss=f"{loss_total / i:.3f}")

                self.on_epoch_done(epoch + 1, loss_total / i)
        except KeyboardInterrupt:
            if i > 0 and loss_total > 0:
                logger.info("interrupted, saving checkpoint")
                self.checkpoint.save(epoch, loss_total / i, self.prev_epoch, self.optimizer, i)


    def test(self):
        dataloader = DataLoader(
            self.testset,
            batch_size=self.batch_size,
            collate_fn=self.collate_fn,
        )
        loss_total = 0
        i = 0

        with torch.no_grad():
            with tqdm(
                dataloader,
                unit="batch",
                total=math.ceil(len(self.testset) / self.batch_size),
            ) as tepoch:
                for x, y in tepoch:
                    tepoch.set_description(f"Test")

                    loss = self.get_loss(x, y)

                    loss_total += loss.item()
                    i += 1

                    tepoch.set_postfix(loss=f"{loss_total / i:.3f}")

        return loss_total / i