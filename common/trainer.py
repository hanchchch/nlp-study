import logging
import math

import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from common.checkpoint import Checkpoint

logger = logging.getLogger(__name__)


class Trainer:
    def __init__(
        self,
        dataset: Dataset,
        model: callable,
        train_dataset_total=None,
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

        self.dataset = dataset
        self.model = model
        self.shuffle = True
        self.device = device
        self.criterion = criterion
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.epoch = epoch
        self.train_dataset_total = train_dataset_total or len(dataset)
        self.checkpoint = Checkpoint(self.model, checkpoint_path)
        self.prev_epoch, self.checkpoint_loaded = self.checkpoint.load()

    def get_loss(self, x, y):
        raise NotImplementedError()

    def train(self):
        dataloader = DataLoader(self.dataset, batch_size=self.batch_size)
        for epoch in range(self.epoch):
            with tqdm(
                dataloader,
                unit="batch",
                total=math.ceil(self.train_dataset_total / self.batch_size),
            ) as tepoch:
                for x, y in tepoch:
                    tepoch.set_description(f"Epoch {epoch+1}")

                    x = x.to(self.device)
                    y = y.to(self.device)

                    self.optimizer.zero_grad()
                    loss = self.get_loss(x, y)

                    loss.backward()
                    self.optimizer.step()

                    tepoch.set_postfix(loss=f"{loss.item():.3f}")

            self.checkpoint.save(epoch, loss.item())
