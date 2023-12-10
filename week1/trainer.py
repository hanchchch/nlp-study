import logging
import math

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from word2vec import Checkpoint, ContextWordsDataset, Word2Vec

logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)


class Trainer:
    def __init__(
        self,
        dataset: ContextWordsDataset,
        model: Word2Vec,
        checkpoint_path="model.pt",
        device=torch.device("cpu"),
        criterion=torch.nn.CrossEntropyLoss(),
        optimizer=None,
        batch_size: int = 512,
        epoch: int = 10,
        train_dataset_total: int = ContextWordsDataset.TOTAL_LENGTH,
    ):
        self.dataset = dataset
        self.model = model
        self.shuffle = True
        self.device = device
        self.criterion = criterion
        self.optimizer = optimizer or torch.optim.AdamW(
            self.model.parameters(), lr=0.0001
        )
        self.batch_size = batch_size
        self.epoch = epoch
        self.train_dataset_total = train_dataset_total
        self.checkpoint = Checkpoint(self.model, checkpoint_path)
        self.prev_epoch, self.checkpoint_loaded = self.checkpoint.load()

    def train(self):
        dataloader = DataLoader(self.dataset, batch_size=self.batch_size)
        for epoch in range(self.epoch):
            with tqdm(
                dataloader,
                unit="batch",
                total=math.ceil(self.train_dataset_total / self.batch_size),
            ) as tepoch:
                for x, y_hat in tepoch:
                    tepoch.set_description(f"Epoch {epoch+1}")

                    x = x.to(self.device)
                    y_hat = y_hat.to(self.device)

                    self.optimizer.zero_grad()

                    output = self.model(x)
                    loss = self.criterion(
                        output, y_hat.view(-1)
                    )  # Reshape y_hat to be 1-dimensional

                    loss.backward()
                    self.optimizer.step()

                    tepoch.set_postfix(loss=f"{loss.item():.3f}")

            self.checkpoint.save(epoch, loss.item())
