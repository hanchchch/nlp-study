import logging
import os

import torch
from torch.utils.data import DataLoader
from torchtext.data.utils import get_tokenizer
from tqdm import tqdm
from word2vec import ContextWordsDataset, Word2Vec

logging.basicConfig(level=logging.INFO)

os.chdir(os.path.dirname(__file__))

tokenizer_name = "basic_english"
vocab_cache_path = f"vocab_{tokenizer_name}.pth"
tokenizer = get_tokenizer(tokenizer_name)

window_size = 2
use_gpu = False


dataset = ContextWordsDataset(root=".data", split="train", tokenizer=tokenizer, window_size=window_size, vocab_cache_path=vocab_cache_path)
model = Word2Vec(word_count=dataset.get_vocab_count())


criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

batch_size = 100 #  math.ceil(train_dataset_total/10000)
dataloader = DataLoader(dataset, batch_size=batch_size)

EPOCH = 1
CHECKPOINT_PATH = "model.pt"

for epoch in range(EPOCH):
    with tqdm(dataloader, unit="batch") as tepoch:
        for (x, y_hat) in tepoch:
            tepoch.set_description(f"Epoch {epoch+1}")

            optimizer.zero_grad()

            output = model(x)
            loss = criterion(output, y_hat.view(-1))  # Reshape y_hat to be 1-dimensional

            loss.backward()
            optimizer.step()

            tepoch.set_postfix(loss=loss.item())



print("done training, saving model")

torch.save({
    'epoch': EPOCH,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': loss,
}, CHECKPOINT_PATH)