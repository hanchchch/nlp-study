import math
import os
import time

import torch
from torch.utils.data import DataLoader
from torchtext.data.utils import get_tokenizer
from torchtext.datasets import WikiText103
from tqdm import tqdm
from word2vec import Preprocessor, Word2Vec, get_word_map

os.chdir(os.path.dirname(__file__))

train_dataset, valid_dataset, test_dataset = WikiText103(
    root=".data", split=("train", "valid", "test")
)
train_dataset_total = 1_801_350

tokenizer_name = "basic_english"
word_map_cache = f"word_map_{tokenizer_name}.json"

tokenizer = get_tokenizer(tokenizer_name)
word_map = get_word_map(train_dataset, tokenizer=tokenizer, total=train_dataset_total, cache_path=word_map_cache)

window_size = 2

preprocessor = Preprocessor(tokenizer=tokenizer, window_size=window_size)
model = Word2Vec(word_map)


criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

batch_size = math.ceil(train_dataset_total/100)
dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)


EPOCH = 1
CHECKPOINT_PATH = "model.pt"

for epoch in range(EPOCH):
    for batchdata in dataloader:
        optimizer.zero_grad()
        loss = 0
        
        for (context_words, word) in tqdm(preprocessor.preprocess(batchdata), total=batch_size):
            start = time.time()
            output = model(context_words)
            label = model.one_hot(word)
            loss += criterion(output, label)
            end = time.time()

        loss.backward()
        optimizer.step()

        print(f"{loss=}")

print("done training, saving model")

torch.save({
    'epoch': EPOCH,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': loss,
}, CHECKPOINT_PATH)
