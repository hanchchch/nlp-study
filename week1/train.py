import logging
import math
import os

import torch
from torch.utils.data import DataLoader
from torchtext.data.utils import get_tokenizer
from tqdm import tqdm
from word2vec import ContextWordsDataset, Word2Vec

logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)

os.chdir(os.path.dirname(__file__))

tokenizer_name = "basic_english"
vocab_cache_path = f"vocab_{tokenizer_name}.pth"
tokenizer = get_tokenizer(tokenizer_name)

window_size = 2
use_gpu = True


logger.info("loading dataset")
dataset = ContextWordsDataset(
    root=".data",
    split="train",
    tokenizer=tokenizer,
    window_size=window_size,
    vocab_cache_path=vocab_cache_path,
)
logger.info(f"dataset loaded, {dataset.get_vocab_count()} vocabs")
train_dataset_total = len(dataset)

device = torch.device("cuda" if use_gpu else "cpu")
model = Word2Vec(word_count=dataset.get_vocab_count()).to(device)

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001)

batch_size = 512
shuffle = True

dataloader = DataLoader(dataset, batch_size=batch_size)

EPOCH = 10
CHECKPOINT_PATH = "model.pt"

if os.path.exists(CHECKPOINT_PATH):
    checkpoint = torch.load(CHECKPOINT_PATH)
    model.load_state_dict(checkpoint["model_state_dict"])
    prev_epoch = checkpoint["epoch"]

def train():
    for epoch in range(EPOCH):
        with tqdm(dataloader, unit="batch", total=math.ceil(train_dataset_total / batch_size)) as tepoch:
            for x, y_hat in tepoch:
                tepoch.set_description(f"Epoch {epoch+1}")

                x = x.to(device)
                y_hat = y_hat.to(device)

                optimizer.zero_grad()

                output = model(x)
                loss = criterion(output, y_hat.view(-1))  # Reshape y_hat to be 1-dimensional

                loss.backward()
                optimizer.step()
                
                tepoch.set_postfix(loss=f"{loss.item():.3f}")

        torch.save(
            {
                "epoch": prev_epoch + epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": loss,
            },
            CHECKPOINT_PATH,
        )

def infer(sentence: str, top_k: int = 10):
    model.eval()

    with torch.no_grad():
        tokens = dataset.sentence_to_token_ids(sentence)

        output = model.input_to_projection(torch.tensor(tokens, dtype=torch.long).to(device))
        # cosine similarity
        similarity = torch.cosine_similarity(output, model.input_to_projection.weight, dim=1)
        top_k_similarities, top_k_indices = torch.topk(similarity, top_k)

        return [(dataset.token_id_to_word(index), similarity.item()) for similarity, index in zip(top_k_similarities, top_k_indices)]
            
    

if __name__ == "__main__":
    train()