import logging
import os

import torch
from inference import Inference
from torchtext.data.utils import get_tokenizer
from trainer import Trainer
from word2vec import ContextWordsDataset, Word2Vec

logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)

os.chdir(os.path.dirname(__file__))

tokenizer_name = "basic_english"
vocab_cache_path = f"vocab_{tokenizer_name}.pth"
tokenizer = get_tokenizer(tokenizer_name)

window_size = 2
use_gpu = True

checkpoint_path = "model.pt"

logger.info("loading dataset")
dataset = ContextWordsDataset(
    root=".data",
    split="train",
    tokenizer=tokenizer,
    window_size=window_size,
    vocab_cache_path=vocab_cache_path,
)
logger.info(f"dataset loaded, {dataset.get_vocab_count()} vocabs")

device = torch.device("cuda" if use_gpu else "cpu")
logger.info(f"device: {device}")

model = Word2Vec(word_count=dataset.get_vocab_count()).to(device)
trainer = Trainer(
    dataset,
    model=model,
    device=device,
    criterion=torch.nn.CrossEntropyLoss(),
    optimizer=torch.optim.AdamW(model.parameters(), lr=0.0001),
    batch_size=512,
    epoch=10,
    train_dataset_total=len(dataset),
    checkpoint_path=checkpoint_path,
)
if trainer.checkpoint_loaded:
    logger.info(f"checkpoint loaded, epoch: {trainer.prev_epoch}")

inference = Inference(
    model,
    sentence_to_token_ids=dataset.sentence_to_token_ids,
    token_id_to_word=dataset.token_id_to_word,
    device=device,
    checkpoint_path=checkpoint_path,
)
