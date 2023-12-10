import json
import os

from tqdm import tqdm


def get_word_map(sentences: list[str], tokenizer: callable, total: int = None, cache_path: str = None):
    if cache_path and os.path.exists(cache_path):
        with open(cache_path) as f:
            return json.load(f)
    
    word_map = {}
    for sentence in tqdm(sentences, total=total):
        for word in tokenizer(sentence):
            if word not in word_map:
                word_map[word] = len(word_map)
    
    if cache_path:
        with open(cache_path, "w") as f:
                json.dump(word_map, f)

    return word_map
