from tqdm import tqdm


def get_word_map(sentences: list[list[str]], tokenizer: callable, total: int = None):
    word_map = {}
    for sentence in tqdm(sentences, total=total):
        for word in tokenizer(sentence):
            if word not in word_map:
                word_map[word] = len(word_map)
    return word_map
