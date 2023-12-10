class Preprocessor:
    def __init__(self, tokenizer: callable, window_size: int = 2):
        self.tokenizer = tokenizer
        self.window_size = window_size

    def get_context_words(self, words: list[str], word_index: int) -> list[str]:
        return [
            words[i]
            for i in range(
                max(word_index - self.window_size, 0),
                min(word_index + self.window_size, len(words)),
            )
            if i != word_index
        ]

    def preprocess(self, sentences: list[str]):
        for sentence in sentences:
            words = self.tokenizer(sentence)
            if len(words) == 0:
                continue

            for word_index, word in enumerate(words):
                context_words = self.get_context_words(words, word_index)
                yield (context_words, word)
