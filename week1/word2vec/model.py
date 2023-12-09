import torch


class Word2Vec(torch.nn.Module):
    def __init__(
        self,
        word_map: dict[str, int],
        window_size: int = 2,
        embedding_dim: int = 100,
    ):
        super().__init__()
        self.word_map = word_map
        self.window_size = window_size
        self.embedding_dim = embedding_dim
        self.input_to_projection = torch.nn.Embedding(len(word_map), embedding_dim)
        self.projection_to_output = torch.nn.Embedding(embedding_dim, len(word_map))
        self.softmax = torch.nn.Softmax(dim=1)

    def one_hot(self, word: str) -> torch.Tensor:
        index = self.word_map[word]
        one_hot = torch.zeros(1, len(self.word_map))
        one_hot[0][index] = 1
        return one_hot

    def get_context_words(self, words: list[str], word_index: int) -> list[str]:
        return [
            words[i]
            for i in range(
                max(word_index - self.window_size, 0),
                min(word_index + self.window_size, len(words)),
            )
            if i != word_index
        ]

    def forward(self, words: list[str], word_index: int) -> torch.Tensor:
        context_words = self.get_context_words(words, word_index)

        # input -> projection
        projection = torch.zeros(1, self.embedding_dim)
        for context_word in context_words:
            projection += self.one_hot(context_word) @ self.input_to_projection.weight
        projection /= len(context_words)

        # projection -> output
        output = projection @ self.projection_to_output.weight
        return self.softmax(output)
