import torch


class Word2Vec(torch.nn.Module):
    def __init__(
        self,
        word_map: dict[str, int],
        embedding_dim: int = 100,
    ):
        super().__init__()
        self.word_map = word_map
        self.embedding_dim = embedding_dim
        self.input_to_projection = torch.nn.Embedding(len(word_map), embedding_dim)
        self.projection_to_output = torch.nn.Embedding(embedding_dim, len(word_map))
        self.softmax = torch.nn.Softmax(dim=1)

    def one_hot(self, word: str) -> torch.Tensor:
        index = self.word_map[word]
        one_hot = torch.zeros(1, len(self.word_map))
        one_hot[0][index] = 1
        return one_hot

    def _forward(self, context_words: list[str]) -> torch.Tensor:
        # input -> projection
        projection = torch.zeros(1, self.embedding_dim)
        for context_word in context_words:
            projection += self.one_hot(context_word) @ self.input_to_projection.weight
        projection /= len(context_words)

        # projection -> output
        output = projection @ self.projection_to_output.weight
        return self.softmax(output)
    
    def forward(self, context_words: list[str]) -> torch.Tensor:
        return self._forward(context_words)

