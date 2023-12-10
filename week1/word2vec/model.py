import torch


class Word2Vec(torch.nn.Module):
    def __init__(
        self,
        word_count: int,
        embedding_dim: int = 300,
    ):
        super().__init__()
        self.word_count = word_count
        self.embedding_dim = embedding_dim
        self.input_to_projection = torch.nn.Embedding(word_count, embedding_dim)
        self.projection_to_output = torch.nn.Linear(
            embedding_dim, word_count, dtype=torch.float32
        )
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # input -> projection
        projection = self.input_to_projection(x)
        projection = torch.mean(projection, dim=1)

        # projection -> output
        output = self.projection_to_output(projection)
        return self.softmax(output)
