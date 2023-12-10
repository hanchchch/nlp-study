import torch


class Word2Vec(torch.nn.Module):
    def __init__(
        self,
        word_count: int,
        embedding_dim: int = 100,
    ):
        super().__init__()
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


class Word2VecParallel(Word2Vec):
    def __init__(
        self,
        word_count: int,
        embedding_dim: int = 100,
    ):
        super().__init__(word_count, embedding_dim)
        self.input_to_projection = self.input_to_projection.to("cuda:0")
        self.projection_to_output = self.projection_to_output.to("cuda:0")

    def create_projection(self):
        return super().create_projection().to("cuda:0")
