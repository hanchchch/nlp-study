import torch


class RNN(torch.nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.Wxh = torch.nn.Embedding(input_size, hidden_size, padding_idx=0)
        self.Whh = torch.nn.Linear(hidden_size, hidden_size, bias=True)
        self.Why = torch.nn.Linear(hidden_size, output_size, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        @image https://i.imgur.com/s8nYcww.png
        """
        batch_size = x.shape[0]
        h = torch.zeros((batch_size, self.hidden_size), requires_grad=True)
        for i in range(x.shape[1]):
            h = torch.tanh(self.Wxh(x[:, i]) + self.Whh(h))
        return self.Why(h)
