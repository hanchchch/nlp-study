import torch


class RNN(torch.nn.Module):
    name = "rnn"

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


class LSTM(torch.nn.Module):
    name = "lstm"

    # without torch.nn.LSTM
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.Wxi = torch.nn.Embedding(input_size, hidden_size, padding_idx=0)
        self.Whi = torch.nn.Linear(hidden_size, hidden_size, bias=True)
        self.Wxf = torch.nn.Embedding(input_size, hidden_size, padding_idx=0)
        self.Whf = torch.nn.Linear(hidden_size, hidden_size, bias=True)
        self.Wxo = torch.nn.Embedding(input_size, hidden_size, padding_idx=0)
        self.Who = torch.nn.Linear(hidden_size, hidden_size, bias=True)
        self.Wxg = torch.nn.Embedding(input_size, hidden_size, padding_idx=0)
        self.Whg = torch.nn.Linear(hidden_size, hidden_size, bias=True)
        self.Why = torch.nn.Linear(hidden_size, output_size, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        h = torch.zeros((batch_size, self.hidden_size), requires_grad=True)
        c = torch.zeros((batch_size, self.hidden_size), requires_grad=True)
        for i in range(x.shape[1]):
            i_t = torch.sigmoid(self.Wxi(x[:, i]) + self.Whi(h))
            f_t = torch.sigmoid(self.Wxf(x[:, i]) + self.Whf(h))
            o_t = torch.sigmoid(self.Wxo(x[:, i]) + self.Who(h))
            g_t = torch.tanh(self.Wxg(x[:, i]) + self.Whg(h))
            c = f_t * c + i_t * g_t
            h = o_t * torch.tanh(c)
        return self.Why(h)
