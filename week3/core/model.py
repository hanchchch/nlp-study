import torch


class PositionalEncoding(torch.nn.Module):
    def __init__(self, position: torch.Tensor, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.dropout = torch.nn.Dropout(dropout)
        self.pos_encoding = self.positional_encoding(position, d_model)

    def get_angles(self, position: torch.Tensor, i: torch.Tensor, d_model: int):
        angles = 1 / torch.pow(10000, (2 * (i // 2)) / d_model)
        return position * angles

    def positional_encoding(self, position: torch.Tensor, d_model: int):
        angle_rads = self.get_angles(
            position=torch.arange(position, dtype=torch.float32)[:, None],
            i=torch.arange(d_model, dtype=torch.float32)[None, :],
            d_model=d_model,
        )

        sines = torch.sin(angle_rads[:, 0::2])
        cosines = torch.cos(angle_rads[:, 1::2])

        angle_rads = torch.zeros(angle_rads.shape)
        angle_rads[:, 0::2] = sines
        angle_rads[:, 1::2] = cosines
        pos_encoding = torch.tensor(angle_rads, dtype=torch.float32)
        pos_encoding = pos_encoding[None, ...]

        return pos_encoding

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.dropout(inputs + self.pos_encoding[:, : inputs.shape[1], :])


class MultiHeadAttention(torch.nn.Module):
    def __init__(self, d_model: int, num_heads: int):
        super().__init__()

        if d_model % num_heads != 0:
            raise ValueError(f"d_model({d_model}) % num_heads({num_heads}) is not zero")

        self.perm = [0, 2, 1, 3]
        self.num_heads = num_heads
        self.d_model = d_model

        self.depth = d_model // self.num_heads

        self.wq = torch.nn.Linear(d_model, d_model)
        self.wk = torch.nn.Linear(d_model, d_model)
        self.wv = torch.nn.Linear(d_model, d_model)

        self.softmax = torch.nn.Softmax(dim=-1)

        self.dense = torch.nn.Linear(d_model, d_model)

    def split_heads(self, inputs: torch.Tensor, batch_size: int) -> torch.Tensor:
        inputs = torch.reshape(
            inputs, shape=(batch_size, -1, self.num_heads, self.depth)
        )
        return inputs.permute(*self.perm)

    def scaled_dot_product_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        softmax((q * k.T) / sqrt(d_k)) * v
        """
        qk = torch.matmul(q, k.transpose(-1, -2))
        scaled_qk = qk / torch.math.sqrt(self.depth)

        if mask is not None:
            scaled_qk += mask * -1e9

        attention_distribution = self.softmax(scaled_qk)
        attention_value = torch.matmul(attention_distribution, v)

        return attention_value

    def forward(
        self, v: torch.Tensor, k: torch.Tensor, q: torch.Tensor, mask: torch.Tensor
    ):
        batch_size = q.shape[0]

        q = self.wq(q)
        k = self.wk(k)
        v = self.wv(v)

        q = self.split_heads(q, batch_size)
        k = self.split_heads(k, batch_size)
        v = self.split_heads(v, batch_size)

        attention_value = self.scaled_dot_product_attention(q, k, v, mask)
        attention_value = attention_value.permute(*self.perm)

        concat_attention = torch.reshape(
            attention_value, (batch_size, -1, self.d_model)
        )

        output = self.dense(concat_attention)

        return output


class PositionWiseFFNN(torch.nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.dense1 = torch.nn.Linear(d_model, d_ff)
        self.dense2 = torch.nn.Linear(d_ff, d_model)
        self.relu = torch.nn.ReLU()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        output = self.dense1(inputs)
        output = self.relu(output)
        output = self.dense2(output)
        return output


class EncoderLayer(torch.nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.mha = MultiHeadAttention(d_model, num_heads)
        self.ffn = PositionWiseFFNN(d_model, d_ff)

        self.layer_norm1 = torch.nn.LayerNorm(d_model)
        self.layer_norm2 = torch.nn.LayerNorm(d_model)

        self.dropout1 = torch.nn.Dropout(dropout)
        self.dropout2 = torch.nn.Dropout(dropout)

    def forward(
        self,
        inputs: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        attention_output = self.mha(inputs, inputs, inputs, mask)
        attention_output = self.dropout1(attention_output)
        attention_output = self.layer_norm1(inputs + attention_output)

        ffn_output = self.ffn(attention_output)
        ffn_output = self.dropout2(ffn_output)
        output = self.layer_norm2(attention_output + ffn_output)

        return output


class DecoderLayer(torch.nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.mha1 = MultiHeadAttention(d_model, num_heads)
        self.mha2 = MultiHeadAttention(d_model, num_heads)
        self.ffn = PositionWiseFFNN(d_model, d_ff)

        self.layer_norm1 = torch.nn.LayerNorm(d_model)
        self.layer_norm2 = torch.nn.LayerNorm(d_model)
        self.layer_norm3 = torch.nn.LayerNorm(d_model)

        self.dropout1 = torch.nn.Dropout(dropout)
        self.dropout2 = torch.nn.Dropout(dropout)
        self.dropout3 = torch.nn.Dropout(dropout)

    def forward(
        self,
        inputs: torch.Tensor,
        encoder_outputs: torch.Tensor,
        look_ahead_mask: torch.Tensor,
        padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        attention_1 = self.mha1(inputs, inputs, inputs, look_ahead_mask)
        attention_1 = self.dropout1(attention_1)
        attention_1 = self.layer_norm1(inputs + attention_1)

        attention_2 = self.mha2(
            attention_1, encoder_outputs, encoder_outputs, padding_mask
        )
        attention_2 = self.dropout2(attention_2)
        attention_2 = self.layer_norm2(attention_1 + attention_2)

        ffn_output = self.ffn(attention_2)
        ffn_output = self.dropout3(ffn_output)
        output = self.layer_norm3(attention_2 + ffn_output)

        return output


class Transformer(torch.nn.Module):
    name = "transformer"

    def __init__(
        self,
        d_model: int = 512,
        num_layers: int = 6,
        num_heads: int = 8,
        d_ff: int = 2048,
        vocab_size: int = 50 * 1000,
        dropout: float = 0.1,
        pad_token_id: int = 0,
    ):
        super().__init__()

        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.pad_token_id = pad_token_id

        self.embedding = torch.nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(vocab_size, d_model, dropout)

        self.enc_layers = torch.nn.ModuleList(
            [EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)]
        )
        self.dec_layers = torch.nn.ModuleList(
            [DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)]
        )
        self.dense = torch.nn.Linear(d_model, vocab_size)

    def create_padding_mask(self, x: torch.Tensor):
        return ~(x == self.pad_token_id)[:, None, None, :]

    def create_look_ahead_mask(self, x: torch.Tensor):
        seq_len = x.shape[1]
        look_ahead_mask = torch.tril(torch.ones((seq_len, seq_len)))
        padding_mask = self.create_padding_mask(x)
        return torch.max(look_ahead_mask, padding_mask)

    def pos_encode(self, x: torch.Tensor):
        embeddings = self.embedding(x) * torch.math.sqrt(self.d_model)
        return self.pos_encoding(embeddings)

    def encoder(self, x: torch.Tensor, padding_mask: torch.Tensor) -> torch.Tensor:
        outputs = self.pos_encode(x)

        for i in range(self.num_layers):
            outputs = self.enc_layers[i](outputs, padding_mask)

        return outputs

    def decoder(
        self, x: torch.Tensor, enc_outputs: torch.Tensor, padding_mask: torch.Tensor
    ) -> torch.Tensor:
        look_ahead_mask = self.create_look_ahead_mask(x)

        outputs = self.pos_encode(x)

        for i in range(self.num_layers):
            outputs = self.dec_layers[i](
                outputs, enc_outputs, look_ahead_mask, padding_mask
            )

        return outputs

    def forward(self, enc_x: torch.Tensor, dec_x: torch.Tensor) -> torch.Tensor:
        enc_padding_mask = self.create_padding_mask(enc_x)
        enc_outputs = self.encoder(enc_x, enc_padding_mask)
        dec_padding_mask = self.create_padding_mask(dec_x)
        dec_outputs = self.decoder(dec_x, enc_outputs, dec_padding_mask)
        outputs = self.dense(dec_outputs)
        return outputs
