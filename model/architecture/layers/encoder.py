import torch
import torch.nn as nn

from model.architecture.layers.attention import MultiHeadAttention
from model.architecture.layers.feedforward import PositionwiseFeedForward


class EncoderLayer(nn.Module):
    """Pre-LN: norm -> sublayer -> dropout -> residual, for training stability."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.dropout1 = nn.Dropout(dropout)

        self.norm2 = nn.LayerNorm(d_model)
        self.ff = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, src_mask: torch.Tensor) -> torch.Tensor:
        normed = self.norm1(x)
        x = x + self.dropout1(self.self_attn(normed, normed, normed, src_mask))

        normed = self.norm2(x)
        x = x + self.dropout2(self.ff(normed))
        return x


class Encoder(nn.Module):
    def __init__(self, n_layers: int, d_model: int, n_heads: int, d_ff: int, dropout: float):
        super().__init__()
        self.layers = nn.ModuleList(
            [EncoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)]
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, src_mask: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, src_mask)
        return self.norm(x)
