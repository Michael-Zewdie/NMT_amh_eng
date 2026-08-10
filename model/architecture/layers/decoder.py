import torch
import torch.nn as nn

from model.architecture.layers.attention import MultiHeadAttention
from model.architecture.layers.feedforward import PositionwiseFeedForward


class DecoderLayer(nn.Module):
    """Pre-LN, with cross-attention over encoder memory between self-attention and the FFN."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.dropout1 = nn.Dropout(dropout)

        self.norm2 = nn.LayerNorm(d_model)
        self.cross_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.norm3 = nn.LayerNorm(d_model)
        self.ff = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: torch.Tensor,
        memory_mask: torch.Tensor,
    ) -> torch.Tensor:
        normed = self.norm1(x)
        x = x + self.dropout1(self.self_attn(normed, normed, normed, tgt_mask))

        normed = self.norm2(x)
        x = x + self.dropout2(self.cross_attn(normed, memory, memory, memory_mask))

        normed = self.norm3(x)
        x = x + self.dropout3(self.ff(normed))
        return x


class Decoder(nn.Module):
    def __init__(self, n_layers: int, d_model: int, n_heads: int, d_ff: int, dropout: float):
        super().__init__()
        self.layers = nn.ModuleList(
            [DecoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)]
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(
        self,
        x: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: torch.Tensor,
        memory_mask: torch.Tensor,
    ) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, memory, tgt_mask, memory_mask)
        return self.norm(x)
