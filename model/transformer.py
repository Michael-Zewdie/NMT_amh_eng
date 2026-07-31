import torch
import torch.nn as nn

from model.layers.decoder import Decoder
from model.layers.embeddings import PositionalEncoding, TokenEmbedding
from model.layers.encoder import Encoder


def make_src_mask(src_pad_mask: torch.Tensor) -> torch.Tensor:
    """[B, S] bool (True=pad) -> [B, 1, 1, S] bool (True=block); broadcasts over heads and query positions."""
    return src_pad_mask[:, None, None, :]


def make_tgt_mask(tgt_pad_mask: torch.Tensor) -> torch.Tensor:
    """[B, T] bool (True=pad) -> [B, 1, T, T] bool (True=block); causal + padding combined."""
    _, T = tgt_pad_mask.shape
    causal = torch.triu(torch.ones(T, T, dtype=torch.bool, device=tgt_pad_mask.device), diagonal=1)
    causal = causal[None, None, :, :]      # [1, 1, T, T]
    pad = tgt_pad_mask[:, None, None, :]   # [B, 1, 1, T]
    return causal | pad


class Seq2SeqTransformer(nn.Module):
    def __init__(
        self,
        src_vocab_size: int,
        tgt_vocab_size: int,
        d_model: int,
        n_heads: int,
        n_encoder_layers: int,
        n_decoder_layers: int,
        d_ff: int,
        dropout: float,
        max_len: int,
        pad_id: int,
        tie_output_projection: bool = True,
    ):
        super().__init__()
        self.src_embed = TokenEmbedding(src_vocab_size, d_model, pad_id)
        self.tgt_embed = TokenEmbedding(tgt_vocab_size, d_model, pad_id)
        self.pos_encoding = PositionalEncoding(d_model, max_len, dropout)

        self.encoder = Encoder(n_encoder_layers, d_model, n_heads, d_ff, dropout)
        self.decoder = Decoder(n_decoder_layers, d_model, n_heads, d_ff, dropout)

        self.generator = nn.Linear(d_model, tgt_vocab_size)
        if tie_output_projection:
            self.generator.weight = self.tgt_embed.embedding.weight

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    @classmethod
    def from_config(cls, cfg, src_vocab_size: int, tgt_vocab_size: int, pad_id: int, device: torch.device) -> "Seq2SeqTransformer":
        model = cls(
            src_vocab_size=src_vocab_size,
            tgt_vocab_size=tgt_vocab_size,
            d_model=cfg.model.d_model,
            n_heads=cfg.model.n_heads,
            n_encoder_layers=cfg.model.n_encoder_layers,
            n_decoder_layers=cfg.model.n_decoder_layers,
            d_ff=cfg.model.d_ff,
            dropout=cfg.model.dropout,
            max_len=cfg.model.max_len,
            pad_id=pad_id,
            tie_output_projection=cfg.model.tie_output_projection,
        )
        return model.to(device)

    def encode(self, src: torch.Tensor, src_pad_mask: torch.Tensor) -> torch.Tensor:
        src_mask = make_src_mask(src_pad_mask)
        x = self.pos_encoding(self.src_embed(src))
        return self.encoder(x, src_mask)

    def decode(
        self,
        tgt_in: torch.Tensor,
        tgt_pad_mask: torch.Tensor,
        memory: torch.Tensor,
        src_pad_mask: torch.Tensor,
    ) -> torch.Tensor:
        tgt_mask = make_tgt_mask(tgt_pad_mask)
        memory_mask = make_src_mask(src_pad_mask)
        x = self.pos_encoding(self.tgt_embed(tgt_in))
        return self.decoder(x, memory, tgt_mask, memory_mask)

    def forward(
        self,
        src: torch.Tensor,
        src_pad_mask: torch.Tensor,
        tgt_in: torch.Tensor,
        tgt_pad_mask: torch.Tensor,
    ) -> torch.Tensor:
        memory = self.encode(src, src_pad_mask)
        decoded = self.decode(tgt_in, tgt_pad_mask, memory, src_pad_mask)
        return self.generator(decoded)
