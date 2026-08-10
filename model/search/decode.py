"""
model.search.decode — picks between model.search.greedy and model.search.beam
from a config.
"""
import torch

from model.search.beam import beam_search_decode
from model.search.greedy import greedy_decode


def decode(model, src: torch.Tensor, src_pad_mask: torch.Tensor, cfg,
           beam_size: int | None = None) -> torch.Tensor:
    """Decode with the strategy `cfg.inference` asks for.

    `beam_size` overrides the config — used by the training loop, which forces
    greedy regardless of what the run will finally be scored with.
    """
    k = cfg.inference.beam_size if beam_size is None else beam_size
    if k > 1:
        return beam_search_decode(model, src, src_pad_mask, cfg.inference.max_len,
                                  k, cfg.inference.length_penalty)
    return greedy_decode(model, src, src_pad_mask, cfg.inference.max_len)
