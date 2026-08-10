"""
model.search.greedy — greedy decoding: always take the argmax token.

No KV cache, so this re-runs the decoder over the full prefix at every step.
It's the strategy training-time eval deliberately stays on; see the note on
evaluate_loader for why.
"""
import torch

from model.common import BOS_ID, EOS_ID, PAD_ID


@torch.no_grad()
def greedy_decode(model, src: torch.Tensor, src_pad_mask: torch.Tensor, max_len: int) -> torch.Tensor:
    model.eval()
    device = src.device
    B = src.size(0)
    memory = model.encode(src, src_pad_mask)

    generated = torch.full((B, 1), BOS_ID, dtype=torch.long, device=device)
    finished = torch.zeros(B, dtype=torch.bool, device=device)

    for _ in range(max_len - 1):
        tgt_pad_mask = generated == PAD_ID
        decoded = model.decode(generated, tgt_pad_mask, memory, src_pad_mask)
        next_token = model.generator(decoded[:, -1, :]).argmax(dim=-1)  # [B]
        next_token = torch.where(finished, torch.full_like(next_token, PAD_ID), next_token)

        generated = torch.cat([generated, next_token.unsqueeze(1)], dim=1)
        finished = finished | (next_token == EOS_ID)
        if finished.all():
            break

    return generated
