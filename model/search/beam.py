"""
model.search.beam — beam search decoding.

No KV cache, so this re-runs the decoder over the full prefix at every step;
beam search costs roughly `beam_size` times greedy on top of that.

Follows Wu et al. (2016) — the reference model/configs/archive/base_v6.yaml
replicates, via Gezmu et al. (arXiv:2104.03543): "we used a beam size of four
and a length penalty of 0.6".
"""
import torch

from model.common import BOS_ID, EOS_ID, PAD_ID


def length_penalty_gnmt(lengths: torch.Tensor, alpha: float) -> torch.Tensor:
    """Wu et al. (2016) eq. 14:  lp(Y) = ((5 + |Y|) / 6) ** alpha.

    A hypothesis' final rank is its cumulative log-probability *divided* by this,
    so alpha > 0 favours longer output (it shrinks the penalty's bite on the
    negative log-prob sum). alpha = 0 reduces to pure log-probability, which
    systematically prefers short hypotheses.
    """
    return ((5.0 + lengths.float()) / 6.0) ** alpha


@torch.no_grad()
def beam_search_decode(model, src: torch.Tensor, src_pad_mask: torch.Tensor, max_len: int,
                       beam_size: int, length_penalty: float) -> torch.Tensor:
    """Batched beam search. Returns the best hypothesis per input, [B, T].

    Beams live flattened as [B * beam_size] rows so the decoder sees one ordinary
    batch. A beam that emits EOS is *frozen* rather than removed: its log-prob
    distribution is replaced by one that can only emit PAD at zero cost, so it
    keeps its score, stops growing, and still competes in the top-k against beams
    that are still running. That makes the search a fixed-shape tensor op the
    whole way through — no ragged finished-pool bookkeeping.

    The length penalty is applied *once, at the end*, when ranking the surviving
    beams. Applying it per step instead would rescale scores that are already
    being compared at equal length, which is not what Wu et al. describe.
    """
    model.eval()
    device = src.device
    B = src.size(0)
    K = beam_size

    memory = model.encode(src, src_pad_mask)                      # [B, S, D]
    S, D = memory.size(1), memory.size(2)
    memory = memory.unsqueeze(1).expand(B, K, S, D).reshape(B * K, S, D)
    mem_pad_mask = src_pad_mask.unsqueeze(1).expand(B, K, S).reshape(B * K, S)

    tokens = torch.full((B * K, 1), BOS_ID, dtype=torch.long, device=device)
    # Only beam 0 is live at step 0. Without this every beam holds an identical
    # BOS prefix, so the first top-k would return the same continuation K times.
    scores = torch.full((B, K), float("-inf"), device=device)
    scores[:, 0] = 0.0
    scores = scores.view(B * K)
    finished = torch.zeros(B * K, dtype=torch.bool, device=device)
    lengths = torch.zeros(B * K, dtype=torch.long, device=device)

    for _ in range(max_len - 1):
        tgt_pad_mask = tokens == PAD_ID
        decoded = model.decode(tokens, tgt_pad_mask, memory, mem_pad_mask)
        logits = model.generator(decoded[:, -1, :])               # [B*K, V]
        V = logits.size(-1)
        logprobs = torch.log_softmax(logits.float(), dim=-1)

        if finished.any():
            frozen = torch.full_like(logprobs, float("-inf"))
            frozen[:, PAD_ID] = 0.0
            logprobs = torch.where(finished.unsqueeze(1), frozen, logprobs)

        candidates = (scores.unsqueeze(1) + logprobs).view(B, K * V)
        top_scores, top_idx = candidates.topk(K, dim=-1)          # [B, K]

        beam_idx = torch.div(top_idx, V, rounding_mode="floor")   # which beam each came from
        token_idx = top_idx % V

        # Flat row indices so the selected beams' histories/state come along.
        rows = (torch.arange(B, device=device).unsqueeze(1) * K + beam_idx).view(-1)
        tokens = torch.cat([tokens[rows], token_idx.view(-1, 1)], dim=1)
        scores = top_scores.view(-1)
        finished = finished[rows]
        lengths = lengths[rows]

        just_ended = (~finished) & (token_idx.view(-1) == EOS_ID)
        lengths = lengths + (~finished).long()   # count this token, EOS included
        finished = finished | just_ended

        if finished.all():
            break

    ranked = (scores / length_penalty_gnmt(lengths, length_penalty)).view(B, K)
    best = ranked.argmax(dim=-1)
    return tokens[torch.arange(B, device=device) * K + best]
