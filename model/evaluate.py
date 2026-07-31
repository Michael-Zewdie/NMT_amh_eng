"""
model.evaluate — greedy decode + sacrebleu corpus BLEU on a data split.

Run (from the project root): python -m model.evaluate [config_path] [checkpoint_path] [split]
Defaults: model/configs/base.yaml, <run_dir>/checkpoints/last.pt, "validation".

v1 uses greedy decoding (argmax at every step, no KV cache) — simple and
correct, not fast. Beam search and KV-caching are deferred fast-follows, not
in scope here. Batch-level early stopping: the decode loop exits once every
sequence in the batch has produced EOS, rather than always running to
inference.max_len.
"""
import sys

import sacrebleu
import torch

from model.checkpoint import load_checkpoint
from model.data.dataset import make_dataloader
from model.data.tokenizer import BOS_ID, EOS_ID, PAD_ID, load_tokenizer
from model.transformer import Seq2SeqTransformer
from model.utils.common import get_device
from model.utils.config import Config, load_config
from processing.utils.paths import RUNS

DEFAULT_CONFIG = "model/configs/base.yaml"


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


def evaluate_loader(model, loader, tgt_tokenizer, cfg: Config, device: torch.device) -> float:
    """Corpus BLEU over an already-built DataLoader. Shared by evaluate_split
    (a full data split, e.g. for a standalone post-training run) and
    model.train's periodic in-training eval (a small fixed subset, since
    greedy decode has no KV cache and is too slow to run on a full split
    every eval_every_steps)."""
    hyps: list[str] = []
    refs: list[str] = []
    model.eval()
    for batch in loader:
        batch = batch.to(device)
        generated = greedy_decode(model, batch.src, batch.src_pad_mask, cfg.inference.max_len)
        hyps.extend(tgt_tokenizer.decode(ids, skip_special_tokens=True) for ids in generated.tolist())

        full_tgt = torch.cat([batch.tgt_in[:, :1], batch.tgt_out], dim=1)  # reconstruct [BOS, ..., EOS]
        refs.extend(tgt_tokenizer.decode(ids, skip_special_tokens=True) for ids in full_tgt.tolist())

    # Default sacrebleu tokenization (13a) is correct for an English target.
    # A future en->am direction will need tokenize="none" — no Amharic-aware
    # tokenizer exists in sacrebleu.
    return sacrebleu.corpus_bleu(hyps, [refs]).score


def evaluate_split(model, split: str, cfg: Config, device: torch.device) -> float:
    tgt_tokenizer = load_tokenizer(cfg.data.tgt_lang)
    loader = make_dataloader(split, cfg, shuffle=False)
    return evaluate_loader(model, loader, tgt_tokenizer, cfg, device)


def main() -> None:
    config_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_CONFIG
    cfg = load_config(config_path)
    checkpoint_path = sys.argv[2] if len(sys.argv) > 2 else str(RUNS / cfg.run_name / "checkpoints" / "last.pt")
    split = sys.argv[3] if len(sys.argv) > 3 else "validation"

    device = get_device()
    src_tokenizer = load_tokenizer(cfg.data.src_lang)
    tgt_tokenizer = load_tokenizer(cfg.data.tgt_lang)
    model = Seq2SeqTransformer.from_config(cfg, src_tokenizer.get_vocab_size(), tgt_tokenizer.get_vocab_size(), PAD_ID, device)
    load_checkpoint(checkpoint_path, model, map_location=device)

    score = evaluate_split(model, split, cfg, device)
    print(f"[evaluate] {split} BLEU: {score:.2f}")


if __name__ == "__main__":
    main()
