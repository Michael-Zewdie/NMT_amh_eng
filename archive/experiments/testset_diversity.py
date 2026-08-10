"""
experiments.testset_diversity — is in-distribution BLEU a property of the MODEL
or of the TEST SET?

Run (from the project root): python -m experiments.testset_diversity [config] [ckpt]

The cleanest possible control: ONE model, one set of weights, one decoder,
evaluated on several equal-sized slices of its own validation split that differ
only in domain composition. Any BLEU difference between slices cannot be model
quality — the model is identical. It can only be a property of the test set.

This replaces the much more expensive approach of training separate models on
narrow vs diverse corpora, which confounds test-set composition with everything
else that changes when you retrain (volume, optimisation trajectory, what the
model actually learned).

Slices, all sampled to the same N so BLEU is comparable:
  single-domain slices  -- gezmu / nllb / quran on their own
  mixed                 -- proportional draw across every source
Plus a diversity statistic per slice (effective vocabulary = exp(unigram entropy))
so the BLEU numbers can be read against how varied each slice actually is.

If narrow slices score far above diverse ones on the SAME model, then comparing
in-distribution BLEU across models trained on different pools is meaningless, and
only fixed-file benchmarks (FLORES/MAFAND) support cross-model comparison.
"""
import collections
import math
import sys

import numpy as np
import pandas as pd
import torch

from model.common import BOS_ID, EOS_ID, PAD_ID, load_checkpoint, load_tokenizer
from model.configs.config import load_config
from model.common import get_device
from model.evaluate import corpus_scores
from model.search import decode
from model.architecture.transformer import Seq2SeqTransformer
from process.utils.paths import FINAL, PROCESSED, RUNS

DEFAULT_CONFIG = "model/configs/base_v6.yaml"
N_PER_SLICE = 1500
SEED = 0


def label_by_source(df: pd.DataFrame) -> pd.Series:
    lab: dict[str, str] = {}
    for path in sorted(PROCESSED.glob("*.csv")):
        d = pd.read_csv(path, usecols=["am", "en"], dtype=str).dropna()
        for k in (d.am + "\x00" + d.en):
            lab.setdefault(k, path.stem)
    return (df.am + "\x00" + df.en).map(lab)


def eff_vocab(sents) -> float:
    toks = [t for s in sents for t in s.lower().split()]
    c = collections.Counter(toks)
    p = np.array(list(c.values())) / len(toks)
    return math.exp(-(p * np.log(p)).sum())


@torch.no_grad()
def score_slice(model, tok_src, tok_tgt, cfg, device, df: pd.DataFrame) -> dict:
    hyps = []
    ids = [[BOS_ID, *e.ids, EOS_ID][: cfg.data.max_src_len] for e in tok_src.encode_batch(df.am.tolist())]
    order = sorted(range(len(ids)), key=lambda i: len(ids[i]))
    out = [None] * len(ids)
    for start in range(0, len(order), cfg.data.batch_size):
        idx = order[start: start + cfg.data.batch_size]
        w = max(len(ids[i]) for i in idx)
        src = torch.full((len(idx), w), PAD_ID, dtype=torch.long)
        for r, i in enumerate(idx):
            src[r, : len(ids[i])] = torch.tensor(ids[i])
        src = src.to(device)
        gen = decode(model, src, src == PAD_ID, cfg)
        for r, i in enumerate(idx):
            out[i] = tok_tgt.decode(gen[r].tolist(), skip_special_tokens=True)
    hyps = out
    s = corpus_scores(hyps, df.en.tolist())
    s["eff_vocab"] = eff_vocab(df.en.tolist())
    return s


def main() -> None:
    cfg_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_CONFIG
    cfg = load_config(cfg_path)
    ckpt = sys.argv[2] if len(sys.argv) > 2 else str(RUNS / cfg.run_name / "checkpoints" / "best.pt")

    device = get_device()
    tok_src, tok_tgt = load_tokenizer(cfg.data.src_lang), load_tokenizer(cfg.data.tgt_lang)
    model = Seq2SeqTransformer.from_config(cfg, tok_src.get_vocab_size(), tok_tgt.get_vocab_size(), PAD_ID, device)
    load_checkpoint(ckpt, model, map_location=device)
    model.eval()

    va = pd.read_csv(FINAL / "validation.csv", usecols=["am", "en"], dtype=str).dropna().reset_index(drop=True)
    va["src"] = label_by_source(va)

    slices: dict[str, pd.DataFrame] = {}
    for name in ("gezmu", "quran", "nllb", "ccaligned"):
        d = va[va.src == name]
        if len(d) >= N_PER_SLICE:
            slices[f"{name} only"] = d.sample(N_PER_SLICE, random_state=SEED)
    # proportional draw across every source
    slices["mixed (all sources)"] = va.sample(N_PER_SLICE, random_state=SEED)

    print(f"model {cfg.run_name}  ckpt {ckpt}")
    print(f"decoding: beam {cfg.inference.beam_size}, lp {cfg.inference.length_penalty}")
    print(f"{N_PER_SLICE} sentences per slice — SAME MODEL throughout\n")
    print(f"{'validation slice':24s} {'BLEU':>7s} {'chrF++':>8s} {'eff-vocab':>11s}")
    print("-" * 54)
    rows = []
    for name, d in slices.items():
        s = score_slice(model, tok_src, tok_tgt, cfg, device, d)
        rows.append((name, s))
        print(f"{name:24s} {s['bleu']:>7.2f} {s['chrf++']:>8.2f} {s['eff_vocab']:>11,.0f}")
    b = [r[1]["bleu"] for r in rows]
    print("-" * 54)
    print(f"spread: {min(b):.2f} - {max(b):.2f} BLEU ({max(b) - min(b):.2f} points) on identical weights")


if __name__ == "__main__":
    main()
