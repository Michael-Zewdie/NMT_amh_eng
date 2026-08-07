"""
experiments.decode_sweep — tune beam_size and length_penalty on FLORES **dev**,
then report the winner once on devtest.

Run (from the project root):
    python -m experiments.decode_sweep [config] [checkpoint]

Why dev and not devtest
-----------------------
collection/collect_benchmark.py deliberately keeps FLORES `dev` alongside
`devtest` precisely so hyperparameters can be tuned without touching the split
everyone reports. Sweeping on devtest and quoting the best cell would be
selecting on the test set — the number would no longer be comparable to any
published system. So: search on dev, then a SINGLE confirmatory run on devtest
using the winning setting.

Costs nothing but inference time and typically buys +0.3-1.0 BLEU, since
am-en-base-v6's beam 4 / lp 0.6 came from Gezmu et al.'s paper rather than from
anything measured on this corpus.
"""
import itertools
import sys
from dataclasses import replace
from pathlib import Path

import pandas as pd
import polars as pl
import torch

from model.common import BOS_ID, EOS_ID, PAD_ID, get_device, load_checkpoint, load_tokenizer
from model.config import load_config
from model.evaluate import corpus_scores
from model.evaluate_benchmark import encode_sources, load_benchmark, translate
from model.transformer import Seq2SeqTransformer
from processing.utils.paths import BENCHMARKS, RUNS

BEAMS = [1, 4, 6, 8]
PENALTIES = [0.0, 0.4, 0.6, 1.0]
DEFAULT_CONFIG = "model/configs/base_v6.yaml"  # stale, see model/train.py's
# DEFAULT_CONFIG comment — moved to archive/ in the 2026-08-07 v2 archive pass.


def score(model, cfg, tok_src, tok_tgt, df, device) -> dict:
    ids, _ = encode_sources(tok_src, df["am"].tolist(), cfg.data.max_src_len)
    hyps = translate(model, ids, tok_tgt, cfg, device)
    return corpus_scores(hyps, df["en"].tolist())


def main() -> None:
    cfg_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_CONFIG
    cfg = load_config(cfg_path)
    ckpt = sys.argv[2] if len(sys.argv) > 2 else str(RUNS / cfg.run_name / "checkpoints" / "best.pt")

    device = get_device()
    tok_src, tok_tgt = load_tokenizer(cfg.data.src_lang), load_tokenizer(cfg.data.tgt_lang)
    model = Seq2SeqTransformer.from_config(cfg, tok_src.get_vocab_size(), tok_tgt.get_vocab_size(), PAD_ID, device)
    load_checkpoint(ckpt, model, map_location=device)
    model.eval()

    bench = BENCHMARKS / "flores200_am_en.csv"
    dev = load_benchmark(bench, "dev")
    print(f"sweeping on FLORES dev ({len(dev)} sentences) — devtest untouched\n")
    print(f"{'beam':>5s} {'lp':>5s} {'BLEU':>7s} {'chrF++':>8s}")
    print("-" * 28)

    best = None
    for beam, lp in itertools.product(BEAMS, PENALTIES):
        if beam == 1 and lp != PENALTIES[0]:
            continue                                  # greedy ignores lp
        cfg_i = replace(cfg, inference=replace(cfg.inference, beam_size=beam, length_penalty=lp))
        s = score(model, cfg_i, tok_src, tok_tgt, dev, device)
        tag = ""
        if best is None or s["bleu"] > best[2]["bleu"]:
            best = (beam, lp, s)
            tag = "  <- best so far"
        print(f"{beam:>5d} {lp:>5.1f} {s['bleu']:>7.2f} {s['chrf++']:>8.2f}{tag}")

    beam, lp, s = best
    print(f"\nbest on dev: beam {beam}, lp {lp} -> BLEU {s['bleu']:.2f}")
    print("confirming once on devtest (single run, no selection):")
    devtest = load_benchmark(bench, "devtest")
    cfg_b = replace(cfg, inference=replace(cfg.inference, beam_size=beam, length_penalty=lp))
    t = score(model, cfg_b, tok_src, tok_tgt, devtest, device)
    print(f"  FLORES devtest  BLEU {t['bleu']:.2f}  chrF++ {t['chrf++']:.2f}")
    print(f"  (v6 shipped setting was beam {cfg.inference.beam_size}, lp {cfg.inference.length_penalty})")


if __name__ == "__main__":
    main()
