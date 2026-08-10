"""
model.evaluate.evaluate_OOD — decode + score one checkpoint on a held-out
benchmark in data/benchmarks/ (FLORES-200, MAFAND-MT) — out of distribution,
never trained on.

Run (from the project root):
    python -m model.evaluate.evaluate_OOD <config> <checkpoint> <benchmark.csv> [split] [--show-worst N]

e.g. python -m model.evaluate.evaluate_OOD model/configs/archive/base_v4.yaml \\
         runs/am-en-base-v4/checkpoints/best.pt \\
         data/benchmarks/flores200_am_en.csv devtest --show-worst 20

Always re-decodes from the checkpoint rather than trusting a stored hypothesis
file, on purpose: am-en-base-v6's hypotheses, written once and read back later,
scored 17.35 BLEU; the same checkpoint/config/benchmark/references reproduce
15.57 today under identical settings. Nothing was wrong with the model — the
stored file was just from an earlier revision of this code. Re-run this script
whenever you need the number; don't cache its output as ground truth.

Why this is a separate module from model.evaluate.evaluate_in_dist rather than
another split of it:

  1. evaluate_in_dist reads data/prepared/*.pkl, which would mean routing a
     benchmark through data/final/ — the one directory that must only ever hold
     training data.
  2. Both model/data/prepare.py (MAX_LEN) and TranslationDataset.__init__
     (max_src_len/max_tgt_len) *silently drop* over-length pairs. Dropping rows
     from a fixed benchmark changes what "FLORES devtest" means and makes the
     number incomparable to every published system. Here long sources are
     truncated instead, and the count in always equals the count out.
  3. evaluate_in_dist rebuilds references by decoding target token ids back to
     text, which round-trips through a lossy tokenizer. A benchmark reference is
     scored exactly as the dataset ships it.

The Amharic source *is* normalized (processing.clean.normalize), because that is
the text distribution the model was trained on — skipping it would measure a
preprocessing mismatch rather than translation quality (costs ~1.8 BLEU on
FLORES devtest). The English reference is never touched.

--show-worst prints the N lowest sentenceBLEU pairs (source/reference/hypothesis)
below the corpus score, for error inspection — entity mangling, dropped clauses,
repetition loops, hallucinated spans. All from this run's own in-memory hyps, so
there's no separate file format or decode-strategy bookkeeping to keep in sync.
"""
import argparse
from pathlib import Path

import pandas as pd
import polars as pl
import sacrebleu
import torch

from model.common import BOS_ID, EOS_ID, PAD_ID, corpus_scores, describe_decoding, load_for_inference
from model.configs.config import Config
from model.search.decode import decode
from process.clean.normalize import normalize

DEFAULT_SPLIT = "devtest"


def load_benchmark(path: Path, split: str | None) -> pd.DataFrame:
    """Read a benchmark CSV and normalize the Amharic side only."""
    df = pd.read_csv(path, dtype=str).fillna("")
    if split and "split" in df.columns:
        df = df[df["split"] == split].reset_index(drop=True)
        if df.empty:
            raise ValueError(f"{path} has no rows with split={split!r}")

    normalized = normalize(pl.DataFrame({"am": df["am"].tolist(), "en": df["en"].tolist()}),
                           "am", "en", name=None)
    df["am"] = normalized["am"].to_list()   # model-facing source
    return df                               # df["en"] stays exactly as shipped


def encode_sources(tokenizer, sentences: list[str], max_src_len: int) -> tuple[list[list[int]], int]:
    """Tokenize with BOS/EOS, truncating (never dropping) to max_src_len.

    Truncation keeps EOS: the model was never trained to continue past one, and a
    source ending mid-sentence without it decodes far worse than one that is
    merely cut short.
    """
    encoded, truncated = [], 0
    for enc in tokenizer.encode_batch(sentences):
        ids = [BOS_ID, *enc.ids, EOS_ID]
        if len(ids) > max_src_len:
            ids = ids[: max_src_len - 1] + [EOS_ID]
            truncated += 1
        encoded.append(ids)
    return encoded, truncated


@torch.no_grad()
def translate(model, src_ids: list[list[int]], tgt_tokenizer, cfg: Config,
              device: torch.device) -> list[str]:
    """Decode every source, returned in the original input order.

    Strategy comes from cfg.inference (greedy, or beam search with a length
    penalty) — see model.search.

    Batches are formed over length-sorted inputs so each one pads to near its own
    longest sequence rather than the benchmark's longest, then the outputs are
    scattered back — a benchmark's length spread is much wider than a shuffled
    training split's, so this matters more here than it does in training.
    """
    order = sorted(range(len(src_ids)), key=lambda i: len(src_ids[i]))
    hyps: list[str | None] = [None] * len(src_ids)

    for start in range(0, len(order), cfg.data.batch_size):
        idx = order[start : start + cfg.data.batch_size]
        width = max(len(src_ids[i]) for i in idx)
        src = torch.full((len(idx), width), PAD_ID, dtype=torch.long)
        for row, i in enumerate(idx):
            src[row, : len(src_ids[i])] = torch.tensor(src_ids[i])
        src = src.to(device)

        generated = decode(model, src, src == PAD_ID, cfg)
        for row, i in enumerate(idx):
            hyps[i] = tgt_tokenizer.decode(generated[row].tolist(), skip_special_tokens=True)
        print(f"\r  decoding: {min(start + len(idx), len(order))}/{len(order)}", end="", flush=True)

    print()
    return hyps


def print_worst(n: int, src: list[str], refs: list[str], hyps: list[str]) -> None:
    scored = [(sacrebleu.sentence_bleu(h, [r]).score, s, r, h) for s, r, h in zip(src, refs, hyps)]
    scored.sort(key=lambda t: t[0])
    print(f"\n{min(n, len(scored))} worst sentences by sentenceBLEU:\n")
    for b, s, r, h in scored[:n]:
        print(f"\033[1msentBLEU {b:.1f}\033[0m")
        print(f"  AM   {s}")
        print(f"  REF  {r}")
        print(f"  HYP  {h}\n")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    ap.add_argument("config")
    ap.add_argument("checkpoint")
    ap.add_argument("benchmark", help="e.g. data/benchmarks/flores200_am_en.csv")
    ap.add_argument("split", nargs="?", default=DEFAULT_SPLIT)
    ap.add_argument("--show-worst", type=int, default=0, metavar="N",
                    help="also print the N worst-scoring sentences by sentenceBLEU")
    args = ap.parse_args()

    benchmark = Path(args.benchmark)
    df = load_benchmark(benchmark, args.split)
    cfg, model, src_tokenizer, tgt_tokenizer, device = load_for_inference(args.config, args.checkpoint)

    src_ids, truncated = encode_sources(src_tokenizer, df["am"].tolist(), cfg.data.max_src_len)
    print(f"[OOD] {benchmark.stem}:{args.split} — {len(df)} sentences, "
          f"{truncated} truncated to max_src_len={cfg.data.max_src_len}")

    hyps = translate(model, src_ids, tgt_tokenizer, cfg, device)
    refs = df["en"].tolist()
    assert len(hyps) == len(refs) == len(df), "benchmark row count changed during evaluation"

    scores = corpus_scores(hyps, refs)
    print(f"[OOD] {cfg.run_name} on {benchmark.stem}:{args.split} ({len(refs)} sentences)")
    print(f"[OOD]   decoding {describe_decoding(cfg)}")
    print(f"[OOD]   BLEU   {scores['bleu']:.2f}")
    print(f"[OOD]   chrF++ {scores['chrf++']:.2f}")

    if args.show_worst:
        print_worst(args.show_worst, df["am"].tolist(), refs, hyps)


if __name__ == "__main__":
    main()
