"""
model.evaluate_benchmark — score a checkpoint on a held-out benchmark in
data/benchmarks/ (FLORES-200, MAFAND-MT), out of distribution.

Run (from the project root):

    python -m model.evaluate_benchmark <config> <checkpoint> <benchmark.csv> [split]

e.g. python -m model.evaluate_benchmark model/configs/base_v4.yaml \\
         runs/am-en-base-v4/checkpoints/best.pt \\
         data/benchmarks/flores200_am_en.csv devtest

Why this is a separate entry point from model.evaluate rather than another split:

  1. model.evaluate reads data/prepared/*.pkl, which would mean routing a
     benchmark through data/final/ — the one directory that must only ever hold
     training data.
  2. Both model/data/prepare.py (MAX_LEN) and TranslationDataset.__init__
     (max_src_len/max_tgt_len) *silently drop* over-length pairs. Dropping rows
     from a fixed benchmark changes what "FLORES devtest" means and makes the
     number incomparable to every published system. Here long sources are
     truncated instead, and the count in always equals the count out.
  3. model.evaluate rebuilds references by decoding target token ids back to
     text, which round-trips through a lossy tokenizer. A benchmark reference is
     scored exactly as the dataset ships it.

The Amharic source *is* normalized (processing.clean.normalize), because that is
the text distribution the model was trained on — skipping it would measure a
preprocessing mismatch rather than translation quality. The English reference is
never touched.
"""
import sys
from pathlib import Path

import pandas as pd
import polars as pl
import torch

from model.common import BOS_ID, EOS_ID, PAD_ID, describe_decoding, load_for_inference
from model.config import Config
from model.evaluate import corpus_scores
from model.search import decode
from processing.clean.normalize import normalize
from processing.utils.paths import RUNS

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


def main() -> None:
    if len(sys.argv) < 4:
        sys.exit(__doc__.strip())
    config_path, checkpoint_path, benchmark_path = sys.argv[1:4]
    split = sys.argv[4] if len(sys.argv) > 4 else DEFAULT_SPLIT

    benchmark = Path(benchmark_path)
    df = load_benchmark(benchmark, split)
    cfg, model, src_tokenizer, tgt_tokenizer, device = load_for_inference(config_path, checkpoint_path)

    src_ids, truncated = encode_sources(src_tokenizer, df["am"].tolist(), cfg.data.max_src_len)
    print(f"[benchmark] {benchmark.stem}:{split} — {len(df)} sentences, "
          f"{truncated} truncated to max_src_len={cfg.data.max_src_len}")

    hyps = translate(model, src_ids, tgt_tokenizer, cfg, device)
    refs = df["en"].tolist()
    assert len(hyps) == len(refs) == len(df), "benchmark row count changed during evaluation"

    out_dir = RUNS / cfg.run_name / "benchmarks"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{benchmark.stem}_{split}.hyp.txt"
    out.write_text("\n".join(h.replace("\n", " ") for h in hyps), encoding="utf-8")

    scores = corpus_scores(hyps, refs)
    print(f"[benchmark] {cfg.run_name} on {benchmark.stem}:{split} ({len(refs)} sentences)")
    print(f"[benchmark]   decoding {describe_decoding(cfg)}")
    print(f"[benchmark]   BLEU   {scores['bleu']:.2f}")
    print(f"[benchmark]   chrF++ {scores['chrf++']:.2f}")
    print(f"[benchmark]   hypotheses → {out}")


if __name__ == "__main__":
    main()
