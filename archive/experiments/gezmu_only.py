"""
experiments.gezmu_only — build a single-domain (gezmu-only) corpus, to test
whether in-distribution BLEU is comparable across data compositions.

Run (from the project root): python -m experiments.gezmu_only

Why this exists
---------------
Every in-distribution BLEU in EXPERIMENTS.md is measured on a validation split
drawn from the same pool the model trained on. That makes the number a
*fit-to-your-own-corpus* score, not a quality score: narrow the pool and the
validation set narrows with it, so the exam gets easier at the same time the
model gets worse at real translation. The degenerate case is a one-sentence
corpus, which scores BLEU 100 and translates nothing.

gezmu is the right corpus to demonstrate this with. It is ~83% Watchtower +
Bible by Gezmu et al.'s own Table 1 (one narrow register), professionally
translated, and — unlike quran, which is only 2,690 unique Amharic verses
wearing 37,380 English costumes (13.9 translations per verse, max 38) — it has
123,838 unique source sentences, so a model trained on it has to generalize
rather than memorize.

Prediction, recorded before running: in-distribution BLEU well ABOVE
am-en-base-v6's, and FLORES/MAFAND well BELOW it. If that holds, in-distribution
BLEU is confirmed unusable for comparing models trained on different pools, and
only the fixed-file benchmarks mean anything.

Everything downstream is identical to v6 — same tiered cutoffs, same tokenizers
(deliberately NOT retrained, so vocabulary is held fixed and data composition is
the only variable), same architecture and recipe. Only the source list differs.

Outputs to staging directories so the full-pool data that am-en-base-v6 trained
on is never touched:
    data/final_gezmu/{train,validation,test}.csv
    data/prepared_gezmu/am-en/{train,validation,test}.pkl
"""
import pickle
import sys
from pathlib import Path

import pandas as pd

from model.common import BOS_ID, EOS_ID, load_tokenizer
from process import pool as pool_mod
from process.utils.paths import DATA, PROCESSED

SOURCE = "gezmu"
MAX_LEN = 150            # keep in sync with model/data/prepare.py
SPLITS = ["validation", "test", "train"]

# Same cutoffs process.py passes for a curated source.
CURATED_COSINE_CUTOFF = 0.0
CURATED_AFRICOMET_CUTOFF = 0.15
SOURCE_LID_CUTOFF = 0.90
TARGET_LID_CUTOFF = 0.90

FINAL_OUT = DATA / "final_gezmu"
PREPARED_OUT = DATA / "prepared_gezmu"


def build_corpus() -> None:
    src = PROCESSED / f"{SOURCE}.csv"
    if not src.exists():
        sys.exit(f"missing {src} — run `python -m processing.process` first")

    df = pool_mod.load_pairs(src, CURATED_COSINE_CUTOFF, CURATED_AFRICOMET_CUTOFF,
                             SOURCE_LID_CUTOFF, TARGET_LID_CUTOFF)
    if df is None:
        sys.exit(f"{src} is not am/en parallel text")
    print(f"[gezmu-only] {len(df):,} pairs kept after cutoffs")

    pooled = pool_mod.decontaminate(pool_mod.pool([df]))

    # split() writes to its module-level FINAL; redirect it so the real
    # data/final/ (what am-en-base-v6 trained on) is left alone.
    FINAL_OUT.mkdir(parents=True, exist_ok=True)
    original = pool_mod.FINAL
    pool_mod.FINAL = FINAL_OUT
    try:
        pool_mod.split(pooled)
    finally:
        pool_mod.FINAL = original


def tokenize_corpus() -> None:
    """Mirror of model/data/prepare.py, writing to the staging directory."""
    src_tok, tgt_tok = load_tokenizer("am"), load_tokenizer("en")
    out_dir = PREPARED_OUT / "am-en"
    out_dir.mkdir(parents=True, exist_ok=True)

    for split in SPLITS:
        df = pd.read_csv(FINAL_OUT / f"{split}.csv", usecols=["am", "en"], dtype=str).dropna()
        src = [[BOS_ID, *e.ids, EOS_ID] for e in src_tok.encode_batch(df["am"].tolist())]
        tgt = [[BOS_ID, *e.ids, EOS_ID] for e in tgt_tok.encode_batch(df["en"].tolist())]
        keep = [i for i, (a, b) in enumerate(zip(src, tgt)) if len(a) <= MAX_LEN and len(b) <= MAX_LEN]
        data = {"src": [src[i] for i in keep], "tgt": [tgt[i] for i in keep]}
        with open(out_dir / f"{split}.pkl", "wb") as f:
            pickle.dump(data, f)
        print(f"[gezmu-only] {split}: {len(df):,} pairs, dropped {len(df) - len(keep)} over "
              f"MAX_LEN={MAX_LEN}, kept {len(keep):,}")
    print(f"[gezmu-only] wrote caches to {out_dir}")


if __name__ == "__main__":
    build_corpus()
    tokenize_corpus()
