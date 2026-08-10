"""
experiments.religious_vs_nllb — domain breadth again, at 3.5x the scale.

Run (from the project root): python -m experiments.religious_vs_nllb

Why repeat this
---------------
am-en-narrow vs am-en-broad confirmed the hypothesis (narrow 26.63 in-dist /
0.68 FLORES; broad 5.05 / 3.02), but both arms held only ~42k training pairs and
both landed near the floor on FLORES. A 2.3 BLEU gap between two models that bad
is a weak foundation for the out-of-distribution half of the claim. These arms
are ~3.5x larger, which is the size at which the OOD numbers start to mean
something.

    NARROW  religious   one Amharic Bible x 7 English Bibles, verse-ID joined
    BROAD   nllb146     the highest-scoring N rows of NLLB mined bitext

Held fixed: architecture, optimizer, LR schedule, warmup, dropout, seed, step
count, tokenizers (NOT retrained, so vocabulary is constant), the 80/10/10
am-grouped split, and corpus size to within whatever decontamination removes.

Selection, stated plainly rather than buried
--------------------------------------------
This is NOT the shared-threshold design domain_breadth.py used. There, both arms
passed the identical numeric cutoff. Here the religious arm keeps everything above
the curated garbage floor (AfriCOMET > 0.15) while the NLLB arm takes its TOP N by
AfriCOMET — i.e. NLLB is served its best rows and religious is served all of its
rows. That is what "best 146,000 NLLB" means, and it is a defensible question
("best available N from each source"), but it is a different question from
domain_breadth's. If NLLB still loses on in-distribution BLEU here, it loses while
being handed its own best data, which makes that direction of the result stronger,
not weaker. The reverse direction is correspondingly weaker.

The religious arm's real constraint: 26,516 unique Amharic verses. Volume comes
from ~5.6 English renderings per verse, not from new source material. The split is
am-grouped, so no verse straddles train and validation — but source-side coverage
is far lower than the pair count suggests.

Outputs:
    data/final_religious/, data/prepared_religious/am-en/
    data/final_nllb146/,   data/prepared_nllb146/am-en/
"""
import pickle

import pandas as pd
import polars as pl

from model.common import BOS_ID, EOS_ID, load_tokenizer
from process import pool as pool_mod
from process.utils.paths import DATA, PROCESSED

CURATED_AFRICOMET = 0.15     # garbage floor, same as production's curated tier
LID_CUTOFF = 0.90
MAX_LEN = 150                # keep in sync with model/data/prepare.py
SPLITS = ["validation", "test", "train"]
SEED = 42


def build(name: str, df: pd.DataFrame) -> int:
    pooled = pool_mod.decontaminate(pool_mod.pool([df], seed=SEED))
    print(f"[{name}] {len(pooled):,} pooled pairs after decontamination")

    final_out = DATA / f"final_{name}"
    final_out.mkdir(parents=True, exist_ok=True)
    original, pool_mod.FINAL = pool_mod.FINAL, final_out
    try:
        pool_mod.split(pooled)
    finally:
        pool_mod.FINAL = original

    src_tok, tgt_tok = load_tokenizer("am"), load_tokenizer("en")
    out_dir = DATA / f"prepared_{name}" / "am-en"
    out_dir.mkdir(parents=True, exist_ok=True)
    n_train = 0
    for sp in SPLITS:
        d = pd.read_csv(final_out / f"{sp}.csv", usecols=["am", "en"], dtype=str).dropna()
        s = [[BOS_ID, *e.ids, EOS_ID] for e in src_tok.encode_batch(d.am.tolist())]
        t = [[BOS_ID, *e.ids, EOS_ID] for e in tgt_tok.encode_batch(d.en.tolist())]
        keep = [i for i, (a, b) in enumerate(zip(s, t)) if len(a) <= MAX_LEN and len(b) <= MAX_LEN]
        with open(out_dir / f"{sp}.pkl", "wb") as f:
            pickle.dump({"src": [s[i] for i in keep], "tgt": [t[i] for i in keep]}, f)
        print(f"[{name}] {sp}: kept {len(keep):,} of {len(d):,}")
        if sp == "train":
            n_train = len(keep)
    return n_train


def best_nllb(n: int) -> pd.DataFrame:
    """Top-n NLLB rows by AfriCOMET, after the same LID floor both arms get.

    polars + a lazy scan because nllb.csv is ~1.2GB; sorting it in pandas would
    materialize the whole thing to throw most of it away.
    """
    lf = (
        pl.scan_csv(PROCESSED / "nllb.csv", infer_schema_length=0)
        .with_columns([
            pl.col("africomet_score").cast(pl.Float32, strict=False),
            pl.col("source_lid").cast(pl.Float32, strict=False),
            pl.col("target_lid").cast(pl.Float32, strict=False),
        ])
        .filter(
            (pl.col("source_lid") > LID_CUTOFF)
            & (pl.col("target_lid") > LID_CUTOFF)
            & pl.col("africomet_score").is_not_null()
        )
        .sort("africomet_score", descending=True)
        .head(n)
        .select(["am", "en", "africomet_score"])
    )
    df = lf.collect()
    print(f"[nllb146] top {len(df):,} by AfriCOMET — "
          f"score range {df['africomet_score'].min():.4f} to {df['africomet_score'].max():.4f}")
    return df.select(["am", "en"]).to_pandas()


def main() -> None:
    rel = pool_mod.load_pairs(PROCESSED / "religious.csv", 0.0, CURATED_AFRICOMET,
                              LID_CUTOFF, LID_CUTOFF)
    if rel is None:
        raise SystemExit("data/processed/religious.csv missing — run collect_religious + process")
    n = len(rel)
    print(f"\n=== religious: {n:,} pairs at africomet>{CURATED_AFRICOMET}, LID>{LID_CUTOFF} ===")
    print(f"=== nllb146: matching that count with NLLB's {n:,} highest-AfriCOMET rows ===\n")

    a = build("religious", rel)
    print()
    b = build("nllb146", best_nllb(n))
    print(f"\n=== train pairs — religious {a:,} | nllb146 {b:,} ===")


if __name__ == "__main__":
    main()
