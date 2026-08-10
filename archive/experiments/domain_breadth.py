"""
experiments.domain_breadth — does the DOMAIN BREADTH of the training corpus
trade in-distribution BLEU against out-of-distribution generalization?

Run (from the project root): python -m experiments.domain_breadth

Hypothesis under test
---------------------
    narrow training domain  -> HIGHER in-distribution BLEU, WORSE OOD (FLORES/MAFAND)
    broad  training domain  -> LOWER  in-distribution BLEU, BETTER OOD

Design: exactly two arms, identical in every respect except the domain breadth
of the training text.

    NARROW  bible-uedin (OPUS)  one book, one register, professionally translated
    BROAD   NLLB mined bitext   ~1,300 effective web domains

Everything else is pinned:
  * same number of training pairs (N below, set by whichever arm is smaller)
  * same tokenizers, NOT retrained  -> vocabulary held fixed
  * same architecture / LR / warmup / dropout / seed / step count
  * same cleaning pipeline (processing.clean.filters.clean) applied to both
  * same 80/10/10 length-stratified, am-grouped split
  * each arm evaluated on ITS OWN validation split (that is what
    "in-distribution" means) and on the SAME fixed FLORES/MAFAND files

Known asymmetry, stated rather than hidden: NLLB rows reaching data/processed/
have already passed collect.py's LASER + LID filters and the pool's
AfriCOMET/LID floors, whereas bible-uedin gets only clean(). Both corpora are
high-quality by construction (one is mined-then-filtered, the other is a
professional translation), but they were not filtered by identical criteria.
This experiment varies domain breadth; it does not control provenance quality.

Outputs:
    data/final_narrow/, data/prepared_narrow/am-en/
    data/final_broad/,  data/prepared_broad/am-en/
"""
import pickle
import sys
import zipfile
from pathlib import Path

import pandas as pd
import polars as pl
import requests

from model.common import BOS_ID, EOS_ID, load_tokenizer
from process.clean.filters import clean
from process import pool as pool_mod
from process.utils.paths import DATA, PROCESSED, RAW

BIBLE_URL = "https://object.pouta.csc.fi/OPUS-bible-uedin/v1/moses/am-en.txt.zip"
BIBLE_DIR = RAW / "bible_uedin"
MAX_LEN = 150
SPLITS = ["validation", "test", "train"]
SEED = 42
AMH_LEN, ENG_LEN = (5, 500), (10, 500)   # same as processing/process.py


def fetch_bible() -> pd.DataFrame:
    am_p = BIBLE_DIR / "bible-uedin.am-en.am"
    en_p = BIBLE_DIR / "bible-uedin.am-en.en"
    if not (am_p.exists() and en_p.exists()):
        BIBLE_DIR.mkdir(parents=True, exist_ok=True)
        z = BIBLE_DIR / "am-en.txt.zip"
        z.write_bytes(requests.get(BIBLE_URL, timeout=180).content)
        with zipfile.ZipFile(z) as zf:
            zf.extract("bible-uedin.am-en.am", BIBLE_DIR)
            zf.extract("bible-uedin.am-en.en", BIBLE_DIR)
        z.unlink()
    return pd.DataFrame({
        "am": am_p.read_text(encoding="utf-8").splitlines(),
        "en": en_p.read_text(encoding="utf-8").splitlines(),
    })


def build(name: str, df: pd.DataFrame, n: int) -> int:
    """Pool -> decontaminate -> subsample to n -> split -> tokenize."""
    pooled = pool_mod.decontaminate(pool_mod.pool([df], seed=SEED))
    if len(pooled) < n:
        sys.exit(f"[{name}] only {len(pooled):,} pairs, need {n:,}")
    pooled = pooled.head(n).reset_index(drop=True)
    print(f"[{name}] using {len(pooled):,} pooled pairs")

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


MIN_PAIRS = 45_000          # each arm must retain at least this many
CANDIDATES = [0.80, 0.70, 0.60, 0.50, 0.40, 0.30, 0.20, 0.15, 0.10, 0.0]


def survivors(path: Path, africomet: float) -> int:
    """Rows kept under a shared numeric threshold. labse is disabled (0.0) for
    BOTH arms because nllb carries no labse_score (score_labse skips rows that
    already have laser_score), so filtering on it could not be symmetric."""
    df = pool_mod.load_pairs(path, 0.0, africomet, 0.90, 0.90)
    return 0 if df is None else len(df)


def main() -> None:
    fetch_bible()   # ensure raw files exist; data/processed/bible.csv is built+scored upstream
    bible_p, nllb_p = PROCESSED / "bible.csv", PROCESSED / "nllb.csv"
    if not bible_p.exists():
        sys.exit("data/processed/bible.csv missing — ingest + score it first")

    # ---- pick the SHARED numeric threshold, data-driven --------------------
    # The production pipeline tiers cutoffs by provenance (curated 0.15 / mined
    # 0.80) because a QE model under-rates archaic register. That is right for
    # building a corpus and WRONG for a controlled experiment: symmetry of intent
    # is not symmetry of treatment. So both arms get the identical number — the
    # highest candidate at which BOTH still retain MIN_PAIRS.
    print("shared-threshold search (identical cutoff applied to both arms):")
    print(f"{'africomet >':>12s} {'bible':>10s} {'nllb':>10s}")
    chosen = None
    for t in CANDIDATES:
        nb, nn = survivors(bible_p, t), survivors(nllb_p, t)
        flag = ""
        if chosen is None and nb >= MIN_PAIRS and nn >= MIN_PAIRS:
            chosen = t
            flag = "  <- CHOSEN"
        print(f"{t:>12.2f} {nb:>10,} {nn:>10,}{flag}")
    if chosen is None:
        sys.exit(f"no threshold leaves {MIN_PAIRS:,} pairs in both arms")

    bible = pool_mod.load_pairs(bible_p, 0.0, chosen, 0.90, 0.90)
    nllb = pool_mod.load_pairs(nllb_p, 0.0, chosen, 0.90, 0.90)
    n = min(len(bible), len(nllb))
    print(f"\n=== shared cutoff africomet>{chosen}, LID>0.90, labse disabled ===")
    print(f"=== matched corpus size: {n:,} pooled pairs per arm ===\n")

    a = build("narrow", bible, n)
    print()
    b = build("broad", nllb, n)
    print(f"\n=== train pairs — narrow {a:,} | broad {b:,} ===")


if __name__ == "__main__":
    main()
