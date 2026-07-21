"""
pool.py — Pool per-source processed data and split it into train/val/test.
Run (from the project root) after the normalize step: python -m processing.pool

Reads every parallel-text CSV in data/processed/ (am/en sources + nllb's amh/eng,
normalized to am/en) → apply the quality cutoffs → pool (concat + cross-source dedup
on am + seeded shuffle) → 80/10/10 split → data/final/. Non-corpus CSVs (e.g.
website-stats) are skipped.

The LaBSE/LID cutoffs are applied *here*, not in the scoring stages: score_labse
and score_lid only annotate, leaving every scored row in data/processed/. So
retuning a threshold is just a re-pool (seconds) rather than a re-embed (hours),
and a *lower* cutoff brings rows back instead of being unrecoverable.

The split is **stratified by Amharic sentence-length bucket** (short/medium/long,
from nmt.lengths — the same buckets explore.length_dist reports), so train,
validation and test all carry the same length proportions.

Outputs:
  data/final/train.csv        80/10/10 split, deduplicated, shuffled, length-stratified
  data/final/validation.csv
  data/final/test.csv
"""
import pandas as pd

from processing.utils.paths import PROCESSED, FINAL
from processing.dist.lengths import LENGTH_CUTOFFS, BUCKETS, bucketize

SEED = 42

# Quality floors, applied per source as it is loaded. A row is kept when it clears
# every cutoff whose score column its source carries: labse_score on the local
# corpora, source_lid/target_lid on everything (nllb.csv ships Meta's own, already
# laser-filtered in collect.py). 0.0 disables a cutoff.
COSINE_CUTOFF     = 0.8
AFRICOMET_CUTOFF  = 0.5   # placeholder — needs empirical tuning against real score distributions
SOURCE_LID_CUTOFF = 0.90
TARGET_LID_CUTOFF = 0.90

FINAL.mkdir(parents=True, exist_ok=True)


def apply_cutoffs(df: pd.DataFrame, name: str, cosine_cutoff: float,
                  africomet_cutoff: float, source_lid_cutoff: float,
                  target_lid_cutoff: float) -> pd.DataFrame:
    """Drop rows below the quality floors, reporting each filter's row loss.

    Each cutoff applies only where its score column exists, so a source is filtered
    on what it actually carries: nllb.csv has laser_score + LID but no labse_score,
    the local corpora the reverse. An unscored source passes through untouched —
    run the score stages first if you meant to filter it.

    laser_score and labse_score are the same kind of gate (semantic alignment
    quality) from different pipelines, so score_labse leaves labse_score NaN on
    rows that already carry laser_score (see score_cache's skip_where) instead of
    computing a redundant one. The labse_score floor below skips those rows rather
    than failing them for lacking a score they were never meant to get — they were
    already gated by laser_score upstream, in collect.py's LASER_CUTOFF.

    africomet_score is a different kind of gate (adequacy/fluency, not semantic
    alignment) so score_africomet scores every source including nllb.csv — the
    floor below applies uniformly, no laser_score exemption.
    """
    floors = [("labse_score", cosine_cutoff), ("africomet_score", africomet_cutoff),
              ("source_lid", source_lid_cutoff), ("target_lid", target_lid_cutoff)]
    for col, cutoff in floors:
        if col not in df.columns or cutoff <= 0.0:
            continue
        before = len(df)
        passes = pd.to_numeric(df[col], errors="coerce") > cutoff
        if col == "labse_score" and "laser_score" in df.columns:
            passes |= df["laser_score"].notna()
        df = df[passes]
        print(f"[pool]   {name}: {col}>{cutoff}: {before} → {len(df)} ({len(df) - before:+d})")
    return df.reset_index(drop=True)


def load_pairs(path, cosine_cutoff: float = COSINE_CUTOFF,
               africomet_cutoff: float = AFRICOMET_CUTOFF,
               source_lid_cutoff: float = SOURCE_LID_CUTOFF,
               target_lid_cutoff: float = TARGET_LID_CUTOFF) -> pd.DataFrame | None:
    """Read one CSV, apply the quality cutoffs, return an am/en frame — or None if
    it isn't parallel text.

    Every source (including nllb.csv) uses am/en columns; the score columns gate the
    rows here and are then trimmed away, along with the rest of the per-source
    metadata (urls, …), so only am/en reaches the pool.
    """
    df = pd.read_csv(path, dtype=str)
    if not {"am", "en"}.issubset(df.columns):
        return None
    df = apply_cutoffs(df, path.stem, cosine_cutoff, africomet_cutoff,
                        source_lid_cutoff, target_lid_cutoff)
    return df[["am", "en"]]


def pool(frames: list[pd.DataFrame], seed: int = SEED) -> pd.DataFrame:
    """Combine sources: concat → cross-source dedup on (am, en) → seeded shuffle.

    Deduping on the pair rather than am alone preserves legitimate multi-reference
    translations (e.g. quran.csv's several independent translator versions of one
    verse) instead of collapsing them to a single row; it only drops an exact
    repeat of both sides, including the same pair reappearing across sources.
    """
    combined = pd.concat(frames, ignore_index=True)
    deduped  = combined.drop_duplicates(subset=["am", "en"], keep="first")
    print(f"[pool] pooled {len(combined)} pairs from {len(frames)} sources")
    print(f"[pool] cross-source dupes removed: {len(combined) - len(deduped)}")
    print(f"[pool] total after dedup: {len(deduped)} pairs")
    return deduped.sample(frac=1, random_state=seed).reset_index(drop=True)


def split(df: pd.DataFrame, ratios=(0.8, 0.1, 0.1), seed: int = SEED) -> dict[str, pd.DataFrame]:
    """Length-stratified train/val/test split (default 80/10/10) → data/final/.

    Each pair is bucketed by Amharic char length (nmt.lengths); the ratios are
    applied *within* every bucket and the pieces recombined, so every split holds
    the same short/medium/long proportions. The pool is already deduped+shuffled,
    so a positional slice inside a bucket is a random draw; each split is then
    reshuffled so buckets interleave rather than sit in blocks.

    The split is grouped by am, not sliced row-by-row: since pool() now keeps
    multiple English references for the same Amharic sentence (e.g. quran.csv's
    ~15 translator versions per verse) instead of collapsing them, splitting by
    row would let the same source sentence land in both train and test with a
    different reference — leaking the answer rather than testing generalization.
    Grouping means the split ratios apply to *distinct sentences*, not rows, so a
    source with heavy duplication can pull the realized row-count ratio away from
    the nominal 80/10/10 — that's the tradeoff for a leak-free split."""
    df = df.copy()
    df["_bucket"] = bucketize(df["am"].str.len().to_numpy())

    parts: dict[str, list[pd.DataFrame]] = {"train": [], "validation": [], "test": []}
    for b in BUCKETS:
        grp = df[df["_bucket"] == b] #keep all of the sentences of type bt
        # Every row sharing an am has identical length, so it's already confined to
        # this bucket — group assignment can't leak a sentence across buckets.
        uniq_am = grp["am"].drop_duplicates().sample(frac=1, random_state=seed).reset_index(drop=True)
        n_train = int(len(uniq_am) * ratios[0])
        n_val   = int(len(uniq_am) * ratios[1])
        train_am = set(uniq_am.iloc[:n_train])
        val_am   = set(uniq_am.iloc[n_train : n_train + n_val])
        parts["train"].append(grp[grp["am"].isin(train_am)])
        parts["validation"].append(grp[grp["am"].isin(val_am)])
        parts["test"].append(grp[~grp["am"].isin(train_am) & ~grp["am"].isin(val_am)])

    splits = {}

    #Shuffle and get rid of _bucket col
    for name, chunks in parts.items():
        s = pd.concat(chunks).sample(frac=1, random_state=seed).reset_index(drop=True)
        splits[name] = s.drop(columns="_bucket")

    # _report(df, splits)
    #Send data to FINAL folder
    for name, part in splits.items():
        path = FINAL / f"{name}.csv"
        part.to_csv(path, index=False)
        print(f"final/{name}: {len(part)} pairs → {path}")
    return splits


# def _report(df: pd.DataFrame, splits: dict[str, pd.DataFrame]) -> None:
#     """Print each split's length-bucket proportions to confirm they match."""
#     lo, hi = LENGTH_CUTOFFS
#     print(f"[pool] length buckets (Amharic chars: short <{lo} | medium | long ≥{hi}) — "
#           f"proportions should match across splits:")
#     header = "         " + "".join(f"{b:>10}" for b in BUCKETS) + f"{'n':>10}"
#     print(header)
#     for name, part in {"pool": df, **splits}.items():
#         lengths = part["am"].str.len().to_numpy()
#         labels = bucketize(lengths)
#         n = len(part)
#         cells = "".join(f"{(labels == b).mean() * 100:9.1f}%" for b in BUCKETS)
#         print(f"  {name:>7}{cells}{n:>10,}")


def main(split_ratios=(0.8, 0.1, 0.1), seed: int = SEED,
         cosine_cutoff: float = COSINE_CUTOFF,
         africomet_cutoff: float = AFRICOMET_CUTOFF,
         source_lid_cutoff: float = SOURCE_LID_CUTOFF,
         target_lid_cutoff: float = TARGET_LID_CUTOFF) -> None:
    """Filter every parallel-text CSV in data/processed/ by the quality cutoffs, pool
    them, and split into data/final/."""
    # sorted() gives a deterministic order so the cross-source dedup keep="first"
    # and seeded shuffle stay reproducible.
    processed_paths = sorted(PROCESSED.glob("*.csv"))
    if not processed_paths:
        raise FileNotFoundError(f"No CSVs in {PROCESSED} — run `python process.py` first.")

    # Pool every parallel-text source: the normalized am/en outputs plus nllb.csv
    # (amh/eng, normalized to am/en). Non-corpus CSVs (e.g. website-stats) are skipped.
    print(f"[pool] cutoffs: labse_score>{cosine_cutoff}, africomet_score>{africomet_cutoff}, "
          f"source_lid>{source_lid_cutoff}, target_lid>{target_lid_cutoff}")
    frames = []
    for p in processed_paths:
        df = load_pairs(p, cosine_cutoff, africomet_cutoff, source_lid_cutoff, target_lid_cutoff)
        if df is None:
            print(f"[pool] skipping {p.name} (not am/en parallel text)")
        else:
            print(f"[pool]   {p.name}: {len(df)} pairs kept")
            frames.append(df)

    if not frames:
        raise ValueError(f"No am/en source CSVs found in {PROCESSED}.")

    split(pool(frames, seed=seed), ratios=split_ratios, seed=seed)


if __name__ == "__main__":
    main()
