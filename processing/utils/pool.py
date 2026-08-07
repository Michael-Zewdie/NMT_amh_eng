"""
pool.py — Pool per-source processed data and split it into train/val/test.
Run (from the project root) after the normalize step: python -m processing.utils.pool

Reads every parallel-text CSV in data/processed/ (am/en sources + nllb's amh/eng,
normalized to am/en) → apply the quality cutoffs → pool (concat + cross-source dedup
on am + seeded shuffle) → drop rows overlapping the held-out benchmarks → 80/10/10
split → data/final/. Non-corpus CSVs (e.g. website-stats) are skipped.

The LaBSE/LID cutoffs are applied *here*, not in the scoring stages: score_labse
and score_lid only annotate, leaving every scored row in data/processed/. So
retuning a threshold is just a re-pool (seconds) rather than a re-embed (hours),
and a *lower* cutoff brings rows back instead of being unrecoverable.

The split is **stratified by Amharic sentence-length bucket** (short/medium/long,
from processing.dist.lengths — the same buckets processing.dist.length_dist reports), so train,
validation and test all carry the same length proportions.

Optionally (main(..., stratify_semantic=True), off by default) it is *also*
stratified by semantic cluster — see split_semantic() below — nesting a k-means
partition of the English side on top of the length buckets, so train/val/test
carry the same topic/domain mix too, not just the same length mix. Off by
default so the plain length-only split (proven, already used to train every
model in EXPERIMENTS.md) stays available for A/B comparison.

Outputs:
  data/final/train.csv        80/10/10 split, deduplicated, shuffled, length-stratified
  data/final/validation.csv
  data/final/test.csv
"""
import numpy as np
import pandas as pd

from processing.utils.paths import PROCESSED, FINAL, SCORES
from processing.dist.lengths import BUCKETS, bucketize
from processing.dist.semantics import N_CLUSTERS, cluster_ids
from processing.utils.manifest import write_manifest, git_info, now

SEED = 42

# Quality floors, applied per source as it is loaded. A row is kept when it clears
# every cutoff whose score column its source carries: labse_score on the local
# corpora, source_lid/target_lid on everything (nllb.csv ships Meta's own, already
# laser-filtered in collect.py). 0.0 disables a cutoff.
COSINE_CUTOFF     = 0.8
AFRICOMET_CUTOFF  = 0.5   # placeholder — needs empirical tuning against real score distributions
SOURCE_LID_CUTOFF = 0.90
TARGET_LID_CUTOFF = 0.90

# labse_score and africomet_score are both automated re-checks of alignment/adequacy
# — the right tool for catching mining errors in nllb.csv/ccaligned.csv, but a poor
# substitute for the professional human translation these sources already carry.
# Gezmu et al. (LREC 2022) got 33.0 BLEU on this exact architecture training on all
# 140k Gezmu pairs, unfiltered — gating 90% of it out on an imperfect QE model's
# opinion throws away already-validated data. LID stays uniform across every source:
# it's a basic sanity check (is this row actually am/en), not a quality judgment.
# religious.csv is an exact verse-ID join of one Amharic Bible against 7 English
# Bibles (collection/collect_religious.py) — every row is professional human
# translation of a known verse, with no mining or alignment guesswork, so it
# belongs in this tier rather than under the mined cutoffs.
CURATED_SOURCES          = frozenset({"gezmu", "afridoc_health", "afridoc_tech", "quran",
                                      "religious"})
CURATED_COSINE_CUTOFF    = 0.0
CURATED_AFRICOMET_CUTOFF = 0.0

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


def decontaminate(df: pd.DataFrame) -> pd.DataFrame:
    """Drop pooled rows that overlap a held-out benchmark in data/benchmarks/.

    Runs on the pooled corpus rather than per source, so one pass covers every
    origin, and runs *before* the split so a leaked sentence can't sneak into
    train via one source while its benchmark twin is being scored.

    Unlike the quality floors above this is not tunable and never optional: the
    whole point of FLORES/MAFAND is a score comparable to published systems, and
    a benchmark sentence sitting in train quietly destroys that. If the benchmark
    directory is empty, processing.utils.decontaminate raises rather than letting
    the pipeline produce a silently-contaminated split.
    """
    from processing.utils.decontaminate import find_contaminated

    mask, matches = find_contaminated(df)
    n = int(mask.sum())
    print(f"[pool] benchmark decontamination: {len(df)} → {len(df) - n} ({-n:+d})")
    if n:
        print(f"[pool]   by benchmark: {matches['benchmark'].value_counts().to_dict()}")
        print(f"[pool]   by match kind: {matches['kind'].value_counts().to_dict()}")
    return df[~mask].reset_index(drop=True)


def split(df: pd.DataFrame, ratios=(0.8, 0.1, 0.1), seed: int = SEED) -> dict[str, pd.DataFrame]:
    """Length-stratified train/val/test split (default 80/10/10) → data/final/.

    Each pair is bucketed by Amharic char length (processing.dist.lengths); the ratios are
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

    #Send data to FINAL folder
    for name, part in splits.items():
        path = FINAL / f"{name}.csv"
        part.to_csv(path, index=False)
        print(f"final/{name}: {len(part)} pairs → {path}")
    return splits


def am_cluster_ids(df: pd.DataFrame, k: int = N_CLUSTERS, seed: int = SEED,
                    embed_tag: str = "mpnet_en") -> dict[str, int]:
    """One semantic-cluster id per unique am, from k-means over mean-pooled English
    embeddings.

    split_semantic() clusters on the *English* side (per the project's own
    request) but still has to group by am for leak-prevention (see split()'s
    docstring) — a single am can carry several independent English references
    (e.g. quran.csv's ~15 translator versions per verse), which would otherwise
    land in different clusters and defeat that grouping. Mean-pooling (then
    L2-renormalizing) every en embedding sharing an am collapses each group back
    to one vector before clustering, so every row sharing an am always gets the
    same cluster id.

    Vectorized, not a per-group Python loop: pd.factorize turns `am` into integer
    codes, np.add.at accumulates every row's vector into its group's running sum
    in one call, then the whole (n_groups, dim) mean matrix is L2-renormalized
    in place. An earlier version used groupby("am").apply(...) — one Python call
    per group — which at production scale (~400k+ unique am) was the actual
    bottleneck in split_semantic(), not the k-means fit that follows it.

    Every array here is (n_groups_or_rows, dim) — at the largest scale this runs
    at (processing.dist.semantic_dist's ~2.8M-row unfiltered pool, ~2.6M unique
    am), each one is several GB, so this was rewritten to reuse one buffer
    in-place (sums -> means -> unit vectors) and drop the row-aligned lookup()
    result the moment it's been accumulated, rather than the more readable but
    much more memory-hungry version that kept sums/means/am_matrix as three
    separate full-size arrays alive simultaneously — measured to blow well past
    available RAM at that scale (a real run got killed here).

    Cache-only and model-free otherwise: raises via embed_cache.lookup() if
    `python -m processing.utils.score_embed` hasn't been run yet for this data.
    Also used by processing.dist.semantic_dist for its distribution report, so
    both the split and the report agree on what a "cluster" is.

    Result is cached to data/scores/semantic_clusters_<hash>.parquet, keyed by
    the sorted set of am strings + k + seed + embed_tag — not just by embed_tag,
    since a re-pool with different cutoffs changes *which* am's are being
    clustered, and the k-means fit is sensitive to that set. A call on the exact
    same am set (e.g. re-running semantic_dist without re-pooling, or re-pooling
    twice with identical cutoffs) hits the cache and skips the embed lookup +
    k-means fit entirely — both the mean-pooling and the fit were measured at
    several minutes at production scale (~2.6M unique am), and the whole point
    of separating scoring from pooling elsewhere in this pipeline is that a
    repeat run shouldn't pay for it again.
    """
    key = _cluster_cache_key(df["am"], k, seed, embed_tag)
    cache_path = SCORES / f"semantic_clusters_{key}.parquet"
    if cache_path.exists():
        cached = pd.read_parquet(cache_path)
        print(f"[pool] am_cluster_ids: cache hit ({cache_path.name}) — "
              f"skipping embed lookup + k-means fit for {len(cached)} am groups")
        return dict(zip(cached["am"], cached["cluster"]))

    from processing.utils.embed_cache import lookup

    en_vecs = lookup(df, embed_tag, key_cols=("en",))  # (len(df), dim), row-aligned to df

    codes, uniques = pd.factorize(df["am"].to_numpy())
    n_groups, dim = len(uniques), en_vecs.shape[1]
    sums = np.zeros((n_groups, dim), dtype=np.float32)  # embeddings are already float32; a handful of them summed doesn't need float64
    np.add.at(sums, codes, en_vecs)
    counts = np.bincount(codes, minlength=n_groups).reshape(-1, 1)
    del en_vecs  # can be as large as the whole embedding cache; only needed for the accumulation above

    sums /= counts  # in-place: sums now holds the per-group mean
    norms = np.linalg.norm(sums, axis=1, keepdims=True)
    np.divide(sums, np.where(norms > 0, norms, 1.0), out=sums)  # in-place L2-renormalize; leaves an (impossible in practice) zero group vector as-is
    am_matrix = sums  # same buffer throughout — no separate means/am_matrix copies

    ids = cluster_ids(am_matrix, k=k, seed=seed)

    SCORES.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"am": uniques, "cluster": ids}).to_parquet(cache_path)
    print(f"[pool] am_cluster_ids: cached {len(uniques)} am groups → {cache_path.name}")

    return dict(zip(uniques, ids))


def _cluster_cache_key(am: pd.Series, k: int, seed: int, embed_tag: str) -> str:
    """Stable content hash for am_cluster_ids' cache: sha1 of every distinct am
    string, sorted (so it's independent of row/encounter order), plus the
    parameters that actually change the result."""
    import hashlib
    joined = "\x00".join(sorted(am.unique().tolist()))
    return hashlib.sha1(f"{embed_tag}|{k}|{seed}|".encode("utf-8") + joined.encode("utf-8")).hexdigest()


def _cut(n: int, ratios) -> tuple[int, int, int]:
    """(n_train, n_val, n_test) for a stratum of size n, guaranteeing at least 1
    example in val and in test whenever there's enough to spare without
    emptying train — the small-cell rule for split_semantic()'s
    (length-bucket × cluster) cells, which can be much smaller than split()'s
    length-only buckets.

    A singleton stratum (n == 1) goes entirely to train: there's nothing to
    guarantee into val/test without either leaking or fabricating data. (The
    plain int(n * ratio) formula would actually route it to test instead, via
    the remainder — n_train=int(0.8)=0, n_val=int(0.1)=0, n_test=1-0-0=1 — so
    this is called out explicitly rather than left as an accident of rounding.)
    """
    if n == 1:
        return 1, 0, 0
    n_train = int(n * ratios[0])
    n_val   = int(n * ratios[1])
    n_test  = n - n_train - n_val
    if n_val == 0 and n >= 2:
        n_val = 1
    if n_test == 0 and n - n_val >= 2:
        n_test = 1
    n_train = n - n_val - n_test
    return n_train, n_val, n_test


def split_semantic(df: pd.DataFrame, ratios=(0.8, 0.1, 0.1), seed: int = SEED,
                    k: int = N_CLUSTERS) -> dict[str, pd.DataFrame]:
    """Length- *and* semantic-cluster-stratified train/val/test split → data/final/.

    Same shape as split() — bucket, shuffle, cut, recombine, reshuffle, write —
    but strata are (length bucket × semantic cluster) cells instead of length
    buckets alone, via am_cluster_ids() above, and cells are cut with _cut()
    instead of a plain int(n * ratio) (see _cut()'s docstring for why: nested
    strata are smaller and more likely to round to 0 for val/test).

    Requires the score_embed cache to already cover every row's English text —
    am_cluster_ids() raises a clear error naming the fix if it doesn't.
    """
    df = df.copy()
    df["_bucket"]  = bucketize(df["am"].str.len().to_numpy())
    am_cluster = am_cluster_ids(df, k=k, seed=seed)
    df["_cluster"] = df["am"].map(am_cluster)

    parts: dict[str, list[pd.DataFrame]] = {"train": [], "validation": [], "test": []}
    for (_b, _c), grp in df.groupby(["_bucket", "_cluster"]):
        uniq_am = grp["am"].drop_duplicates().sample(frac=1, random_state=seed).reset_index(drop=True)
        n_train, n_val, _n_test = _cut(len(uniq_am), ratios)
        train_am = set(uniq_am.iloc[:n_train])
        val_am   = set(uniq_am.iloc[n_train : n_train + n_val])
        parts["train"].append(grp[grp["am"].isin(train_am)])
        parts["validation"].append(grp[grp["am"].isin(val_am)])
        parts["test"].append(grp[~grp["am"].isin(train_am) & ~grp["am"].isin(val_am)])

    splits = {}
    for name, chunks in parts.items():
        s = pd.concat(chunks).sample(frac=1, random_state=seed).reset_index(drop=True)
        splits[name] = s.drop(columns=["_bucket", "_cluster"])

    for name, part in splits.items():
        path = FINAL / f"{name}.csv"
        part.to_csv(path, index=False)
        print(f"final/{name}: {len(part)} pairs → {path}")
    return splits


def main(split_ratios=(0.8, 0.1, 0.1), seed: int = SEED,
         cosine_cutoff: float = COSINE_CUTOFF,
         africomet_cutoff: float = AFRICOMET_CUTOFF,
         source_lid_cutoff: float = SOURCE_LID_CUTOFF,
         target_lid_cutoff: float = TARGET_LID_CUTOFF,
         curated_sources: frozenset[str] = CURATED_SOURCES,
         curated_cosine_cutoff: float = CURATED_COSINE_CUTOFF,
         curated_africomet_cutoff: float = CURATED_AFRICOMET_CUTOFF,
         stratify_semantic: bool = False,
         n_clusters: int = N_CLUSTERS) -> None:
    """Filter every parallel-text CSV in data/processed/ by the quality cutoffs, pool
    them, and split into data/final/. Also writes data/final/manifest.json — every
    cutoff actually used, per-source survivor counts, split sizes, and the git SHA —
    so a run trained off this data can be traced back to exactly what built it
    (model/data/prepare.py copies this manifest forward into data/prepared/<pair>/,
    and model/train.py embeds it into runs/<name>/manifest.json). See
    processing.utils.manifest for why this exists.

    curated_sources get curated_cosine_cutoff/curated_africomet_cutoff instead of the
    standard cutoffs — see the CURATED_SOURCES comment for why. LID cutoffs stay the
    same for every source.

    stratify_semantic=True nests a semantic-cluster dimension on top of the
    default length-only stratification — see split_semantic()."""
    # sorted() gives a deterministic order so the cross-source dedup keep="first"
    # and seeded shuffle stay reproducible.
    processed_paths = sorted(PROCESSED.glob("*.csv"))
    if not processed_paths:
        raise FileNotFoundError(f"No CSVs in {PROCESSED} — run `python -m processing.process` first.")

    # Pool every parallel-text source: the normalized am/en outputs plus nllb.csv
    # (amh/eng, normalized to am/en). Non-corpus CSVs (e.g. website-stats) are skipped.
    print(f"[pool] cutoffs (mined): labse_score>{cosine_cutoff}, africomet_score>{africomet_cutoff}, "
          f"source_lid>{source_lid_cutoff}, target_lid>{target_lid_cutoff}")
    print(f"[pool] cutoffs (curated {sorted(curated_sources)}): "
          f"labse_score>{curated_cosine_cutoff}, africomet_score>{curated_africomet_cutoff}, "
          f"source_lid>{source_lid_cutoff}, target_lid>{target_lid_cutoff}")
    frames = []
    source_counts = {}  # per-source survivor count, for the manifest below
    for p in processed_paths:
        curated = p.stem in curated_sources
        cos_c = curated_cosine_cutoff if curated else cosine_cutoff
        afc_c = curated_africomet_cutoff if curated else africomet_cutoff
        df = load_pairs(p, cos_c, afc_c, source_lid_cutoff, target_lid_cutoff)
        if df is None:
            print(f"[pool] skipping {p.name} (not am/en parallel text)")
        else:
            print(f"[pool]   {p.name}: {len(df)} pairs kept")
            source_counts[p.stem] = len(df)
            frames.append(df)

    if not frames:
        raise ValueError(f"No am/en source CSVs found in {PROCESSED}.")

    pooled_all = pool(frames, seed=seed)
    pooled = decontaminate(pooled_all)
    if stratify_semantic:
        splits = split_semantic(pooled, ratios=split_ratios, seed=seed, k=n_clusters)
    else:
        splits = split(pooled, ratios=split_ratios, seed=seed)

    write_manifest(FINAL, {
        "final_dir": str(FINAL),
        "built_at": now(),
        "git": git_info(),
        "cutoffs": {
            "mined": {"labse_score": cosine_cutoff, "africomet_score": africomet_cutoff,
                      "source_lid": source_lid_cutoff, "target_lid": target_lid_cutoff},
            "curated_sources": sorted(curated_sources),
            "curated": {"labse_score": curated_cosine_cutoff, "africomet_score": curated_africomet_cutoff,
                        "source_lid": source_lid_cutoff, "target_lid": target_lid_cutoff},
        },
        "split_ratios": list(split_ratios),
        "seed": seed,
        "stratify_semantic": stratify_semantic,
        "n_clusters": n_clusters if stratify_semantic else None,
        "source_survivor_counts": source_counts,
        "pooled_after_dedup": len(pooled_all),
        "pooled_after_decontam": len(pooled),
        "split_sizes": {name: len(df) for name, df in splits.items()},
    })


if __name__ == "__main__":
    main()
