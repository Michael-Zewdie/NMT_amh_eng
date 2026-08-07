"""
embed_cache.py — content-keyed cache for raw sentence embeddings (vector-valued),
the vector analogue of score_cache.py's scalar score cache.

score_cache.py's cache is `key -> float columns` in a parquet file — a good fit
for scalar scores (labse_score, africomet_score, LID confidences) but not for
fixed-width embedding vectors. This module reuses score_cache.row_keys()'s content
hashing (same stability guarantees: a cached embedding survives anything that
rewrites data/processed/ without changing the text) but stores the vectors
themselves in a dense .npy matrix, with a companion parquet file recording which
key occupies which row.

Cache lives at data/scores/<name>_embed.npy (float32, shape (n, dim)) +
data/scores/<name>_embed_keys.parquet (ordered key index) — regenerable, gitignored
with data/, same as score_cache.py's caches. Delete both to force a re-embed.
"""
import numpy as np
import pandas as pd

from processing.utils.paths import PROCESSED, SCORES
from processing.utils.score_cache import row_keys


def _paths(name: str):
    return SCORES / f"{name}_embed.npy", SCORES / f"{name}_embed_keys.parquet"


# In-process memo: {name: (vecs, keys)}. A full np.load() of an 8+GB cache is a
# one-time, fast *sequential* read — the problem was never that load by itself,
# it was paying for it twice in one process (once from pool.split_semantic, once
# from semantic_dist), which pushed a real run to 61/62GB RAM and got it
# OOM-killed. An earlier fix tried np.load(mmap_mode="r") instead, trading that
# for scattered random-access page faults across the whole file for lookup()'s
# ~659k/2.76M-row subset — which timed out instead of OOMing. Load once, keep it
# resident for the rest of the process, same as every score_cache.py consumer
# implicitly relies on the OS/pandas not re-reading a CSV it already has open.
_CACHE: dict[str, tuple[np.ndarray, pd.Index]] = {}


def _load(name: str, dim: int) -> tuple[np.ndarray, pd.Index]:
    """The cache as (vectors, key index); empty (but correctly shaped) if absent
    or the vector/key counts disagree (a corrupted or half-written cache).
    Memoized per name for the lifetime of the process — see _CACHE above."""
    if name in _CACHE:
        return _CACHE[name]
    npy_path, keys_path = _paths(name)
    empty = (np.empty((0, dim), dtype=np.float32), pd.Index([], name="key"))
    if not npy_path.exists() or not keys_path.exists():
        return empty
    vecs = np.load(npy_path)
    keys = pd.Index(pd.read_parquet(keys_path)["key"], name="key")
    if len(vecs) != len(keys) or vecs.shape[1] != dim:
        print(f"[embed_cache] {name}: cache shape mismatch — ignoring it and re-embedding")
        return empty
    _CACHE[name] = (vecs, keys)
    return vecs, keys


def embedded(df: pd.DataFrame, name: str, key_cols: tuple[str, ...], compute,
             dim: int = 768) -> np.ndarray:
    """Return a (len(df), dim) matrix, row i = the embedding of df.iloc[i]'s
    key_cols text, computing only cache misses.

    `compute(texts: list[str]) -> np.ndarray` must take a list of distinct texts
    and return their embeddings in the same order — it is called once, with every
    cache miss, and skipped entirely when everything hits (the model never loads).
    This is the only function in this module allowed to call compute(); pool.py's
    stratification step uses lookup() below instead, which never does.
    """
    keys = row_keys(df, key_cols)
    cache_vecs, cache_keys = _load(name, dim)

    distinct = keys.drop_duplicates()
    miss_keys = distinct[cache_keys.get_indexer(distinct) == -1]
    print(f"[embed_cache] {name}: {len(distinct) - len(miss_keys)}/{len(distinct)} distinct "
          f"key(s) hit cache, {len(miss_keys)} to embed")

    if len(miss_keys):
        text_col = key_cols[0]
        # First occurrence of each miss key gives the text to embed.
        miss_rows = df.loc[miss_keys.index]
        fresh = np.asarray(compute(miss_rows[text_col].tolist()), dtype=np.float32)
        cache_vecs = np.concatenate([cache_vecs, fresh], axis=0)
        cache_keys = cache_keys.append(pd.Index(miss_keys.to_numpy(), name="key"))
        SCORES.mkdir(parents=True, exist_ok=True)
        npy_path, keys_path = _paths(name)
        np.save(npy_path, cache_vecs)
        pd.DataFrame({"key": cache_keys}).to_parquet(keys_path)
        _CACHE[name] = (cache_vecs, cache_keys)  # keep the memo in sync with what's now on disk
        print(f"[embed_cache] {name}: +{len(miss_keys)} embedded → {npy_path} ({len(cache_keys)} total)")

    # Vectorized index lookup (pandas Index.get_indexer, hash-based C implementation)
    # rather than a Python dict, which at millions of entries was real overhead too.
    idx = cache_keys.get_indexer(keys)
    return cache_vecs[idx]


def lookup(df: pd.DataFrame, name: str, key_cols: tuple[str, ...], dim: int = 768) -> np.ndarray:
    """Strict, cache-only read: a (len(df), dim) matrix, never calling a model.

    Raises if any row's key_cols text isn't already cached — used by pool.py at
    split time, which must stay model-free and fast (see pool.py's split_semantic
    docstring). Run `python -m processing.utils.score_embed` first to populate
    the cache.

    The underlying cache load is memoized per process (see _CACHE), so calling
    this more than once in the same run — e.g. once from pool.split_semantic,
    once from processing.dist.semantic_dist — only pays the disk-read cost once.
    """
    keys = row_keys(df, key_cols)
    cache_vecs, cache_keys = _load(name, dim)

    idx = cache_keys.get_indexer(keys)
    missing_mask = idx == -1
    if missing_mask.any():
        missing = keys[missing_mask]
        raise RuntimeError(
            f"[embed_cache] {name}: {len(missing)} row(s) ({missing.nunique()} distinct "
            f"text(s)) have no cached embedding. Run "
            f"`python -m processing.utils.score_embed` (or enable RUN_EMBED in "
            f"processing/process.py) before stratifying by semantic cluster."
        )

    return cache_vecs[idx]


def annotate_missing(tag: str, key_cols: tuple[str, ...], compute, dim: int = 768) -> None:
    """Ensure every am/en CSV in data/processed/ has its key_cols text embedded and
    cached — the vector analogue of score_cache.annotate_processed(). Unlike that
    function, there's no column to write back into the CSVs: embeddings live only
    in the cache, looked up later by pool.py via lookup().
    """
    annotated = 0
    for path in sorted(PROCESSED.glob("*.csv")):
        df = pd.read_csv(path, dtype=str)
        if not {"am", "en"}.issubset(df.columns):
            print(f"[{tag}] skipping {path.name} (not am/en parallel text)")
            continue
        print(f"[{tag}] embedding {path.name}: {len(df)} rows")
        embedded(df, tag, key_cols, compute, dim=dim)
        annotated += 1
    print(f"\n[{tag}] embedded {annotated} source(s) → {SCORES}")
