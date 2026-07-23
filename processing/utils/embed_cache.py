"""
embed_cache.py — a content-keyed cache for sentence-embedding vectors.

score_cache.py caches scalar model scores and coerces every score column via
pd.to_numeric before storing, which can't hold a high-dimensional embedding
vector. This module is the vector analogue: rows are keyed by a hash of their
own text (not a whole dataframe's columns), and the cache stores a float32
matrix alongside a parquet of keys instead of a single-column parquet of floats.

Caches live in data/embeddings/<name>.npy + <name>_keys.parquet — regenerable,
and gitignored with data/. Delete both to force a re-embed (e.g. after changing
the model).
"""
import hashlib

import numpy as np
import pandas as pd

from processing.utils.paths import EMBEDDINGS


def _text_keys(texts: pd.Series) -> pd.Series:
    """A stable content hash per text value: sha1 of the UTF-8 bytes."""
    return texts.astype(str).map(lambda s: hashlib.sha1(s.encode("utf-8")).hexdigest())


def _matrix_path(name: str):
    return EMBEDDINGS / f"{name}.npy"


def _keys_path(name: str):
    return EMBEDDINGS / f"{name}_keys.parquet"


def _load(name: str) -> tuple[np.ndarray, pd.Index]:
    """The cache as (matrix, key_index); empty (but correctly shaped) if absent."""
    mpath, kpath = _matrix_path(name), _keys_path(name)
    if not mpath.exists() or not kpath.exists():
        return np.empty((0, 0), dtype=np.float32), pd.Index([], name="key")
    matrix = np.load(mpath)
    keys = pd.read_parquet(kpath)["key"]
    return matrix, pd.Index(keys, name="key")


def embedded(texts: pd.Series, name: str, compute) -> np.ndarray:
    """Return an (len(texts), dim) float32 matrix aligned to texts' positional
    order, computing only the uncached rows.

    `compute(list[str]) -> np.ndarray` is called once with every distinct cache
    miss across `texts`; it is skipped entirely when everything hits, so a run
    where every text has already been embedded never loads the model.
    """
    keys = _text_keys(texts)
    matrix, cached_keys = _load(name)
    cache = pd.Series(range(len(cached_keys)), index=cached_keys)

    # One compute per *distinct* miss: identical text embedded once.
    misses = keys[~keys.isin(cache.index) & ~keys.duplicated()]
    hits = int(keys.isin(cache.index).sum())
    print(f"[embed_cache] {name}: {hits}/{len(keys)} rows hit cache, "
          f"{len(misses)} distinct text(s) to embed")

    if len(misses):
        fresh = np.asarray(compute(texts.loc[misses.index].astype(str).tolist()), dtype=np.float32)
        matrix = np.vstack([matrix, fresh]) if matrix.size else fresh
        cached_keys = cached_keys.append(pd.Index(misses.to_numpy(), name="key"))
        cache = pd.Series(range(len(cached_keys)), index=cached_keys)

        EMBEDDINGS.mkdir(parents=True, exist_ok=True)
        np.save(_matrix_path(name), matrix)
        pd.DataFrame({"key": cached_keys}).to_parquet(_keys_path(name))
        print(f"[embed_cache] {name}: +{len(misses)} embedded → {_matrix_path(name)} "
              f"({len(cached_keys)} total)")

    idx = cache.reindex(keys).to_numpy()
    return matrix[idx]
