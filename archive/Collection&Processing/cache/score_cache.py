"""
score_cache.py — a content-keyed cache for expensive model scores (LaBSE, LID).

The scoring models cost minutes-to-hours per full run, but a *cutoff* is just a
comparison. This module lets the two be separated: score once, threshold as often
as you like.

Rows are keyed by a hash of their am/en text, not by file or row position, so a
cached score survives anything that rewrites data/processed/ without changing the
sentences — re-running the clean step, reordering, adding a source. Only genuinely
new (or re-normalized) text reaches the model. The cache is shared across sources,
so a sentence appearing in two corpora is embedded once.

Cutoff changes therefore cost nothing: re-run the stage, every key hits the cache,
and the model is never loaded.

Caches live in data/scores/<name>.parquet — regenerable, and gitignored with data/.
Delete one to force a re-score (e.g. after changing the model or normalize()).
"""
import hashlib

import pandas as pd

from process.utils.paths import PROCESSED, SCORES


def row_keys(df: pd.DataFrame, key_cols: tuple[str, ...] = ("am", "en")) -> pd.Series:
    """A stable content hash per row: sha1 of the key columns joined by NUL.

    NUL can't occur in the text, so it can't be confused with a separator inside a
    sentence. The hash is over the *normalized* text the model actually sees, so
    editing processing.clean.normalize correctly invalidates nothing automatically —
    delete the cache by hand if normalization changes.
    """
    joined = df[list(key_cols)].astype(str).agg("\x00".join, axis=1)
    return joined.map(lambda s: hashlib.sha1(s.encode("utf-8")).hexdigest())


def _cache_path(name: str):
    return SCORES / f"{name}.parquet"


def _load(name: str, score_cols: list[str]) -> pd.DataFrame:
    """The cache as a key-indexed frame; empty (but correctly shaped) if absent or
    stale — a cache written for different columns is ignored rather than merged."""
    path = _cache_path(name)
    empty = pd.DataFrame(columns=score_cols, index=pd.Index([], name="key"), dtype=float)
    if not path.exists():
        return empty
    cached = pd.read_parquet(path)
    if not set(score_cols).issubset(cached.columns):
        print(f"[cache] {path.name} lacks {score_cols} — ignoring it and re-scoring")
        return empty
    return cached[score_cols]


def scored(df: pd.DataFrame, name: str, score_cols: list[str], compute,
           key_cols: tuple[str, ...] = ("am", "en"),
           skip_where: str | None = None) -> pd.DataFrame:
    """Return `df` with `score_cols` attached, computing only the uncached rows.

    `compute(frame)` must take a frame of unscored rows and return it with
    `score_cols` filled — it is called once, with every cache miss across the
    frame, and skipped entirely when everything hits. That skip is what keeps a
    cutoff change cheap: the model never loads.

    `skip_where`, if given, names a column that marks a row as already covered by
    an *equivalent* score (e.g. nllb.csv's laser_score plays the same role as
    labse_score — both gate semantic alignment quality, so a row that already
    carries one doesn't need the model run for the other). Rows where that column
    is non-null are left out of the cache lookup and the compute() call entirely;
    they come back with `score_cols` unset (NaN) rather than a fabricated value,
    since the two scores aren't on the same scale.
    """
    keys = row_keys(df, key_cols)
    cache = _load(name, score_cols)

    eligible = df.index
    if skip_where is not None and skip_where in df.columns:
        has_equivalent = df[skip_where].notna()
        if has_equivalent.any():
            print(f"[cache] {name}: {int(has_equivalent.sum())} row(s) already carry "
                  f"{skip_where} — treated as equivalent, not scored")
        eligible = df.index[~has_equivalent]

    eligible_keys = keys.loc[eligible]
    # One compute per *distinct* miss: a sentence repeated across rows is scored once.
    misses = eligible_keys[~eligible_keys.isin(cache.index) & ~eligible_keys.duplicated()]
    hits = int(eligible_keys.isin(cache.index).sum())
    print(f"[cache] {name}: {hits}/{len(eligible_keys)} eligible rows hit cache, "
          f"{len(misses)} distinct row(s) to score")

    if len(misses):
        fresh = compute(df.loc[misses.index].copy())
        added = pd.DataFrame(
            {c: pd.to_numeric(fresh[c]).to_numpy() for c in score_cols},
            index=pd.Index(misses.to_numpy(), name="key"),
        )
        cache = pd.concat([cache, added])
        SCORES.mkdir(parents=True, exist_ok=True)
        cache.to_parquet(_cache_path(name))
        print(f"[cache] {name}: +{len(added)} scored → {_cache_path(name)} ({len(cache)} total)")

    out = df.copy()
    for col in score_cols:
        out[col] = cache[col].reindex(keys).to_numpy()
    return out


def annotate_processed(tag: str, score_cols: list[str], compute,
                       skip_where: str | None = None,
                       skip_files: tuple[str, ...] = ()) -> None:
    """Add `score_cols` to every am/en CSV in data/processed/, writing back in place.

    The shared body of score_labse / score_africomet / score_lid's main(), which
    were three verbatim copies of this loop. Each now supplies only what actually
    differs: its log tag, the columns it produces, the compute function, and
    whichever rows or files it skips.

    `skip_files` names CSVs that already ship these columns from upstream
    (nllb.csv carries Meta's own source_lid/target_lid). `skip_where` is passed
    through to scored() — see its docstring for the equivalent-score logic.
    """
    annotated = 0
    for path in sorted(PROCESSED.glob("*.csv")):
        if path.name in skip_files:
            continue
        df = pd.read_csv(path, dtype=str)
        if not {"am", "en"}.issubset(df.columns):
            print(f"[{tag}] skipping {path.name} (not am/en parallel text)")
            continue
        print(f"[{tag}] scoring {path.name}: {len(df)} rows")
        scored(df, tag, score_cols, compute, skip_where=skip_where).to_csv(path, index=False)
        annotated += 1
    print(f"\n[{tag}] annotated {annotated} source(s) → {PROCESSED}")
