"""
processing.clean.filters — shared row-dropping filters for every source (polars, column-parameterized).

Used by both the NLLB path (amh/eng) and the non-NLLB path (am/en) so filtering
is identical across the pooled corpus. These DROP rows; normalization
(processing.clean.normalize) runs before them.
"""
import re
import string

import polars as pl

from processing.clean.normalize import normalize

# ሀ-፿ is the Ethiopic block (U+1200-137F): every Amharic fidel, Ethiopic digit,
# and Ethiopic punctuation mark. "Allowed extra" chars (digits/punctuation/
# whitespace) are shared by both sides; only the letter script differs.
_ALLOWED_EXTRA = re.escape(string.punctuation) + r"0-9\s"
_AMH_DISALLOWED = f"[^ሀ-፿{_ALLOWED_EXTRA}]"
_ENG_DISALLOWED = f"[^A-Za-z{_ALLOWED_EXTRA}]"


def length_normalization(df: pl.DataFrame, am_col: str = "am", en_col: str = "en",
                         am_len=(5, 500), en_len=(10, 500)) -> pl.DataFrame:
    """Remove extra short and extra long sentences. am_len/en_len are (min, max) chars."""
    return df.filter(
        pl.col(am_col).str.len_chars().is_between(am_len[0], am_len[1], closed="none")
        & pl.col(en_col).str.len_chars().is_between(en_len[0], en_len[1], closed="none")
    )


def script_purity(df: pl.DataFrame, am_col: str = "am", en_col: str = "en") -> pl.DataFrame:
    """Keep only rows where the Amharic column is Ethiopic script and the English
    column is Latin script (both plus digits/punctuation/whitespace) — drops rows
    with stray characters from the other side, or any third script, leaking in."""
    return df.filter(
        ~pl.col(am_col).str.contains(_AMH_DISALLOWED)
        & ~pl.col(en_col).str.contains(_ENG_DISALLOWED)
    )


def dedupe(df: pl.DataFrame, keys=("am",)) -> pl.DataFrame:
    """Drop duplicates on `keys`, keeping the first occurrence in current order.

    keys=("am",) keeps one target per Amharic sentence (NLLB: retains the highest
    laser_score, since the frame is sorted descending). keys=("am","en") keeps
    every distinct pair — e.g. multiple human translations of one Amharic verse.
    """
    return df.unique(subset=list(keys), keep="first", maintain_order=True)


def clean(df: pl.DataFrame, name: str, am_col: str = "am", en_col: str = "en",
          am_len=(5, 500), en_len=(10, 500), dedupe_keys=("am",)) -> pl.DataFrame:
    """The full per-source cleaning pipeline in one call: shared normalize, then
    length → script_purity → dedupe, logging per-step row loss under `name`.
    Shared by process.py for both the am/en sources and the NLLB stream."""
    df = normalize(df, am_col, en_col, name)
    steps = [
        ("length_normalization", lambda d: length_normalization(d, am_col, en_col, am_len, en_len)),
        ("script_purity",        lambda d: script_purity(d, am_col, en_col)),
        ("dedupe",               lambda d: dedupe(d, dedupe_keys)),
    ]
    for label, fn in steps:
        before = len(df)
        df = fn(df)
        print(f"[{name}]   {label}: {before} → {len(df)} ({len(df) - before:+d})")
    return df
