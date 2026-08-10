"""
processing.clean.filters — shared row-dropping filters for every source (polars, column-parameterized).

Used by both the NLLB path (amh/eng) and the non-NLLB path (am/en) so filtering
is identical across the pooled corpus. These DROP rows; normalization
(processing.clean.normalize) runs before them.
"""
import re
import string

import polars as pl

from process.clean.normalize import normalize

# ሀ-፿ is the Ethiopic block (U+1200-137F): every Amharic fidel, Ethiopic digit,
# and Ethiopic punctuation mark. Digits/whitespace are shared by both sides; ASCII
# punctuation is not — processing.clean.normalize maps Latin punctuation on the
# Amharic side to its Ethiopic equivalent, so a stray Latin mark there means
# normalize didn't cover it (or it's genuinely mixed-script) rather than something
# to pass through. Deliberate exceptions:
#   "\"'"  — normalize collapses ‹›′«»‘’“” to ASCII "/' on *both* sides before this
#            filter runs (the English side used to be skipped, which silently
#            dropped every row with a smart quote instead of just leaving it
#            un-normalized — see processing.clean.normalize._QUOTE_MAP), so
#            those two chars stay allowed here.
#   "."    — normalize only converts sentence-final periods to ። and intentionally
#            leaves abbreviation (አ.ክ.ም) and decimal (3.14) dots as literal periods
#            — without this exception every row containing one would be silently
#            dropped here instead of just mis-punctuated.
#   "/"    — regulation/proclamation number format (ቁጥር ፫፻፵፭/፪ሺ፯), seen throughout
#            Negarit-style legal text; normalize doesn't rewrite it.
#   "()"   — sub-clause numbering ((፩), (a)); paren_balance() below still catches
#            a genuinely dangling one, so allowing the balanced case here is safe.
#   "°º″"  — degree/second marks in geographic-coordinate text (Negarit heritage-
#            site regulations cite site boundaries this way); minutes are already
#            covered by the plain ' above.
_ASCII_PUNCT = re.escape(string.punctuation)
_AMH_ALLOWED_PUNCT = re.escape("\"'./()°º″")
_DIGITS_SPACE = r"0-9\s"
_AMH_DISALLOWED = f"[^ሀ-፿{_AMH_ALLOWED_PUNCT}{_DIGITS_SPACE}]"
_ENG_DISALLOWED = f"[^A-Za-z{_ASCII_PUNCT}{_DIGITS_SPACE}]"


def length_normalization(df: pl.DataFrame, am_col: str = "am", en_col: str = "en",
                         am_len=(5, 500), en_len=(10, 500)) -> pl.DataFrame:
    """Remove extra short and extra long sentences. am_len/en_len are (min, max) chars."""
    return df.filter(
        pl.col(am_col).str.len_chars().is_between(am_len[0], am_len[1], closed="none")
        & pl.col(en_col).str.len_chars().is_between(en_len[0], en_len[1], closed="none")
    )


def script_purity(df: pl.DataFrame, am_col: str = "am", en_col: str = "en") -> pl.DataFrame:
    """Keep only rows where the Amharic column is Ethiopic script (plus digits,
    whitespace, and the two quote chars normalize leaves as ASCII) and the English
    column is Latin script (plus digits/ASCII punctuation/whitespace) — drops rows
    with stray characters from the other side, unmapped Latin punctuation, or any
    third script, leaking in."""
    return df.filter(
        ~pl.col(am_col).str.contains(_AMH_DISALLOWED)
        & ~pl.col(en_col).str.contains(_ENG_DISALLOWED)
    )


# An ordinal list marker ("ሀ)" / "a)") carries one closing paren with no matching
# '(' by convention on *both* sides of this corpus (Negarit sub-item lettering) —
# not an unbalanced fragment, just how these languages number list items without
# a companion opening paren. Anchored to the start of the string, so it never
# touches a real mid-sentence closing paren.
_LIST_MARKER_RE = r"^\s*\S\)"


def paren_balance(df: pl.DataFrame, am_col: str = "am", en_col: str = "en") -> pl.DataFrame:
    """Drop rows where either column has unbalanced parentheses, once a leading
    ordinal list marker is discounted (checked on a throwaway copy — the stored
    text is untouched).

    Seen in gezmu.csv/quran.csv: a parenthetical split across a verse boundary at
    the source, leaving one side with a dangling '(' or ')' that refers to content
    in a neighboring row — a fragment, not a complete sentence pair."""
    am_check = pl.col(am_col).str.replace(_LIST_MARKER_RE, "")
    en_check = pl.col(en_col).str.replace(_LIST_MARKER_RE, "")
    return df.filter(
        (am_check.str.count_matches(r"\(") == am_check.str.count_matches(r"\)"))
        & (en_check.str.count_matches(r"\(") == en_check.str.count_matches(r"\)"))
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
    length → script_purity → paren_balance → dedupe, logging per-step row loss
    under `name`. Shared by process.py for both the am/en sources and the NLLB
    stream."""
    df = normalize(df, am_col, en_col, name)
    steps = [
        ("length_normalization", lambda d: length_normalization(d, am_col, en_col, am_len, en_len)),
        ("script_purity",        lambda d: script_purity(d, am_col, en_col)),
        ("paren_balance",        lambda d: paren_balance(d, am_col, en_col)),
        ("dedupe",               lambda d: dedupe(d, dedupe_keys)),
    ]
    for label, fn in steps:
        before = len(df)
        df = fn(df)
        print(f"[{name}]   {label}: {before} → {len(df)} ({len(df) - before:+d})")
    return df
