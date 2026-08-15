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
# and Ethiopic punctuation mark. Plus Arabic numerals and whitespace.
#
# The four ASCII punctuation marks are not decoration — normalize leaves exactly
# these as ASCII on the Amharic side (quotes via _QUOTE_MAP, periods in
# abbreviations like አ.ክ.ም and decimals like 3.14, parens around sub-clause
# numbering), so excluding them here would drop correctly-aligned rows for
# punctuation rather than for script. Measured: 41,419 gezmu rows and 199,135
# NLLB rows, none of which contain a Latin letter.
_ASCII_PUNCT = re.escape(string.punctuation)
_AMH_ALLOWED_PUNCT = re.escape("\"'.()")
_DIGITS_SPACE = r"0-9\s"
_AMH_DISALLOWED = f"[^ሀ-፿{_AMH_ALLOWED_PUNCT}{_DIGITS_SPACE}]"
_ENG_DISALLOWED = f"[^A-Za-z{_ASCII_PUNCT}{_DIGITS_SPACE}]"

# Sources that skip script_purity entirely. AfriDoc is medical/technical prose
# where Amharic legitimately embeds the English term inline — "( Haemoglobin)",
# "ቫይታሚን B9(folate)" — along with % and hyphenated ranges (ከ6-59 ወር). Any script
# rule tight enough to be useful on mined web text deletes about a quarter of
# these two corpora, and it deletes precisely the domain terminology that makes
# them worth including. gezmu is deliberately NOT here: it loses only 2.7%, and
# its larger drop is dedupe correctly collapsing repeated verses.
SCRIPT_PURITY_EXEMPT = frozenset({"afridoc_health", "afridoc_tech"})


def length_normalization(df: pl.DataFrame, am_col: str = "am", en_col: str = "en",
                         am_len=(5, 500), en_len=(10, 500)) -> pl.DataFrame:
    """Remove extra short and extra long sentences. am_len/en_len are (min, max) chars."""
    return df.filter(
        pl.col(am_col).str.len_chars().is_between(am_len[0], am_len[1], closed="none")
        & pl.col(en_col).str.len_chars().is_between(en_len[0], en_len[1], closed="none")
    )


def script_purity(df: pl.DataFrame, am_col: str = "am", en_col: str = "en") -> pl.DataFrame:
    """Keep rows whose Amharic column is Ethiopic (fidel, Ethiopic numerals and
    punctuation) plus Arabic numerals, whitespace and the four ASCII marks
    normalize leaves behind, and whose English column is Latin plus
    digits/ASCII punctuation/whitespace.

    Drops rows with the other side's script, an unmapped Latin mark, or a third
    script leaking in. Not applied to SCRIPT_PURITY_EXEMPT sources — see clean().
    """
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

    Seen in gezmu.csv: a parenthetical split across a verse boundary at
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
    stream.

    script_purity is skipped for SCRIPT_PURITY_EXEMPT sources. Keyed on `name`
    rather than a caller-passed flag so every consumer of this pipeline gets the
    same answer for the same source — an experiment that re-cleans AfriDoc by
    hand is how the two drifted apart before."""
    df = normalize(df, am_col, en_col, name)
    steps = [
        ("length_normalization", lambda d: length_normalization(d, am_col, en_col, am_len, en_len)),
        ("paren_balance",        lambda d: paren_balance(d, am_col, en_col)),
        ("dedupe",               lambda d: dedupe(d, dedupe_keys)),
    ]
    if name not in SCRIPT_PURITY_EXEMPT:
        steps.insert(1, ("script_purity", lambda d: script_purity(d, am_col, en_col)))
    else:
        print(f"[{name}]   script_purity: skipped (SCRIPT_PURITY_EXEMPT)")
    for label, fn in steps:
        before = len(df)
        df = fn(df)
        print(f"[{name}]   {label}: {before} → {len(df)} ({len(df) - before:+d})")
    return df
