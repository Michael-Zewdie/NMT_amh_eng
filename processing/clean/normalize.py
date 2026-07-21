"""
processing.clean.normalize — shared Amharic/English text normalization for every source.

One polars, column-parameterized `normalize()` used by process.py for both the
NLLB stream and the csv_raw am/en sources, so the pooled corpus is normalized
identically no matter which stream a pair came from.

Order matters: labialization must run before the homophone merge, because the
homophone step later collapses ኅ (used here in ኅዋ→ኋ) away.
"""
import polars as pl

# ── Amharic homophone fidel series → one canonical form (amh only) ─────────────
_HOMOPHONE_MAP = {
    # ሐ,ኀ,ኸ → ሀ
    "ሐ": "ሀ", "ሑ": "ሁ", "ሒ": "ሂ", "ሓ": "ሃ", "ሔ": "ሄ", "ሕ": "ህ", "ሖ": "ሆ",
    "ኀ": "ሀ", "ኁ": "ሁ", "ኂ": "ሂ", "ኃ": "ሃ", "ኄ": "ሄ", "ኅ": "ህ", "ኆ": "ሆ",
    "ሗ": "ኋ",  
    "ዃ": "ኋ",
    # "ኸ": "ሀ", "ኹ": "ሁ", "ኺ": "ሂ", "ኻ": "ሃ", "ኼ": "ሄ", "ኽ": "ህ", "ኾ": "ሆ",
    # ሠ → ሰ
    "ሠ": "ሰ", "ሡ": "ሱ", "ሢ": "ሲ", "ሣ": "ሳ", "ሤ": "ሴ", "ሥ": "ስ", "ሦ": "ሶ",
    "ሧ": "ሷ",
    # ዐ → አ
    "ዐ": "አ", "ዑ": "ኡ", "ዒ": "ኢ", "ዓ": "ኣ", "ዔ": "ኤ", "ዕ": "እ", "ዖ": "ኦ",
    # ፀ → ጸ
    "ፀ": "ጸ", "ፁ": "ጹ", "ፂ": "ጺ", "ፃ": "ጻ", "ፄ": "ጼ", "ፅ": "ጽ", "ፆ": "ጾ",
}
_HOMOPHONE_FROM = list(_HOMOPHONE_MAP.keys())
_HOMOPHONE_TO = list(_HOMOPHONE_MAP.values())

# Labialized fidels typed/OCR'd as base 6th-order consonant + ዋ (waa) instead of
# the single precomposed glyph, e.g. ልዋ -> ሏ. Must run before the homophone
# merge, since that step merges ኅ (used here in ኅዋ->ኋ) away.
_LABIALIZATION_MAP = {
    "ሉዋአ": "ሏ",  # LU  + WAA → LWA
    "ሑዋአ": "ሗ",  # HHU + WAA → HHWA  (ሐ-series)
    "ሙዋአ": "ሟ",  # MU  + WAA → MWA
    "ሡዋአ": "ሧ",  # SZU + WAA → SZWA  (ሠ-series, non-standard Amharic)
    "ሩዋአ": "ሯ",  # RU  + WAA → RWA
    "ሱዋአ": "ሷ",  # SU  + WAA → SWA
    "ሹዋአ": "ሿ",  # SHU + WAA → SHWA
    "ቡዋአ": "ቧ",  # BU  + WAA → BWA
    "ቩዋአ": "ቯ",  # VU  + WAA → VWA
    "ቱዋአ": "ቷ",  # TU  + WAA → TWA
    "ቹዋአ": "ቿ",  # CU  + WAA → CWA
    "ኑዋአ": "ኗ",  # NU  + WAA → NWA
    "ኙዋአ": "ኟ",  # NYU + WAA → NYWA
    "ዙዋአ": "ዟ",  # ZU  + WAA → ZWA
    "ዡዋአ": "ዧ",  # ZHU + WAA → ZHWA
    "ዱዋአ": "ዷ",  # DU  + WAA → DWA
    "ዹዋአ": "ዿ",  # DDU + WAA → DDWA  (ዸ-series, non-Amharic)
    "ጁዋአ": "ጇ",  # JU  + WAA → JWA
    "ጡዋአ": "ጧ",  # THU + WAA → THWA
    "ጩዋአ": "ጯ",  # CHU + WAA → CHWA
    "ጱዋአ": "ጷ",  # PHU + WAA → PHWA
    "ጹዋአ": "ጿ",  # TSU + WAA → TSWA
    "ፉዋአ": "ፏ",  # FU  + WAA → FWA
    "ፑዋአ": "ፗ",  # PU  + WAA → PWA
    "ቁዋአ": "ቋ",  # QU  + WAA → QWAA
    "ቑዋአ": "ቛ",  # QHU + WAA → QHWAA (ቐ-series, non-Amharic)
    "ኁዋአ": "ኋ",  # XU  + WAA → XWAA
    "ኩዋአ": "ኳ",  # KU  + WAA → KWAA
    "ኹዋአ": "ዃ",  # KXU + WAA → KXWAA (ኸ-series, non-Amharic)
    "ጉዋአ": "ጓ",  # GU  + WAA → GWAA
}
_LABIALIZATION_FROM = list(_LABIALIZATION_MAP.keys())
_LABIALIZATION_TO = list(_LABIALIZATION_MAP.values())

# Latin punctuation → its Ge'ez/Ethiopic equivalent, so the Amharic column ends up
# with Ethiopic punctuation only. Native Ethiopic marks already in the text (፠, ።,
# ፦, ፨, …) are left untouched — this only rewrites the Latin marks that leak in
# from typing/OCR. Where several Ethiopic marks could map to one Latin character
# (., : each had two candidates), we pick the common one (። full stop, ፥ colon);
# the rarer stylistic marks (፠ section, ፨ paragraph separator, ፦ preface colon)
# simply aren't produced by this map, though they pass through unchanged if
# already present. Wordspace (፡→space) and quote-mark cleanup (‹›′«»→"/') aren't
# true punctuation pairs, so they stay pointed at their ASCII/plain form.
_PUNCTUATION_MAP = {
    "፡": " ",  # ETHIOPIC WORDSPACE
    ".": "።",  # → ETHIOPIC FULL STOP
    ",": "፣",  # → ETHIOPIC COMMA
    ";": "፤",  # → ETHIOPIC SEMICOLON
    ":": "፥",  # → ETHIOPIC COLON
    "?": "፧",  # → ETHIOPIC QUESTION MARK
    "‹": '"',  # SINGLE LEFT-POINTING ANGLE QUOTATION MARK
    "›": '"',  # SINGLE RIGHT-POINTING ANGLE QUOTATION MARK
    "′": "'",  # PRIME (used as apostrophe in this corpus)
    "«": '"',  # LEFT-POINTING DOUBLE ANGLE QUOTATION MARK
    "»": '"',  # RIGHT-POINTING DOUBLE ANGLE QUOTATION MARK
}
_PUNCTUATION_FROM = list(_PUNCTUATION_MAP.keys())
_PUNCTUATION_TO = list(_PUNCTUATION_MAP.values())

# Two (or more) ETHIOPIC WORDSPACE marks in a row is the traditional full-stop
# convention that predates the precomposed ። (ETHIOPIC FULL STOP) character —
# a single ፡ is just the word-separator, but ፡፡ ends a sentence. OCR/typing often
# introduces a stray space between the two (፡ ፡), so this tolerates whitespace
# between repeats. Must run before the punctuation step below, which maps a lone
# ፡ to a plain space — by then the doubled pattern would already be gone.
_DOUBLE_WORDSPACE_RE = r"(?:፡\s*){2,}"


def normalize(df: pl.DataFrame, am_col: str = "am", en_col: str = "en",
              name: str | None = None) -> pl.DataFrame:
    """Full text normalization applied to both languages / all sources.

    Both columns: NFC-normalize + strip. Amharic column only: collapse labialized
    fidels, merge homophone fidel series, collapse doubled wordspace marks (፡፡) to
    a full stop, map stray Latin punctuation to Ge'ez, and collapse whitespace
    runs. Order is fixed (labialization before homophone; doubled-wordspace before
    the punctuation step, which would otherwise erase it one mark at a time).

    If `name` is given, logs per-step how many rows each step *modified* (a row
    counts once even if several of its columns changed), with a percentage of the
    input — the modification analogue of clean()'s per-step row-drop log.
    """
    total = len(df)
    steps = [
        ("nfc_strip", (am_col, en_col), (
            pl.col(am_col).str.normalize("NFC").str.strip_chars(),
            pl.col(en_col).str.normalize("NFC").str.strip_chars(),
        )),
        ("labialization", (am_col,), (pl.col(am_col).str.replace_many(_LABIALIZATION_FROM, _LABIALIZATION_TO),)),
        ("homophone",     (am_col,), (pl.col(am_col).str.replace_many(_HOMOPHONE_FROM, _HOMOPHONE_TO),)),
        ("double_wordspace", (am_col,), (pl.col(am_col).str.replace_all(_DOUBLE_WORDSPACE_RE, "። "),)),
        ("punctuation",   (am_col,), (pl.col(am_col).str.replace_many(_PUNCTUATION_FROM, _PUNCTUATION_TO),)),
        ("whitespace",    (am_col,), (pl.col(am_col).str.replace_all(r"\s+", " ").str.strip_chars(),)),
    ]
    for label, cols, exprs in steps:
        before = {}
        for c in cols:
            before[c] = df[c]                       # pre-step values (polars frames are immutable)
        df = df.with_columns(*exprs)
        if name is not None:
            changed = pl.Series([False] * total)    # rows where any touched column differs (null-safe)
            for c in cols:
                changed = changed | df[c].ne_missing(before[c])
            n = int(changed.sum())
            pct = (n / total * 100) if total else 0.0
            print(f"[{name}]   {label}: {n}/{total} modified ({pct:.1f}%)")
    return df
