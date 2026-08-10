"""
processing.clean.normalize — shared Amharic/English text normalization for every source.

One polars, column-parameterized `normalize()` used by process.py for both the
NLLB stream and the csv_raw am/en sources, so the pooled corpus is normalized
identically no matter which stream a pair came from.

Order matters: labialization must run before the homophone merge, because the
homophone step later collapses ዃ (produced by labialization from ኹዋአ→ዃ→ኋ) away.
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

# Labialized fidels typed/OCR'd as 2nd-order ("u") base consonant + ዋ (waa) +
# አ (glottal a) instead of the single precomposed glyph, e.g. ሉዋአ (lu+wa+a) ->
# ሏ (lwa). NOTE: this is a 3-character pattern, not the simpler "bare 6th-order
# consonant + ዋ" (e.g. ልዋ, 2 chars) you might expect from the sound alone — the
# map below is exhaustive and uniform across all 30 entries, so the 3-char shape
# looks like the actual artifact this corpus's OCR/input pipeline produces, but
# it hasn't been checked against whether the simpler 2-char form also occurs and
# would currently pass through unmerged. Must run before the homophone merge,
# since that step merges ዃ (produced here from ኹዋአ->ዃ->ኋ) away.
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

# Typographic quote/prime look-alikes → plain ASCII "/'. Unlike the Amharic-only
# map below, this runs on *both* columns: curly quotes, guillemets, and primes are
# a font/OCR/word-processor artifact of the typing pipeline, not something
# specific to either script — English prose from Word/web sources uses “curly”
# quotes just as often as OCR'd Amharic does. Originally this only touched
# am_col, which meant every row whose English side had a smart quote or prime
# silently vanished at script_purity (whose _ENG_DISALLOWED class is ASCII-only)
# instead of just being mis-punctuated — the same class of bug the sentence-final
# period fix below ran into on the Amharic side. Not true punctuation pairs, so
# they map straight to their ASCII/plain form rather than an Ethiopic equivalent.
_QUOTE_MAP = {
    "‹": '"',  # SINGLE LEFT-POINTING ANGLE QUOTATION MARK
    "›": '"',  # SINGLE RIGHT-POINTING ANGLE QUOTATION MARK
    "«": '"',  # LEFT-POINTING DOUBLE ANGLE QUOTATION MARK
    "»": '"',  # RIGHT-POINTING DOUBLE ANGLE QUOTATION MARK
    "′": "'",  # PRIME (used as apostrophe in this corpus)
    "‘": "'",  # LEFT SINGLE QUOTATION MARK (curly)
    "’": "'",  # RIGHT SINGLE QUOTATION MARK (curly) — also the common smart apostrophe
    "“": '"',  # LEFT DOUBLE QUOTATION MARK (curly) — seen from Gemini OCR output
    "”": '"',  # RIGHT DOUBLE QUOTATION MARK (curly) — quoting defined terms in
               # Negarit-style legal text
}
_QUOTE_FROM = list(_QUOTE_MAP.keys())
_QUOTE_TO = list(_QUOTE_MAP.values())

# Latin punctuation → its Ge'ez/Ethiopic equivalent, so the Amharic column ends up
# with Ethiopic punctuation only. Native Ethiopic marks already in the text (፠, ።,
# ፦, ፨, …) are left untouched — this only rewrites the Latin marks that leak in
# from typing/OCR. Where several Ethiopic marks could map to one Latin character
# (., : each had two candidates), we pick the common one (። full stop, ፥ colon);
# the rarer stylistic marks (፠ section, ፨ paragraph separator, ፦ preface colon)
# simply aren't produced by this map, though they pass through unchanged if
# already present. Amharic-only (unlike _QUOTE_MAP above): English keeps its own
# comma/semicolon/colon/question-mark, so this map only ever runs on am_col.
#
# "." is deliberately NOT here — see _SENTENCE_PERIOD_RE below. Ge'ez script has
# no letter case, so unlike Latin text we can't tell a sentence-final period from
# an abbreviation dot (አ.ክ.ም) or a decimal point (3.14) by looking at what follows
# it. The only signal we actually have is spacing: abbreviation/decimal dots are
# glued tight to the characters on both sides, sentence-final periods are followed
# by whitespace or end-of-string. replace_many is a blind literal swap and can't
# express that condition, so "." gets its own regex step instead.
_PUNCTUATION_MAP = {
    "፡": " ",  # ETHIOPIC WORDSPACE
    ",": "፣",  # → ETHIOPIC COMMA
    ";": "፤",  # → ETHIOPIC SEMICOLON
    ":": "፥",  # → ETHIOPIC COLON
    "?": "፧",  # → ETHIOPIC QUESTION MARK
}
_PUNCTUATION_FROM = list(_PUNCTUATION_MAP.keys())
_PUNCTUATION_TO = list(_PUNCTUATION_MAP.values())

# A "." followed by optional closing quote/paren and then whitespace or
# end-of-string is sentence-final and mapped to ። (e.g. `he said "no."` → the
# quote is between the period and the boundary, but it's still a sentence end).
# A "." glued to a non-period, non-space character on both sides (አ.ክ.ም, 3.14) is
# an abbreviation/decimal dot and is left as a literal period. The leading
# `(^|[^.])` also protects runs of 2+ dots (ellipses, "..." / "..") from being
# partially converted — the char right before a "." in a run is itself a ".", so
# it can never satisfy `[^.]`, and no single dot in the run gets touched. Capture
# groups stand in for a lookaround — Polars' regex engine (Rust `regex` crate)
# doesn't support lookaround. Runs after `punctuation` below (not before), so
# curly quotes are already ASCII "/' by the time the closer class here checks for
# them.
_SENTENCE_PERIOD_RE = r'(^|[^.])\.(["\')]*)(\s|$)'

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

    Both columns: NFC-normalize + strip, map typographic quote/prime look-alikes
    (curly quotes, guillemets, primes) to plain ASCII. Amharic column only:
    collapse labialized fidels, merge homophone fidel series, collapse doubled
    wordspace marks (፡፡) to a full stop, map stray Latin punctuation to Ge'ez, and
    convert sentence-final "." to ። (leaving abbreviation/decimal dots like
    አ.ክ.ም or 3.14 alone). Order is fixed: labialization before homophone;
    doubled-wordspace before punctuation, which would otherwise erase a lone ፡
    one mark at a time; quote-normalization before sentence_period, since the
    latter treats a trailing ASCII quote/paren as still sentence-final and needs
    curly quotes already converted to see it.

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
        ("quote",     (am_col, en_col), (
            pl.col(am_col).str.replace_many(_QUOTE_FROM, _QUOTE_TO),
            pl.col(en_col).str.replace_many(_QUOTE_FROM, _QUOTE_TO),
        )),
        ("labialization", (am_col,), (pl.col(am_col).str.replace_many(_LABIALIZATION_FROM, _LABIALIZATION_TO),)),
        ("homophone",     (am_col,), (pl.col(am_col).str.replace_many(_HOMOPHONE_FROM, _HOMOPHONE_TO),)),
        ("double_wordspace", (am_col,), (pl.col(am_col).str.replace_all(_DOUBLE_WORDSPACE_RE, "። "),)),
        ("punctuation",   (am_col,), (pl.col(am_col).str.replace_many(_PUNCTUATION_FROM, _PUNCTUATION_TO),)),
        ("sentence_period", (am_col,), (pl.col(am_col).str.replace_all(_SENTENCE_PERIOD_RE, "${1}።${2}${3}"),)),
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
