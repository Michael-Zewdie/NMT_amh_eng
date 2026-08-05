"""
decontaminate.py — Strip training rows that overlap a held-out benchmark.

data/benchmarks/ holds fixed, human-translated eval sets (FLORES-200, MAFAND-MT).
Keeping them outside csv_raw/ stops them being *pooled* as training data, but it
can't stop the same sentence arriving independently through a mined source —
CCAligned and NLLB are both Common Crawl-derived and FLORES is Wikimedia-sourced,
so genuine overlap is plausible. A benchmark the model has memorized reports a
score that isn't comparable to anyone else's.

The fix goes on the *training* side, never the benchmark side: benchmarks stay
byte-for-byte intact so our numbers stay comparable to published ones, and the
pooled corpus loses the overlapping rows.

Two levels of matching, because exact string equality is not enough — a mined
pair can differ from the benchmark by a stray quote or a rewrapped clause and
still be the same sentence:

  exact  — normalized key equality (see _key)
  fuzzy  — shares a 5-token n-gram with a benchmark sentence AND scores
           >= FUZZY_RATIO on difflib's sequence ratio

FUZZY_RATIO is set deliberately loose. Observed behaviour on this corpus: hits
above ~0.85 are genuine near-duplicates, while 0.70-0.80 hits are usually two
different sentences sharing a stock clause ("… fell short of international
standards"). Those false positives are kept anyway because the trade is wildly
asymmetric — over-dropping costs a few dozen rows out of ~366k, under-dropping
silently inflates a benchmark score. Read a fuzzy hit's ratio before treating it
as evidence of real leakage.

A row is contaminated if *either* side matches. A shared English reference leaks
the answer just as much as a shared Amharic source.

Run standalone for a report (writes data/benchmarks/reports/contamination.csv):

    python -m processing.utils.decontaminate

processing.utils.pool calls find_contaminated() on the pooled corpus before
splitting, unconditionally — unlike the quality cutoffs this is not a tunable,
and a silent no-op here would quietly invalidate every benchmark number.
"""
import re
import sys
from collections import Counter, defaultdict
from difflib import SequenceMatcher

import pandas as pd
import polars as pl

from processing.clean.normalize import normalize
from processing.utils.paths import BENCHMARKS, FINAL

NGRAM = 5           # n-gram size, in tokens, for the near-duplicate index
FUZZY_RATIO = 0.7   # difflib ratio at or above which a candidate counts as a match
MAX_CANDIDATES = 8  # per row/side, ranked by shared n-grams — caps the ratio work

REPORTS = BENCHMARKS / "reports"   # a subdirectory, so BENCHMARKS.glob("*.csv") can't read it back

_NON_WORD = re.compile(r"[^\w]+", flags=re.UNICODE)


def _key(text: str) -> str:
    """Comparison key: casefold, drop every punctuation/whitespace character.

    Matching happens on this rather than raw text so cosmetic differences — a
    stray quote, doubled spaces, a trailing period — can't hide a real overlap.
    """
    return _NON_WORD.sub("", str(text).casefold())


def _tokens(text: str) -> list[str]:
    """Casefolded word tokens, for n-gram extraction. Amharic is space-separated
    like English, so one tokenizer serves both sides."""
    return _NON_WORD.sub(" ", str(text).casefold()).split()


def _ngrams(text: str) -> set[str]:
    """n-grams of a sentence. Sentences shorter than NGRAM tokens yield a single
    whole-sentence gram instead of nothing, so they stay findable in the index."""
    toks = _tokens(text)
    if len(toks) < NGRAM:
        return {" ".join(toks)} if toks else set()
    return {" ".join(toks[i:i + NGRAM]) for i in range(len(toks) - NGRAM + 1)}


def _normalize_pairs(am: pd.Series, en: pd.Series) -> tuple[pd.Series, pd.Series]:
    """Run benchmark text through the same normalization the training corpus got.

    This is the step that makes the comparison meaningful: everything in
    data/final/ has been through processing.clean.normalize (Amharic homophone
    merging, labialized-fidel collapse, Latin→Ge'ez punctuation), while a raw
    benchmark file has not. Comparing raw benchmark text against normalized
    training text would miss real overlaps on the Amharic side entirely.
    """
    frame = pl.DataFrame({"am": am.astype(str).tolist(), "en": en.astype(str).tolist()})
    out = normalize(frame, "am", "en", name=None)
    return pd.Series(out["am"].to_list()), pd.Series(out["en"].to_list())


class BenchmarkIndex:
    """Exact-key sets + an n-gram inverted index over every benchmark sentence.

    The benchmarks are ~4k sentences and the pooled corpus is hundreds of
    thousands of rows, so the small side gets indexed and the large side streamed
    past it — not the other way around.
    """

    def __init__(self, bench: pd.DataFrame):
        am, en = _normalize_pairs(bench["am"], bench["en"])
        self.labels = [f"{b}:{s}" for b, s in zip(bench["benchmark"], bench["split"])]
        self.am_text = am.tolist()
        self.en_text = en.tolist()

        self.keys: dict[str, dict[str, int]] = {"am": {}, "en": {}}
        self.grams: dict[str, dict[str, list[int]]] = {"am": defaultdict(list), "en": defaultdict(list)}
        for side, texts in (("am", self.am_text), ("en", self.en_text)):
            for i, text in enumerate(texts):
                k = _key(text)
                if not k:
                    continue
                self.keys[side].setdefault(k, i)
                for g in _ngrams(text):
                    self.grams[side][g].append(i)

    def __len__(self) -> int:
        return len(self.labels)

    def match(self, text: str, side: str) -> tuple[int, str, float] | None:
        """Best benchmark match for one sentence, or None.

        Returns (benchmark row index, "exact"|"fuzzy", ratio).
        """
        k = _key(text)
        if not k:
            return None
        hit = self.keys[side].get(k)
        if hit is not None:
            return hit, "exact", 1.0

        shared = Counter()
        for g in _ngrams(text):
            for i in self.grams[side].get(g, ()):
                shared[i] += 1
        if not shared:
            return None

        texts = self.am_text if side == "am" else self.en_text
        best, best_ratio = None, 0.0
        for i, _ in shared.most_common(MAX_CANDIDATES):
            ratio = SequenceMatcher(None, k, _key(texts[i]), autojunk=False).ratio()
            if ratio > best_ratio:
                best, best_ratio = i, ratio
        return (best, "fuzzy", best_ratio) if best_ratio >= FUZZY_RATIO else None


def load_benchmarks() -> pd.DataFrame:
    """Every CSV directly in data/benchmarks/, tagged with its filename.

    Non-recursive on purpose: reports/ and cache/ live one level down precisely
    so generated files can never be mistaken for a benchmark.
    """
    paths = sorted(BENCHMARKS.glob("*.csv"))
    if not paths:
        raise FileNotFoundError(
            f"No benchmark CSVs in {BENCHMARKS} — run `python -m collection.collect_benchmark` first. "
            "Refusing to build a training split with no decontamination."
        )
    frames = []
    for p in paths:
        df = pd.read_csv(p, dtype=str).fillna("")
        if not {"am", "en"}.issubset(df.columns):
            raise ValueError(f"{p} has no am/en columns — is it a benchmark?")
        df["benchmark"] = p.stem
        df["split"] = df.get("split", pd.Series([""] * len(df)))
        frames.append(df[["am", "en", "benchmark", "split"]])
    return pd.concat(frames, ignore_index=True)


def find_contaminated(df: pd.DataFrame, index: BenchmarkIndex | None = None) -> tuple[pd.Series, pd.DataFrame]:
    """Flag rows of `df` (am/en) that overlap a benchmark sentence.

    Returns (boolean mask aligned to df.index, a frame of the matches found).
    """
    index = index or BenchmarkIndex(load_benchmarks())
    flags = [False] * len(df)
    matches = []
    for pos, (am, en) in enumerate(zip(df["am"].astype(str), df["en"].astype(str))):
        for side, text in (("am", am), ("en", en)):
            hit = index.match(text, side)
            if hit is None:
                continue
            i, kind, ratio = hit
            flags[pos] = True
            matches.append({
                "side": side, "kind": kind, "ratio": round(ratio, 3),
                "benchmark": index.labels[i],
                "train_am": am, "train_en": en,
                "bench_am": index.am_text[i], "bench_en": index.en_text[i],
            })
            break  # one hit is enough to drop the row
    return pd.Series(flags, index=df.index), pd.DataFrame(matches)


def main() -> None:
    """Report benchmark overlap in the current data/final/ splits."""
    index = BenchmarkIndex(load_benchmarks())
    print(f"[decontam] indexed {len(index)} benchmark sentences from {BENCHMARKS}")

    all_matches, total_rows, total_hits = [], 0, 0
    for split in ("train", "validation", "test"):
        path = FINAL / f"{split}.csv"
        if not path.exists():
            print(f"[decontam] {split}: missing ({path}) — skipping")
            continue
        df = pd.read_csv(path, dtype=str).fillna("")
        mask, matches = find_contaminated(df, index)
        n = int(mask.sum())
        total_rows += len(df)
        total_hits += n
        breakdown = ""
        if n:
            kinds = matches["kind"].value_counts().to_dict()
            sides = matches["side"].value_counts().to_dict()
            breakdown = f"  [{kinds} {sides} benchmarks={matches['benchmark'].value_counts().to_dict()}]"
        print(f"[decontam] {split}: {n}/{len(df)} contaminated ({n / len(df) * 100:.3f}%){breakdown}")
        if n:
            matches.insert(0, "split", split)
            all_matches.append(matches)

    print(f"[decontam] total: {total_hits}/{total_rows} ({total_hits / max(total_rows, 1) * 100:.3f}%)")
    if all_matches:
        REPORTS.mkdir(parents=True, exist_ok=True)
        out = REPORTS / "contamination.csv"
        pd.concat(all_matches, ignore_index=True).to_csv(out, index=False)
        print(f"[decontam] matched pairs → {out}")
        print("[decontam] re-pool (`python -m processing.process`) to strip these, then re-run model.data.prepare")


if __name__ == "__main__":
    sys.exit(main())
