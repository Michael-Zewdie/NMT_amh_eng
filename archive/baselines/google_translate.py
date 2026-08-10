"""
baselines.google_translate — score Google Translate on the same benchmark
sentences our own model is scored on, with the same scorer.

Run (from the project root):

    export GOOGLE_TRANSLATE_API_KEY=...
    python -m baselines.google_translate --dry-run    # character count, no quota spent
    python -m baselines.google_translate --confirm    # translate + score
    python -m baselines.google_translate data/benchmarks/mafand_en_amh.csv test --confirm

Uses Cloud Translation **v2** (the basic edition) — an API key is enough, no
service-account JSON or ADC. FLORES-200 devtest is ~1k sentences / ~150k
characters, inside the 500k-characters/month free tier, so a full run costs
nothing; the script prints the exact count and refuses to spend quota without
--confirm.

Two deliberate choices, both about fairness of the comparison:

  - Google is sent the **raw** benchmark Amharic, not the output of
    processing.clean.normalize. Our normalization exists to match our own
    training distribution; imposing it on a black-box system trained on its own
    data would handicap it and make the gap look larger than it is.
  - Scoring goes through model.evaluate.corpus_scores, the identical function
    model.evaluate_benchmark uses. Different metric settings between a system and
    its baseline is the standard way MT comparisons quietly become meaningless.

Translations are cached per source sentence under data/benchmarks/cache/, so
re-scoring is free and a partial/failed run resumes instead of re-spending quota.
"""
import argparse
import html
import os
import sys
from pathlib import Path

import pandas as pd
import requests

from model.evaluate import corpus_scores
from process.utils.paths import BENCHMARKS

API_URL = "https://translation.googleapis.com/language/translate/v2"
API_KEY_ENV = "GOOGLE_TRANSLATE_API_KEY"
CACHE = BENCHMARKS / "cache"   # a subdirectory, so BENCHMARKS.glob("*.csv") can't read it back

DEFAULT_BENCHMARK = BENCHMARKS / "flores200_am_en.csv"
DEFAULT_SPLIT = "devtest"
BATCH = 64          # segments per request; v2 caps at 128 and also limits total request size
FREE_TIER_CHARS = 500_000


def load_split(path: Path, split: str | None) -> pd.DataFrame:
    df = pd.read_csv(path, dtype=str).fillna("")
    if split and "split" in df.columns:
        df = df[df["split"] == split].reset_index(drop=True)
        if df.empty:
            raise ValueError(f"{path} has no rows with split={split!r}")
    return df


def translate_batch(sentences: list[str], api_key: str) -> list[str]:
    """One v2 request. `q` repeats per segment; responses come back in order."""
    resp = requests.post(API_URL, params={"key": api_key}, timeout=120,
                         data={"q": sentences, "source": "am", "target": "en", "format": "text"})
    if not resp.ok:
        # Google's error body names the actual problem (API not enabled, key
        # restricted to the wrong referrer, billing off) — far more useful than
        # the bare status code raise_for_status would give.
        sys.exit(f"[google] HTTP {resp.status_code}: {resp.text}")
    # format=text still lets entities through in practice (&#39; for an apostrophe),
    # and those would score as mismatched tokens against a clean reference.
    return [html.unescape(t["translatedText"]) for t in resp.json()["data"]["translations"]]


def translate_all(sentences: list[str], cache_path: Path, api_key: str) -> list[str]:
    """Translate every sentence, reading and extending the on-disk cache.

    Keyed by source sentence rather than row position so the cache survives a
    benchmark being re-fetched or re-ordered.
    """
    cached: dict[str, str] = {}
    if cache_path.exists():
        prior = pd.read_csv(cache_path, dtype=str).fillna("")
        cached = dict(zip(prior["am"], prior["hyp"]))
        print(f"[google] cache: {len(cached)} sentences already translated ({cache_path})")

    todo = [s for s in dict.fromkeys(sentences) if s not in cached]
    for start in range(0, len(todo), BATCH):
        chunk = todo[start : start + BATCH]
        for src, hyp in zip(chunk, translate_batch(chunk, api_key)):
            cached[src] = hyp
        print(f"\r  translating: {min(start + len(chunk), len(todo))}/{len(todo)}", end="", flush=True)
        # Persist after every batch so an interruption never re-spends quota.
        CACHE.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"am": list(cached), "hyp": list(cached.values())}).to_csv(cache_path, index=False)
    if todo:
        print()

    return [cached[s] for s in sentences]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1].strip())
    parser.add_argument("benchmark", nargs="?", default=str(DEFAULT_BENCHMARK))
    parser.add_argument("split", nargs="?", default=DEFAULT_SPLIT)
    parser.add_argument("--dry-run", action="store_true",
                        help="report the character count and exit without calling the API")
    parser.add_argument("--confirm", action="store_true",
                        help="required to actually spend translation quota")
    args = parser.parse_args(argv)

    path = Path(args.benchmark)
    df = load_split(path, args.split)
    sources = df["am"].tolist()          # raw, deliberately un-normalized
    cache_path = CACHE / f"google_{path.stem}_{args.split}.csv"

    chars = sum(len(s) for s in sources)
    print(f"[google] {path.stem}:{args.split} — {len(sources)} sentences, {chars:,} characters "
          f"({chars / FREE_TIER_CHARS * 100:.1f}% of the {FREE_TIER_CHARS:,}-char monthly free tier)")

    if args.dry_run:
        return
    if not args.confirm:
        sys.exit("[google] refusing to spend quota without --confirm (or use --dry-run)")

    api_key = os.environ.get(API_KEY_ENV)
    if not api_key:
        sys.exit(f"[google] ${API_KEY_ENV} is not set — create a Cloud Translation API key and export it")

    hyps = translate_all(sources, cache_path, api_key)
    scores = corpus_scores(hyps, df["en"].tolist())
    print(f"[google] Google Translate on {path.stem}:{args.split} ({len(hyps)} sentences)")
    print(f"[google]   BLEU   {scores['bleu']:.2f}")
    print(f"[google]   chrF++ {scores['chrf++']:.2f}")
    print(f"[google]   translations cached → {cache_path}")


if __name__ == "__main__":
    main(sys.argv[1:])
