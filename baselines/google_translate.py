"""
baselines.google_translate — score Google Cloud Translate on the same
benchmarks our own model is scored on (FLORES-200 devtest, MAFAND-MT test),
with the same scorer.

Run (from the project root), after putting a key in .env:

    python -m baselines.google_translate

Uses Cloud Translation v2 (the basic edition) — an API key is enough, no
service-account JSON or ADC. Both benchmarks combined are well inside the
500k-characters/month free tier.

Two deliberate choices, both about fairness of the comparison:

  - Google is sent the **raw** benchmark Amharic, not the output of
    processing.clean.normalize. Our normalization exists to match our own
    training distribution; imposing it on a black-box system trained on its own
    data would handicap it and make the gap look larger than it is.
  - Scoring goes through model.evaluate.corpus_scores, the identical function
    model.evaluate.evaluate_OOD uses. Different metric settings between a system
    and its baseline is the standard way MT comparisons quietly become
    meaningless.

Translations are cached per source sentence under data/benchmarks/cache/, so
re-scoring is free and a re-run resumes instead of re-spending quota.
"""
import html
import os
import sys
from pathlib import Path

import pandas as pd
import requests

from model.common import corpus_scores
from process.utils.paths import BENCHMARKS

API_URL = "https://translation.googleapis.com/language/translate/v2"
API_KEY_ENV = "GOOGLE_TRANSLATE_API_KEY"
ENV_FILE = Path(__file__).resolve().parent.parent / ".env"
CACHE = BENCHMARKS / "cache"   # a subdirectory, so BENCHMARKS.glob("*.csv") can't read it back
BATCH = 64                     # segments per request; v2 caps at 128 and also limits total request size

# The same OOD benchmarks/splits model.evaluate.evaluate_OOD is scored on.
EVAL_SETS = [
    (BENCHMARKS / "flores200_am_en.csv", "devtest"),
    (BENCHMARKS / "mafand_en_amh.csv", "test"),
]


def load_api_key() -> str:
    key = os.environ.get(API_KEY_ENV)
    if not key and ENV_FILE.exists():
        for line in ENV_FILE.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and line.split("=", 1)[0] == API_KEY_ENV:
                key = line.split("=", 1)[1].strip()
    if not key:
        sys.exit(f"[google] {API_KEY_ENV} is not set — put it in .env or export it")
    return key


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


def evaluate_one(path: Path, split: str, api_key: str) -> None:
    df = pd.read_csv(path, dtype=str).fillna("")
    df = df[df["split"] == split].reset_index(drop=True)
    sources = df["am"].tolist()   # raw, deliberately un-normalized
    cache_path = CACHE / f"google_{path.stem}_{split}.csv"

    print(f"[google] {path.stem}:{split} — {len(sources)} sentences")
    hyps = translate_all(sources, cache_path, api_key)
    scores = corpus_scores(hyps, df["en"].tolist())
    print(f"[google] {path.stem}:{split} ({len(hyps)} sentences)")
    print(f"[google]   BLEU   {scores['bleu']:.2f}")
    print(f"[google]   chrF++ {scores['chrf++']:.2f}\n")


def main() -> None:
    api_key = load_api_key()
    for path, split in EVAL_SETS:
        evaluate_one(path, split, api_key)


if __name__ == "__main__":
    main()
