"""
process.dist.diversity — how semantically diverse is a corpus?

The diversity analogue of process.dist.length (short/medium/long) and
process.dist.semantic (k-means topic regions): those describe how a corpus is
*distributed*, this reduces it to comparable scalars.

Run (from the project root):
    python -m process.dist.diversity data/final_broad/train.csv
    python -m process.dist.diversity data/final_*/train.csv --n 20000
    python -m process.dist.diversity my_new_corpus.csv --embed      # embed uncached text

Accepts .csv (uses the `en` column by default, override with --column) or a
plain-text file, one sentence per line.

Metrics
-------
Vendi Score (Friedman & Dieng, 2023) — THE headline number.
    exp(Shannon entropy of the eigenvalues of the sentence-similarity matrix).
    Interpretable as the EFFECTIVE NUMBER OF DISTINCT ITEMS: n copies of one
    sentence score ~1, n mutually orthogonal sentences score n. Unlike a raw
    average distance it is not fooled by a corpus that is mostly one topic plus a
    few far-flung outliers — entropy weights by how much mass sits in each
    direction, so a lopsided corpus scores low even when its mean distance is high.

    Computed exactly via the d x d Gram matrix: X X^T (n x n) and X^T X (d x d)
    share their non-zero spectrum, so this costs O(n*d^2) instead of an n x n
    eigendecomposition, and n=100k stays cheap.

mean pairwise cosine distance — the intuitive version.
    Computed in O(n*d), not O(n^2), via the identity for unit-norm vectors:
        sum_{i != j} v_i . v_j = ||sum_i v_i||^2 - n
    Reported alongside Vendi because it is easy to sanity-check by hand; prefer
    Vendi when the two disagree.

distinct trigram ratio — a purely lexical control.
    Embeddings call paraphrases similar; this catches literal repetition
    (boilerplate, formulaic register) that the semantic metrics smooth over.

IMPORTANT — always compare at equal n. All three metrics grow with sample size,
so an unmatched comparison measures corpus size rather than diversity. --n
(default 20,000) is applied identically to every input, and any corpus smaller
than it is reported with its own n and flagged, not silently compared.

Embeddings come from the shared mpnet_en cache (process.cache.embed_cache, the
same vectors process.dist.semantic clusters on). Default is cache-only and never
loads a model; pass --embed to encode uncached sentences, which needs the GPU.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from process.cache.embed_cache import _load, lookup
from process.cache.score_cache import row_keys

TAG = "mpnet_en"
DIM = 768
DEFAULT_N = 20_000
SEED = 0


def read_sentences(path: Path, column: str) -> list[str]:
    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path, dtype=str)
        if column not in df.columns:
            raise SystemExit(f"{path}: no '{column}' column (has: {', '.join(df.columns)})")
        return df[column].dropna().tolist()
    return [l for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def get_vectors(sents: list[str], embed: bool) -> tuple[np.ndarray, float]:
    """(vectors, cache coverage). Cache-only unless embed=True."""
    df = pd.DataFrame({"en": sents})
    if embed:
        from process.dist.semantic.en_encoder import get_en_encoder

        def compute(texts: list[str]) -> np.ndarray:
            return get_en_encoder().encode(texts, batch_size=128,
                                           normalize_embeddings=True, show_progress_bar=True)

        from process.cache.embed_cache import embedded
        return embedded(df, TAG, ("en",), compute, DIM), 1.0

    _, cache_keys = _load(TAG, DIM)
    keys = row_keys(df, ("en",))
    hit = keys.isin(cache_keys)
    if not hit.any():
        raise SystemExit("no cached embeddings for this corpus — rerun with --embed")
    return lookup(df[hit.to_numpy()], TAG, ("en",), DIM), float(hit.mean())


def vendi(V: np.ndarray) -> float:
    n = len(V)
    ev = np.linalg.eigvalsh((V.T @ V) / n)
    ev = np.clip(ev, 0.0, None)
    ev = ev[ev > 1e-12]
    ev = ev / ev.sum()
    return float(np.exp(-(ev * np.log(ev)).sum()))


def mean_pairwise_distance(V: np.ndarray) -> float:
    n = len(V)
    s = V.sum(0)
    return 1.0 - (float(s @ s) - n) / (n * (n - 1))


def distinct_trigram_ratio(sents: list[str]) -> float:
    seen, total = set(), 0
    for s in sents:
        w = s.split()
        for i in range(len(w) - 2):
            seen.add((w[i], w[i + 1], w[i + 2]))
            total += 1
    return len(seen) / max(total, 1)


def score(path: Path, column: str, n: int, embed: bool) -> dict:
    sents = read_sentences(path, column)
    V, coverage = get_vectors(sents, embed)
    V = V / np.linalg.norm(V, axis=1, keepdims=True)   # score_embed normalizes; be defensive

    rng = np.random.default_rng(SEED)
    short = len(V) < n
    if not short:
        take = np.sort(rng.choice(len(V), size=n, replace=False))
        V = V[take]
    return {
        "corpus": path.parent.name + "/" + path.name if path.parent.name else path.name,
        "n": len(V), "short": short, "coverage": coverage,
        "vendi": vendi(V),
        "mean_pairwise_distance": mean_pairwise_distance(V),
        "distinct_trigram_ratio": distinct_trigram_ratio(sents[:n]),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    ap.add_argument("paths", nargs="+", type=Path)
    ap.add_argument("--column", default="en", help="CSV column to score (default: en)")
    ap.add_argument("--n", type=int, default=DEFAULT_N,
                    help=f"sample size, applied identically to every corpus (default {DEFAULT_N})")
    ap.add_argument("--embed", action="store_true",
                    help="encode uncached sentences (loads the model, needs GPU)")
    args = ap.parse_args()

    rows = [score(p, args.column, args.n, args.embed) for p in args.paths]
    w = max(len(r["corpus"]) for r in rows) + 2
    print(f"\nsample size n={args.n:,}, held constant across corpora "
          f"(all three metrics grow with n)\n")
    print(f"{'corpus':<{w}}{'n':>8}{'cov':>6}{'Vendi (eff. #)':>16}"
          f"{'mean pair dist':>16}{'distinct 3gram':>16}")
    print("-" * (w + 62))
    for r in rows:
        flag = " *" if r["short"] else ""
        print(f"{r['corpus']:<{w}}{r['n']:>8,}{r['coverage']:>5.0%}{r['vendi']:>16.1f}"
              f"{r['mean_pairwise_distance']:>16.4f}{r['distinct_trigram_ratio']:>16.3f}{flag}")
    if any(r["short"] for r in rows):
        print("\n* corpus smaller than --n; its scores use its own n and are NOT "
              "comparable to the others without re-running at a matched, smaller --n.")


if __name__ == "__main__":
    main()
