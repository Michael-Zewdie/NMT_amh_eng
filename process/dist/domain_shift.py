"""
process.dist.domain_shift — is a benchmark drawn from a DIFFERENT distribution
than the training corpus, i.e. was the model actually forced to generalize?

Run (from the project root):
    python -m process.dist.domain_shift --train data/final_broad/train.csv \
        --benchmark flores200_am_en:devtest mafand_en_amh:test

Companion to process.dist.diversity, which measures spread WITHIN one corpus;
this measures distance BETWEEN two. Both read the shared mpnet_en English
embedding cache.

Three statistics, because each fails differently
------------------------------------------------
domain-classifier AUC
    Fit logistic regression to tell training sentences from benchmark sentences,
    scored by cross-validated ROC-AUC on balanced samples. 0.5 = the two are
    indistinguishable in embedding space (same distribution); 1.0 = trivially
    separable (disjoint distributions). This is the headline number: it is a
    direct, interpretable measure of "are these different domains".

MMD (maximum mean discrepancy, RBF kernel) + permutation test
    A proper two-sample statistic with a p-value, so "different" is a hypothesis
    test rather than an eyeballed gap. AUC can look high on small samples by
    chance; the permutation test says whether the gap is real.

nearest-neighbour max cosine similarity
    For each benchmark sentence, its single most similar training sentence. This
    is the LEAKAGE check, and it catches what the other two cannot: two corpora
    can be distributionally distinct overall while still sharing a handful of
    near-identical sentences. Decontamination (process.pool.decontaminate) removes
    exact and fuzzy string matches; this catches semantic near-duplicates that
    survive string matching. Values near 1.0 on many rows would mean the benchmark
    is partly memorizable and its score overstates generalization.

Why the control matters
-----------------------
A raw AUC of 0.95 means nothing without knowing what "same distribution" scores on
this data and this sample size. `--control` runs the identical machinery on two
splits of the SAME benchmark (FLORES dev vs devtest), which are same-distribution
by construction. That is the floor every other row should be read against.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from process.cache.embed_cache import _load, embedded
from process.cache.score_cache import row_keys
from process.utils.paths import BENCHMARKS

TAG = "mpnet_en"
DIM = 768
SEED = 0


def _encoder(texts: list[str]) -> np.ndarray:
    from process.dist.semantic.en_encoder import get_en_encoder
    return get_en_encoder().encode(texts, batch_size=128, normalize_embeddings=True,
                                   show_progress_bar=False)


def vectors(sents: list[str], allow_embed: bool) -> np.ndarray:
    df = pd.DataFrame({"en": sents})
    if allow_embed:
        V = embedded(df, TAG, ("en",), _encoder, DIM)
    else:
        _, ck = _load(TAG, DIM)
        keep = row_keys(df, ("en",)).isin(ck).to_numpy()
        if keep.sum() < 200:
            raise SystemExit("too few cached embeddings — rerun with --embed")
        from process.cache.embed_cache import lookup
        V = lookup(df[keep], TAG, ("en",), DIM)
    return V / np.linalg.norm(V, axis=1, keepdims=True)


def load_benchmark_en(spec: str) -> list[str]:
    name, _, split = spec.partition(":")
    df = pd.read_csv(BENCHMARKS / f"{name}.csv", dtype=str).fillna("")
    if split:
        df = df[df["split"] == split]
    return df["en"].tolist()


def classifier_auc(A: np.ndarray, B: np.ndarray, n: int) -> float:
    """Cross-validated ROC-AUC separating A from B, on balanced samples."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score
    rng = np.random.default_rng(SEED)
    a = A[rng.choice(len(A), n, replace=False)]
    b = B[rng.choice(len(B), n, replace=False)]
    X = np.vstack([a, b])
    y = np.r_[np.zeros(n), np.ones(n)]
    clf = LogisticRegression(max_iter=2000, C=1.0)
    return float(cross_val_score(clf, X, y, cv=5, scoring="roc_auc").mean())


def mmd_permutation(A: np.ndarray, B: np.ndarray, n: int, perms: int = 200) -> tuple[float, float]:
    """Squared MMD with an RBF kernel, plus a permutation p-value."""
    rng = np.random.default_rng(SEED)
    a = A[rng.choice(len(A), n, replace=False)]
    b = B[rng.choice(len(B), n, replace=False)]
    Z = np.vstack([a, b])
    d2 = np.maximum(2 - 2 * (Z @ Z.T), 0)          # unit vectors -> squared euclidean
    gamma = 1.0 / np.median(d2[d2 > 0])
    K = np.exp(-gamma * d2)

    def stat(idx_a, idx_b):
        Kaa = K[np.ix_(idx_a, idx_a)].mean()
        Kbb = K[np.ix_(idx_b, idx_b)].mean()
        Kab = K[np.ix_(idx_a, idx_b)].mean()
        return Kaa + Kbb - 2 * Kab

    ia, ib = np.arange(n), np.arange(n, 2 * n)
    observed = stat(ia, ib)
    count = 0
    for _ in range(perms):
        p = rng.permutation(2 * n)
        if stat(p[:n], p[n:]) >= observed:
            count += 1
    return float(observed), (count + 1) / (perms + 1)


def nn_similarity(train: np.ndarray, bench: np.ndarray, pool: int = 60_000) -> np.ndarray:
    """For each benchmark row, the max cosine similarity to any training row."""
    rng = np.random.default_rng(SEED)
    T = train[rng.choice(len(train), min(pool, len(train)), replace=False)]
    out = np.empty(len(bench), dtype=np.float32)
    for i in range(0, len(bench), 512):
        out[i:i + 512] = (bench[i:i + 512] @ T.T).max(axis=1)
    return out


def report(label: str, T: np.ndarray, B: np.ndarray, n: int) -> None:
    n = min(n, len(T), len(B))
    auc = classifier_auc(T, B, n)
    mmd, p = mmd_permutation(T, B, min(n, 800))
    nn = nn_similarity(T, B)
    print(f"\n=== {label} ===")
    print(f"  domain-classifier AUC : {auc:.3f}   (0.5 = same distribution, 1.0 = disjoint)")
    print(f"  MMD^2                 : {mmd:.4f}   permutation p = {p:.3f}")
    print(f"  nearest-neighbour cosine similarity to training data:")
    print(f"      median {np.median(nn):.3f} | p95 {np.percentile(nn, 95):.3f} | max {nn.max():.3f}")
    for thr in (0.9, 0.95):
        print(f"      rows with NN sim > {thr}: {(nn > thr).sum()} / {len(nn)} ({(nn > thr).mean():.2%})")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    ap.add_argument("--train", type=Path, required=True, help="training CSV with an `en` column")
    ap.add_argument("--benchmark", nargs="+", default=["flores200_am_en:devtest", "mafand_en_amh:test"])
    ap.add_argument("--control", default="flores200_am_en:dev",
                    help="same-distribution baseline, compared against the first benchmark")
    ap.add_argument("--n", type=int, default=1000, help="balanced sample size per side")
    ap.add_argument("--embed", action="store_true", help="encode uncached sentences (needs GPU)")
    args = ap.parse_args()

    if args.train.suffix.lower() == ".csv":
        train_en = pd.read_csv(args.train, usecols=["en"], dtype=str).dropna()["en"].tolist()
    else:   # plain text, one sentence per line (e.g. Gezmu's released .en split)
        train_en = [l for l in args.train.read_text(encoding="utf-8").splitlines() if l.strip()]
    T = vectors(train_en, args.embed)
    print(f"[train] {args.train}: {len(T):,} embedded sentences")

    for spec in args.benchmark:
        report(f"{args.train.parent.name} vs {spec}", T, vectors(load_benchmark_en(spec), args.embed), args.n)

    if args.control:
        first = args.benchmark[0]
        A, B = vectors(load_benchmark_en(first), args.embed), vectors(load_benchmark_en(args.control), args.embed)
        report(f"CONTROL — {first} vs {args.control} (same distribution by construction)", A, B, args.n)


if __name__ == "__main__":
    main()
