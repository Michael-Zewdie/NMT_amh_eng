"""
semantic_dist.py — bar chart of the corpus's semantic-cluster distribution.

Reads data/final/{train,validation,test}.csv (the actual post-cutoff,
post-decontamination training pool — their union is exactly the same set
processing.utils.pool.split_semantic() clustered) and computes the same per-am
mean-pooled English-side cluster assignment it used
(processing.utils.pool.am_cluster_ids, over processing.dist.semantics'
N_CLUSTERS/cluster_ids), then writes a bar chart of the per-cluster distinct-am
counts.

Deliberately *not* data/processed/ (the full pre-cutoff pool, ~4x bigger at
last count: 2.8M rows vs. ~659k in data/final) — this file exists to report on
what's actually in the training data, not the raw material before quality
filtering, and reading the smaller, already-final set also means this call
shares am_cluster_ids()'s on-disk cache with whatever pool.split_semantic()
already computed for the same data, instead of tripling the work with a
different (larger) am set that can't hit that cache.

Only meaningful when STRATIFY_SEMANTIC is on. Running it standalone requires
the score_embed cache to already be populated: python -m
processing.utils.score_embed first.

Also dumps N_SAMPLES random English sentences per cluster — a chart tells you
the clusters are balanced, not what they *mean*; the sample CSV lets you
eyeball whether cluster 3 is "medical text" and cluster 9 is "religious text",
or whether k=16 is cutting the space somewhere nonsensical.

Run (from the project root): python -m processing.dist.semantic_dist
Outputs: data/processed/figs/semantic_clusters_bar.png
         data/processed/figs/semantic_clusters_samples.csv
"""
from collections import Counter

import matplotlib
matplotlib.use("Agg")            # headless: write PNGs without a display
import matplotlib.pyplot as plt
import pandas as pd

from processing.utils.paths import FINAL, FIGS
from processing.utils.pool import am_cluster_ids
from processing.dist.semantics import N_CLUSTERS

_BAR_COLOR = "#4C6FE1"
N_SAMPLES = 8   # sentences shown per cluster, both in the CSV and on stdout


def bar(counts: dict, path) -> None:
    """Bar chart of per-cluster distinct-am counts."""
    path.parent.mkdir(parents=True, exist_ok=True)
    clusters = sorted(counts)
    sizes = [counts[c] for c in clusters]
    n = sum(sizes)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar([str(c) for c in clusters], sizes, color=_BAR_COLOR)
    ax.set_xlabel("semantic cluster")
    ax.set_ylabel("distinct Amharic sentences")
    ax.set_title(f"Semantic cluster distribution (k={N_CLUSTERS}, n={n:,} am groups)")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def sample_sentences(df: pd.DataFrame, clusters: dict, n: int = N_SAMPLES,
                      seed: int = 42) -> pd.DataFrame:
    """n random (cluster, am, en) rows per cluster, for eyeballing what a cluster
    actually contains.

    One row per *am* (not per English reference) — a multi-reference am (e.g.
    quran.csv's ~15 translator versions per verse) would otherwise flood the
    sample with near-identical rows and make a cluster look more homogeneous
    than the underlying sentence diversity actually is.
    """
    reps = df.drop_duplicates(subset=["am"]).copy()
    reps["cluster"] = reps["am"].map(clusters)
    # Explicit loop rather than groupby(...).apply(...): pandas' apply silently
    # drops the grouping column from the reassembled frame in some versions when
    # group_keys=False and the function returns a same-shaped slice of the group.
    picked = [grp.sample(n=min(n, len(grp)), random_state=seed)
              for _, grp in reps.groupby("cluster")]
    return (pd.concat(picked, ignore_index=True)[["cluster", "am", "en"]]
              .sort_values("cluster")
              .reset_index(drop=True))


def main() -> None:
    # train/validation/test are already disjoint on am and each internally
    # deduped (pool.py's own pool() did that before the split), so concatenating
    # them just reconstructs the exact set split_semantic() clustered — same
    # am_cluster_ids() cache key, no re-dedup needed.
    frames = [pd.read_csv(FINAL / f"{name}.csv", dtype=str, usecols=["am", "en"])
              for name in ("train", "validation", "test")]
    pooled = pd.concat(frames, ignore_index=True)

    clusters = am_cluster_ids(pooled)  # {am: cluster_id}
    label_counts = Counter(clusters.values())
    counts = {c: label_counts.get(c, 0) for c in range(N_CLUSTERS)}

    fig_path = FIGS / "semantic_clusters_bar.png"
    bar(counts, fig_path)

    samples = sample_sentences(pooled, clusters)
    samples_path = FIGS / "semantic_clusters_samples.csv"
    samples.to_csv(samples_path, index=False)

    summary = ", ".join(f"{c}:{n:,}" for c, n in counts.items())
    print(f"semantic clusters (k={N_CLUSTERS}) -> {summary}")
    print(f"chart -> {fig_path}")
    print(f"samples ({N_SAMPLES}/cluster) -> {samples_path}\n")
    for c in range(N_CLUSTERS):
        print(f"--- cluster {c} ({counts[c]:,} am groups) ---")
        for en in samples.loc[samples.cluster == c, "en"]:
            print(f"  {en[:110]}")
        print()


if __name__ == "__main__":
    main()
