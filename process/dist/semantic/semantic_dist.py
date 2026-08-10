"""
semantic_dist.py — bar chart of the corpus's semantic-cluster distribution.

Pools every am/en CSV in data/processed/ (same source data length_dist.py
reports on) and computes the same per-am mean-pooled English-side cluster
assignment processing.utils.pool.split_semantic uses
(processing.utils.pool.am_cluster_ids, over processing.dist.semantics'
N_CLUSTERS/cluster_ids), then writes a bar chart of the per-cluster distinct-am
counts.

Only meaningful when STRATIFY_SEMANTIC is on — process.py chains this after
pool only in that case. Running it standalone requires the score_embed cache to
already be populated: python -m processing.utils.score_embed first.

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

from process.utils.paths import PROCESSED, FIGS
from process.pool import am_cluster_ids
from process.dist.semantic.semantics import N_CLUSTERS

_BAR_COLOR = "#4C6FE1"
N_SAMPLES = 8   # sentences shown per cluster, both in the CSV and on stdout


def load_am_en(path) -> pd.DataFrame | None:
    """Read one CSV's am/en columns, or None if it isn't am/en parallel text."""
    df = pd.read_csv(path, dtype=str)
    return df[["am", "en"]] if {"am", "en"}.issubset(df.columns) else None


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
    return (reps.groupby("cluster", group_keys=False)
                .apply(lambda g: g.sample(n=min(n, len(g)), random_state=seed))
                .sort_values("cluster")[["cluster", "am", "en"]]
                .reset_index(drop=True))


def main() -> None:
    frames = []
    for path in sorted(PROCESSED.glob("*.csv")):
        df = load_am_en(path)
        if df is not None:
            frames.append(df)
    # Cross-source dedup on the pair, same as pool.py's own pool() — duplicate
    # rows shouldn't inflate a single am group's weight in the cluster fit.
    pooled = pd.concat(frames, ignore_index=True).drop_duplicates(subset=["am", "en"])

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
