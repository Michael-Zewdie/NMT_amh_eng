"""
length_dist.py — Pie chart of the corpus's Amharic sentence-length distribution.

Pools every am/en CSV in data/processed/, buckets each pair by its Amharic
character length into short / medium / long (LENGTH_CUTOFFS — edit that one
global to reshape the buckets), and writes a pie chart of the split.

Run (from the project root): python -m processing.dist.length_dist
Outputs: data/processed/figs/length_buckets_pie.png
"""
from collections import Counter

import matplotlib
matplotlib.use("Agg")            # headless: write PNGs without a display
import matplotlib.pyplot as plt
import polars as pl

from processing.utils.paths import PROCESSED, FIGS
from processing.dist.lengths import LENGTH_CUTOFFS, BUCKETS, bucketize

_PIE_COLORS = ("#4C9F70", "#E1B84B", "#C1543B")   # short / medium / long


def load_am(path) -> pl.Series | None:
    """Read one CSV's Amharic column, or None if it isn't am/en parallel text."""
    df = pl.read_csv(path, infer_schema_length=0)
    return df["am"] if {"am", "en"}.issubset(df.columns) else None


def pie(counts: dict, path) -> None:
    """Pie chart of short/medium/long bucket counts."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lo, hi = LENGTH_CUTOFFS
    n = sum(counts[b] for b in BUCKETS)
    sizes = [counts[b] for b in BUCKETS]
    labels = [f"{b}\n{counts[b]:,}" for b in BUCKETS]
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.pie(sizes, labels=labels, colors=_PIE_COLORS, autopct="%1.1f%%",
           startangle=90, counterclock=False,
           wedgeprops={"edgecolor": "white", "linewidth": 1.5})
    ax.set_title(f"Amharic sentence-length buckets (n={n:,})\n"
                 f"short < {lo}  ·  medium  ·  long ≥ {hi} chars", fontsize=11)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    am_columns = []
    for path in sorted(PROCESSED.glob("*.csv")):
        am = load_am(path)
        if am is not None:
            am_columns.append(am)
    am = pl.concat(am_columns)

    labels = bucketize(am.str.len_chars().to_numpy())
    label_counts = Counter(labels)          # how many sentences landed in each bucket
    counts = {}
    for bucket in BUCKETS:
        counts[bucket] = label_counts[bucket]

    fig_path = FIGS / "length_buckets_pie.png"
    pie(counts, fig_path)
    print(f"short {counts['short']:,} / medium {counts['medium']:,} / long {counts['long']:,} "
          f"-> {fig_path}")


if __name__ == "__main__":
    main()
