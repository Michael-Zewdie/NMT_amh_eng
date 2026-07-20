"""
domain_dist.py — NLLB source-domain distribution, before vs after processing.

Standalone analysis — NOT a stage of the process.py pipeline. It bars the top-N
registered source domains in the raw mined parquet (before) against the cleaned
nllb.csv (after), so you can see which sites the laser/length/script/dedupe
filtering kept or dropped.

Reading the raw parquet is heavy (~16M rows); only source_url/target_url are
loaded. Give a top_n on the command line to widen/narrow the bars.

Run (from the project root): python -m explore.domain_dist [top_n]
Output: data/processed/figs/nllb_domains_bar.png
"""
import sys

import matplotlib
matplotlib.use("Agg")            # headless: write PNGs without a display
import matplotlib.pyplot as plt
import polars as pl

from processing.utils.paths import NLLB_FULL, PROCESSED, FIGS
from processing.dist.websites import domains

TOP_N = 15
_RAW_COLOR, _PROC_COLOR = "#8C8C8C", "#4C78A8"


def domain_counts(path) -> pl.DataFrame:
    """Registered-domain counts for one file, sorted most-common first."""
    read = pl.read_parquet if path.suffix == ".parquet" else pl.read_csv
    df = read(path, columns=["source_url", "target_url"])
    dom = domains(df).rename("domain")
    return dom.value_counts(sort=True)          # columns: [domain, count]


def _panel(ax, vc: pl.DataFrame, title: str, top_n: int, color: str) -> None:
    top = vc.tail(top_n)
    names = top["domain"].to_list() #[::-1]       # reverse: biggest bar on top
    counts = top["count"].to_list() #[::-1]
    total = int(vc["count"].sum())
    ax.barh(names, counts, color=color)
    ax.set_title(f"{title}\ntop {top_n} of {vc.height:,} domains · {total:,} pairs", fontsize=11)
    ax.set_xlabel("pairs")
    for i, c in enumerate(counts):
        ax.text(c, i, f" {c:,} ({c / total * 100:.1f}%)", va="center", fontsize=8)
    ax.margins(x=0.18)                          # headroom for the value labels


def main(top_n: int = TOP_N) -> None:
    print(f"reading raw parquet ({NLLB_FULL.name})")
    raw = domain_counts(NLLB_FULL)
    print(f"reading processed ({PROCESSED / 'nllb.csv'})…")
    proc = domain_counts(PROCESSED / "nllb.csv")

    fig, axes = plt.subplots(1, 2, figsize=(15, 7))
    _panel(axes[0], raw,  "NLLB source domains — raw (before)",       top_n, _RAW_COLOR)
    _panel(axes[1], proc, "NLLB source domains — processed (after)",  top_n, _PROC_COLOR)
    fig.suptitle("NLLB source-domain distribution: before vs after processing", fontsize=13)
    fig.tight_layout()

    FIGS.mkdir(parents=True, exist_ok=True)
    out = FIGS / "nllb_domains_bar.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"raw:       {int(raw['count'].sum()):>12,} pairs / {raw.height:,} domains")
    print(f"processed: {int(proc['count'].sum()):>12,} pairs / {proc.height:,} domains")
    print(f"-> {out}")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else TOP_N)
