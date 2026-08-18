"""Where the pairs in the broad v1 corpus came from — by source, and inside NLLB by site.

Run (from the project root): python -m experiments.domain_breadth chart [--top N]
Outputs: <process.utils.paths.FIGS>/final_broad_sources.png (currently data/figs/)

FINAL_BROAD/*.csv (archived 2026-08-17, see paths.py) carries only am/en — corpus.py drops the source column
before writing — so provenance is recovered by joining each pair back against
data/processed/*.csv on (am, en). The lookup dedupes keep-first over the sorted
glob, mirroring process.pool.pool()'s own dedup, so a pair carried by two sources
is attributed the same way the pool attributed it.

This is the chart that says whether the BROAD arm is still broad. A domain-breadth
experiment whose corpus has collapsed to one source has quietly stopped testing
its own hypothesis — see corpus.py's CUTOFFS note on why a uniform AfriCOMET
floor does exactly that.

But "nllb" names a collection method, not a domain, so a single 62.5% wedge cannot
settle that question either way: web-mined text is broad if it came from thousands
of separate sites and narrow if one crawl dominates it. The right-hand panel opens
that wedge up by registered domain — the same source_url/target_url mapping
process.dist.nllb_domain_dist uses on the raw corpus — so the wedge reads as real
breadth or as one site wearing a big number.
"""
import matplotlib
matplotlib.use("Agg")            # headless: write PNGs without a display
import matplotlib.pyplot as plt
import polars as pl

from experiments.domain_breadth.paths import FINAL_BROAD
from process.utils.paths import FIGS, PROCESSED
from process.utils.websites import domains

SPLITS = ["train", "validation", "test"]
FIG_PATH = FIGS / "final_broad_sources.png"
TOP_N = 15                       # NLLB sites to bar; the tail is a subtitle line, not a bar
                                 # — folding ~23k sites into one "other" would dwarf every
                                 # real bar and hide the shape the panel exists to show.

SURFACE = "#fcfcfb"
TEXT_PRIMARY, TEXT_SECONDARY = "#1a1a19", "#63625c"

# Wedges are drawn in this order, and each source keeps its hue regardless of how
# large its slice is, so a re-tuned corpus is comparable to an old chart.
#
# The order is load-bearing: in a pie only neighbouring wedges touch, so the ring
# is validated as a circular adjacent pairlist rather than all-pairs (which these
# six cannot clear — green↔orange falls to protan ΔE 3.2). As listed, including the
# green→blue wrap, worst adjacent CVD ΔE is 9.1 light / 8.4 dark and worst
# normal-vision ΔE 19.6 light / 19.3 dark. Reordering these keys can silently break
# that; re-run the palette validator if you do.
SOURCE_COLORS = {
    "nllb":           "#2a78d6",   # blue
    "ccaligned":      "#eb6834",   # orange
    "religious":      "#1baf7a",   # aqua
    "quran":          "#eda100",   # yellow
    "afridoc_health": "#e87ba4",   # magenta
    "afridoc_tech":   "#008300",   # green
}
UNATTRIBUTED = "#8b8a83"
DIRECT_LABEL_MIN = 0.05          # only wedges >= 5% get an on-chart label; the rest
                                 # would collide, and the legend carries their numbers

# The site bars are one series, so they take the NLLB wedge's own hue: that panel IS
# the wedge, magnified, and sharing the colour is what says so. No new categorical
# steps enter, so the validated ring above is untouched. "unknown" is not a site —
# those rows carry no URL — and wears the neutral so it can't be read as one.
UNKNOWN_DOMAIN = "unknown"


def source_lookup() -> pl.DataFrame:
    """(am, en) -> (source, domain), deduped keep-first over the sorted glob like pool() does.

    ``domain`` is the registered site the pair was mined from, for the sources that
    record a URL; null for the curated corpora, which have no site to report.
    """
    frames = []
    for p in sorted(PROCESSED.glob("*.csv")):
        df = pl.read_csv(p, infer_schema_length=0)
        if not {"am", "en"}.issubset(df.columns):
            continue
        domain = (domains(df).alias("domain") if {"source_url", "target_url"} <= set(df.columns)
                  else pl.lit(None, dtype=pl.String).alias("domain"))
        frames.append(df.select("am", "en").with_columns(pl.lit(p.stem).alias("source"), domain))
    if not frames:
        raise ValueError(f"no am/en sources in {PROCESSED}")
    return pl.concat(frames).unique(subset=["am", "en"], keep="first", maintain_order=True)


def counts() -> tuple[dict[str, int], int, pl.DataFrame]:
    if not FINAL_BROAD.exists():
        raise SystemExit(f"no {FINAL_BROAD} — run "
                         f"python -m experiments.domain_breadth corpus --write")
    corpus = pl.concat([
        pl.read_csv(FINAL_BROAD / f"{s}.csv", infer_schema_length=0).select("am", "en")
        for s in SPLITS
    ])
    tagged = corpus.join(source_lookup(), on=["am", "en"], how="left")
    tally = dict(tagged["source"].fill_null("unattributed").value_counts().iter_rows())
    ordered = {s: tally[s] for s in SOURCE_COLORS if tally.get(s)}
    if tally.get("unattributed"):
        ordered["unattributed"] = tally["unattributed"]
    # Counted off the same tagged frame as the wedges, so the sites sum to the NLLB
    # wedge and not to "every NLLB pair" — the latter re-counts pairs that the pool's
    # keep-first dedup already handed to an earlier source.
    sites = tagged.filter(pl.col("source") == "nllb")["domain"].value_counts(sort=True)
    return ordered, len(corpus), sites


def pie(ax, tally: dict[str, int], total: int) -> None:
    sizes = list(tally.values())
    colors = [SOURCE_COLORS.get(s, UNATTRIBUTED) for s in tally]
    # Direct-label only the wedges with room; every source is named in the legend,
    # so identity is never carried by colour alone.
    labels = [s if n / total >= DIRECT_LABEL_MIN else "" for s, n in tally.items()]

    ax.set_facecolor(SURFACE)
    wedges, texts, autotexts = ax.pie(
        sizes, labels=labels, colors=colors, startangle=90, counterclock=False,
        autopct=lambda p: f"{p:.1f}%" if p >= DIRECT_LABEL_MIN * 100 else "",
        pctdistance=0.72, labeldistance=1.06,
        wedgeprops={"edgecolor": SURFACE, "linewidth": 2},   # 2px surface gap between fills
    )
    for t in texts:
        t.set_color(TEXT_PRIMARY)
        t.set_fontsize(10)
    for t in autotexts:
        t.set_color("white")
        t.set_fontsize(10)
        t.set_fontweight("bold")

    # Legend under the pie rather than beside it: the site panel owns the space to the
    # right, and two columns keep it from squeezing the wedges.
    ax.legend(wedges,
              [f"{s}  {n:,} ({n / total:.1%})" for s, n in tally.items()],
              loc="upper center", bbox_to_anchor=(0.5, 0.06), ncol=2, frameon=False,
              fontsize=9, labelcolor=TEXT_PRIMARY, handlelength=1.2, columnspacing=1.2)
    ax.set_title("by source", fontsize=11, color=TEXT_PRIMARY, pad=8)


def sites_bar(ax, sites: pl.DataFrame, nllb_total: int, top_n: int) -> None:
    top = sites.head(top_n).reverse()            # reverse: biggest bar on top under barh
    names, values = top["domain"].to_list(), top["count"].to_list()

    ax.set_facecolor(SURFACE)
    ax.barh(names, values, height=0.62,          # < 1: the band's leftover stays air
            color=[UNATTRIBUTED if n == UNKNOWN_DOMAIN else SOURCE_COLORS["nllb"] for n in names])
    for i, c in enumerate(values):               # every bar labelled, so the x-axis can go
        ax.text(c, i, f"  {c:,}  ({c / nllb_total:.1%})", va="center",
                fontsize=8.5, color=TEXT_SECONDARY)

    ax.set_title(f"inside the NLLB wedge — top {top_n} of {sites.height:,} sites\n"
                 f"the top {top_n} are {sum(values) / nllb_total:.1%} of NLLB pairs · "
                 f"{int((sites['count'] == 1).sum()):,} sites contribute one pair each",
                 fontsize=11, color=TEXT_PRIMARY, pad=8, linespacing=1.6)
    ax.margins(x=0.24)                           # headroom for the value labels
    ax.tick_params(labelsize=9, length=0)
    for label in ax.get_yticklabels():
        label.set_color(TEXT_PRIMARY)
    ax.xaxis.set_visible(False)
    for side in ("top", "right", "bottom", "left"):
        ax.spines[side].set_visible(False)


def figure(tally: dict[str, int], total: int, sites: pl.DataFrame, top_n: int) -> None:
    FIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig, (ax_pie, ax_bar) = plt.subplots(1, 2, figsize=(14.5, 7.4), facecolor=SURFACE,
                                         width_ratios=[1, 1.25])
    pie(ax_pie, tally, total)
    sites_bar(ax_bar, sites, tally.get("nllb", 0), top_n)
    fig.suptitle(f"{FINAL_BROAD.name} — where the pairs came from  (n={total:,})\n"
                 f"{len(tally)} sources · gezmu excluded by construction",
                 fontsize=13, color=TEXT_PRIMARY, linespacing=1.8)
    fig.tight_layout()
    fig.savefig(FIG_PATH, dpi=150, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)


def cmd_chart(args=None) -> None:
    top_n = getattr(args, "top", None) or TOP_N
    tally, total, sites = counts()
    width = max(len(s) for s in tally)
    for src, n in tally.items():                 # table view: the numbers, not just the wedges
        print(f"  {src:<{width}}  {n:>8,}  {n / total:>6.1%}")
    print(f"  {'TOTAL':<{width}}  {total:>8,}")
    if tally.get("unattributed"):
        print(f"\n  note: {tally['unattributed']:,} pairs matched no processed source — "
              f"{FINAL_BROAD} is stale relative to data/processed/")

    nllb = tally.get("nllb", 0)
    if nllb:
        print(f"\n  NLLB by site — {sites.height:,} registered domains over {nllb:,} pairs")
        dwidth = max(len(d) for d in sites.head(top_n)["domain"])
        for dom, n in sites.head(top_n).iter_rows():
            print(f"    {dom:<{dwidth}}  {n:>7,}  {n / nllb:>6.1%}")
        for k in (top_n, 50, 100, 1000):         # how slowly the long tail accumulates
            if k <= sites.height:
                print(f"    top {k:>5} sites:  {int(sites.head(k)['count'].sum()) / nllb:>6.1%}")
        print(f"    {int((sites['count'] == 1).sum()):,} sites contribute exactly one pair")

    figure(tally, total, sites, top_n)
    print(f"\n-> {FIG_PATH}")


if __name__ == "__main__":
    cmd_chart()
