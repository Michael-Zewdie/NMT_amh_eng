"""Builds data/final_clean/ — the corpus this experiment trains on.

    python -m experiments.clean_recipe corpus            # report counts only
    python -m experiments.clean_recipe corpus --write    # commit the split

Corpus selection lives here; lowercasing is a tokenization-time concern and lives
in arms.py, so these CSVs stay CASED. That split matters: it means a second arm
(a cased control, or a different LASER floor) reuses this exact file and
therefore this exact test set, rather than re-partitioning the pool. That is the
lesson from the abandoned stratification experiment (archive/README.md), where
three arms each partitioned the same pool independently, ended up with ~99%
disjoint test sets, and produced nothing comparable.

WHAT CHANGED FROM am-en-base-v4, and why
----------------------------------------
Every choice below traces to a measurement in EXPERIMENTS.md's "v4 data audit":

1. NLLB gated on laser_score > 1.08, not the upstream 1.06.
   LASER is the only score in this pipeline that can see mined misalignment.
   Measured on the reliable subpopulation (both sides carrying a multi-digit
   Arabic number, no Ge'ez numerals), misalignment falls monotonically across
   LASER deciles, 38.2% -> 4.1%. Raising the floor to 1.08 halves the mined
   noise, 16.0% -> 7.0%, for 46% of the rows. Domain breadth survives the cut
   (5,405 -> 3,765 distinct source sites), which matters because breadth is what
   gave am-en-broad its +6.2 FLORES over am-en-narrow.

2. AfriCOMET stays at 0.80. Raising it is a pure loss: 0.80 -> 0.92 discards
   97.6% of mined rows and moves misalignment 28.5% -> 26.9% (naive detector,
   but the flatness is the point). AfriCOMET grades adequacy/fluency of a whole
   sentence, so a fluent pair with one swapped entity scores well; it saturates
   near its error floor and cannot be tuned past it.

3. quran EXCLUDED. Worst source in the audit by a wide margin -- 14.13 in-dist
   BLEU against gezmu's 27.59 and nllb's 27.80 -- because its English carries
   bracketed exegetical commentary the Amharic does not, which no aligned model
   can reproduce. 7.4% of v4's training data spent teaching a register that is
   not translation.

4. ccaligned EXCLUDED at the project's request.

5. religious EXCLUDED: the audit found 100% of the rows that reached v4's train
   split were also present in nllb.csv, so it contributes no distinct text.

6. gezmu and both afridoc corpora enter WHOLE -- every floor disabled, LID
   included. They are already human-curated, so each of these filters can only
   subtract; the audit found LID in particular was a length artifact costing
   7,751 correctly-aligned pairs (see the SOURCE_LID comment below). This
   extends the Gezmu et al. root-cause fix that made v4 the best model in the
   project: v4 disabled LaBSE/AfriCOMET on curated sources but still ran LID.

    gezmu 124,408 / 124,409     afridoc_tech 6,040 / 6,040
    afridoc_health 5,809 / 5,809    (the one missing gezmu row is a cross-source
                                     duplicate, removed by pool's dedup)
"""
import json
import subprocess
import sys
from datetime import datetime, timezone

import pandas as pd

from process import pool as pool_mod
from process.utils.paths import DATA, PROCESSED, ROOT

FINAL_CLEAN = DATA / "final_clean"
RESULTS = ROOT / "experiments" / "clean_recipe" / "results.json"

# ── CUTOFFS ────────────────────────────────────────────────────────────────────
# Per source, never uniform: AfriCOMET is not calibrated across registers, so one
# shared floor is a register filter wearing a quality filter's clothes.
AFRICOMET = {
    "nllb":           0.80,    # unchanged from v4 -- raising it is a pure loss, see docstring
    # 0.15 on the curated tier is a garbage trap, not a quality filter, and is
    # deliberately set far below anything that could act as a register filter.
    # Measured: it drops 44 of gezmu's 124,409 rows (0.04%) and ZERO from either
    # afridoc corpus, both of which sit entirely above it (p01 = 0.566 and 0.484).
    # What it does catch on gezmu is real -- verse-split misalignments
    # ("አንተ ቋጥኜና ምሽጌ ስለሆንክ"/"where i can always enter.") and, at 0.101, English
    # transliterated into Ge'ez script: "ዶ ኖት ሰት ትሂስ ፓገ ..."/"do not set this page
    # will not contain any web tags or seo information." -- web boilerplate that
    # belongs in no corpus. Cheap insurance; it will not move any headline number.
    "gezmu":          0.15,
    "afridoc_health": 0.15,
    "afridoc_tech":   0.15,
}
LABSE = {
    # No nllb entry by construction: score_labse leaves labse_score NaN wherever
    # laser_score already exists, and apply_cutoffs passes those rows through on
    # laser_score. The floor filtered 0 of 2,361,879 nllb rows every time it ran.
    # LASER is the equivalent gate, and LASER_FLOOR below is where it is applied.
    #
    # 0.15 on the curated tier earns its place where the AfriCOMET floor does not:
    # it removes 144 gezmu rows, 133 of which AfriCOMET 0.15 happily keeps, and
    # 0 from either afridoc corpus (min 0.207 and 0.178). The 133 are flatly
    # misaligned, and their AfriCOMET scores show why the second floor is needed:
    #   labse -0.037 / afri 0.349  "የሰዋስው ስርአቱና የአረፍተ ነገር አወቃቀሩ..." (on grammar and
    #                              sentence structure) -> "so joseph began to open
    #                              up the granaries where the surplus grain was..."
    #   labse -0.013 / afri 0.507  "የተለያዩ ርእሶች" (various topics)
    #                              -> "kindness melts bitterness, 6 / 15"
    # This is the LASER-vs-AfriCOMET result from the v4 audit repeating one level
    # down: an ALIGNMENT score ("are these the same sentence") sees misalignment,
    # an adequacy/fluency score does not, at any threshold.
    "gezmu":          0.15,
    "afridoc_health": 0.15,
    "afridoc_tech":   0.15,
}
# LID is per-source too, and DISABLED on the curated tier. On mined text it is a
# useful sanity check ("is this row actually am/en"); on gezmu/afridoc it is a
# length artifact wearing a quality filter's clothes. Measured before switching it
# off: the rows it would drop have median Amharic length 14-15 characters against
# 53-83 for the rows it keeps, and inspection found them correctly aligned --
# "ስራ 10፥ 35"/"acts 10: 35.", "የአየር ንብረት ለውጥ"/"Climate change",
# "ኣለም ድርብ ሃላፊነት ተጋርጦባታል።"/"The world is facing a double mandate." (source_lid 0.01).
# LID classifiers are simply unreliable on short strings, and it was costing 5.9%
# of gezmu, 4.4% of afridoc_tech and 2.8% of afridoc_health -- 7,751 correct pairs.
#
# Nothing is lost on the mined side by making this per-source: collect/nllb.py
# already applies its own LID floors upstream, so the pool-time nllb filter drops
# exactly 0 rows either way.
#
# One judgment call left visible rather than hidden: a large share of gezmu's
# recovered rows are bare Bible verse references. They are correct pairs, and
# Gezmu et al. trained on their full corpus, so they are kept -- but they are
# short, formulaic and repetitive, and if a later run wants them gone the filter
# to reach for is a minimum length, not LID.
SOURCE_LID = {"nllb": 0.90, "gezmu": 0.0, "afridoc_health": 0.0, "afridoc_tech": 0.0}
TARGET_LID = {"nllb": 0.90, "gezmu": 0.0, "afridoc_health": 0.0, "afridoc_tech": 0.0}

# The one genuinely new knob in this experiment. Applied here rather than by
# re-running collect/nllb.py (whose LASER_CUTOFF=1.06 produced nllb.csv): raising
# it there would mean re-downloading and re-filtering 16.1M pairs, and the column
# is already on every row, so this is the same filter one stage later.
LASER_FLOOR = 1.08
LASER_SOURCES = frozenset({"nllb"})

EXCLUDE = frozenset({"quran", "ccaligned", "religious"})
SEED = 42

# ── AFRIDOC ────────────────────────────────────────────────────────────────────
# This experiment used to re-clean afridoc_health and afridoc_tech from
# data/raw/csv_raw/ with a local copy of the pipeline, because the shared
# script_purity step was deleting a quarter of both for the wrong reason. That
# fix now lives where it belongs: process.clean.filters.SCRIPT_PURITY_EXEMPT
# skips the step for these two sources, so data/processed/ already holds the
# recovered rows and load_sources() just reads them like anything else.
#
# Re-run `python -m process` to regenerate data/processed/ under the new rule.
AMH_LEN, ENG_LEN = (5, 500), (10, 500)

# The pipeline default, and what v4 used — so the in-distribution number here is
# at least measured the same WAY as v4's 26.39, even though it is measured on a
# different corpus and therefore still not directly comparable to it.
#
# Briefly built at ~96/2/2 on 2026-08-14 to preserve training data, and reverted:
# that was an unrequested deviation from the project's stated 80/10/10, and it was
# caught ~50k steps into a run that then had to be restarted. The ~48k pairs it
# moves out of train buy a ~30k-pair test set, which is what makes an in-dist BLEU
# stable enough to quote.
#
# pool.split() applies these ratios WITHIN each Amharic-length bucket and groups by
# `am`, so the split is length-stratified and leak-free by construction. Because
# grouping is by distinct sentence rather than by row, a source with heavy
# duplication pulls the realized row counts slightly off the nominal 80/10/10 —
# that is the documented tradeoff for not leaking a source sentence across splits.
RATIOS = (0.8, 0.1, 0.1)


def load_sources() -> tuple[list[pd.DataFrame], dict]:
    """Every processed am/en source except EXCLUDE, filtered by its own cutoffs."""
    frames, survivors = [], {}
    for p in sorted(PROCESSED.glob("*.csv")):
        if p.stem in EXCLUDE:
            print(f"[clean] excluding {p.name} (see EXCLUDE)")
            continue
        df = pd.read_csv(p, dtype=str)
        if not {"am", "en"}.issubset(df.columns):
            print(f"[clean] skipping {p.name} (not am/en parallel text)")
            continue
        if p.stem not in AFRICOMET:
            sys.exit(f"[clean] {p.name} has no AFRICOMET entry — add one, or add it to "
                     f"EXCLUDE (a new source appeared in {PROCESSED})")
        before = len(df)
        if p.stem in LASER_SOURCES:
            if "laser_score" not in df.columns:
                sys.exit(f"[clean] {p.name} is in LASER_SOURCES but carries no laser_score")
            keep = pd.to_numeric(df["laser_score"], errors="coerce") > LASER_FLOOR
            df = df[keep]
            print(f"[clean]   {p.stem}: laser_score>{LASER_FLOOR}: {before} → {len(df)} "
                  f"({len(df) - before:+d})")
        df = pool_mod.apply_cutoffs(df, p.stem, LABSE.get(p.stem, 0.0), AFRICOMET[p.stem],
                                    SOURCE_LID.get(p.stem, 0.0), TARGET_LID.get(p.stem, 0.0))
        survivors[p.stem] = len(df)
        frames.append(df[["am", "en"]].assign(_source=p.stem))
    if not frames:
        raise ValueError(f"no am/en sources survived in {PROCESSED}")
    return frames, survivors


def report(df: pd.DataFrame, label: str) -> None:
    print(f"\n[clean] {label}: {len(df):,} pairs")
    for src, n in df["_source"].value_counts().items():
        print(f"[clean]     {src:16s} {n:>8,}  ({n / len(df):6.1%})")


def git_info() -> dict:
    def run(*a):
        return subprocess.run(a, capture_output=True, text=True).stdout.strip()
    return {"sha": run("git", "rev-parse", "--short", "HEAD"),
            "dirty": bool(run("git", "status", "--porcelain"))}


def cmd_corpus(args) -> None:
    frames, survivors = load_sources()
    pooled_full = pool_mod.pool(frames, seed=SEED)
    pooled = pool_mod.decontaminate(pooled_full)
    report(pooled, "after cutoffs + dedup + decontamination")

    if not args.write:
        print("\n[clean] report only — re-run with --write to commit")
        return

    FINAL_CLEAN.mkdir(parents=True, exist_ok=True)
    original, pool_mod.FINAL = pool_mod.FINAL, FINAL_CLEAN
    try:
        splits = pool_mod.split(pooled[["am", "en"]], ratios=RATIOS, seed=SEED)
    finally:
        pool_mod.FINAL = original

    (FINAL_CLEAN / "manifest.json").write_text(json.dumps({
        "experiment": "clean_recipe",
        "built_at": datetime.now(timezone.utc).isoformat(),
        "git": git_info(),
        "excluded_sources": sorted(EXCLUDE),
        "cutoffs": {"africomet": AFRICOMET, "labse": LABSE, "laser": LASER_FLOOR,
                    "laser_sources": sorted(LASER_SOURCES),
                    "source_lid": SOURCE_LID, "target_lid": TARGET_LID, "lid_note": "disabled on curated tier — length artifact, see corpus.py"},
        "source_survivor_counts": survivors,
        "pooled_after_dedup": len(pooled_full),
        "after_decontamination": len(pooled),
        "source_mix": pooled["_source"].value_counts().to_dict(),
        "split_ratios": list(RATIOS),
        "split_sizes": {k: len(v) for k, v in splits.items()},
        "seed": SEED,
    }, indent=2))
    print(f"\n[clean] wrote {FINAL_CLEAN} — {({k: len(v) for k, v in splits.items()})}")
    print("[clean] next: python -m experiments.clean_recipe build --arm lower")
