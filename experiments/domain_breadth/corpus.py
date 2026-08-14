"""Builds data/final_broad/ — the BROAD arm's corpus.

Restored from commit 0ffcabb (`experiments/broad_8k_matched.py`), which built the
CSVs on disk but was dropped in an earlier refactor, leaving data/final_broad/
unreproducible.

    python -m experiments.domain_breadth corpus            # report counts only
    python -m experiments.domain_breadth corpus --write    # commit the split

Tune the CUTOFFS block below, run without --write to see how many pairs survive,
repeat until the total is at or a little above TARGET_TOTAL, then --write.

Design: one arm, size-matched to the already-finished narrow arm (am-en-gezmu-8k)
rather than rebuilding both. gezmu.csv is EXCLUDED rather than down-weighted, so
the two arms share no source data at all — a clean narrow-vs-complement comparison
instead of "narrow" vs "narrow-plus-five-others", which would confound the result
with train-domain overlap.

Cutoffs live here rather than being imported from process.process so retuning this
experiment can't silently change the production pipeline, or vice versa. They are
applied to data/processed/*.csv, which process.process has already annotated with
labse_score / africomet_score / source_lid / target_lid — scoring only annotates,
cutoffs are applied at pool time, so re-selecting never reloads a scoring model.

Tokenization is NOT done here — `build --arm broad` fits the arm's own shared 8k
vocabulary over these CSVs (arms.py).
"""
import json
import subprocess
import sys
from datetime import datetime, timezone

import pandas as pd

from experiments.domain_breadth.paths import FINAL_BROAD
from process import pool as pool_mod
from process.utils.paths import PROCESSED

# ── CUTOFFS ────────────────────────────────────────────────────────────────────
# Per source, not uniform. AfriCOMET is not calibrated across registers, so one
# shared floor across sources of different register is a register filter wearing a
# quality filter's clothes (see process.process's CURATED_AFRICOMET_CUTOFF note).
# With the curated religious corpora now excluded that hazard is mostly moot here,
# but the per-source shape is kept: it is what makes the afridoc corpora survivable
# alongside a 0.85-ish mined floor.
#
# To reproduce the corpus currently on disk instead, use the production tiering over
# all six sources: africomet 0.80 / labse 0.70 for nllb+ccaligned, africomet 0.15 /
# labse 0.0 for the rest, which yields 541k and then randomly subsamples to target.
AFRICOMET = {
    "nllb":           0.8465,  # set by TARGET_TOTAL, not by taste — see EXCLUDE below
    "afridoc_health": 0.0,     # 0.0 disables the floor (apply_cutoffs skips it): these two
    "afridoc_tech":   0.0,     # are small and curated, so all ~11.4k rows are kept.
}
LABSE = {
    # No nllb entry: a LaBSE floor on nllb is a no-op by construction. score_labse
    # leaves labse_score NaN wherever laser_score already exists (score_cache's
    # skip_where), and apply_cutoffs passes those rows through on laser_score — so
    # the floor filtered 0 of 2,361,879 nllb rows every time it ran. LASER is the
    # equivalent gate and was already applied upstream, in collect.py's LASER_CUTOFF.
    "afridoc_health": 0.0,     # curated: LaBSE's tail here is not separable — correct
    "afridoc_tech":   0.0,     # rows score as low as real misalignments, so no floor works.
}
SOURCE_LID = 0.90             # uniform: a sanity check (is this row am/en), not a quality judgment
TARGET_LID = 0.90

# religious/quran/ccaligned are excluded alongside gezmu. The first two matter most:
# the NARROW arm is ~83% Watchtower/Bible, so keeping a 34% religious+quran share in
# the BROAD arm put the same register on both sides of the comparison and blunted the
# contrast the experiment is trying to measure. Dropping them makes the arms cleanly
# disjoint in register, not merely in source file.
#
# The cost is stated rather than hidden: what remains (nllb + afridoc) cannot reach
# TARGET_TOTAL at a 0.86 AfriCOMET floor — it tops out near 103k — so nllb's floor is
# lowered to 0.8465 to keep the size match with the narrow arm, which is a control the
# experiment cannot give up. That trades a little mined-row quality for size parity,
# and leaves the corpus ~92% nllb. Whether that is still "broad" is a question about
# NLLB's internal site spread, which is exactly what source_dist.py's right-hand panel
# now measures.
#
# Re-including a source means restoring its AFRICOMET/LABSE entry; load_sources exits
# with a clear message if one is missing.
EXCLUDE = frozenset({"gezmu", "religious", "quran", "ccaligned"})
SEED = 42

# am-en-gezmu-8k's exact split sizes (data/final/manifest.json: the authors' own
# Table 2 counts, 140,000/2,864/2,500, minus one degenerate empty-source row dropped
# from train). Matched as ratios of a same-sized pool rather than hardcoded counts,
# since length-stratified splitting can't hit an exact integer target per bucket.
GEZMU_TRAIN, GEZMU_VAL, GEZMU_TEST = 139_999, 2_864, 2_500
TARGET_TOTAL = GEZMU_TRAIN + GEZMU_VAL + GEZMU_TEST
RATIOS = (GEZMU_TRAIN / TARGET_TOTAL, GEZMU_VAL / TARGET_TOTAL, GEZMU_TEST / TARGET_TOTAL)


def load_sources() -> list[pd.DataFrame]:
    """Every processed am/en source except EXCLUDE, filtered by its own cutoffs.

    Calls apply_cutoffs directly rather than pool.load_pairs so africomet_score
    survives into the frame — the overshoot trim needs it. Dropped before writing.
    """
    frames = []
    for p in sorted(PROCESSED.glob("*.csv")):
        if p.stem in EXCLUDE:
            why = "not Gezmu" if p.stem == "gezmu" else "see EXCLUDE"
            print(f"[broad] excluding {p.name} ({why})")
            continue
        df = pd.read_csv(p, dtype=str)
        if not {"am", "en"}.issubset(df.columns):
            print(f"[broad] skipping {p.name} (not am/en parallel text)")
            continue
        if p.stem not in AFRICOMET:
            sys.exit(f"[broad] {p.name} has no AFRICOMET entry — add one, or add it to "
                     f"EXCLUDE (a new source appeared in {PROCESSED})")
        # A missing LABSE entry means no LaBSE gate, which is the right default: only
        # sources whose rows actually carry labse_score can be filtered on it.
        df = pool_mod.apply_cutoffs(df, p.stem, LABSE.get(p.stem, 0.0), AFRICOMET[p.stem],
                                    SOURCE_LID, TARGET_LID)
        frames.append(df[["am", "en", "africomet_score"]].assign(_source=p.stem))
    if not frames:
        raise ValueError(f"no am/en sources survived in {PROCESSED}")
    return frames


def report(df: pd.DataFrame, label: str) -> None:
    mix = df["_source"].value_counts()
    print(f"\n[broad] {label}: {len(df):,} pairs")
    for src, n in mix.items():
        print(f"[broad]     {src:16s} {n:>8,}  ({n / len(df):6.1%})")


def git_info() -> dict:
    def run(*a):
        return subprocess.run(a, capture_output=True, text=True).stdout.strip()
    return {"sha": run("git", "rev-parse", "--short", "HEAD"),
            "dirty": bool(run("git", "status", "--porcelain"))}


def cmd_corpus(args) -> None:
    frames = load_sources()
    survivors = {f["_source"].iloc[0]: len(f) for f in frames}

    pooled_full = pool_mod.pool(frames, seed=SEED)
    pooled = pool_mod.decontaminate(pooled_full)
    report(pooled, "after cutoffs + dedup + decontamination")

    print(f"\n[broad] target {TARGET_TOTAL:,} (am-en-gezmu-8k's size)")
    if len(pooled) < TARGET_TOTAL:
        sys.exit(f"[broad] SHORT by {TARGET_TOTAL - len(pooled):,} — lower a cutoff and re-run")

    over = len(pooled) - TARGET_TOTAL
    if over:
        # Trim the overshoot by dropping the globally lowest-scoring rows. Safe only
        # because the per-source cutoffs above have already done the selecting: this
        # is a remainder operation. If `over` is a large fraction of the pool, this
        # trim becomes the real selector and will preferentially delete the curated
        # sources — retune AFRICOMET instead of leaning on it.
        pct = over / len(pooled)
        print(f"[broad] over by {over:,} ({pct:.1%}) — dropping the lowest-scoring {over:,}")
        if pct > 0.10:
            print(f"[broad] WARNING: trimming {pct:.0%} globally will skew the source mix "
                  f"toward whichever source scores highest (nllb). Tighten AFRICOMET.")
        pooled = (pooled.assign(_afc=pd.to_numeric(pooled["africomet_score"], errors="coerce"))
                  .sort_values("_afc", ascending=False, kind="mergesort")
                  .head(TARGET_TOTAL).drop(columns="_afc")
                  .sample(frac=1, random_state=SEED).reset_index(drop=True))
        report(pooled, "after trim")

    floors = {s: round(float(pd.to_numeric(g["africomet_score"], errors="coerce").min()), 4)
              for s, g in pooled.groupby("_source", sort=True)}
    print(f"\n[broad] effective AfriCOMET floors: {floors}")

    if not args.write:
        print("\n[broad] report only — re-run with --write to commit")
        return

    FINAL_BROAD.mkdir(parents=True, exist_ok=True)
    original, pool_mod.FINAL = pool_mod.FINAL, FINAL_BROAD
    try:
        splits = pool_mod.split(pooled[["am", "en"]], ratios=RATIOS, seed=SEED)
    finally:
        pool_mod.FINAL = original

    (FINAL_BROAD / "manifest.json").write_text(json.dumps({
        "arm": "broad",
        "experiment": "domain_breadth",
        "counterpart_run": "am-en-narrow",
        "built_at": datetime.now(timezone.utc).isoformat(),
        "git": git_info(),
        "excluded_sources": sorted(EXCLUDE),
        "cutoffs": {"africomet": AFRICOMET, "labse": LABSE,
                    "source_lid": SOURCE_LID, "target_lid": TARGET_LID},
        "effective_africomet_floors": floors,
        "source_survivor_counts": survivors,
        "pooled_after_dedup": len(pooled_full),
        "matched_total": TARGET_TOTAL,
        "matched_against": {"train": GEZMU_TRAIN, "validation": GEZMU_VAL, "test": GEZMU_TEST},
        "split_ratios": list(RATIOS),
        "split_sizes": {k: len(v) for k, v in splits.items()},
        "seed": SEED,
    }, indent=2))
    print(f"\n[broad] wrote {FINAL_BROAD} — {({k: len(v) for k, v in splits.items()})}")
    print("[broad] next: python -m experiments.domain_breadth build --arm broad")
