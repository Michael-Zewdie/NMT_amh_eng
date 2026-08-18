"""Builds data/final_broad_v2/ — the BROAD arm's corpus, register-disjoint rebuild.

    python -m experiments.domain_breadth corpus            # report counts only
    python -m experiments.domain_breadth corpus --write    # commit the split

WHY V2 EXISTS
-------------
`data/final_broad/` (v1) is FROZEN: it is the corpus `am-en-broad` trained on and
every published number traces to it. Its manifest records `excluded_sources:
["gezmu"]` and a pool of religious 147,048 / quran 37,068 / ccaligned 7,492
alongside nllb 338,334 — the trained 140k came out nllb 62.6%, religious 27.1%,
quran 6.9%, ccaligned 1.4%, afridoc 2.2%. Roughly a THIRD religious register, the
same register as the narrow arm it is contrasted against, which blunts the very
contrast the experiment measures. So the published +6.23 FLORES is a floor.

v2 removes that overlap and fixes two other things found alongside it:

    religious/quran/ccaligned OUT   arms become disjoint in register, not just in
                                    source file
    afridoc enters WHOLE            19,497 rows, up from 14,404 — the script_purity
                                    exemption recovered 5,093 the old filter ate.
                                    13% of the corpus, against v1's 2.2%
    LASER replaces the subsample    v1 gated nllb on africomet 0.8465, a number
                                    picked to hit a row target, then randomly
                                    trimmed. nllb carries no africomet_score at all,
                                    so that gate was a no-op and selection was
                                    effectively random. LASER_FLOOR selects the
                                    best-aligned rows instead.

Size stays matched to the narrow arm (145,363) — that control is what makes the
comparison mean anything, and LASER_FLOOR is tuned to land there.

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

from experiments.domain_breadth.paths import FINAL_BROAD_V2
from process import pool as pool_mod
from process.utils.paths import PROCESSED

# ── CUTOFFS ────────────────────────────────────────────────────────────────────
# LASER IS THE SIZE DIAL, not AfriCOMET plus a random subsample. v1 gated nllb on
# africomet 0.8465 — a number chosen to hit TARGET_TOTAL rather than for any quality
# reason — and then randomly trimmed the overshoot. Both halves of that are gone:
#
#   * data/processed/nllb.csv carries NO africomet_score column at all, so the v1
#     floor was silently a no-op and the selection was effectively random.
#   * LASER is the only score in this pipeline that can see mined misalignment
#     (v4 audit: misalignment falls 38.2% -> 4.1% across LASER deciles). Using it as
#     the dial means the rows that survive are the best-aligned ones, not an
#     arbitrary sample.
#
# LASER_FLOOR is tuned so nllb + ALL of afridoc lands just above TARGET_TOTAL:
# 1.108 keeps 129,547 nllb (the top 18.3% by alignment) for 149,044 total against a
# 145,363 target. The distribution is steep and truncated (1.080-1.250, median
# 1.0914), so this floor is sensitive — 1.100 gives 207k and 1.112 gives ~100k.
AFRICOMET = {
    "nllb":           0.0,     # no africomet_score column on nllb; LASER_FLOOR is the gate
    "afridoc_health": 0.0,     # 0.0 disables the floor (apply_cutoffs skips it): these two
    "afridoc_tech":   0.0,     # are curated, and now enter WHOLE — all 19,497 rows.
}
LASER_FLOOR = 1.108
LASER_SOURCES = frozenset({"nllb"})
LABSE = {
    # No nllb entry: a LaBSE floor on nllb is a no-op by construction. score_labse
    # leaves labse_score NaN wherever laser_score already exists (score_cache's
    # skip_where), and apply_cutoffs passes those rows through on laser_score — so
    # the floor filtered 0 of 2,361,879 nllb rows every time it ran. LASER is the
    # equivalent gate and was already applied upstream, in collect.py's LASER_CUTOFF.
    "afridoc_health": 0.0,     # curated: LaBSE's tail here is not separable — correct
    "afridoc_tech":   0.0,     # rows score as low as real misalignments, so no floor works.
}
# LID is per-source now, and DISABLED on the curated tier — the v4 audit found a 0.90
# floor there is a LENGTH artifact, not an am/en sanity check: the rows it drops have
# median Amharic length 14-15 chars against 53-83 for the rows it keeps, and hand
# inspection found them correctly aligned. It was costing 4.4% of afridoc_tech and
# 2.8% of afridoc_health. On mined text it still does the job it was written for.
# (Moot in practice for the re-cleaned afridoc CSVs, which carry no LID columns yet —
# apply_cutoffs skips a floor whose column is absent — but stated so the intent does
# not depend on which score stages last ran.)
SOURCE_LID = {"nllb": 0.90, "afridoc_health": 0.0, "afridoc_tech": 0.0}
TARGET_LID = {"nllb": 0.90, "afridoc_health": 0.0, "afridoc_tech": 0.0}

# religious/quran/ccaligned are excluded alongside gezmu — BUT NOT IN THE RUN THAT
# WAS SCORED. See the ⚠ block in the module docstring: am-en-broad trained on a
# corpus that excluded gezmu only, and was ~34% religious+quran+ccaligned.
#
# The reason to exclude them on a rerun: the NARROW arm is ~83% Watchtower/Bible, so
# a 34% religious+quran share in the BROAD arm puts the same register on both sides
# of the comparison and blunts the contrast the experiment is trying to measure.
# Dropping them makes the arms cleanly disjoint in register, not merely in source
# file. That is a better experiment — it has simply never been run.
#
# The size match with the narrow arm is kept — it is the control the experiment
# cannot give up — but LASER_FLOOR now buys it instead of a lowered AfriCOMET floor
# plus a random trim. The corpus lands ~87% nllb / ~13% afridoc, against v1's 92%/2.2%:
# afridoc goes in whole (19,497 rows after the script_purity exemption recovered
# 5,093 of them) because it is curated text and every floor on it was measured to
# subtract only. Whether ~87% nllb is still "broad" is a question about NLLB's
# internal site spread, which source_dist.py's right-hand panel measures.
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

    Calls apply_cutoffs directly rather than pool.load_pairs so `_laser` survives into
    the frame — the overshoot trim ranks on it. Dropped before writing.

    Curated rows carry no laser_score, so `_laser` is NaN for them and the trim treats
    NaN as +inf: afridoc is never the thing that gets cut. That is deliberate. It is the
    scarce tier (19,497 rows against nllb's 129,547) and the whole point of this rebuild
    is to keep all of it; trimming it to hit a row target would undo the change.
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
        before = len(df)
        if p.stem in LASER_SOURCES:
            if "laser_score" not in df.columns:
                sys.exit(f"[broad] {p.name} is in LASER_SOURCES but carries no laser_score")
            df = df[pd.to_numeric(df["laser_score"], errors="coerce") > LASER_FLOOR]
            print(f"[broad]   {p.stem}: laser_score>{LASER_FLOOR}: {before} → {len(df)} "
                  f"({len(df) - before:+d})")
        # A missing LABSE entry means no LaBSE gate, which is the right default: only
        # sources whose rows actually carry labse_score can be filtered on it.
        df = pool_mod.apply_cutoffs(df, p.stem, LABSE.get(p.stem, 0.0), AFRICOMET[p.stem],
                                    SOURCE_LID.get(p.stem, 0.0), TARGET_LID.get(p.stem, 0.0))
        laser = (pd.to_numeric(df["laser_score"], errors="coerce")
                 if "laser_score" in df.columns else pd.Series(float("nan"), index=df.index))
        frames.append(df[["am", "en"]].assign(_laser=laser, _source=p.stem))
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
        print(f"[broad] over by {over:,} ({pct:.1%}) — dropping the lowest-LASER {over:,}")
        if pct > 0.10:
            print(f"[broad] WARNING: trimming {pct:.0%} means LASER_FLOOR is set too low "
                  f"and this trim, not the floor, is doing the selecting. Raise it.")
        # NaN (curated, no laser_score) sorts as +inf, so only mined rows are ever cut.
        pooled = (pooled.assign(_rank=pooled["_laser"].fillna(float("inf")))
                  .sort_values("_rank", ascending=False, kind="mergesort")
                  .head(TARGET_TOTAL).drop(columns="_rank")
                  .sample(frac=1, random_state=SEED).reset_index(drop=True))
        report(pooled, "after trim")

    floors = {s: (None if g["_laser"].isna().all()
                  else round(float(g["_laser"].min()), 4))
              for s, g in pooled.groupby("_source", sort=True)}
    print(f"\n[broad] effective LASER floors: {floors}  (None = curated, no laser_score)")

    if not args.write:
        print("\n[broad] report only — re-run with --write to commit")
        return

    FINAL_BROAD_V2.mkdir(parents=True, exist_ok=True)
    original, pool_mod.FINAL = pool_mod.FINAL, FINAL_BROAD_V2
    try:
        splits = pool_mod.split(pooled[["am", "en"]], ratios=RATIOS, seed=SEED)
    finally:
        pool_mod.FINAL = original

    (FINAL_BROAD_V2 / "manifest.json").write_text(json.dumps({
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
    print(f"\n[broad] wrote {FINAL_BROAD_V2} — {({k: len(v) for k, v in splits.items()})}")
    print("[broad] next: python -m experiments.domain_breadth build --arm broad")
