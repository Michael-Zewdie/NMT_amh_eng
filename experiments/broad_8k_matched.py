"""
experiments.broad_8k_matched — does DOMAIN BREADTH of the training corpus trade
in-distribution BLEU against out-of-distribution generalization, holding every
other variable fixed against the already-completed am-en-gezmu-8k run?

Run (from the project root): python -m experiments.broad_8k_matched

Hypothesis under test (see EXPERIMENTS.md's "v2" section for am-en-gezmu-8k's
numbers, the other half of this comparison):
    narrow domain (Gezmu, ~83% Watchtower+Bible) -> higher in-distribution BLEU,
                                                     worse OOD (FLORES/MAFAND)
    broad  domain (6 other sources, mixed registers) -> lower in-distribution
                                                          BLEU, better OOD

Design: one new arm, size-matched to the existing am-en-gezmu-8k arm rather
than rebuilding both from scratch (unlike archive/experiments/domain_breadth.py,
that run's counterpart already exists and is finished):

    NARROW (existing) am-en-gezmu-8k   Gezmu et al.'s own 140,000-pair split,
                                        zero filtering, one domain
    BROAD  (this run)  am-en-broad-8k  6 other processed sources (gezmu.csv
                                        itself EXCLUDED — see below), pooled
                                        under this pipeline's normal tiered
                                        AfriCOMET/LaBSE/LID cutoffs, subsampled
                                        down to the SAME split sizes as Gezmu's

Pinned identical between the two arms: architecture, all training
hyperparameters, step count, seed (model/configs/broad_8k_matched.yaml mirrors
model/configs/gezmu_8k.yaml line for line except run_name/data.prepared_dir/
resume_from), split sizes, and — deliberately, not retrained — the tokenizer:
data/tokenizer/{am,en} is the 8k vocabulary fit on Gezmu's own train split,
reused here unchanged (same convention as domain_breadth.py's "same tokenizers,
NOT retrained -> vocabulary held fixed"). That is a real, stated asymmetry: a
tokenizer fit on Gezmu's narrow register will fragment medical/tech/web/
religious vocabulary in the broad arm's data more than a tokenizer fit on that
data itself would — i.e. if anything this HURTS the broad arm's fluency
relative to a from-scratch broad tokenizer, biasing against the OOD-generalizes-
better hypothesis rather than for it. Documented rather than hidden.

gezmu.csv is EXCLUDED from the broad pool (not just down-weighted) so the two
arms share no source data at all — a clean narrow-vs-complement comparison
instead of "narrow" vs "narrow-plus-five-others", which would confound the
result with partial train-domain overlap.

Cutoffs are this pipeline's actual PRODUCTION tiered cutoffs (imported from
processing.process / processing.utils.pool), not a synthetic shared threshold
like domain_breadth.py used — this arm is meant to represent "the best broad-
domain corpus this project actually knows how to build," the real alternative
to Gezmu-only, not an artificially-matched-filtering ablation.

Outputs:
    data/final_broad/, data/prepared_broad/am-en/   (NOT data/final,
    data/prepared/am-en — those are am-en-gezmu-8k's; left untouched)
"""
import sys

import pandas as pd

from model.common import load_tokenizer
from processing.utils import pool as pool_mod
from processing.utils.paths import DATA, PROCESSED
from processing.utils.manifest import write_manifest, git_info, now
from processing.process import (
    AFRICOMET_CUTOFFS, DEFAULT_AFRICOMET_CUTOFF, COSINE_CUTOFF,
    CURATED_COSINE_CUTOFF, SOURCE_LID_CUTOFF, TARGET_LID_CUTOFF,
)
from processing.utils.pool import CURATED_SOURCES
from model.data.prepare import _tokenize_column, MAX_LEN

SEED = 42
EXCLUDE_SOURCES = frozenset({"gezmu"})
SPLITS = ["validation", "test", "train"]

# am-en-gezmu-8k's exact split sizes (data/final/manifest.json as of this run's
# construction: train.after=139999, validation.after=2864, test.after=2500 —
# the authors' own Table 2 counts, 140,000/2,864/2,500, minus one degenerate
# empty-source row dropped from train). Matched as ratios of a same-sized
# broad pool below, not hardcoded counts, since length-stratified splitting
# can't hit an exact integer target per bucket.
GEZMU_TRAIN, GEZMU_VAL, GEZMU_TEST = 139_999, 2_864, 2_500
TARGET_TOTAL = GEZMU_TRAIN + GEZMU_VAL + GEZMU_TEST
RATIOS = (GEZMU_TRAIN / TARGET_TOTAL, GEZMU_VAL / TARGET_TOTAL, GEZMU_TEST / TARGET_TOTAL)


def load_broad_sources() -> list[pd.DataFrame]:
    """Every processed am/en source except EXCLUDE_SOURCES, each filtered by
    ITS OWN production cutoff (curated vs. mined tier) — mirrors
    processing.utils.pool.main()'s per-source loop exactly, just skipping
    gezmu.csv."""
    frames = []
    for p in sorted(PROCESSED.glob("*.csv")):
        if p.stem in EXCLUDE_SOURCES:
            print(f"[broad] excluding {p.name} (this arm's whole point is 'not Gezmu')")
            continue
        curated = p.stem in CURATED_SOURCES
        cos_c = CURATED_COSINE_CUTOFF if curated else COSINE_CUTOFF
        afc_c = AFRICOMET_CUTOFFS.get(p.stem, DEFAULT_AFRICOMET_CUTOFF)
        df = pool_mod.load_pairs(p, cos_c, afc_c, SOURCE_LID_CUTOFF, TARGET_LID_CUTOFF)
        if df is None:
            print(f"[broad] skipping {p.name} (not am/en parallel text)")
            continue
        print(f"[broad]   {p.name}: {len(df)} pairs kept (labse>{cos_c}, africomet>{afc_c})")
        frames.append(df.assign(_source=p.stem))
    if not frames:
        raise ValueError(f"No am/en sources survived in {PROCESSED} (besides excluded {EXCLUDE_SOURCES})")
    return frames


def main() -> None:
    frames = load_broad_sources()
    source_counts_before_subsample = {f["_source"].iloc[0]: len(f) for f in frames}
    frames = [f.drop(columns="_source") for f in frames]

    pooled_full = pool_mod.pool(frames, seed=SEED)
    pooled = pool_mod.decontaminate(pooled_full)
    if len(pooled) < TARGET_TOTAL:
        sys.exit(f"[broad] only {len(pooled):,} pairs after decontam, need {TARGET_TOTAL:,} "
                  f"to match am-en-gezmu-8k's size")

    # Pure random subsample (already shuffled+deduped by pool()) down to the
    # matched total — preserves each surviving source's natural relative share
    # rather than forcing artificial per-source quotas.
    subsampled = pooled.sample(n=TARGET_TOTAL, random_state=SEED).reset_index(drop=True)
    print(f"[broad] subsampled {len(pooled):,} -> {len(subsampled):,} pairs "
          f"(matching am-en-gezmu-8k's {TARGET_TOTAL:,})")

    final_out = DATA / "final_broad"
    final_out.mkdir(parents=True, exist_ok=True)
    original, pool_mod.FINAL = pool_mod.FINAL, final_out
    try:
        splits = pool_mod.split(subsampled, ratios=RATIOS, seed=SEED)
    finally:
        pool_mod.FINAL = original

    src_tok, tgt_tok = load_tokenizer("am"), load_tokenizer("en")
    out_dir = DATA / "prepared_broad" / "am-en"
    out_dir.mkdir(parents=True, exist_ok=True)
    prepared_sizes = {}
    for sp in SPLITS:
        d = pd.read_csv(final_out / f"{sp}.csv", usecols=["am", "en"], dtype=str).dropna()
        s = _tokenize_column(src_tok, d["am"].tolist())
        t = _tokenize_column(tgt_tok, d["en"].tolist())
        keep = [i for i, (a, b) in enumerate(zip(s, t)) if len(a) <= MAX_LEN and len(b) <= MAX_LEN]
        import pickle
        with open(out_dir / f"{sp}.pkl", "wb") as f:
            pickle.dump({"src": [s[i] for i in keep], "tgt": [t[i] for i in keep]}, f)
        print(f"[broad] {sp}: kept {len(keep):,} of {len(d):,} (MAX_LEN={MAX_LEN})")
        prepared_sizes[sp] = len(keep)

    manifest = {
        "arm": "broad",
        "experiment": "broad_8k_matched",
        "counterpart_run": "am-en-gezmu-8k",
        "built_at": now(),
        "git": git_info(),
        "excluded_sources": sorted(EXCLUDE_SOURCES),
        "cutoffs": {
            "note": "production tiered cutoffs (processing.process), not a synthetic shared threshold",
            "labse_score": {"default": COSINE_CUTOFF, "curated_sources": sorted(CURATED_SOURCES),
                             "curated_value": CURATED_COSINE_CUTOFF},
            "africomet_score": {"default": DEFAULT_AFRICOMET_CUTOFF, "per_source": AFRICOMET_CUTOFFS},
            "source_lid": SOURCE_LID_CUTOFF, "target_lid": TARGET_LID_CUTOFF,
        },
        "tokenizer_note": "reused data/tokenizer/{am,en} as-is (fit on Gezmu's own train split for "
                           "am-en-gezmu-8k) rather than retraining on this corpus — see module docstring",
        "source_survivor_counts_pre_subsample": source_counts_before_subsample,
        "pooled_after_dedup": len(pooled_full),
        "pooled_after_decontam": len(pooled),
        "matched_total": TARGET_TOTAL,
        "matched_against": {"train": GEZMU_TRAIN, "validation": GEZMU_VAL, "test": GEZMU_TEST},
        "split_ratios": list(RATIOS),
        "split_sizes": {k: len(v) for k, v in splits.items()},
        "prepared_sizes": prepared_sizes,
        "seed": SEED,
    }
    write_manifest(final_out, manifest)
    write_manifest(out_dir, manifest)  # travels with data/prepared_broad/ too
    print(f"\n[broad] done — final/{final_out.name}, prepared/{out_dir.relative_to(DATA)}")
    print(f"[broad] split sizes: { {k: len(v) for k, v in splits.items()} } "
          f"(target train/val/test: {GEZMU_TRAIN}/{GEZMU_VAL}/{GEZMU_TEST})")


if __name__ == "__main__":
    main()
