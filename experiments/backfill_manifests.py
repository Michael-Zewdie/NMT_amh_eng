"""
experiments.backfill_manifests — ONE-TIME backfill of runs/<name>/manifest.json
for every run trained before processing.utils.manifest existed (added
2026-08-06, alongside model/train.py + the data-build scripts writing their own
manifests going forward — see that module's docstring for why).

Run (from the project root): python -m experiments.backfill_manifests

Every field below is sourced and cited in DATA[name]["source"] — from
EXPERIMENTS.md, the run's own model/configs/*.yaml (including archive/), or a
specific runs/*.log build log. Anything not stated in one of those is left None
rather than guessed; EXPERIMENTS.md itself already does this (see its "Known
gaps" paragraph) and this script preserves those gaps instead of papering over
them.

Two corrections made in the course of writing this (documented here, not just
silently applied):
  - am-en-narrow / am-en-broad: model/configs/{narrow,broad}.yaml said "47,756 /
    47,761 train" in their header comment. That was runs/breadth.log's FIRST
    build (swapped in ~18:33), superseded ~25 min later by the rebuild in
    runs/overnight.log (~18:58) — confirmed by checkpoint mtimes (best.pt
    written 20:12, i.e. after the second build started, not the first). The
    actual trained counts are 41,832 / 41,850. The yaml comments were corrected
    to match; this script uses the correct (overnight.log) numbers.
  - am-en-base-v5: EXPERIMENTS.md's run-history entry says "Not yet trained" for
    the *current* (Gezmu-replication) v5 config, but results/benchmarks.csv and
    runs/am-en-base-v5/checkpoints/ show it WAS trained and scored. EXPERIMENTS.md
    is stale on this specific line; flagged in this run's manifest rather than
    silently corrected (the prose narrative is out of this script's scope).

This does NOT try to reconstruct data/final_<name>/manifest.json for runs built
straight into the (repeatedly overwritten) production data/final/ — those counts
genuinely aren't recoverable, same as EXPERIMENTS.md already says. Where a
per-arm data/final_<name>/ directory still exists on disk, a matching data
manifest is written there too (and copied into data/prepared_<name>/), using the
same schema pool.py/domain_breadth.py now write going forward.
"""
from pathlib import Path

from model.config import load_config
from processing.utils.manifest import write_manifest, git_info, now
from processing.utils.paths import DATA, RUNS

BACKFILLED_AT = now()
GIT = git_info()  # current HEAD — NOT the commit that actually produced these
                   # runs (most predate git history here); recorded as
                   # "backfill_git", distinct from a live run's "git" field.

# run_name -> config path (every run maps 1:1 via each yaml's own run_name: field)
# NOTE: as of the 2026-08-07 v2 archive pass (extended same day to cover EVERY
# v1 run/config, not just the data-composition tracks), every config below
# lives under archive/model/configs/ — model/configs/archive/ is the one
# exception, pre-existing and left in place since model/common.py's
# find_config() reads it directly. This script was a ONE-TIME backfill that
# already ran (see every archived run's manifest.json, "backfilled": true) —
# paths updated so a future re-run loads correctly rather than crashing on a
# moved file, not because this needs to run again.
CONFIG_PATHS = {
    "am-en-base":            "model/configs/archive/base.yaml",
    "am-en-baseline":        "model/configs/archive/baseline.yaml",
    "am-en-base-v2":         "model/configs/archive/base_v2.yaml",
    "am-en-base-v3":         "model/configs/archive/base_v3.yaml",
    "am-en-base-v4":         "archive/model/configs/base_v4.yaml",
    "am-en-base-v5":         "model/configs/archive/base_v5.yaml",
    "am-en-base-v6":         "archive/model/configs/base_v6.yaml",
    "am-en-broad":           "archive/model/configs/broad.yaml",
    "am-en-gezmu-only":      "archive/model/configs/gezmu_only.yaml",
    "am-en-mixed-93k":       "archive/model/configs/mixed_93k.yaml",
    "am-en-narrow":          "archive/model/configs/narrow.yaml",
    "am-en-nllb-146k":       "archive/model/configs/nllb_146k.yaml",
    "am-en-nllb-93k":        "archive/model/configs/nllb_93k.yaml",
    "am-en-religious":       "archive/model/configs/religious.yaml",
    "am-en-small":           "model/configs/archive/small.yaml",
    "am-en-small-moderate":  "model/configs/archive/small_moderate.yaml",
    "am-en-small-strict":    "model/configs/archive/small_strict.yaml",
    "en-am-base-v1":         "archive/model/configs/reverse_en_am.yaml",
}

_EXP_MD = "EXPERIMENTS.md"

# Data provenance actually known per run. `cutoffs=None` means genuinely not
# recoverable (stated as such in EXPERIMENTS.md), not "wasn't looked up".
DATA_PROVENANCE = {
    "am-en-base": {
        "cutoffs": {"africomet_score_uniform": 0.5},
        "note": "AFRICOMET_CUTOFF=0.5, uniform across all sources — later found to "
                "let through ~50% misaligned/garbage NLLB pairs. Exact train/val/test "
                "counts not recoverable, data/final/ has been overwritten by every "
                "re-pool since.",
        "source": f"{_EXP_MD} §Run history: am-en-base",
    },
    "am-en-baseline": {
        "cutoffs": None,
        "note": "Data cutoff at the time not recorded and not recoverable — short "
                "smoke-test run, not a real data/quality experiment.",
        "source": f"{_EXP_MD} §Run history: am-en-baseline",
    },
    "am-en-base-v2": {
        "cutoffs": {"africomet_score_uniform": 0.62},
        "split_sizes": {"train": 1818721, "validation": 227201, "test": 227297},
        "note": "Raised AFRICOMET_CUTOFF 0.5->0.62, uniform across all sources.",
        "source": f"{_EXP_MD} §Run history: am-en-base-v2",
    },
    "am-en-base-v3": {
        "cutoffs": {"africomet_score_uniform": 0.80},
        "sources_added": ["ccaligned"],
        "total_pairs": 365729,
        "train_pairs": 292614,
        "note": "AFRICOMET_CUTOFF=0.80 uniform across all sources (still pre-tiering).",
        "source": f"{_EXP_MD} §Run history: am-en-base-v3",
    },
    "am-en-base-v4": {
        "cutoffs": {
            "tiered": True,
            "curated_sources": ["gezmu", "afridoc_health", "afridoc_tech", "quran"],
            "curated": {"labse_score": 0.0, "africomet_score": 0.0},
            "mined": {"labse_score": 0.7, "africomet_score": 0.8},
            "lid": {"source_lid": 0.90, "target_lid": 0.90},
        },
        "total_pairs": 511021,
        "train_pairs_pooled": 409455,
        "train_pairs_prepared": 409454,
        "note": "First tiered-cutoff run — curated sources (Gezmu et al. root-cause fix) "
                "get liberal/disabled LaBSE+AfriCOMET, mined sources keep the strict floors.",
        "source": f"{_EXP_MD} §Root-cause check against Gezmu et al. + §am-en-base-v4",
    },
    "am-en-base-v5": {
        "cutoffs": {
            "tiered": True,
            "curated_sources": ["gezmu", "afridoc_health", "afridoc_tech", "quran"],
            "curated": {"labse_score": 0.0, "africomet_score": 0.0},
            "mined": {"labse_score": 0.7, "africomet_score": 0.8},
            "lid": {"source_lid": 0.90, "target_lid": 0.90},
        },
        "train_pairs_prepared": 409443,
        "note": "Same tiered scheme as v4 (CURATED_AFRICOMET_CUTOFF still 0.0 — the "
                "0.0->0.15 change happened for v6, see base_v6.yaml). "
                "DISCREPANCY FLAGGED: EXPERIMENTS.md's am-en-base-v5 entry says "
                "'Not yet trained', but results/benchmarks.csv and this run's own "
                "checkpoints show it was trained and scored (FLORES 12.40/36.57 beam4, "
                "MAFAND 5.97/26.42 beam4) — EXPERIMENTS.md is stale on this line.",
        "source": "model/configs/base_v6.yaml comment ('Train is now 408,835, was "
                  "409,443') + EXPERIMENTS.md §am-en-base-v5 (current)",
    },
    "am-en-base-v6": {
        "cutoffs": {
            "tiered": True,
            "curated_sources": ["gezmu", "afridoc_health", "afridoc_tech", "quran"],
            "curated": {"labse_score": 0.0, "africomet_score": 0.15},
            "mined": {"labse_score": 0.7, "africomet_score": 0.8},
            "lid": {"source_lid": 0.90, "target_lid": 0.90},
        },
        "train_pairs": 408835,
        "note": "CURATED_AFRICOMET_CUTOFF raised 0.0->0.15 vs v4/v5 — a garbage floor "
                "(catches genuine off-by-one misalignment), not a quality gate. Dropped "
                "53 rows, all quran.",
        "source": "model/configs/base_v6.yaml inline comment",
    },
    "am-en-narrow": {
        "cutoffs": {"africomet_score": 0.5, "source_lid": 0.90, "target_lid": 0.90,
                    "labse_score": None},
        "source_name": "bible-uedin",
        "train_pairs": 41832, "validation_pairs": 5247, "test_pairs": 5234,
        "note": "Shared-threshold design (experiments/domain_breadth.py) — see that "
                "module's docstring on why the cutoff is a controlled-experiment "
                "symmetry choice, not a tuned optimum for bible-uedin. Superseded an "
                "earlier 47,756-pair build; see this module's docstring.",
        "source": "runs/overnight.log (shared-threshold search + build) + "
                  "data/final_narrow/manifest.json",
    },
    "am-en-broad": {
        "cutoffs": {"africomet_score": 0.5, "source_lid": 0.90, "target_lid": 0.90,
                    "labse_score": None},
        "source_name": "nllb",
        "train_pairs": 41850, "validation_pairs": 5230, "test_pairs": 5233,
        "note": "Shared-threshold design, same run as am-en-narrow. nllb rows were "
                "already laser_score>1.06-filtered at collection (collection/collect.py).",
        "source": "runs/overnight.log + data/final_broad/manifest.json",
    },
    "am-en-gezmu-only": {
        "cutoffs": {"labse_score": 0.0, "africomet_score": 0.15,
                    "source_lid": 0.90, "target_lid": 0.90},
        "cutoff_tier": "curated (same as production process.py curated tier)",
        "source_name": "gezmu",
        "train_pairs": 93649,
        "note": "Single-source diagnostic — tests whether in-distribution BLEU is "
                "comparable across data compositions (it isn't).",
        "source": "model/configs/gezmu_only.yaml + experiments/gezmu_only.py",
    },
    "am-en-mixed-93k": {
        "cutoffs": {
            "mined": {"labse_score": 0.7, "africomet_score": 0.80},
            "curated": {"labse_score": 0.0, "africomet_score": 0.15},
            "lid": {"source_lid": 0.90, "target_lid": 0.90},
        },
        "sources": "all six processed sources",
        "pool_target": 117049,
        "train_pairs": 93619,
        "note": "Arm C of the diversity-vs-volume experiment — v6's exact recipe and "
                "source mix, subsampled to gezmu-only's pool size.",
        "source": "model/configs/mixed_93k.yaml + experiments/fixed_size_corpus.py",
    },
    "am-en-nllb-93k": {
        "cutoffs": {"labse_score": 0.7, "africomet_score": 0.80,
                    "source_lid": 0.90, "target_lid": 0.90},
        "cutoff_tier": "mined",
        "source_name": "nllb",
        "pool_target": 117049,
        "train_pairs": 93638,
        "note": "Arm B of the diversity-vs-volume experiment — NLLB only, subsampled "
                "to gezmu-only's pool size.",
        "source": "model/configs/nllb_93k.yaml + experiments/fixed_size_corpus.py",
    },
    "am-en-religious": {
        "cutoffs": {"africomet_score": 0.15, "source_lid": 0.90, "target_lid": 0.90,
                    "labse_score": None},
        "cutoff_tier": "curated garbage floor (all rows above it kept, not top-N)",
        "source_name": "religious (verse-ID joined, 1 Amharic Bible x 7 English Bibles)",
        "pooled_after_decontam": 147048,
        "train_pairs": 117658, "validation_pairs": 14648, "test_pairs": 14742,
        "note": "Narrow arm of the large-scale domain-breadth replication. NOT the "
                "shared-threshold design — this arm keeps everything above the curated "
                "floor while nllb146 (its counterpart) gets NLLB's best rows by AfriCOMET. "
                "dropout raised to 0.2 vs the 0.1 used elsewhere (118k pairs overfits at 0.1).",
        "source": "runs/build_religious.log + experiments/religious_vs_nllb.py",
    },
    "am-en-nllb-146k": {
        "cutoffs": {"africomet_score": None, "source_lid": 0.90, "target_lid": 0.90,
                    "labse_score": None},
        "cutoff_tier": "top-N by AfriCOMET score (not a floor) — score range 0.8428-1.0195",
        "source_name": "nllb",
        "pooled_after_decontam": 147031,
        "train_pairs": 117624, "validation_pairs": 14702, "test_pairs": 14705,
        "note": "Broad arm, paired with am-en-religious. Selection asymmetry stated "
                "up front in model/configs/nllb_146k.yaml: this arm is handed NLLB's "
                "cleanest rows, religious just clears a garbage floor.",
        "source": "runs/build_religious.log + experiments/religious_vs_nllb.py",
    },
    "am-en-small": {
        "cutoffs": {"africomet_score_uniform": 0.85},
        "train_pairs": 102699,
        "note": "Parallel session's own run (separate Claude Code session on this repo), "
                "uniform cutoff sweep point 0.85 (its best of 0.62/0.85/0.88, all uniform "
                "incl. curated sources — later shown to be the wrong call for curated "
                "data, same lesson this log found independently via Gezmu et al.).",
        "source": f"{_EXP_MD} §Small-architecture track + parallel session's own memory "
                  "(data-quality-over-quantity, per that section)",
    },
    "am-en-small-moderate": {
        "cutoffs": None,
        "note": "Exact cutoff not confirmed from this log's side (parallel session's "
                "own artifact). Plausibly a loosened-curated-filtering point, given the "
                "gap between its weak in-distribution score and strong OOD score "
                "(FLORES 13.94, MAFAND 6.99 — best of any run in this project). "
                "See EXPERIMENTS.md's own 'Known gaps' note.",
        "source": f"{_EXP_MD} §Small-architecture track ('Known gaps' paragraph)",
    },
    "am-en-small-strict": {
        "cutoffs": None,
        "note": "Name implies the 0.88-or-stricter sweep point; exact value not "
                "confirmed from this log's side. Worst OOD result of any model trained "
                "(FLORES 5.46, MAFAND 3.02).",
        "source": f"{_EXP_MD} §Small-architecture track ('Known gaps' paragraph)",
    },
    "en-am-base-v1": {
        "cutoffs": "same as am-en-base-v6 (identical recipe, src/tgt swapped)",
        "note": "Reverse direction (en->am), built for backtranslation. Reads "
                "data/prepared/en-am/, built by experiments/prepare_reverse.py from the "
                "same production data/final/ that fed am-en-base-v6. CAVEAT: its own "
                "val_bleu uses sacrebleu's English 13a tokenizer on an Amharic target — "
                "not a meaningful score, not comparable to any am->en number.",
        "source": "model/configs/reverse_en_am.yaml",
    },
}


# run_name -> (final_<x>, prepared_<x>) stem, for the arms whose own data
# directory still exists on disk (unlike production data/final/, which has been
# overwritten by every re-pool and genuinely can't be backfilled).
ARM_DATA_DIRS = {
    "am-en-narrow":     "narrow",
    "am-en-broad":      "broad",
    "am-en-gezmu-only": "gezmu",
    "am-en-religious":  "religious",
    "am-en-nllb-146k":  "nllb146",
    "am-en-mixed-93k":  "mixed",
    "am-en-nllb-93k":   "nllb",
}


def backfill_data_manifest(run_name: str, stem: str) -> None:
    final_dir = DATA / f"final_{stem}"
    prepared_dir = DATA / f"prepared_{stem}" / "am-en"
    if not final_dir.exists():
        print(f"[backfill]   {final_dir} doesn't exist, skipping its data manifest")
        return
    d = DATA_PROVENANCE[run_name]
    manifest = {
        "arm": stem,
        "run_name": run_name,
        "backfilled": True,
        "backfilled_at": BACKFILLED_AT,
        "backfill_git": GIT,
        **{k: v for k, v in d.items() if k != "note"},
        "note": d.get("note"),
    }
    write_manifest(final_dir, manifest)
    if prepared_dir.exists():
        write_manifest(prepared_dir, manifest)
    print(f"[backfill]   data manifest -> {final_dir}/manifest.json"
          f"{' + ' + str(prepared_dir) + '/manifest.json' if prepared_dir.exists() else ''}")


def main() -> None:
    missing = set(CONFIG_PATHS) - set(DATA_PROVENANCE)
    if missing:
        raise SystemExit(f"no DATA entry for: {sorted(missing)}")

    for run_name, config_path in CONFIG_PATHS.items():
        cfg = load_config(config_path)
        run_dir = RUNS / run_name
        if not run_dir.exists():
            print(f"[backfill] skipping {run_name}: {run_dir} doesn't exist")
            continue

        manifest = {
            "run_name": run_name,
            "config_path": config_path,
            "config": dict(cfg),
            "backfilled": True,
            "backfilled_at": BACKFILLED_AT,
            "backfill_git": GIT,  # current HEAD, NOT the commit that trained this run
            "data_provenance": DATA_PROVENANCE[run_name],
            "status": "finished (backfilled — see data_provenance.source for what's "
                      "confirmed vs. unrecoverable)",
        }
        path = write_manifest(run_dir, manifest)
        print(f"[backfill] wrote {path}")

        stem = ARM_DATA_DIRS.get(run_name)
        if stem:
            backfill_data_manifest(run_name, stem)


if __name__ == "__main__":
    main()
