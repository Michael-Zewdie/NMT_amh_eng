# Archive — v1 experiments

Everything here was moved out of the active tree on 2026-08-07 to start a
clean **v2 ("2.0")** round of experiments from scratch. First v2 goal: a true
unfiltered reproduction of Gezmu et al. (LREC 2022) — train on their ~140k
pairs, no cutoffs, and see if this pipeline hits their reported BLEU.

**Nothing was deleted.** `runs/` and `data/` are gitignored, so those moves
are filesystem-only and don't touch git history; the code moves under
`experiments/` and `model/configs/` were done with `git mv` and are visible in
the commits that built this folder. Structure mirrors the original relative
paths, so restoring something is `git mv archive/experiments/foo.py
experiments/` (or plain `mv` for `archive/runs/*` / `archive/data/*`).

This was done in two passes the same day: the first archived only the
data-composition investigation scripts; the user then clarified they meant
*every* v1 experiment, including the current-best/production track
(`base_v4`, `base_v6`) and the gezmu-only precursor. This README describes
the final, full state.

## What's here

**`experiments/`** — every experiment script from v1:
`domain_breadth.py`, `fixed_size_corpus.py`, `religious_vs_nllb.py`,
`testset_diversity.py` (narrow-vs-broad / composition-vs-BLEU track);
`memorization.py` (FLORES memorization check); `prepare_reverse.py`
(en→am reverse-direction data prep); `singleref_eval.py` (multi-reference
eval fix for the `religious` track); `gezmu_only.py` (single-domain gezmu
corpus build — see caveat below).

**`model/configs/`** — every per-experiment config: `broad.yaml`,
`narrow.yaml`, `mixed_93k.yaml`, `nllb_93k.yaml`, `nllb_146k.yaml`,
`religious.yaml`, `reverse_en_am.yaml`, `gezmu_only.yaml`, and the
current-best/production pair `base_v4.yaml` / `base_v6.yaml`.

(Note: `model/configs/archive/` — the *pre-existing* archive of superseded
`base*`/`small*` configs from before this project even had a `results/`
system — was left where it is; `model/common.py`'s `_CONFIG_DIRS` reads from
it directly, so moving it would break `find_config`/`discover_runs`.)

**`runs/`** — checkpoints + tensorboard logs for every trained run:
`am-en-base`, `am-en-baseline`, `am-en-base-v2`, `am-en-base-v3`,
`am-en-base-v4` (former current-best), `am-en-base-v5`, `am-en-base-v6`
(former production), `am-en-broad`, `am-en-narrow`, `am-en-mixed-93k`,
`am-en-nllb-93k`, `am-en-nllb-146k`, `am-en-religious`, `am-en-gezmu-only`,
`am-en-small`, `am-en-small-moderate`, `am-en-small-strict`, `en-am-base-v1`
— plus every run-specific log (`breadth.log`, `large_breadth.log`,
`build_religious.log`, `score_religious.log`, `v5_train.superseded.log`,
`arms_train.log`, `gezmu_train.log`, `indist_recheck.log`, `overnight.log`,
`queue2.log`, `recover_v6.log`, `rescore.log`, `v6_train.log`). `runs/` in
the active tree is now empty.

**`data/`** — every `final_*`/`prepared_*` dataset built for a specific
experiment: `final`/`prepared` (former production pool), `final_gezmu`/
`prepared_gezmu`, `prepared_v4_32k`, `final_broad`/`prepared_broad`,
`final_narrow`/`prepared_narrow`, `final_mixed`/`prepared_mixed`,
`final_nllb`/`prepared_nllb`, `final_nllb146`/`prepared_nllb146`,
`final_religious`/`prepared_religious`, and `final_length_only_backup` (no
code references it and it has no `manifest.json` — origin unclear, kept
rather than deleted).

## What's still in the active tree (deliberately not archived)

Reusable pipeline infrastructure that any v2 experiment — Gezmu reproduction
or otherwise — will still need as input, plus the historical record:

- **Model/processing code**: `model/` (architecture, training loop, data
  pipeline, tokenizer loading — none of it experiment-specific),
  `processing/`, `collection/`, `baselines/`.
- **Generic tooling** (not tied to any one track): `experiments/report.py`,
  `experiments/backfill_manifests.py`, `experiments/decode_sweep.py`. Note
  `model/train.py` / `model/evaluate.py` / `model/translate.py` /
  `experiments/decode_sweep.py` each had a `DEFAULT_CONFIG` pointing at the
  now-archived `base_v6.yaml` — flagged inline with a comment rather than
  repointed, since no v2 config exists yet to repoint it to. Pass a config
  path explicitly until v2 has its own default.
- **Source/derived data, not experiment-specific**: `data/raw/`,
  `data/processed/` (per-source cleaned CSVs — `gezmu.csv` etc.),
  `data/scores/` (AfriCOMET/LaBSE/LID/semantic-cluster scores, expensive to
  recompute), `data/tokenizer/` (trained subword tokenizers — note the
  Gezmu paper found 8k vocab optimal on this exact corpus, so a v2 gezmu
  reproduction may want to train a fresh tokenizer rather than reuse these),
  `data/benchmarks/` (FLORES-200, MAFAND-MT held-out sets).
- **Historical record** (results, not experiment artifacts): `EXPERIMENTS.md`,
  `results/benchmarks.csv`, `results/indist.csv`, `results/experiments.yaml`,
  `results/v6_flores_devtest.{csv,tsv}`.

## Caveat for the v2 Gezmu-reproduction experiment specifically

`archive/experiments/gezmu_only.py` / `archive/data/final_gezmu` are **not** a
true unfiltered reproduction — that build still applied `AFRICOMET_CUTOFF=0.15`
and `LID>0.90` cutoffs to the Gezmu corpus. Gezmu et al. trained on all ~140k
pairs completely unfiltered. `model/configs/archive/base_v5.yaml` documents
every hyperparameter matched to their paper (batch size in tokens, warmup,
dropout, label smoothing, etc.) and is the right starting point for a v2
config — it was previously run against the pooled ~409k-pair recipe, not
Gezmu's own corpus alone, so point its data at a freshly-built, truly
unfiltered Gezmu-only pool.
