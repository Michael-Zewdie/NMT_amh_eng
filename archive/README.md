# Archive — v1 investigation tracks

Everything here was moved out of the active tree on 2026-08-07 to start a clean
**v2** round of experiments, first goal: reproduce Gezmu et al. (LREC 2022) and
see if this pipeline can hit their reported BLEU. Nothing was deleted —
`runs/` and `data/` are gitignored, so those moves are filesystem-only and
don't touch git history; the code moves under `experiments/` and
`model/configs/` were done with `git mv` and are visible in the commit that
added this folder.

Mirrors the original relative paths, so restoring something is
`git mv archive/experiments/foo.py experiments/` (or plain `mv` for
`archive/runs/*` / `archive/data/*`).

## What's here and why

**`experiments/`** — one-off hypothesis-test scripts from the v1 investigation
into data composition (domain breadth, corpus size, source mix), plus two
single-purpose utilities. Full rationale for each in its own docstring and in
`EXPERIMENTS.md`.
- `domain_breadth.py`, `fixed_size_corpus.py`, `religious_vs_nllb.py`,
  `testset_diversity.py` — the narrow-vs-broad / composition-vs-BLEU track.
- `memorization.py` — FLORES memorization check for `am-en-base-v6`.
- `prepare_reverse.py` — en→am reverse-direction data prep (backtranslation
  prerequisite; not part of the Gezmu-reproduction goal).
- `singleref_eval.py` — multi-reference-corpus eval fix, specific to the
  `religious` track's 7-references-per-verse data.

**`model/configs/`** — the run configs paired 1:1 with the archived scripts
above: `broad.yaml`, `narrow.yaml`, `mixed_93k.yaml`, `nllb_93k.yaml`,
`nllb_146k.yaml`, `religious.yaml`, `reverse_en_am.yaml`.

(Note: `model/configs/archive/` — the *pre-existing* archive of superseded
`base*`/`small*` configs — was left where it is; `model/common.py`'s
`_CONFIG_DIRS` reads from it directly, so moving it would break
`find_config`/`discover_runs`.)

**`runs/`** — checkpoints + tensorboard logs for:
- `am-en-base`, `am-en-baseline`, `am-en-base-v2`, `am-en-base-v3` — superseded
  early base-track versions (config already lived in `model/configs/archive/`).
- `am-en-base-v5` — the *trained* Gezmu-hyperparameter-matching attempt, but on
  the pooled ~409k-pair recipe, not Gezmu's own corpus alone. Superseded by
  whatever v2's true reproduction run ends up being; its config
  (`model/configs/archive/base_v5.yaml`) documents every matched hyperparameter
  and is worth reading before setting up the v2 config.
- `am-en-broad`, `am-en-narrow`, `am-en-mixed-93k`, `am-en-nllb-93k`,
  `am-en-nllb-146k`, `am-en-religious` — outputs of the archived experiment
  scripts above.
- `am-en-small`, `am-en-small-moderate`, `am-en-small-strict` — the separate
  256d small-architecture track (see `~/.claude/.../memory/` for that session's
  notes). Not the 512d architecture Gezmu et al. used.
- `en-am-base-v1` — reverse-direction model, paired with `prepare_reverse.py`.
- Plus the logs specific to those runs: `breadth.log`, `large_breadth.log`,
  `build_religious.log`, `score_religious.log`, `v5_train.superseded.log`,
  `arms_train.log`.

**`data/`** — `final_*`/`prepared_*` variants that only ever fed the archived
runs above (`broad`, `narrow`, `mixed`, `nllb`, `nllb146`, `religious`), plus
`final_length_only_backup`, which no code references and has no
`manifest.json` — origin unclear, kept rather than deleted.

## What's still in the active tree (not archived)

- `experiments/gezmu_only.py` — closest existing precursor to v2's first goal,
  but **not** a true reproduction: it still applies `AFRICOMET_CUTOFF=0.15` and
  `LID>0.90` cutoffs to the Gezmu corpus. Gezmu et al. trained on all ~140k
  pairs unfiltered. Useful as a template, not as-is.
- `data/final_gezmu`, `data/prepared_gezmu` — output of the above; same caveat.
- `experiments/backfill_manifests.py`, `experiments/report.py`,
  `experiments/decode_sweep.py` — reusable infra, not tied to any one track.
- `model/configs/base_v4.yaml`, `base_v6.yaml`, `gezmu_only.yaml` — current
  production recipe + the gezmu-only starting point.
- `runs/am-en-base-v4`, `runs/am-en-base-v6`, `runs/am-en-gezmu-only` — current
  best model, current production run, and the closest existing gezmu baseline.
- `data/final`, `data/prepared`, `data/prepared_v4_32k` — current production
  pools (the latter is what `base_v4.yaml` evaluates against).
- `data/processed`, `data/raw`, `data/scores`, `data/tokenizer`,
  `data/benchmarks` — source/derived data and benchmarks, not track-specific.
