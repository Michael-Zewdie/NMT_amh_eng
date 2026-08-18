# archive/

Superseded work. Nothing here is deleted — it is moved out of the active
workspace so `runs/`, `experiments/` and `data/` show only what is current.
All of it still runs if restored.

Archived 2026-08-12, after experiment #3 answered the domain-breadth question.
**The results themselves are NOT archived** — they live in
`results/domain_breadth_results.md` and `EXPERIMENTS.md`, which remain the
authoritative record and still cite these runs' numbers.

## What is active

| kept | why |
|---|---|
| `runs/am-en-narrow` | experiment #3/#5 narrow arm — in-domain 32.07, the fixed baseline both broad arms are measured against |
| `runs/am-en-broad-v2` | experiment #5 broad arm — **best OOD model**, FLORES 14.53 ci, supersedes broad v1 |
| `runs/am-en-clean-lower` | the clean recipe — FLORES 18.55 ci / MAFAND 8.46 ci (avg12); `model/translate.py`'s default |
| `experiments/domain_breadth/` | builds/trains the arms (`arms.py`) and owns the 3×3 eval (`evaluate.py`) |
| `data/{final,prepared}_translit`, `data/{final,prepared}_broad_v2_translit`, `data/{final,prepared}_clean_lower` | their data |
| `data/tokenizer/shared_translit{,_broad_v2,_clean_lower}` | their vocabularies |
| `data/{raw,processed,scores,final,benchmarks}` | shared pipeline, not experiment leftovers — see below |

`am-en-base-v4` and `am-en-broad` were **archived 2026-08-17** — see below.

## Datasets archived 2026-08-13

| moved | to | why |
|---|---|---|
| `data/prepared_broad` | `archive/data/prepared_broad_8k` | experiment #2's broad arm (`am-en-broad-8k`, already archived). Renamed on the way in — `archive/data/prepared_broad` was already taken by a v1 arm. |
| `data/archive/` | `archive/data/from_data_archive/` | an archive nested inside `data/`. Kept as one directory because its `prepared/` would collide with `archive/data/prepared`. |
| `data/tokenizer/{am,en}_32k_v4` | `archive/data/tokenizer/` | v1 `am-en-base-v4`'s 32k vocabularies — that run is already archived. **Reversed 2026-08-14 — see below.** |

`am-en-base-v4`'s and `am-en-broad-8k`'s `manifest.json` were patched to the new
paths, so both stay re-scoreable from the archive without hand-editing.

### `am-en-base-v4` restored 2026-08-14

Brought back out of the archive as the project's best model (`best.pt`, step
157,000 — FLORES 14.97/41.00, MAFAND 6.30/28.44, full-validation 26.39):

| restored | to |
|---|---|
| `archive/runs/am-en-base-v4` | `runs/am-en-base-v4` |
| `archive/data/prepared_v4_32k` | `data/prepared_v4_32k` (409,454 train / 50,720 validation pairs) |
| `archive/data/tokenizer/{am,en}_32k_v4` | `data/tokenizer/` |
| `model/configs/archive/base_v4.yaml` | `model/configs/base_v4.yaml` |

The run directory had no `config.yaml` of its own (only a backfilled
`manifest.json`), so `base_v4.yaml` was copied in as one. Both it and the
manifest now carry an explicit `prepared_dir: data/prepared_v4_32k` — v4's
config previously relied on the default `data/prepared`, which no longer exists.
Restoring under the `_v4_32k` name rather than the generic `prepared/` keeps it
from being mistaken for an 8k-era cache. Verified end-to-end: loads on CUDA at
32000/32000 vocab and decodes FLORES devtest.

**Deliberately left in `data/`:** `raw/` (6.2G — includes `raw/local/Gezmu`, which
both current arms read at build *and* eval time), `processed/` (1.3G, input to
`process/pool.py`), `scores/` (8.4G of content-keyed AfriCOMET/LaBSE/LID caches —
many GPU-hours to rebuild), `final/` (referenced by 20 active modules incl.
`paths.py`'s `FINAL`), `benchmarks/` (FLORES/MAFAND, needed by every eval), and
`data/tokenizer/{am,en}` (1.8M, the live default in `model/common.py:25`).

## `am-en-base-v4` and `am-en-broad` archived 2026-08-17

Both superseded, 5.1G of checkpoints between them. Nothing deleted; both were
**verified to load and decode from their archived paths** after the move.

| moved | to |
|---|---|
| `runs/am-en-base-v4` | `archive/runs/am-en-base-v4` |
| `data/prepared_v4_32k` | `archive/data/prepared_v4_32k` |
| `data/tokenizer/{am,en}_32k_v4` | `archive/data/tokenizer/` |
| `model/configs/base_v4.yaml` | `model/configs/archive/base_v4.yaml` |
| `runs/am-en-broad` | `archive/runs/am-en-broad` |
| `runs/broad_{build,eval,launch}.log` | `archive/runs/` |
| `data/final_broad` | `archive/data/final_broad_translit` — **renamed**, see below |
| `data/prepared_broad_translit` | `archive/data/prepared_broad_translit` |
| `data/tokenizer/shared_translit_broad` | `archive/data/tokenizer/` |

This reverses the 2026-08-14 restore of v4 recorded above. Both runs'
`config.yaml` and `manifest.json`, and `model/configs/archive/base_v4.yaml`,
were repointed at the archived data paths, so each stays scoreable in place:

```
python -m model.evaluate.evaluate_OOD archive/runs/am-en-base-v4/config.yaml \
        archive/runs/am-en-base-v4/checkpoints/best.pt \
        data/benchmarks/flores200_am_en.csv devtest
```

**Why v4:** `am-en-clean-lower` now dominates it — FLORES 18.55 / MAFAND 8.46
case-insensitive against v4's 14.97 / 6.30, and even v4's own cased metric is
within noise of clean-lower's 14.63 cased floor, which is a floor only because
that model structurally cannot emit a capital letter. `model/translate.py`'s
`DEFAULT_CONFIG` had already moved to `am-en-clean-lower`. EXPERIMENTS.md's TL;DR
still names v4 on the cased FLORES column, which is accurate and unchanged — the
model is archived, not demoted.

**Why broad v1:** superseded by `am-en-broad-v2` on every fixed benchmark
(FLORES 14.53 vs 12.46 ci, MAFAND 6.84 vs 5.95 ci) *and* on v1's own test split
(24.73 vs 20.09), at matched size and matched steps. Experiment #5's write-up in
EXPERIMENTS.md is the authoritative record.

**`final_broad` → `final_broad_translit`:** `archive/data/final_broad` was
already taken by `am-en-broad-old`'s v1 corpus (see the `-old` section below),
so this one took the `*_translit` suffix its `prepared_`/tokenizer siblings
already carry. Same rename-on-the-way-in as `prepared_broad_8k`.

**One live dependency, deliberately kept pointing into the archive.**
`experiments/domain_breadth/paths.py` still has a `broad` entry — `FINAL_BROAD`,
`TOK_DIRS["broad"]` and `PREPARED_DIRS["broad"]` now resolve under
`archive/data/`. `test_broad` is a published column of the 3×3, so `narrow` and
`broad_v2` must keep scoring on it; only the *checkpoints* went away. `cmd_eval`
already skips an arm with no `best.pt` and prints why, so re-running the eval
today produces the 3×3 minus the BROAD v1 row. To get that row back:
`mv archive/runs/am-en-broad runs/` and revert nothing else.

`prepared_dir()` in that module now joins `ROOT` instead of `DATA`, since
`PREPARED_DIRS` no longer holds `data/`-prefixed strings for every arm.

## The `-old` suffix

`archive/runs/am-en-narrow-old` and `am-en-broad-old` are the **v1** arms
(~42k pairs, bible-uedin vs NLLB mined, FLORES 0.68 / 3.02). They were plain
`am-en-narrow` / `am-en-broad` until 2026-08-13, when the current arms took those
names — `-old`, not `-v1`, so they can't be mistaken for the `am-en-base-v2…v6`
series. Their 8 rows in `results/benchmarks.csv` were re-keyed to match, and
their `manifest.json`s record `renamed_from`. If you see `am-en-narrow` with a
FLORES under 1, it is one of these, not the current arm.

## What was archived, and what it was for

**`runs/am-en-gezmu-8k`, `runs/am-en-broad-8k`** — experiment #2, the first 140k
domain-breadth comparison (broad +6.37 FLORES / +2.50 MAFAND). Superseded by
experiment #3, which repeated it with the paper's preprocessing on both arms and
got +7.03 / +2.57. `am-en-gezmu-8k` is also the reproduction baseline whose
30.75 / 31.22 the transliteration result is measured against.

**`runs/dd-*` (10 runs), `experiments/domain_dist_10k.py`, `data/*_health10k`,
`data/*_diverse10k`** — experiment #1, the 10k-pair domain-distribution test
(2 arms × 5 seeds). A **null result caused by a floor effect**: both arms landed
at ~1.2 FLORES BLEU, where nothing is measurable. Kept because the null is a real
finding — it establishes that below ~50k pairs this architecture cannot resolve
domain questions at all — and because its 5-seed variance estimates
(±0.06–0.14 on OOD metrics) are the only ones in the project.

## Stratification experiment, archived 2026-08-14

**`runs/am-en-splitstrat-{length,domain,semantic}`, `experiments/stratification/`,
`data/{final,prepared}_splitstrat_*`, `data/tokenizer/shared_translit_splitstrat`** —
the split-strategy comparison (length vs domain vs semantic stratification).
Abandoned before it produced a scored result: `experiments/stratification/results.json`
was never written, so the only numbers are in-training `val_bleu` proxies, and
those are not comparable across arms for two independent reasons.

1. **Different vocabularies.** `data/tokenizer/{am,en}` were moved to
   `archive/` on 2026-08-13 08:06, *after* the length arm (finished 07:00) and
   domain arm (started 07:39) had already loaded them at startup, and both
   pre-date the `load_tokenizer` fix that honours `data.src_tokenizer`. Their
   val BLEU was decoded through the old am/en id map; only 18 of 8000 id→token
   positions agree with `shared_translit_splitstrat`. The scores aren't garbage
   — `evaluate_loader` decodes hyps *and* refs with the same wrong tokenizer, a
   consistent relabelling — but they sit on a different surface scale than the
   semantic arm, which ran after the fix.
2. **Different test sets.** Each arm partitions the same 432k pool
   independently, so the three test splits are ~99% disjoint: only **97**
   `am` sentences are held out by all three, and the union of the three train
   sets covers 108,909 of ~110k unique `am`. There is no leak-free common
   subset large enough to score on.

Recorded numbers, for whatever they are worth: length `best.pt` @245k = 23.97,
domain @235k = 23.86, semantic @70k = 21.00. The 0.11 gap between the two
genuinely comparable arms (length vs domain — same tokenizer, same bug) is the
useful signal here: it suggests the whole axis has a dynamic range near the
noise floor, which is unsurprising given all three arms train on ~80% of the
same pool. Semantic's apparent −3 is almost certainly the measurement artifact
above, not a real effect — its `val_loss` was **2.99 at 70k** against the other
two arms' ~3.15 at 250k.

If this is ever picked up again, the design needs a shared holdout carved out
of the pool *before* stratification, plus a random-split control and a
seed-repeat to establish the noise floor. Checkpoint retention is also a rolling
window of the last 12 `checkpoints/avg/step*.pt`, which is why length and domain
have nothing below step 195k.

**Not archived, deliberately:** `process/dist/semantic/`, `process/pool.py`'s
`split_semantic()`, and `process/process.py`'s `STRATIFY_SEMANTIC` flag are the
general pipeline's optional semantic-stratification feature (off by default),
not this experiment. `process/pool.py:36` imports `semantics.py` at module level,
so moving it would break every importer of `process.pool`.

## ccaligned + quran removed from the codebase, 2026-08-14

Both sources are out of the corpus and out of the active tree entirely, at the
project's direction. Nothing is deleted:

| moved | to |
|---|---|
| `collect/{ccaligned,quran}.py` | `archive/collect/` |
| `data/raw/csv_raw/{ccaligned,quran}.csv` | `archive/data/csv_raw/` |
| `data/processed/{ccaligned,quran}.csv` | `archive/data/processed_removed/` |
| `data/raw/ccaligned_full/` (extracted OPUS release) | `archive/data/ccaligned_full/` |
| `data/raw/local/Quran/` (Tanzil source files) | `archive/data/Quran/` |

Code changes that went with it — the first is the one that would have broken on
import, the rest are correctness rather than cleanup:

- `collect/__main__.py` no longer imports either collector; `SOURCES` is now
  `{afridoc, gezmu, nllb}`.
- `process/pool.py`: `quran` dropped from `CURATED_SOURCES`.
- `process/process.py`: the am-only `dedupe_keys` branch is now `nllb` alone.
- `process/utils/paths.py`: `CCALIGNED_FULL` removed (its only consumer was the
  archived collector).
- Docstrings across `process/` that used quran as the worked example of a
  multi-reference source now use `religious.csv` (one Amharic Bible against 7
  English ones), which is the remaining source with that shape — the behaviour
  they describe still exists, only the example moved.

**One consequence worth knowing:** nllb is now the ONLY mined source, and it is
already LID-gated upstream in `collect/nllb.py`. Combined with the curated tier's
LID exemption (also 2026-08-14), the pool-time LID filter now drops 0 rows from
every source. It is kept as a live guard for the next mined source, not because it
does work today.

Restoring either source is a `mv` back plus re-adding its `SOURCES` entry and, for
quran, its `CURATED_SOURCES` membership.

## One live dependency

`experiments/domain_breadth/arms.py` reads `am-en-gezmu-8k`'s `manifest.json` to pin
its hyperparameters to the baseline. It falls back to `archive/runs/` when the
run is not in `runs/`, so archiving does not break it — verified. Only the
manifest is needed, not the checkpoints.

## Dead code archived 2026-08-17

Not superseded data or a superseded run — orphaned *code*, found by checking
every `.py` file in the active tree for references from anything else.

| moved | to | why |
|---|---|---|
| `model/transformer.py` | `archive/model/transformer.py` | stale duplicate of `model/architecture/transformer.py`; its own imports (`model.layers.decoder`) point at a package that doesn't exist, so it couldn't even run |
| `model/tokenize/train_tokenizer.py` | `archive/model/tokenize/train_tokenizer.py` | v1's separate English ByteLevel-BPE tokenizer, writing to `data/tokenizer/en/` (now itself archived at `data/tokenizer/archive/en/`). Every experiment since the Gezmu-repro recipe fits one shared vocab inline (`arms.py`'s `cmd_build`) instead |
| `model/tokenize/train_tokenizer_am.py` | `archive/model/tokenize/train_tokenizer_am.py` | same, Amharic Unigram side, `data/tokenizer/am/` |
| `model/data/prepare.py` | `archive/model/data/prepare.py` | v1's `data/final/*.csv` → `data/prepared/am-en/*.pkl` step; `data/prepared/` (unsuffixed) no longer exists on disk, superseded by the same inline `cmd_build` |

`model/common.py`'s `TOKENIZER_AM`/`TOKENIZER_EN` default path (`load_tokenizer`'s
fallback when a run's config omits `data.src_tokenizer`) is untouched — it's a
real fallback, just one no live run exercises anymore since `am-en-base-v4`
(the last run that relied on it) was itself archived. `EXPERIMENTS.md` cites
`train_tokenizer*.py` once, for a vocab-frequency analysis; that citation still
resolves via `archive/`.

## Restoring

`mv archive/runs/<name> runs/` — that is all. Configs inside each run are
absolute-path-free apart from tokenizer paths, and every run directory carries
its own `config.yaml` + `manifest.json`, so a restored run can be re-scored
without reconstructing anything.
