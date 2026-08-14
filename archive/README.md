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
| `runs/am-en-narrow` | experiment #3 narrow arm — in-domain 32.07 |
| `runs/am-en-broad` | experiment #3 broad arm — **best OOD model**, FLORES 11.78 |
| `experiments/domain_breadth/` | builds/trains both arms (`arms.py`) and owns the 2×2 eval (`evaluate.py`) |
| `data/{final,prepared}_translit`, `data/{final,prepared}_broad_translit` | their data |
| `data/tokenizer/shared_translit{,_broad}` | their vocabularies |
| `runs/am-en-base-v4` | **best model in the project** — restored 2026-08-14, see below |
| `data/prepared_v4_32k`, `data/tokenizer/{am,en}_32k_v4` | v4's data and vocabularies |
| `data/{raw,processed,scores,final,benchmarks}` | shared pipeline, not experiment leftovers — see below |

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

## Restoring

`mv archive/runs/<name> runs/` — that is all. Configs inside each run are
absolute-path-free apart from tokenizer paths, and every run directory carries
its own `config.yaml` + `manifest.json`, so a restored run can be re-scored
without reconstructing anything.
