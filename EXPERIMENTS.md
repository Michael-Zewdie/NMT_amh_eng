# Experiment log — Amharic→English NMT

## TL;DR

Best model so far on the **cased** benchmark columns: **`am-en-base-v4` / `best.pt` (step 157,000)**
— archived 2026-08-17, now at `archive/runs/am-en-base-v4/`, still scoreable in place
(`archive/README.md`). On the case-insensitive comparison `am-en-clean-lower` (avg12)
is ahead at 18.55 FLORES / 8.46 MAFAND; v4 holds this row only because it can emit
capitals and clean-lower cannot. `model/translate.py` defaults to `am-en-clean-lower`.

| metric | score |
|---|---|
| in-distribution (own validation, 50,720 pairs) | **26.39 BLEU** |
| FLORES-200 devtest (out-of-distribution, 1,012 pairs) | **14.97 BLEU** / 41.00 chrF++ |
| MAFAND-MT test (out-of-distribution, 1,037 pairs) | **6.30 BLEU** / 28.44 chrF++ |

Architecture for the `base*`/`baseline` runs below: Transformer base — `d_model=512, n_heads=8, 6 encoder + 6 decoder layers, d_ff=2048`. This matches the architecture in Gezmu et al. (LREC 2022), the only other published from-scratch system on this exact language pair we found. The `small*` runs (separate track, see below) use a smaller `d_model=256, n_heads=4, 4+4 layers, d_ff=1024, dropout=0.3`. All runs: batch size 128, inverse-sqrt LR schedule, greedy decoding at eval. **Every BLEU/chrF++ number in this file predates beam search** (added later in `model/search.py`) and is therefore a greedy score — see the `am-en-base-v5` entry below.

**Cross-model comparison, fixed benchmarks** (FLORES/MAFAND are stable files, so these are valid apples-to-apples regardless of when each model trained — unlike in-distribution validation BLEU, which isn't comparable across runs since the training pool's composition changed multiple times):

| run | architecture | FLORES devtest BLEU / chrF++ | MAFAND test BLEU / chrF++ |
|---|---|---|---|
| am-en-baseline* | base, 10k steps only | 2.40 / 20.39 | 1.80 / 17.79 |
| am-en-base-v3 | base | 14.25 / 39.81 | 6.17 / 28.13 |
| **am-en-base-v4 (best.pt)** | base | **14.97 / 41.00** | 6.30 / 28.44 |
| am-en-small | small | 10.09 / 34.95 | 4.84 / 25.92 |
| am-en-small-moderate | small | 13.94 / 38.02 | **6.99 / 27.45** |
| am-en-small-strict | small | 5.46 / 28.31 | 3.02 / 22.23 |

Notable: `am-en-small-moderate` — a much smaller model (256d vs 512d) — nearly matches v4 on FLORES and *beats* it on MAFAND. Worth a real look before assuming bigger-is-better here; see "small-architecture track" below.

*am-en-baseline was a short smoke test, not a real experiment — see its entry in Run history.

**Known gaps (not fully recorded, flagged rather than guessed):** `am-en-base`'s exact train/val/test pair counts at training time; `am-en-baseline`'s data cutoff and full in-distribution BLEU; `am-en-small-moderate` and `am-en-small-strict`'s exact data-cutoff values (architecture, step counts, and all BLEU/chrF++ numbers *are* confirmed for both). These are from a parallel session and/or predate this log — not reconstructable from currently-available files (`data/final/` has been overwritten many times since). If precision on any of these matters later, ask before assuming.

---

## Run history

### am-en-base (pre-existing, before this log started)
Trained on data pooled at `AFRICOMET_CUTOFF=0.5`. Later discovered this cutoff let through ~50% misaligned/garbage pairs (mined NLLB bitext). 100,000 steps, warmup 4000 (`model/configs/archive/base.yaml`). Periodic subset eval peaked 12.78 BLEU @ step 83,000. Full validation BLEU **12.57** — deflated by bad references as much as by model quality; several hand-checked "wrong" translations were actually reasonable, just scored against a nonsense reference. Exact train/val/test pair counts at the time aren't recoverable — `data/final/` has been overwritten by every re-pool since.

### am-en-baseline (early smoke test, base architecture)
10,000 steps, warmup 1000 (`model/configs/archive/baseline.yaml`) — a short smoke-test run, not a serious data/quality experiment. Periodic subset eval peaked 3.33 BLEU @ step 9,500, final 2.64. No full in-distribution validation BLEU was ever computed for it (only the training-time subset above, and FLORES/MAFAND gathered retroactively: 2.40 / 1.80). Data cutoff at the time not recorded and not recoverable.

### am-en-base-v2
Fresh retrain after raising `AFRICOMET_CUTOFF` 0.5→0.62. Data: 1,818,721 train / 227,201 val / 227,297 test. 200,000 steps (~14 epochs). Full validation BLEU **16.60**. No out-of-distribution benchmark existed yet at this point.

### Data expansion + benchmark infrastructure
- Researched additional AM-EN sources (OPUS, Masakhane, HuggingFace). Added **CCAligned** (OPUS, 346,518 raw web-mined pairs) as a training source. **Correction — the effective contribution is far smaller than that headline: 346,518 → 70,350 after `clean()` (script purity alone kills 242,419) → 7,492 after the pool cutoffs. A 2.2% survival rate, ~1.5% of the pool.** Not a filtering failure: only 43% of its Amharic side is ≥99% Ethiopic, 13.9% is majority-Latin, and its `labse_score` distribution is sharply bimodal (p5 0.199, p50 0.849) — a solid core with a long garbage tail that the 0.7 cutoff correctly removes. Note also that OPUS's Moses release ships **no** scores or URLs (just two line-aligned text files, verified against the TMX and XML releases too), so every score on CCAligned is computed locally, unlike NLLB which arrives with `laser_score` + LID + URLs. The original statmt.org sentence-level release *does* carry a LASER column (`Source_Sentence \t Target_Sentence \t LASER_similarity`, threshold 1.04) and would allow the same cheap up-front filtering `collect.py` does for NLLB; domain/URL provenance exists only in the 344GB document-level release.
- Added **MAFAND-MT** (1,936 pairs, human-translated news domain, Masakhane) and **FLORES-200** (2,009 pairs, Wikipedia domain, Meta) as held-out benchmarks in `data/benchmarks/` — deliberately kept out of the training pool.
- Built `processing.utils.decontaminate`: strips any pooled training row that overlaps a benchmark (exact or fuzzy match), so benchmark scores stay valid. Runs unconditionally inside `pool.py`.

### AFRICOMET_CUTOFF=0.85→0.88 (rejected in this run's context, but see correction below)
Tested a stricter uniform cutoff at 0.85, later pushed to 0.88 in a code comment. Applied to the *full* base-track pool (all sources uniformly), 0.85 collapsed it to 131,266 total pairs (train 104,994) — a 94% loss from v2's data, quran down to 45 survivors from 37,380 — so it was reverted to 0.80 as a middle ground.

**Correction:** at the time, the code comment citing "0.85 won the quality-vs-quantity comparison, see memory/data_quality_over_quantity.md" was checked against the *repo's* filesystem only and, not being found there, was treated as unverified. That was a mistake — the file exists in Claude's own persistent memory system (`~/.claude/projects/.../memory/`), written by a **separate, parallel Claude Code session** running concurrently on this same repo (the "other model" training mentioned mid-project) that ran a real controlled 0.62/0.85/0.88 sweep on the *small* (256d) architecture and found 0.85 genuinely best — see `am-en-small` below. That sweep's cutoffs were applied uniformly (curated sources included), same over-filtering issue this log's own investigation found independently — and the memory itself already flags that its ranking was decided on in-distribution BLEU only, before FLORES existed, and needs re-adjudicating. It's re-adjudicated below: this log's tiered approach beats that sweep's best point on every out-of-distribution metric. Full detail in memory files `data-quality-over-quantity`, `nllb-corpus-noise`, `flores-ood-baseline`.

### am-en-base-v3
`AFRICOMET_CUTOFF=0.80` (uniform across all sources), plus CCAligned mixed in. Data: 365,729 total (train 292,614). 35,000 steps (~15 epochs), fresh init.

- In-distribution (full validation): **26.24 BLEU**
- FLORES-200 devtest: **14.25 BLEU** / 39.81 chrF++
- MAFAND-MT test: **6.17 BLEU** / 28.13 chrF++

### Root-cause check against Gezmu et al. (LREC 2022)
The paper reports **33.0 BLEU** for am→en — but on their *own* 2,500-sentence in-domain test split (directly comparable to our own-validation number, not FLORES), using the exact same architecture as ours, trained on **all 140,000** Gezmu pairs, unfiltered.

Our pipeline was applying the same automated AfriCOMET-QE/LaBSE cutoffs meant for noisy *mined* data (NLLB, CCAligned) uniformly to already-curated, professionally-translated sources — gutting Gezmu to 13,638 of 124,409 cleaned pairs and Quran to 446 of 37,380. An imperfect QE model's opinion was overriding validated human translation.

**Fix:** tiered cutoffs in `processing.utils.pool` — `CURATED_SOURCES = {gezmu, afridoc_health, afridoc_tech, quran}` get liberal/disabled LaBSE + AfriCOMET-QE cutoffs; `nllb`/`ccaligned` keep the strict ones (0.7 / 0.8). LID cutoffs stay uniform everywhere (basic sanity check, not a quality judgment). Result: 511,021 total pairs (train 409,455) — Gezmu now contributes 117,087 (vs. 13,638), Quran 37,120 (vs. 446).

### am-en-base-v4
Trained on the tiered-cutoff data (409,454 prepared train pairs). Configured for 224,000 steps (~70 epochs) to fill an overnight run; added `best.pt` checkpoint tracking (saves separately from `last.pt` whenever the periodic eval-subset BLEU improves) as a safety net against unattended overfitting.

Training plateaued in the periodic 500-example subset eval around 28.5–29.7 BLEU starting near step 86k, with val_loss drifting up slightly through step 180k (mild overfitting) — no further real gains. **Stopped manually at step 180,000/224,000.** Best subset checkpoint: step 157,000 (29.69 on the 500-example subset).

**Full evaluation of `best.pt`:**
- In-distribution (full validation, 50,720 pairs): **26.39 BLEU** — only +0.15 over v3, despite 40% more training data and the curated-source fix. The subset (29.69) was optimistic — selected via 157 repeated comparisons against the *same* fixed 500 examples, which biases the "best" checkpoint toward whichever happened to score well on that specific subset (classic adaptive-overfitting-to-a-small-held-out-set effect). The full 50,720-example number is the trustworthy one.
- FLORES-200 devtest: **14.97 BLEU** / 41.00 chrF++ (+0.72 / +1.19 over v3)
- MAFAND-MT test: **6.30 BLEU** / 28.44 chrF++ (+0.13 / +0.31 over v3)

Small, real improvements across all three metrics, but far more modest than the subset trajectory suggested during training.

### am-en-base-v5 (first attempt — abandoned, not scored)
Four coupled recipe changes at once (vocab 32k→8k per side, effective batch 128→1024 sentences via `accum_steps=8`, warmup 4000→2000, dropout 0.1→0.3). Stopped manually at step 6,350 of 15,000; periodic subset eval reached `val_bleu` 23.66 at step 6,000 on the 3,000-example subset. **Never benchmarked, run directory deleted** — the training log survives at `runs/v5_train.superseded.log`. Its one durable contribution is the 8k tokenizers, which the vocab-frequency analysis in `model/tokenize/train_tokenizer*.py` justifies independently and which the Gezmu paper turns out to corroborate (see below).

### am-en-base-v5 (current — Gezmu et al. replication)
Deliberate reset: rather than keep hand-tuning a recipe, reproduce the only published from-scratch system on this language pair and use it as the baseline that later optimizations are measured against. Config `model/configs/archive/base_v5.yaml`, which documents every matched value, every inferred value, and every remaining deviation inline.

Source: Gezmu, Nürnberger & Bati, *Extended Parallel Corpus for Amharic-English Machine Translation*, arXiv:2104.03543v3 (LREC 2022), §4.2. Their reported hyperparameters: Adam, "varied learning rate over the course of training", dropout 0.1, label smoothing 0.1, batch size 1024, six Transformer blocks, eight heads, filter size 2048, hidden size 512, 250,000 steps, decoding with beam 4 and length penalty 0.6 over an average of the last twelve checkpoints, implemented in tensor2tensor.

Two things worth recording from that paper that weren't in this log before:

- **Their headline 33.0 BLEU (am→en) is specifically the NMT-8K system.** Table 3 sweeps subword vocabulary from 1K to 32K; am→en peaks at 8K (33.0) and *declines* at 16K (32.9) and 32K (32.2). Our own move to 8k vocab was derived independently from token-frequency statistics — the paper corroborates it on this exact language pair.
- **"Batch size of 1024" is in tokens, not sentences,** because tensor2tensor's `batch_size` counts subwords. Our mean target sequence is 21.13 tokens over 409,443 prepared training pairs, so the config uses `batch_size: 48` (≈1,014 target tokens) with `accum_steps: 1`. At 48 sentences/step, 250k steps ≈ 29 epochs over our pool.

Remaining deviations, all documented in the config: no checkpoint averaging (they decoded an average of the last 12, we decode a single `best.pt`); `model/optim.py` uses AdamW, so torch's default `weight_decay=0.01` applies where Vaswani/t2t used none (**since fixed — `model/training/optim.py` now passes `weight_decay=0.0` explicitly, making AdamW identical to plain Adam; this is no longer a deviation as of the v2 runs**); different corpus (their 140k vs our pooled ~409k); different subword algorithm (their shared word-piece vs our separate ByteLevel BPE / Unigram, both 8k); `warmup_steps: 4000` is inferred from Vaswani et al. since the paper doesn't state it, and in this repo that also pins peak LR at 6.99e-4.

**Comparability warning:** their 33.0 is on *their own* 2,500-pair in-domain test split. That is comparable to our in-distribution validation BLEU, **not** to FLORES or MAFAND.

Not yet trained.

---

## v2 domain breadth at 140k pairs (`am-en-gezmu-8k` vs `am-en-broad-8k`) — HYPOTHESIS CONFIRMED

Both runs finished 2026-08-08 but were never scored out-of-domain until 2026-08-11. Same architecture (512d/6+6), same 250,000 steps, same recipe, size-matched at ~140k train pairs; the only difference is corpus breadth — `gezmu` alone (narrow, ~83% Watchtower+Bible register) vs everything *except* gezmu pooled (`afridoc_health/tech, ccaligned, nllb, quran, religious`), so the two arms share zero source overlap. Both scored below from `best.pt`, beam 4, length penalty 0.6, same eval code, same day.

| arm (140k pairs) | in-distribution | FLORES devtest | MAFAND test |
|---|---|---|---|
| gezmu-8k (**narrow**) | **28.70** val / 31.22 test (own, checkpoint-averaged) | 4.36 / 25.98 | 2.44 / 21.79 |
| broad-8k (**broad**) | 17.85 val (own) | **10.73 / 34.98** | **4.94 / 25.47** |

**This is the predicted trade, cleanly:** the narrow arm wins in-distribution by ~11 BLEU, the broad arm wins out-of-distribution by **+6.37 FLORES** and **+2.50 MAFAND** — more than doubling the narrow arm on FLORES. Because FLORES/MAFAND are identical files for both arms, those two columns are directly comparable; the in-distribution column is not (each arm scores its own test set).

**The token confound that wrecked the 10k experiment does not apply here.** Source-token budgets are 3.09M (gezmu) vs 3.52M (broad) — 14% apart, versus 2.9× in the 10k run. A 14% difference cannot explain a 2.5× FLORES gap.

**Caveat on the tokenizer, which biases *against* the winner:** both arms use `data/tokenizer/{am,en}`, the 8k vocab fit on Gezmu's own narrow register. The broad arm's medical/tech/web vocabulary therefore gets worse subword coverage than a corpus-matched tokenizer would give it. Broad wins anyway, which makes the result stronger, not weaker.

---

## Experiment #3 (`am-en-broad`, 2026-08-12) — domain breadth HOLDS at higher model capability

The domain-breadth question asked a third time, with both arms trained under the Gezmu preprocessing recipe (Moses + AT4MT transliteration + shared 8k vocab + tied embeddings). Same corpora and split sizes as experiment #2; each arm fits its own 8k shared Unigram vocabulary on its own transliterated train split (symmetric treatment — see `experiments/domain_breadth/arms.py`). Scored best.pt vs best.pt, beam 4, detokenized.

| arm | own-domain | cross-domain | FLORES devtest | MAFAND test |
|---|---|---|---|---|
| **NARROW** `am-en-narrow` | **32.07** / 48.76 | 9.12 / 30.73 | 4.75 / 26.70 | 2.84 / 22.37 |
| **BROAD** `am-en-broad` | 19.17 / 41.22 | 10.57 / 32.61 | **11.78 / 35.82** | **5.41 / 25.98** |
| broad − narrow | −12.90 | +1.45 | **+7.03** | **+2.57** |

### ⚠ Those numbers are confounded by capitalisation — use the table below

Discovered 2026-08-14. **Gezmu ships lowercased** — 0.0% of its English has any uppercase, in train, dev *and* test — while the broad corpus (91% of sentences start capitalised), FLORES (98.7%) and MAFAND (92.1%) are all cased. Nothing in the pipeline restores case; `detok_en` only rejoins punctuation, so the narrow model emits lowercase permanently. That inflates the gap from **both** ends: narrow gets a free pass in-domain (lowercase output vs its own lowercase references — its 32.07 is literally unchanged by case-insensitive scoring) and is over-penalised on the cased benchmarks. Measured ceiling on that penalty: a *perfect* translation that is merely lowercase scores **80.1 BLEU on FLORES and 71.6 on MAFAND**, not 100.

Re-scored case-insensitively (same checkpoints, same decoding, same beam — only `.lower()` on both sides before scoring; the cased column reproduced every published number exactly):

| arm | own-domain | cross-domain | FLORES devtest | MAFAND test |
|---|---|---|---|---|
| **NARROW** `am-en-narrow` | **32.07** / 48.76 | 12.17 / 33.30 | 6.22 / 28.61 | 4.23 / 24.31 |
| **BROAD** `am-en-broad` | 20.09 / 42.26 | 14.72 / 35.75 | **12.46 / 36.84** | **5.95 / 27.13** |
| broad − narrow | −11.98 | +2.55 | **+6.23** | **+1.72** |

**The conclusion survives; the magnitude was overstated.** FLORES +7.03 → **+6.23**, MAFAND +2.57 → **+1.72** (a third of the MAFAND effect was casing). The 2×2 asymmetry gets *sharper* in relative terms: narrow loses **19.9 BLEU** leaving its domain (32.07 → 12.17), broad loses **5.4** (20.09 → 14.72) — a 3.7× difference against 2.7× on the cased numbers. Broad still wins cross-domain transfer, by more than before (+2.55 vs +1.45).

Both scorings are now computed in one pass and stored in `experiments/domain_breadth/results.json` under `bleu`/`chrf++` and `bleu_ci`/`chrf++_ci`; `python -m experiments.domain_breadth eval --lowercase` prints the confound-free table. Retraining narrow on cased text is not possible — Gezmu releases only the lowercased `*.base.*` files — so case-insensitive scoring is the fix, not a data rebuild. **The `am-en-gezmu-8k` reproduction comparison (31.22 vs the paper's 33.0) is unaffected**: Gezmu et al. trained and scored on this same lowercased data, so that one is like-for-like.

**The effect held and did not shrink** (experiment #2 measured +6.37 / +2.50, cased and subject to the same confound). Across all three tests:

| experiment | narrow FLORES | broad FLORES | broad − narrow |
|---|---|---|---|
| #1 — 10k pairs, small arch | 1.37 ± 0.09 | 1.20 ± 0.14 | −0.17 (null, both floored) |
| #2 — 140k, base arch | 4.36 | 10.73 | +6.37 |
| #3 — 140k + paper preprocessing | 4.75 | 11.78 | +7.03 |
| #3 again, case-insensitive | 6.22 | 12.46 | +6.23 |

(#1 and #2 have not been re-scored case-insensitively, and #2 has the identical confound — its narrow arm is the same lowercased Gezmu corpus — so the +6.37 there is overstated by roughly the same margin. Only the last row is confound-free.)

**The 2×2 asymmetry is the sharpest form of the result:** off its own domain the narrow model loses **19.9 BLEU** (32.07 → 12.17); the broad model loses **5.4** (20.09 → 14.72), and beats the narrow model on cross-domain transfer. A narrow corpus buys in-domain score that does not survive leaving the domain. (Case-insensitive figures; the cased numbers were 23.0 and 8.6 — see the capitalisation note above.)

**Do not read #2 → #3 as a measured trend.** n=1 per arm, no variance estimate, and #3 deliberately changed the vocabulary design (per-arm fitted, vs #2's shared Gezmu-fit tokenizer), so part of the larger gap is vocabulary rather than domain. The defensible claim is that the effect is robust to model capability, not that it grew.

`am-en-broad` is the **best OOD model at this data scale** — FLORES 11.78 vs `broad-8k`'s 10.73 on identical data, purely from preprocessing. Wall clock 4.27h (resumed once after a power outage at step 63,500; `best_bleu` was correctly restored, so nothing was lost). Full write-up in `results/domain_breadth_results.md`.

**Renamed 2026-08-13.** These two runs were `am-en-gezmu-translit` and `am-en-broad-translit`; the code was `experiments/{gezmu,broad}_translit.py`. They are now `runs/am-en-{narrow,broad}` and `experiments/domain_breadth/{narrow,broad}.py`. Each `manifest.json` carries a `renamed_from` field, and both `train.log`s still print the old name — the numbers above are unchanged. Data and tokenizer dirs kept their `*_translit` names.

### ⚠ The broad arm is ~34% religious register — and the code no longer says so

Found 2026-08-16. `experiments/domain_breadth/corpus.py` declares `EXCLUDE = {gezmu, religious, quran, ccaligned}`, but that is **not** what built `data/final_broad/`. Its `manifest.json` records `excluded_sources: ["gezmu"]` and a pre-subsample pool of religious 147,048 / quran 37,068 / ccaligned 7,492 alongside nllb 338,334. On the trained 140k the mix was **nllb 62.6%, religious 27.1%, quran 6.9%, ccaligned 1.4%, afridoc 2.2%** (`results/domain_breadth_results.md`, measured from build-time `_source` labels). The four-way EXCLUDE was introduced when the script was restored from commit 0ffcabb; running `corpus --write` today would build a *different, register-disjoint* corpus and silently break comparability with everything above.

**This does not weaken the result — it means the published gap is a floor.** Narrow is ~83% Watchtower/Bible and broad is ~34% religious, so the two arms overlap in register. A genuinely register-disjoint broad arm should widen the +6.23 FLORES, not shrink it. The docstring now carries this warning; the corpus on disk is the artifact of record.

---

## Experiment #5 (`am-en-broad-v2`, 2026-08-16) — the register-disjoint broad arm

The rerun the note above called for. Same architecture, same 250,000 steps, same batch, same 8k shared Unigram, same Moses + AT4MT recipe, same beam 4 / lp 0.6, config chained from `am-en-narrow`'s own manifest — **every config key identical to narrow except `run_name` and the three data paths**. Size held at 145,363 so the match with the narrow arm survives. Four corpus changes:

| change | effect |
|---|---|
| religious / quran / ccaligned dropped | arms now disjoint in register, not just in source file |
| afridoc enters WHOLE | 19,497 rows (**13.4%**, up from 2.2%) — the `SCRIPT_PURITY_EXEMPT` fix recovered 5,093 |
| LASER replaces the AfriCOMET-plus-subsample | `laser > 1.108`, the top 18.3% of nllb by alignment |
| English lowercased | narrow / broad-v2 are now **both** case-free — the 2×2's model-side casing confound is gone, not corrected |

`data/processed/nllb.csv` carries **no `africomet_score` column at all**, so v1's `africomet 0.8465` gate was a silent no-op and its selection was effectively random. That is what LASER replaces.

### The 3×3, case-insensitive

| model | test_gezmu | test_broad | test_broad_v2 | FLORES | MAFAND |
|---|---|---|---|---|---|
| **NARROW** | **32.07** | 12.17 | 16.96 | 6.22 | 4.23 |
| **BROAD v1** | 14.72 | **20.09** | 29.01 | 12.46 | 5.95 |
| **BROAD v2** | 17.68 | **24.73** | **29.28** | **14.53** | **6.84** |

- **broad_v2 − narrow: FLORES +8.31, MAFAND +2.61.** The domain-breadth result holds and **strengthens** — v1 measured +6.23 / +1.72. Removing the register overlap widened the gap, exactly as the note above predicted.
- **broad_v2 − broad_v1: FLORES +2.07, MAFAND +0.89**, at fixed size and fixed step count. Quality of selection beat the loss of source diversity.
- **v2 wins on v1's own home ground: 24.73 vs 20.09 on `test_broad`** (+4.64). That cannot be explained by test-set difficulty, and is the strongest single line here.

**Read the case-insensitive column only.** On the cased column v2 and v1 look identical on FLORES — 11.72 vs 11.78 — because v2 is lowercase and structurally cannot emit capitals while v1 could. The entire +2.07 is invisible in the cased table. This is the clearest demonstration yet of why the `_ci` correction was necessary.

**`test_broad_v2` is an easier test set than `test_broad`** — every model scores higher on it (narrow 12.17 → 16.96, v1 20.09 → 29.01). So the diagonal is not comparable across arms; use FLORES, MAFAND, and `test_gezmu`, which are fixed.

### A prediction on record that was wrong

Before the run, the stated expectation was that broad_v2 would score **9–13 on `test_gezmu`**, down from v1's 14.72, on the reasoning that v1's cross-domain transfer was partly an artifact of its 34% religious share sitting in the same register as gezmu. Removing it should have cost transfer.

It scored **17.68 — up 2.96, not down.** The named falsification condition (">16 on test_gezmu") was met. So v1's cross-domain transfer was **not** propped up by register overlap. Better-aligned mined data plus 6× the AfriDoc produced a model that transfers better even onto gezmu's own register, which it has never seen. Register overlap was diluting the broad arm rather than flattering it.

FLORES 13–16 and a +7 to +10 gap over narrow were both predicted correctly (14.53, +8.31); the transfer direction was not.

---

## `am-en-narrow` (2026-08-11) — the paper's preprocessing recovers +1.32 BLEU

Tests whether Gezmu et al.'s §4.1 preprocessing explains the 1.78 BLEU gap between our reproduction (31.22) and their reported 33.0. A **bundle of five coupled changes** against `am-en-gezmu-8k` — Moses tokenization both sides, Amharic transliterated to Latin via the authors' own AT4MT code, one shared 8k vocabulary replacing two separate 8k, tied source/target embeddings, and English's algorithm forced ByteLevel BPE → Unigram as a rider on sharing. Every other hyperparameter is read from `am-en-gezmu-8k`'s own `manifest.json`, so nothing can drift. Code: `experiments/domain_breadth/arms.py`, `model/tokenize/preprocess.py`.

| metric | `gezmu-8k` | `narrow` | delta |
|---|---|---|---|
| in-dist test, un-averaged (like-for-like) | 30.75 | **32.07** | **+1.32** |
| FLORES devtest | 4.36 / 25.98 | 4.75 / 26.70 | +0.39 |
| MAFAND test | 2.44 / 21.79 | 2.84 / 22.37 | +0.40 |

Also **~8× faster to converge**: matched the baseline's *final* 250k-step validation (28.70) by step 30,000.

**Mechanism, measured before committing GPU time:** transliteration raises cross-lingual shared subword types 180 → 1,003 (2.3% → 12.5% of types in use), and the shared pieces change from digit strings (`2012`, `144,000`) to named entities (`ethiopia`, `protestant`, `benjamin`, `hospital`) — the paper's stated rationale, confirmed empirically.

**Free partial ablation.** The first build of this run lacked Moses, leaving 21.4% of punctuation glued to words and 376 duplicate vocab entries (`congregation.` vs `congregation`). It scored **4.40 vs 15.81** val BLEU at step 5,000, against the baseline's 9.89 — so Moses does a large share of the work and a shared vocabulary alone is *not sufficient*. Root cause worth remembering: **ByteLevel BPE splits punctuation natively (0.00% glued); SentencePiece Unigram does not (21.4%)**. Unifying the vocabulary silently removed a property English had been getting for free.

**~0.93 BLEU still unexplained.** Top remaining candidate is the subword algorithm: t2t's `SubwordTextEncoder` is greedy-merge/wordpiece, closer to BPE, and on this exact transliterated corpus BPE measured **+59% cross-lingual sharing** (946 → 1,501) with lower fertility. (Avoid ByteLevel BPE here — it shreds the transliteration's non-ASCII `ə ɨ ṗ š ṣ ṭ ž ʷ` into byte pairs.) Then checkpoint averaging, which this run could not use. **Batch size is demoted**: "1024 sentences" would imply 1,808 epochs over 140k pairs (Vaswani did 22.7 on WMT14), and the baseline gained only +0.19 over its last 65,000 steps — an asymptote, not starvation.

**Two cautions recorded from this run.** (1) Mid-run validation hit 33.97 and the baseline's val→test offset suggested ~35.6 on test; actual was 32.07. **Val→test offsets do not transfer across differently-preprocessed models.** (2) Semitic root-and-pattern morphology is *non-concatenative* — `s-b-r` is discontinuous in `səbərə` — so **no** contiguous-substring tokenizer (BPE, Unigram, or WordPiece) can extract Amharic roots. Transliteration's benefit is cross-lingual sharing and finer granularity, not root extraction; an earlier claim in this project that transliteration makes the root "a literal character sequence" was wrong.

Full write-up incl. diversity quantification: `results/domain_breadth_results.md`.

---

## Domain-distribution experiment at 10k pairs (`dd-*`, 2026-08-10) — NULL RESULT

**Question.** Does the domain distribution of a small training corpus trade in-distribution BLEU against out-of-distribution generalization? Two arms matched at 10,000 pooled pairs, 5 seeds each, everything else pinned (same tokenizers — reused, not retrained — same 256d/4+4 architecture, LR, warmup, dropout 0.3, 10,000 steps).

- **HEALTH** — AfriDocMT health, all 10,000 raw pairs, deliberately *unfiltered* (human-translated; a QE model's opinion of it is not evidence). 7,918 train.
- **DIVERSE** — NLLB mined bitext, top 10,000 by `africomet_score`, 1,116 distinct source web domains. 7,997 train.

Everything reproducible from `experiments/domain_dist_10k.py` (`build` / `train` / `eval` / `report`); per-run configs + manifests under `runs/dd-*/`, scores in `experiments/domain_dist_10k_results.json`.

**Results** (greedy, mean ± std over 5 seeds):

| arm | in-dist (own test) | cross-domain test | FLORES devtest | MAFAND test |
|---|---|---|---|---|
| health10k | 8.77 ± 0.34 | 2.93 ± 0.13 | **1.37 ± 0.09** | **0.83 ± 0.06** |
| diverse10k | 11.15 ± 0.60 | 0.78 ± 0.07 | 1.20 ± 0.14 | 0.77 ± 0.10 |

**The hypothesis was not confirmed.** The diverse arm did *not* generalize better out-of-domain. On FLORES the narrow arm is marginally *higher* (1.37 vs 1.20) — the opposite of the prediction — and on MAFAND the two are tied within noise. Seed variance is small (±0.06–0.14 on the OOD metrics), so this is a real null, not noise.

**Why it's a null rather than a refutation: both arms are on the floor.** ~1.2 BLEU on FLORES is not a weak translation system, it's a non-functional one. Hand-checked hypotheses are fluent English almost entirely decoupled from the source (the health model does correctly emit "diabetes" and a `Dr. … Professor … University` frame on medical sentences, confirming the pipeline is wired correctly and domain signal exists — it just cannot carry a sentence). For scale, `am-en-small-moderate` reaches FLORES 13.94 on the *same architecture* with ~100k+ pairs. **At 8k training pairs, data scale dominates domain composition so completely that the domain effect is unmeasurable.** A domain-breadth question cannot be asked of two models that both fail.

**Two confounds, both discovered during the run and neither fixable after the fact:**

1. **Matching on pair count did not match training signal.** Health carries 348,794 train tokens vs diverse's 119,262 — **2.9×** — because ranking NLLB by `africomet_score` is strongly biased toward short sentences (mean 14.9 vs 44.1 source tokens). QE models rate short simple pairs highly. Any future top-k-by-QE selection should expect this and match on tokens, not rows.
2. **In-distribution numbers are not comparable between arms.** Each arm's test split has its own intrinsic difficulty, and diverse's sentences are ~3× shorter, hence easier. So diverse's higher own-test BLEU (11.15 vs 8.77) is **not** evidence it is the better model, and the cross-domain column inherits the same asymmetry (health→diverse 2.93 vs diverse→health 0.78 is partly a sentence-length artifact). Only FLORES/MAFAND — identical files for both arms — are clean comparisons here, which is precisely why they are the columns the verdict rests on.

**What would actually answer the question:** rerun at 50–100k pairs per arm, **token-matched rather than row-matched**, with a single-domain source large enough to reach that size (AfriDocMT health caps at 10k total; `religious` has 149k and `gezmu` 124k). Below roughly 50k pairs this architecture does not clear the noise floor on FLORES, so the comparison has no resolving power.

**Superseded 2026-08-11 — see the 140k-pair section above.** That rerun effectively already existed: `am-en-gezmu-8k` (narrow) vs `am-en-broad-8k` (broad), size-matched at 140k pairs, needed only an OOD scoring pass. It confirms the hypothesis decisively (broad +6.37 FLORES, +2.50 MAFAND) and is near token-matched, so **no retrain is needed to answer the domain-breadth question.** What remains unanswerable from scratch is the narrower question of *AfriDocMT health specifically* as the single domain — 10k pairs is its hard ceiling, which is below this architecture's floor. That one needs fine-tuning a pretrained multilingual model rather than training from scratch.

**Process note:** benchmark decontamination caught 3 contaminated rows in the DIVERSE arm, one of them a MAFAND *test* sentence. On a metric that landed at 0.77 BLEU, a single leaked test pair would have been a visible fraction of the score.

---

## v4 data audit (2026-08-14) — what `am-en-base-v4` was actually trained on

Prompted by nothing breaking: v4 is the best model in the project, and the question was
simply whether its data deserved the credit. Decoded all 511,020 prepared pairs back to
text through the restored 32k tokenizers (so this is exactly what v4 saw, not a
reconstruction) and attributed 100% of rows to a source corpus.

**Integrity: clean, no caveats.** 0 duplicate pairs in train, 0 train→validation or
train→test leakage (0 shared `am`, so the grouping held), 0 empty/`am`==`en`/sub-5-char
rows, 0 script contamination either direction, and train/val/test composition matched
within 0.8pp on every source. **The thing most likely to invalidate 26.39 is genuinely
absent.** The 32,435 repeated `am` in train are Quran's multi-translation structure.

**Composition, and where the noise lives:**

| source | pairs | share | misaligned* |
|---|---|---|---|
| nllb | 265,931 | 64.9% | **16.1%** |
| gezmu | 97,105 | 23.7% | low |
| quran | 30,360 | 7.4% | n/a (73 rows w/ digits) |
| ccaligned | 6,659 | 1.6% | — |
| afridoc_tech / _health | 9,233 | 2.2% | — |
| religious | 166 | 0.0% | — |

\* Measured on the reliable subpopulation only: both sides carrying a multi-digit Arabic
numeral, no Ge'ez numerals. **A naive `\d+` detector roughly doubles the rate** (it said
28.6%) because it flags Ge'ez numerals (`፰`/`፴፪`), Amharic spelled-out numbers
(*ስምንት*→"8"), and separator formatting (`1፣200` vs `1200`) as mismatches. Roughly **1 in
10 of all v4 training data is degraded**, not the 1 in 6 the naive detector implied.

**The quality scores cannot see this.** Across deciles, on NLLB:

| | decile 1 | decile 5 | decile 10 | top 1% |
|---|---|---|---|---|
| by **AfriCOMET** | 30.9% | 29.3% | **27.1%** | 27.5% |
| by **LASER** | 38.2% | 25.8% | **4.1%** | 6.5% |

AfriCOMET is flat — it grades adequacy/fluency of a whole sentence, so a fluent pair with
one swapped entity scores well. It saturates at an error floor near 27% and **cannot be
tuned past it**: raising the floor 0.80 → 0.92 discards 97.6% of mined rows to move
misalignment 1.6 points. LASER is a bitext *alignment* score — "are these the same
sentence" — and separates 9x. At matched retention, `laser>1.08` (45.8% kept, 16.3%
misaligned) beats `afri>0.84` (46.6% kept, 27.4%) outright.

**The data is dirty because `LASER_CUTOFF = 1.06` sits at the bottom of the useful
range**, not because the signal was missing. Noise also concentrates by sentence length
(27.5% under 40 chars → 12.2% at 100–140) and by source site (4.2% to 55%), but LASER
alone does nearly all the removable work.

**Noise depresses v4's headline number rather than inflating it.** Scoring v4 on
validation split by whether the reference is intact:

| validation subset | BLEU | chrF++ |
|---|---|---|
| NLLB, digits **aligned** | **31.32** | 53.30 |
| NLLB, no digits | 28.10 | 49.51 |
| curated | 23.46 | 43.99 |
| NLLB, digits **mismatched** | **18.62** | 41.59 |

v4 is *better* than 26.39 suggests — ~31 where the reference is trustworthy. When the
reference says "6 months" and the model correctly says "9 months," the model is punished
for being right. The errors are also near-misses rather than random pairs: 18.62 on the
misaligned bucket only happens if the model produces largely-correct output differing on
one detail. Uncorrelated noise of that kind adds gradient variance, not a learnable false
pattern — which is why ~10% degraded data does not produce a broken model.

**Per-source in-distribution BLEU** (greedy, 1k-sentence samples), which is why quran was
dropped from the next experiment:

| ccaligned | nllb | gezmu | afridoc_health | afridoc_tech | **quran** |
|---|---|---|---|---|---|
| 31.22 | 27.80 | 27.59 | 25.75 | 23.26 | **14.13** |

Quran's English carries bracketed exegetical commentary the Amharic does not — correctly
aligned, but unreproducible by any faithful model, and 7.4% of training data spent on it.

**Two pipeline over-filters found while building the follow-up, both costing good data:**

1. **LID 0.9 on curated sources is a length artifact.** Rows it drops have median Amharic
   length 14–15 chars vs 53–83 for rows it keeps, and inspection found them correctly
   aligned (`ስራ 10፥ 35`/"acts 10: 35.", `የአየር ንብረት ለውጥ`/"Climate change"). LID classifiers
   are unreliable on short strings. Cost: 7,751 correct pairs (5.9% of gezmu).
2. **`script_purity` deletes ~26% of both AfriDoc corpora.** Its Amharic rule
   `[^ሀ-፿"'\./\(\)°º″0-9\s]` fails any row with a Latin letter in the Amharic column — but
   medical/technical Amharic embeds the English term inline: `( Haemoglobin)`,
   `( physiologic)`, `ቫይታሚን B9(folate)`. It also forbids `%` and `-`, so `40%` and
   `ከ6-59 ወር` fail, and the English rule deletes 140 rows for containing an en-dash.
   Relaxing it to "predominantly Ge'ez" recovers health 5,809 → 9,798 and tech
   6,040 → 9,623. **gezmu is not affected** (2.7%; its larger drop is dedupe, correctly
   collapsing repeated verses).

   **Superseded 2026-08-15.** The fix originally lived in `experiments/clean_recipe/` as
   a `reclean()` that re-ran a private copy of the pipeline from raw. It now lives in the
   production pipeline: `process.clean.filters.SCRIPT_PURITY_EXEMPT` skips the step for
   these two sources outright, so `data/processed/` carries the recovered rows and no
   experiment needs its own cleaning path. The Amharic rule was simplified at the same
   time to `[^ሀ-፿"'.()0-9\s]` — Ethiopic block, Arabic numerals, whitespace, and only the
   four ASCII marks `normalize` deliberately leaves behind. Measured against the old rule,
   holding every other step fixed: gezmu −1,028 (loses `/` and `°º″`), afridoc_health
   +2,597, afridoc_tech +2,496, religious unchanged.

   **`am-en-clean-lower` was trained before this change** and its corpus counts below are
   the `reclean()` ones. Rebuilding `data/processed/` now yields health 9,832 and tech
   9,665 instead of 9,798 and 9,623 — close, but not identical, so a rebuilt corpus is not
   bit-for-bit the one that produced the FLORES 18.55 result.

## Experiment #4 (`am-en-clean-lower`, 2026-08-14) — the audit's recipe, COMPLETE

One arm, **250,000 steps** — step-matched to `am-en-narrow` and `am-en-broad`, so the
FLORES/MAFAND comparison against them is clean. (Launched at a 60,000-step screening
budget and raised mid-run once val_bleu was still climbing +0.7 per 5k at step 40k. Safe
to extend because the schedule is Vaswani inverse-sqrt with no `max_steps` term
— `model/training/optim.py:36` — so a resumed run sees bit-identical learning rates to a
native one. That would NOT hold under cosine or linear decay.)

**Split: 80/10/10, length-stratified, grouped by `am`** — 238,967 train / 29,826
validation / 29,876 test. Verified after building: length-bucket shares match across all
three splits to within 0.1pp (short 37.0/36.9/37.0, medium 57.4/57.4/57.4, long
5.6/5.6/5.6) and there are **0 shared `am`** between train and either held-out split.
An earlier build used ~96/2/2 to preserve training data; that was an unrequested
deviation from the project's standard ratios, caught ~50k steps in, and the run was
restarted from scratch on the corrected split.

Corpus 298,669 pairs:

| source | pairs | share |
|---|---|---|
| nllb (`laser>1.08`, `afri>0.80`) | 155,017 | 51.9% |
| gezmu (floors 0.15, no LID) | 124,231 | 41.6% |
| afridoc_health (re-cleaned) | 9,798 | 3.3% |
| afridoc_tech (re-cleaned) | 9,623 | 3.2% |

Changes from v4: NLLB gated on LASER 1.08 rather than 1.06; quran, ccaligned and
religious dropped (religious contributed no distinct text — 100% of its rows were already
in nllb); curated tier enters whole apart from 0.15 LaBSE/AfriCOMET garbage traps; AfriDoc
re-cleaned; English lowercased following Gezmu's own `*.base.*` release; Gezmu recipe
(Moses + AT4MT + shared 8k Unigram + tied embeddings, beam 4 / lp 0.6).

**The 0.15 curated floors are near-no-ops but not useless.** AfriCOMET 0.15 drops 44 gezmu
rows and 0 from either AfriDoc corpus. LaBSE 0.15 drops 144, **133 of which AfriCOMET
keeps** — and those are flatly misaligned at AfriCOMET scores of 0.35–0.51
(`የሰዋስው ስርአቱና...` on grammar → "so joseph began to open up the granaries"). The v4 audit's
LASER-vs-AfriCOMET result repeating one level down: alignment scores see misalignment,
adequacy scores do not, at any threshold.

**Casing is handled by case-insensitive scoring**, matching `experiments.domain_breadth`
rather than adding a truecaser — its `_ci` numbers already exist, so this arm drops
straight into that comparison (`am-en-broad` FLORES 12.46 / MAFAND 5.95, `am-en-narrow`
6.22 / 4.23). The cased column is reported as a floor and measures missing capital
letters, not translation quality; a perfect-but-lowercase translation caps at ~80.1 BLEU
on FLORES. Neither column is comparable to this file's main cased table, and the in-dist
column is on a different corpus than v4's so it is **not** comparable to 26.39.

**Prediction on record before results:** in-dist will look much higher and most of that is
artifact (easier corpus + lowercasing); FLORES/MAFAND are fixed and are the real test,
where +1 to +3 FLORES over `am-en-broad`'s 11.78 cased / 12.46 case-insensitive is the
honest expectation.

**Still open at eval time:** checkpoint averaging. `checkpoint_avg_n: 12` is populating
`checkpoints/avg/`, and at 250k the full rolling window exists. Gezmu et al. decoded an
average of the last 12; this log lists "no checkpoint averaging" as one of the last
remaining deviations from their recipe, and it was worth part of the gap on
`am-en-gezmu-8k` (31.22 averaged vs 30.75 single-checkpoint). Cheap to add at eval.

### Results

Training completed all 250,000 steps cleanly. Best in-training val BLEU **31.08 at step
245,000** (the tail was flat: 30.2–31.1 from step 150k on, while val loss rose from 2.4458
to 2.4723 — mild overfitting with BLEU holding, the usual label-smoothing signature). The
run could have stopped around 150k for ~the same model.

Scored with `python -m experiments.clean_recipe eval`, which transliterates the source and
detokenizes the hypothesis before scoring — `model.evaluate.evaluate_OOD` on its own does
neither and would have measured a preprocessing mismatch against this arm's translit 8k
vocabulary.

| checkpoint | in-dist\* | FLORES | MAFAND |
|---|---|---|---|
| `best.pt` (245k), cased floor | 28.43 | 14.17 | 5.82 |
| `best.pt` (245k), **case-insensitive** | 32.10 | 18.07 | 8.34 |
| `avg12.pt` (195k–250k), cased floor | 29.08 | 14.63 | 5.94 |
| `avg12.pt` (195k–250k), **case-insensitive** | **32.83** | **18.55** | **8.46** |

chrF++ for the averaged model: 53.69 in-dist / 44.14 FLORES / 31.10 MAFAND.

\* in-dist is this corpus's own 29,876-sentence test split — **not** comparable to v4's 26.39.

**Against the step-matched arms, case-insensitive:**

| model | FLORES | MAFAND |
|---|---|---|
| `am-en-clean-lower` (avg12) | **18.55** | **8.46** |
| `am-en-broad` | 12.46 | 5.95 |
| `am-en-narrow` | 6.22 | 4.23 |

**The prediction on record was wrong, in the favourable direction.** +1 to +3 FLORES over
`am-en-broad` was the honest expectation; the actual gap is **+6.09 FLORES / +2.51
MAFAND** — comparable in size to the domain-breadth effect itself (+6.23 FLORES
broad-over-narrow). The two stack rather than overlap: breadth decides what distribution
the model covers, the clean recipe decides how much signal survives inside it. The cased
FLORES floor alone (14.63) is within noise of `am-en-base-v4`'s 14.97 while being
structurally unable to emit a capital letter.

**Checkpoint averaging replicates.** +0.48 FLORES, +0.12 MAFAND, +0.73 in-dist over the
single best checkpoint — same direction and magnitude as the +0.47 measured on
`am-en-gezmu-8k`. It is no longer an open deviation from the Gezmu recipe.

**Caveat on the averaged checkpoint:** `model.common.average_checkpoints` still has **no
caller**. `avg12.pt` was materialized by an out-of-repo script that means the averaged
number is not reproducible from a committed entry point. Folding a `--checkpoint avg` path
into the eval commands is the obvious follow-up; every run is already paying to store 12
snapshots for it.

---

## Key findings

1. **Data alignment bugs can masquerade as "the model is bad."** The original 12.57 BLEU was mostly a stale/noisy-reference artifact, not primarily a model quality problem — always spot-check hypothesis/reference pairs by hand before trusting an aggregate metric.
2. **Automated quality filters need to match the kind of data they're filtering.** A QE/alignment cutoff tuned for noisy mined bitext (NLLB, CCAligned) is actively harmful applied to already-curated, professionally-translated corpora — it throws away validated good data based on an imperfect model's opinion.
3. **In-domain validation BLEU is not the full picture.** Our own validation set is drawn from the same pool as training, so it under-reports how narrow the model actually is. FLORES/MAFAND (genuinely out-of-domain, decontaminated) are much harder and are the numbers comparable to published systems.
4. **A small fixed eval subset used repeatedly for checkpoint selection overfits to itself.** `best.pt`'s subset score (29.69) overstated true quality by ~3.3 BLEU vs. the full validation set (26.39) — always confirm a "best" checkpoint against the full held-out set before trusting it.
5. **From-scratch training on ~400-500k pairs plateaus well short of what pretrained-multilingual fine-tuning could likely reach.** Not yet tested in this project — see next steps.
6. **When a code comment or claim can't be verified, check Claude's persistent memory (`~/.claude/projects/.../memory/`), not just the repo.** A parallel Claude Code session on this same repo writes there; a repo-only search will wrongly conclude a well-founded claim is unverified. This log's own "the memory file doesn't exist" claim (made mid-project) was itself wrong for exactly this reason — corrected above.

## Small-architecture track (parallel, separate from the base-model runs above)

Run by a **separate, parallel Claude Code session** working the same repo concurrently (not this log's conversation) — configs in `model/configs/archive/small*.yaml`, `d_model=256, n_heads=4, 4+4 layers, d_ff=1024, dropout=0.3`, meaningfully smaller and more regularized than the base architecture. That session's own memory (`data-quality-over-quantity`, `nllb-corpus-noise`, `flores-ood-baseline` — see `~/.claude/projects/-home-michael-NMT-amh-eng/memory/`) is the authoritative source for what it tried; summarized here for a single combined record.

- **am-en-small** = that session's uniform `AFRICOMET_CUTOFF=0.85` point in a controlled 0.62/0.85/0.88 sweep (all uniform, curated sources included — same over-filtering issue this log found independently via the Gezmu paper). 102,699 train pairs, 80,000 steps. In-distribution BLEU **24.97** (their best of the three sweep points; 0.62 got only ~17-18 in-dist with 1.15M pairs, 0.88 overfit hard and got 21.09 with 40,863 pairs). FLORES 10.09/34.95, MAFAND 4.84/25.92.
- **am-en-small-moderate**: resumed once, 287,000 steps total, much longer budget (`max_steps: 300000` in its config) than the other two sweep points. Periodic subset eval peaked only 18.21 @ step 224,000 — much lower than `small`'s in-distribution peak — but generalizes best of any model trained in this entire project: FLORES 13.94/38.02 (close to v4's 14.97), **MAFAND 6.99/27.45 (best of any run, beats v4's 6.30)**. Exact cutoff/data snapshot not confirmed from this log's side, but given the name and the gap between its weak in-distribution score and strong OOD score, this is plausibly that session's own attempt at loosening the curated-source filtering — i.e. independently converging on the same fix this log arrived at via the Gezmu paper.
- **am-en-small-strict**: 80,000 steps, name implies the 0.88 (or stricter) point. FLORES 5.46/28.31, MAFAND 3.02/22.23 — the worst OOD result of any model trained. Corroborates the same lesson as the 0.85→0.88 rejection above, independently, in a completely different architecture/run.

**This log's tiered-cutoff result re-adjudicates that session's own open question** (its `flores-ood-baseline` memory explicitly says the 0.62/0.85/0.88 ranking was never checked on FLORES): v4's FLORES 14.97 and MAFAND 6.30 beat `am-en-small`'s uniform-0.85 FLORES 10.09/MAFAND 4.84 outright — though part of that gap is architecture size (512d vs 256d) as well as the tiered-cutoff fix, so it isn't a clean single-variable comparison. `am-en-small-moderate` beating v4 on MAFAND despite the size difference is the strongest single data point that the tiering idea (however it was arrived at) matters more than raw model size.

**Takeaway:** `small-moderate`'s out-of-domain generalization despite a much lower in-domain/subset score suggests the small, heavily-regularized architecture may generalize better per-parameter than scaling up the base model further — worth a real head-to-head (same data snapshot, both architectures) before committing to "bigger is better" as the next step.

## Not yet tried / possible next steps

### Free — no retraining required, do these the day a run finishes
- ~~**Checkpoint averaging over the last 12 checkpoints.**~~ **DONE (twice).** Built for `am-en-gezmu-8k`, where it was worth **+0.47** (30.75 → 31.22). Silently **removed by the "Refactor" commit** along with `experiments/average_checkpoints.py`, which is why `am-en-narrow` has no rolling snapshots and cannot be averaged. **Restored 2026-08-11** as `model.common.save_weights_only` + `average_checkpoints`, driven by `training.checkpoint_avg_n` in `model/training/train.py` (weight-only snapshots under `<run>/checkpoints/avg/`, pruned to the newest N). Verified: exact arithmetic mean, dtype preserved, pruning keeps the newest N.
- **Decoding sweep.** `data/benchmarks/flores200_am_en.csv` reserves a `dev` split precisely so `devtest` stays untouched — sweep `beam_size` 4-8 and `length_penalty` 0.4-1.0 on `dev`, report on `devtest`. ~+0.3-1.0.
- **Re-score every pre-beam checkpoint under beam 4.** Every BLEU/chrF++ in this file is a *greedy* score (beam search landed after them), so the cross-model table is not comparable to any run scored with beam. Cheap, and it should happen before v6 is compared to anything above.

### Training-loop fixes
- **Length bucketing.** `model/data/dataset.py` / `make_dataloader` batch by raw index, so a 5-token sentence and a 150-token one land in the same batch and everything pads to the longest. Two costs: wasted compute on padding, and batch-to-batch variance in effective token count. It is also what forces `batch_size` to be set by the worst case (`base_v5.yaml`'s comment measures 128 sentences peaking at 11.2 GiB purely because one long sentence can drag a batch to `max_src_len`). A length-bucketed sampler would cut padding waste substantially and let the physical batch grow. **Note: this is NOT the cause of the epoch-periodic ripple in `train/loss`** — that was measured (autocorrelation peak at lag 8,500 steps vs 8,517 steps/epoch, ratio 1.00) and is the ordinary within-epoch recency effect, amplitude ±0.02 on a loss of 2.81, absent from `val_loss`. Benign.
- ~~**`best_bleu` is not restored on resume.**~~ **FIXED, REVERTED, RE-FIXED.** `train.py` set `best_bleu = -1.0` after `load_checkpoint`, so the first eval after any resume overwrote `best.pt` even when it scored *worse* than the pre-interruption best. Harmless while a curve is still rising; silently destroys the best checkpoint when resuming a plateaued run. Fixed once during v6 (the step-115,000 best was hand-copied to `best_step115000.pt` as insurance), then **the "Refactor" commit reverted it** — `save_checkpoint` stopped persisting the field. **Re-fixed 2026-08-11**: `save_checkpoint` stores `best_bleu`, `model.common.load_best_bleu` restores it, defaulting to -1.0 for older checkpoints (backward compatible, verified both paths).

**Both of the above are a warning about this log.** Two items recorded here as *done* were silently undone by a refactor, and nothing caught it until a run needed them. When relying on a fix documented in this file, grep the code and confirm it is still there.
- Effective batch size is now handled (`accum_steps` exists; v6 deliberately uses 48 sentences ≈ 1024 target tokens to match tensor2tensor's token-counted batch).

### Data
- **Neighbour-relative margin scoring to replace raw LaBSE cosine on curated sources.** Deferred from v6 (needs a re-embed of ~174k rows, ~10 min on GPU, cached afterward). Raw cosine cannot work here: a floor provably cannot separate correct Quran verses from real Gezmu misalignments — they occupy the same score band, because absolute cosine confounds "is this pair wrong" with "is this domain hard to embed". A margin cancels the domain term by comparing a pair against its own positional neighbours. Verified on labelled examples: 5/5 known Gezmu misalignments caught (the winning neighbour was `j±1` in four of five), while correct Quran verses that any absolute floor would have deleted came back positive. Caveats: needs a same-verse guard for multi-reference sources like quran (`dedupe_keys=("am","en")` puts near-duplicate translations adjacent, making margins noisy); flag on `margin < 0` rather than a tuned threshold; treat as *review*, not silent deletion. Requires document order, which `clean()` preserves today but nothing enforces — add an assert or an explicit `doc_order` column.
- **Quran is shipped pre-tokenized.** 96.24% of its English side has a space before punctuation (`Allah , most benevolent , ever-merciful .`) — Moses-style tokenization baked into the Tanzil source, not something this pipeline did. Every other source is clean (gezmu 0.00%, afridoc ~0.05%, both benchmarks <2.5%). **Score-neutral** — sacrebleu's 13a tokenizer splits punctuation on both sides, so `"mat ."` vs `"mat."` scores BLEU 100.00 / chrF++ 100.00 — but it is ~9% of the training signal teaching unnatural spacing, which shows up in `model.translate` output and never in BLEU. One-line regex in `processing/clean/normalize.py` for a future data rebuild. This is also the true cause of the recurring sacrebleu "you forgot to detokenize your test data" warning during training: that check counts *hypotheses* ending in `" ."` against an **absolute** threshold of 100, and a 3,000-example eval subset carries ~143 such lines.
- More curated data: full JW300 (flagged misalignment issues in the literature, needs heavy filtering), EthioMT, AfroLingu-MT.

### Structural — the actual levers
- **Fine-tuning a pretrained multilingual model** (e.g. NLLB-200-distilled-600M) instead of training from scratch. Still the single biggest lever, and the one that targets the real problem: v4 scores 26.39 in-distribution but only 14.97 on FLORES, and that ~11-point gap *is* the low-resource ceiling. Everything else on this page is worth 1-2 BLEU; this is worth substantially more, specifically out-of-domain.
- **Backtranslation.** Gezmu et al. measured this on this exact language pair — Table 4, NMT 26.7 → NMT+CACO 27.8, so +1.1 BLEU. Raw material is already on disk: the pipeline **discards ~15.8M NLLB pairs** plus ~63k CCAligned, whose *pairings* are bad but whose individual sentences are perfectly good monolingual text in both languages. Note the direction — improving am→en needs monolingual *English* translated into Amharic, so it requires training the reverse en→am model first.
- **Minimum Risk Training / RL on a BLEU reward.** Deliberately last. MRT (Shen et al. 2016) needs *k* sampled decodes per training example, and with no KV cache in the decoder that dominates everything; gains are ~+0.5-1.5 on an already-converged model. It also optimizes the yardstick directly, which stops it being a yardstick. Not worth it before the two items above.
