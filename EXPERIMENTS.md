# Experiment log — Amharic→English NMT

## TL;DR

Best model so far: **`am-en-base-v4` / `best.pt` (step 157,000)**

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

Remaining deviations, all documented in the config: no checkpoint averaging (they decoded an average of the last 12, we decode a single `best.pt`); `model/optim.py` uses AdamW, so torch's default `weight_decay=0.01` applies where Vaswani/t2t used none; different corpus (their 140k vs our pooled ~409k); different subword algorithm (their shared word-piece vs our separate ByteLevel BPE / Unigram, both 8k); `warmup_steps: 4000` is inferred from Vaswani et al. since the paper doesn't state it, and in this repo that also pins peak LR at 6.99e-4.

**Comparability warning:** their 33.0 is on *their own* 2,500-pair in-domain test split. That is comparable to our in-distribution validation BLEU, **not** to FLORES or MAFAND.

Not yet trained.

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
- **Checkpoint averaging over the last 12 checkpoints.** Gezmu et al. decode from an average, we decode a single `best.pt`. Open deviation in `base_v6.yaml`. Typically +0.3-0.8 BLEU for an afternoon's work on weights already saved to disk.
- **Decoding sweep.** `data/benchmarks/flores200_am_en.csv` reserves a `dev` split precisely so `devtest` stays untouched — sweep `beam_size` 4-8 and `length_penalty` 0.4-1.0 on `dev`, report on `devtest`. ~+0.3-1.0.
- **Re-score every pre-beam checkpoint under beam 4.** Every BLEU/chrF++ in this file is a *greedy* score (beam search landed after them), so the cross-model table is not comparable to any run scored with beam. Cheap, and it should happen before v6 is compared to anything above.

### Training-loop fixes
- **Length bucketing.** `model/data/dataset.py` / `make_dataloader` batch by raw index, so a 5-token sentence and a 150-token one land in the same batch and everything pads to the longest. Two costs: wasted compute on padding, and batch-to-batch variance in effective token count. It is also what forces `batch_size` to be set by the worst case (`base_v5.yaml`'s comment measures 128 sentences peaking at 11.2 GiB purely because one long sentence can drag a batch to `max_src_len`). A length-bucketed sampler would cut padding waste substantially and let the physical batch grow. **Note: this is NOT the cause of the epoch-periodic ripple in `train/loss`** — that was measured (autocorrelation peak at lag 8,500 steps vs 8,517 steps/epoch, ratio 1.00) and is the ordinary within-epoch recency effect, amplitude ±0.02 on a loss of 2.81, absent from `val_loss`. Benign.
- **`best_bleu` is not restored on resume.** `model/train.py` sets `best_bleu = -1.0` after `load_checkpoint`, so the first eval after any resume overwrites `best.pt` even when it scores *worse* than the pre-interruption best. Harmless while a curve is still rising; silently destroys the best checkpoint when resuming a plateaued run. Fix: persist `best_bleu` in the checkpoint dict and restore it, defaulting to -1.0 when absent (backward compatible). Bit by this during v6 — the step-115,000 best was hand-copied to `best_step115000.pt` as insurance.
- Effective batch size is now handled (`accum_steps` exists; v6 deliberately uses 48 sentences ≈ 1024 target tokens to match tensor2tensor's token-counted batch).

### Data
- **Neighbour-relative margin scoring to replace raw LaBSE cosine on curated sources.** Deferred from v6 (needs a re-embed of ~174k rows, ~10 min on GPU, cached afterward). Raw cosine cannot work here: a floor provably cannot separate correct Quran verses from real Gezmu misalignments — they occupy the same score band, because absolute cosine confounds "is this pair wrong" with "is this domain hard to embed". A margin cancels the domain term by comparing a pair against its own positional neighbours. Verified on labelled examples: 5/5 known Gezmu misalignments caught (the winning neighbour was `j±1` in four of five), while correct Quran verses that any absolute floor would have deleted came back positive. Caveats: needs a same-verse guard for multi-reference sources like quran (`dedupe_keys=("am","en")` puts near-duplicate translations adjacent, making margins noisy); flag on `margin < 0` rather than a tuned threshold; treat as *review*, not silent deletion. Requires document order, which `clean()` preserves today but nothing enforces — add an assert or an explicit `doc_order` column.
- **Quran is shipped pre-tokenized.** 96.24% of its English side has a space before punctuation (`Allah , most benevolent , ever-merciful .`) — Moses-style tokenization baked into the Tanzil source, not something this pipeline did. Every other source is clean (gezmu 0.00%, afridoc ~0.05%, both benchmarks <2.5%). **Score-neutral** — sacrebleu's 13a tokenizer splits punctuation on both sides, so `"mat ."` vs `"mat."` scores BLEU 100.00 / chrF++ 100.00 — but it is ~9% of the training signal teaching unnatural spacing, which shows up in `model.translate` output and never in BLEU. One-line regex in `processing/clean/normalize.py` for a future data rebuild. This is also the true cause of the recurring sacrebleu "you forgot to detokenize your test data" warning during training: that check counts *hypotheses* ending in `" ."` against an **absolute** threshold of 100, and a 3,000-example eval subset carries ~143 such lines.
- More curated data: full JW300 (flagged misalignment issues in the literature, needs heavy filtering), EthioMT, AfroLingu-MT.

### Structural — the actual levers
- **Fine-tuning a pretrained multilingual model** (e.g. NLLB-200-distilled-600M) instead of training from scratch. Still the single biggest lever, and the one that targets the real problem: v4 scores 26.39 in-distribution but only 14.97 on FLORES, and that ~11-point gap *is* the low-resource ceiling. Everything else on this page is worth 1-2 BLEU; this is worth substantially more, specifically out-of-domain.
- **Backtranslation.** Gezmu et al. measured this on this exact language pair — Table 4, NMT 26.7 → NMT+CACO 27.8, so +1.1 BLEU. Raw material is already on disk: the pipeline **discards ~15.8M NLLB pairs** plus ~63k CCAligned, whose *pairings* are bad but whose individual sentences are perfectly good monolingual text in both languages. Note the direction — improving am→en needs monolingual *English* translated into Amharic, so it requires training the reverse en→am model first.
- **Minimum Risk Training / RL on a BLEU reward.** Deliberately last. MRT (Shen et al. 2016) needs *k* sampled decodes per training example, and with no KV cache in the decoder that dominates everything; gains are ~+0.5-1.5 on an already-converged model. It also optimizes the yardstick directly, which stops it being a yardstick. Not worth it before the two items above.
