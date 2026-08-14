# Domain breadth vs. generalization — Amharic→English

**Question:** does the domain distribution of the training corpus trade in-distribution BLEU against out-of-distribution generalization?

**Answer: yes, decisively — at sufficient data scale.** A narrow corpus wins in-distribution by ~11 BLEU; a broad corpus more than doubles it out-of-distribution.

Generated 2026-08-11. Full narrative in [`EXPERIMENTS.md`](../EXPERIMENTS.md).

---

## 1. Headline result — replicated at two capability levels

The domain-breadth effect was tested three times, at increasing model quality. It is **absent when models are broken, present when they work, and slightly larger when they work well.**

| experiment | pairs/arm | narrow FLORES | broad FLORES | **broad − narrow** |
|---|---|---|---|---|
| #1 — 10k, small architecture | 10k | 1.37 ± 0.09 | 1.20 ± 0.14 | **−0.17** (null — both floored) |
| #2 — 140k, base architecture | 140k | 4.36 | 10.73 | **+6.37** |
| #3 — 140k + paper preprocessing | 140k | 4.75 | **11.78** | **+7.03** |

> **⚠ Every number in this file is CASED and therefore confounded** (found 2026-08-14).
> Gezmu ships lowercased (0.0% uppercase); the broad corpus and both benchmarks are
> cased, and nothing restores case at inference. Case-insensitive re-scoring moves the
> headline to **FLORES +6.23 / MAFAND +1.72** (from +7.03 / +2.57) and the 2×2 drops to
> 19.9 vs 5.4 BLEU. The conclusion holds; the magnitudes here are inflated. Corrected
> table and full reasoning: EXPERIMENTS.md, experiment #3. Both scorings live in
> `experiments/domain_breadth/results.json` (`bleu` vs `bleu_ci`).

**Experiment #3's full 2×2** (`am-en-narrow` vs `am-en-broad`, best.pt, beam 4, detokenized):

| arm | own-domain test | cross-domain test | FLORES devtest | MAFAND test |
|---|---|---|---|---|
| **NARROW** (Gezmu) | **32.07** / 48.76 | 9.12 / 30.73 | 4.75 / 26.70 | 2.84 / 22.37 |
| **BROAD** (pooled) | 19.17 / 41.22 | 10.57 / 32.61 | **11.78 / 35.82** | **5.41 / 25.98** |
| | *narrow +12.90* | *broad +1.45* | **broad +7.03** | **broad +2.57** |

**The asymmetry in the 2×2 is the sharpest statement of the result.** Moving off its own domain, the narrow model loses **23.0 BLEU** (32.07 → 9.12). The broad model loses **8.6** (19.17 → 10.57) — and its cross-domain score on Gezmu's *own* test set (10.57) beats what the narrow model manages on broad's (9.12). Training narrowly buys a large in-domain score that does not survive contact with anything else.

**Caveats on reading #2 → #3 as a trend:**
- **n=1 per arm** at 140k. The +6.37 → +7.03 change is small and has no variance estimate behind it; the defensible claim is that the effect **held and did not shrink**, not that it measurably grew.
- **#3 changed the vocabulary design.** #2 gave both arms the Gezmu-fit tokenizer; #3 fits a fresh 8k shared vocabulary on each arm's own training split (same algorithm, size, and procedure — symmetric treatment). Under #2's design broad won *despite* a foreign vocabulary; here that handicap is removed, so part of the larger gap is vocabulary rather than domain.

**`am-en-broad` is the best out-of-distribution model trained at this data scale** — FLORES 11.78 vs `broad-8k`'s 10.73 (+1.05) on identical data, from preprocessing alone.

---

## 1b. Original 140k comparison (`am-en-gezmu-8k` vs `am-en-broad-8k`)

Size-matched at ~140k train pairs, 250,000 steps, **29 of 32 config keys byte-identical** (only `prepared_dir`, `run_name`, `resume_from` differ). Scored from `best.pt`, beam 4, length penalty 0.6, same eval code, same day.

| arm | corpus | in-distribution | FLORES devtest | MAFAND test |
|---|---|---|---|---|
| `am-en-gezmu-8k` | **narrow** — Gezmu only (~83% Watchtower/Bible) | **28.70** val · 31.22 test | 4.36 / 25.98 | 2.44 / 21.79 |
| `am-en-broad-8k` | **broad** — all sources *except* Gezmu | 17.85 val | **10.73 / 34.98** | **4.94 / 25.47** |
| | | *narrow +10.85* | **broad +6.37** | **broad +2.50** |

BLEU / chrF++. The two arms share **zero source overlap** by construction.

### Robustness checks — the result survives all three

| check | finding |
|---|---|
| **Token budget** | 3.09M vs 3.52M source tokens (14% apart). Cannot explain a 2.5× FLORES gap. |
| **Source normalization** | Removing `normalize()` from the Amharic side: gap is **+6.34** FLORES (vs +6.37). Unchanged. |
| **Tokenizer bias** | Shared 8k vocab was fit on *Gezmu's own* narrow register — it handicaps the broad arm. Broad wins anyway. |

The tokenizer point matters: every bias in the setup runs **against** the winning arm, so the true effect is likely larger than measured.

---

## 2. The same experiment at 10k pairs — NULL RESULT

Run first (`dd-*`, 2 arms × 5 seeds). AfriDocMT health (unfiltered, single-domain) vs NLLB top-10k by AfriCOMET (1,116 web domains). Small architecture (256d/4+4), greedy decoding.

| arm | in-dist (own test) | cross-domain test | FLORES devtest | MAFAND test |
|---|---|---|---|---|
| `dd-health10k-s{1..5}` | 8.77 ± 0.34 | 2.93 ± 0.13 | 1.37 ± 0.09 | 0.83 ± 0.06 |
| `dd-diverse10k-s{1..5}` | 11.15 ± 0.60 | 0.78 ± 0.07 | 1.20 ± 0.14 | 0.77 ± 0.10 |

Mean ± std over 5 seeds. **No OOD advantage for diversity** — narrow is marginally *higher* on FLORES.

**This is a floor effect, not a refutation.** At ~1.2 BLEU both arms are non-functional: hypotheses are fluent English decoupled from the source. `am-en-small-moderate` reaches FLORES 13.94 on the *same architecture* with ~100k pairs. At 8k pairs, data scale dominates domain composition so completely the domain effect is unmeasurable.

Seed variance is ±0.06–0.14 on the OOD metrics, so this is a genuine null rather than noise — the 5-seed design did its job.

---

## 2b. Were the models actually forced to generalize? (distribution shift, measured)

Both benchmarks are drawn from a genuinely different distribution than either training corpus, and neither model could have memorized them. Measured with `process/dist/domain_shift.py` on cached MPNet English embeddings.

| training corpus | vs FLORES devtest | vs MAFAND test |
|---|---|---|
| | AUC / MMD² / max-NN | AUC / MMD² / max-NN |
| **broad** (`final_broad`) | 0.910 / 0.0248 / 0.824 | 0.942 / 0.0350 / 0.852 |
| **Gezmu** | 0.962 / 0.0559 / 0.868 | 0.971 / 0.0626 / 0.979 |
| *control: FLORES dev vs devtest (same distribution)* | 0.642 / 0.0030 / — | — |

- **AUC** — cross-validated ROC-AUC of a logistic classifier separating training text from benchmark text. 0.5 = indistinguishable, 1.0 = disjoint.
- **MMD²** — squared maximum mean discrepancy (RBF kernel), permutation p = 0.005 for every row.
- **max-NN** — the highest cosine similarity between any benchmark sentence and any of 60,000 sampled training sentences (the leakage check).

**Different distributions, decisively.** Read against the same-distribution control (AUC 0.642, MMD² 0.0030), both corpora are far from both benchmarks — broad at 8–12× the control's MMD², Gezmu at 19–21×.

**No leakage for the broad arm.** Zero of 1,012 FLORES rows and zero of 1,037 MAFAND rows exceed 0.9 cosine similarity to any training sentence; the closest anywhere is 0.824 / 0.852, median ~0.53. This catches semantic near-duplicates that string-based decontamination cannot. So `broad`'s FLORES 11.78 / MAFAND 5.41 is genuine out-of-distribution performance.

⚠️ **One leaked row in the NARROW arm.** A single MAFAND sentence sits at 0.979 similarity to Gezmu training text — Gezmu was never decontaminated (its repro bypasses the pooling pipeline). It is 1 row in 1,037, and it would *inflate* the narrow arm's MAFAND score, which the narrow arm nonetheless lost.

### What this means for the hypothesis — and the alternative it does not exclude

**Gezmu is further from both benchmarks than broad is** (2.3× by MMD² on FLORES, 1.8× on MAFAND). That raises a real alternative explanation:

> **(a) breadth causes generalization** vs **(b) broad merely resembles the benchmarks more**, making the result "train on data like your test set" rather than anything about diversity.

Three things weigh against (b) being the whole story: broad wins on **two independent benchmarks** (Wikipedia-derived and news, not one lucky match); broad also wins **cross-domain transfer** (10.57 on Gezmu's test vs narrow's 9.12 on broad's); and **breadth and proximity are not independent** — a corpus covering more of semantic space is necessarily closer in expectation to *any* arbitrary target, so proximity is plausibly the mechanism by which breadth works rather than a confound contaminating it.

The strongest evidence that proximity matters is in the main table: on its own test split the narrow model wins by **22.9 BLEU**. Proximity dominates where you have it.

**Refined claim, better supported than the loose one:** narrow training concentrates all of your proximity in one place; broad training spreads it — losing in-domain BLEU and gaining everywhere else.

**What would falsify (a):** a held-out benchmark *closer to Gezmu than to broad* on which the narrow model still won. No such benchmark exists here — FLORES and MAFAND are both general-domain. This question is not closed.

---

## 3. Limitations

**Of the 140k result (the one the conclusion rests on):**

1. **Provenance is confounded with breadth.** The narrow arm is one curated human-translated corpus; the broad arm pools mined and curated sources. "Breadth" and "source mixture" are not separated. A clean test would sample multiple narrow corpora of matched provenance.
2. **"Narrow" here means religious register** (Gezmu, ~83% Watchtower/Bible), not an arbitrary domain. How far this generalizes to other single domains is untested.
3. **In-distribution BLEU is not comparable across arms** — each scores its own test split, with its own intrinsic difficulty. Only FLORES/MAFAND (identical files) support cross-arm claims. The +10.85 in-distribution figure is indicative, not a measurement.
4. **`gezmu-8k` was resumed twice mid-run** and was affected by a since-fixed `best_bleu`-on-resume bug. It self-healed, but checkpoint selection was perturbed where `broad-8k`'s was not.
5. **Single seed per arm.** Unlike the 10k experiment, these are n=1. The effect is far larger than plausible seed noise, but no variance estimate exists at this scale.

**Of the 10k result:**

6. **Row-matching ≠ token-matching.** Ranking NLLB by AfriCOMET selects short sentences (mean 14.9 vs 44.1 source tokens), so equal row counts gave **2.9× different token budgets**. Any future top-k-by-QE selection must match on tokens.
7. **AfriDocMT health caps at 10,000 pairs total**, which is below this architecture's floor — so health-as-single-domain is not answerable by training from scratch. It needs fine-tuning a pretrained multilingual model.

8. **Breadth and benchmark-proximity are not separated.** The broad corpus is measurably closer to both benchmarks than Gezmu is (§2b), and no held-out benchmark in this project sits closer to Gezmu than to broad. So "breadth helps" and "broad happens to resemble general-domain benchmarks" cannot be fully disentangled with the evaluation sets available.

---

## 4. Where the models are

| model | checkpoint | size | config |
|---|---|---|---|
| `am-en-gezmu-8k` (narrow) | `runs/am-en-gezmu-8k/checkpoints/best.pt` | 600M | `runs/am-en-gezmu-8k/config.yaml` |
| | `runs/am-en-gezmu-8k/checkpoints/averaged.pt` | 216M | last-12 average → 31.22 in-dist |
| `am-en-broad-8k` (broad) | `runs/am-en-broad-8k/checkpoints/best.pt` | 600M | `runs/am-en-broad-8k/config.yaml` |
| | `runs/am-en-broad-8k/checkpoints/last.pt` | 600M | step 250,000 |
| `am-en-narrow` (narrow, paper preprocessing) | `runs/am-en-narrow/checkpoints/best.pt` | ~550M | 48.2M params — tied embeddings halve the embedding table |
| `am-en-broad` (broad, paper preprocessing) | `runs/am-en-broad/checkpoints/best.pt` | — | **experiment #3, training** |
| `dd-{health,diverse}10k-s{1..5}` | `runs/dd-*/checkpoints/best.pt` | 10 × ~140M | `runs/dd-*/config.yaml` |

Both `config.yaml` files were reconstructed from each run's `manifest.json` on 2026-08-11 — the originals (`model/configs/broad_8k_matched.yaml`, `gezmu_8k.yaml`) were deleted in a refactor. `runs/` and `data/` are gitignored.

**Reproduce:** `python -m experiments.domain_dist_10k {build,train,eval,report}` (10k arms) · `python -m experiments.domain_breadth {build,train} --arm narrow|broad` then `python -m experiments.domain_breadth eval` (experiment #3, incl. the full 2×2). Scores in `experiments/domain_dist_10k_results.json`, `runs/am-en-narrow/results.json`, `experiments/domain_breadth/results.json`.

**Corpus diversity** is measurable with `python -m process.dist.diversity <corpus.csv> --n 20000`, which reports the Vendi Score (effective number of distinct items), mean pairwise cosine distance, and distinct-trigram ratio. It refuses to silently compare corpora at mismatched sample sizes.

---

## 6. Corpus diversity, quantified

All at **n=4,600 held constant** (every one of these metrics grows with n, so unmatched samples measure corpus size rather than diversity):

| corpus | **Vendi (effective #)** | mean pair dist | distinct 3-gram |
|---|---|---|---|
| religious | 71.7 | 0.7211 | 0.745 |
| health10k (exp #1 narrow) | 108.5 | 0.8311 | 0.867 |
| gezmu (exp #2/#3 narrow) | 133.5 | 0.8623 | 0.909 |
| broad pool (exp #2/#3 broad) | 174.4 | 0.9144 | 0.925 |
| diverse10k (exp #1 broad) | 203.9 | 0.9381 | 0.906 |
| NLLB full pool | 221.1 | 0.9406 | 0.808 |

**This confirms experiment #1's null was a floor effect, not a weak manipulation.** Its diversity contrast was **1.88×** (108.5 → 203.9) — *larger* than experiment #2's **1.31×** (133.5 → 174.4) — and it still found nothing, because both arms were non-functional.

**The broad arm is still ~1/3 religious.** Measured on the trained 140k: nllb 62.6%, religious 27.1%, quran 6.9%, ccaligned 1.4%, afridoc_health 1.1%, afridoc_tech 1.1% — and `jw.org` is the single largest web domain *inside* the NLLB portion. Across the NLLB rows carrying a URL, there are **3,317 distinct web domains**, top-5 concentration 25.4%.

Headroom, if diversity is worth maximizing (measured, not estimated): random NLLB reaches 213.0 and per-domain capping only 218.6, versus the pool's 221.1 ceiling. **The gain comes almost entirely from removing the religious dilution, not from clever sampling.**

⚠️ **Vendi is an unvalidated proxy.** Nothing here shows that maximizing it maximizes OOD BLEU — we have two data points relating the two, one of which is a null. It also trades directly against alignment quality, since the diversity headroom is all mined NLLB text with documented misalignment and entity-swap errors that survive AfriCOMET/LASER/LID filtering. A third arm at Vendi ≈215 (NLLB-only 140k) would establish whether OOD performance is monotonic in diversity or peaks and falls off.

**Caveat:** two Gezmu rows appear in the broad training set despite Gezmu being excluded by source, and the source-match total (140,152) exceeds the row count (139,989) — the same text reaches multiple source files. The arms have zero *source* overlap, not zero *content* overlap.

---

## 5. Side finding — the Gezmu et al. reproduction gap

`am-en-gezmu-8k` reaches **31.22** BLEU on the paper's own test split vs their reported **33.0** (NMT-8K, Table 3) — 94.6% of the published number, on their data and their split.

Metric convention does **not** explain the 1.78 gap:

| BLEU convention | score |
|---|---|
| sacrebleu 13a (what we report) | **31.22** |
| sacrebleu intl | 31.30 (+0.08) |
| lowercased (any tokenizer) | no change — **the corpus is already fully lowercased** |

Because the corpus is lowercased throughout, our BLEU is already *case-insensitive* — the more generous convention. If the paper reports cased BLEU, our setup is advantaged, which makes the gap harder to explain by measurement, not easier.

### TESTED — the preprocessing recipe closes most of the gap (`am-en-narrow`, 2026-08-11)

The paper's §4.1 describes Moses tokenization on both sides, then Amharic transliterated to Latin via AT4MT "to share named-entities between the languages," feeding one shared subword vocabulary. Reproducing that bundle — **five coupled changes**: Moses both sides, AT4MT transliteration, one shared 8k vocab replacing two separate 8k, tied source/target embeddings, and English's algorithm forced from ByteLevel BPE to Unigram — with **every** hyperparameter read from `am-en-gezmu-8k`'s manifest so nothing could drift:

| metric | `gezmu-8k` | `narrow` | delta |
|---|---|---|---|
| in-dist test, un-averaged (like-for-like) | 30.75 | **32.07** | **+1.32** |
| in-dist test, baseline checkpoint-averaged | 31.22 | 32.07 | +0.85 |
| FLORES devtest | 4.36 | 4.75 | +0.39 |
| MAFAND test | 2.44 | 2.84 | +0.40 |
| paper's reported | — | — | 33.0 (**−0.93 remaining**) |

It also converged **~8× faster**: matched the baseline's *final* 250,000-step validation score (28.70) by step 30,000.

**Mechanism, measured before spending GPU time.** Transliteration raises cross-lingual shared subword types **180 → 1,003** (2.3% → 12.5% of types in use). Without it the only shared pieces are digit strings (`2012`, `144,000`); with it they are named entities and loanwords (`ethiopia`, `protestant`, `benjamin`, `hospital`) — the paper's stated rationale, confirmed.

**A partial ablation came free.** The first attempt of this run was identical except it lacked Moses, so 21.4% of punctuation stayed glued to words (376 duplicate vocab entries like `congregation.` vs `congregation`). It scored **4.40 vs 15.81** val BLEU at step 5,000 — against the baseline's 9.89. So Moses tokenization does a large share of the work, and a shared vocabulary alone is **not sufficient**. Note the baseline never needed Moses: ByteLevel BPE splits punctuation natively (0.00% glued); SentencePiece Unigram does not (21.4%). Unifying the vocabulary silently removed a property English had been getting for free.

**Caveat on attribution:** the five changes move together and cannot be separated by this run alone — a shared vocabulary is meaningless without script overlap, and tying embeddings is meaningless without a shared vocabulary. This measures "the paper's tokenization approach" as a unit.

Still unexplained, ~0.93 BLEU:

1. **Subword algorithm** (now the top candidate). We use SentencePiece Unigram; t2t's `SubwordTextEncoder` is a greedy-merge/wordpiece method, closer to BPE. Measured on this exact transliterated corpus, BPE gives **+59% cross-lingual sharing** (946 → 1,501 types) and lower fertility (am 1.79 → 1.57, en 1.46 → 1.19). Avoid *ByteLevel* BPE specifically: the transliteration emits non-ASCII (`ə ɨ ṗ š ṣ ṭ ž ʷ`) that ByteLevel shreds into byte pairs.
2. **Checkpoint averaging.** This run has none — the feature had been removed by a refactor and was only restored afterwards. Worth ~+0.5 on the baseline, which would put this at ~32.6. Still short of 33.0.
3. **Effective batch size.** 46 sentences ≈ 1,017 target tokens. Identical to the paper on the reading that t2t's `batch_size` counts *tokens*; 22× smaller if it counts *sentences*. Demoted as a candidate: 1024 sentences implies 1,808 epochs over a 140k corpus (Vaswani did 22.7 on WMT14), and the baseline gained only +0.19 over its last 65,000 steps — an asymptote, not a starved model.
4. **`warmup_steps: 4000` is inferred** from Vaswani et al.; the paper never states it.
5. **No length bucketing** (we pad to batch max, t2t packs by token count); **bf16** vs 2018-era fp32.

**A projection recorded here as a caution:** mid-run, validation reached 33.97 and the baseline's val→test offset (+2.52) suggested a test score near ~35.6. Actual was **32.07** — validation *overstated* test by 1.9, where the baseline's *understated* it by 2.5. Val→test offsets do not transfer across differently-preprocessed models.

⚠️ **A preprocessing mismatch was found while investigating this.** `am-en-gezmu-8k` was trained on raw Gezmu Amharic (its repro script bypasses the pipeline), but `load_benchmark` applies `normalize()` to the Amharic side at eval. Scoring it through the normalizing path gives 29.47 instead of 31.22 — a **1.75 BLEU** train/inference mismatch. This affects `gezmu-8k` only; `broad-8k` was built through the production pipeline. Both numbers above are reported through the matching path, and §1's robustness check confirms the domain conclusion is unaffected.
