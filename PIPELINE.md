# Pipeline dataflow

End-to-end dataflow, raw sources → trained checkpoint → BLEU. Two diagrams: the
**corpus build** (collect → process → pool → `data/final/`) and the **model track**
(tokenizers → prepare → train → evaluate).

Experiment results and per-run history live in [EXPERIMENTS.md](EXPERIMENTS.md);
this file is only about *what flows where*.

---

## A. Corpus build

Everything here is driven by two entry points: `python -m collection.collect`
(acquire) and `python -m processing.process` (clean → score → pool → split).

```mermaid
flowchart TD
    %% ---------------- 1. collect ----------------
    subgraph COL["1 · collect &mdash; python -m collection.collect"]
        direction TB
        HF["AfriDocMT health + tech<br/>HuggingFace"]
        LOC["Gezmu · Quran / Tanzil<br/>data/raw/local/"]
        NLP[("NLLB mined bitext<br/>16.1M pairs · parquet")]
        CCP[("CCAligned<br/>~346k pairs · OPUS")]
    end

    CSVR[("data/raw/csv_raw/*.csv")]
    HF --> CSVR
    LOC --> CSVR
    NLP -- "laser_score &ge; 1.06<br/>source/target LID &ge; 0.95" --> CSVR
    CCP --> CSVR

    %% ---------------- 2. clean ----------------
    CSVR --> CLEAN["2 · process.py · RUN_CLEAN<br/><b>clean()</b> — normalize → length → script purity → dedupe<br/>mined: dedupe on <i>am</i> · curated: dedupe on <i>am+en</i>"]
    CLEAN --> PROCD[("data/processed/*.csv")]

    %% ---------------- 3. annotate (never drops rows) ----------------
    PROCD -. "RUN_LABSE" .-> LAB["score_labse → labse_score"]
    PROCD -. "RUN_AFRICOMET" .-> AFR["score_africomet → africomet_score"]
    PROCD -. "RUN_LID" .-> LIDS["score_lid → source_lid / target_lid"]
    PROCD -. "RUN_EMBED (optional)" .-> EMB["score_embed → English-side embeddings<br/>(all-mpnet-base-v2, not LaBSE)"]
    LAB -. "rewrite in place" .-> PROCD
    AFR -. "rewrite in place" .-> PROCD
    LIDS -. "rewrite in place" .-> PROCD
    LAB <-. "hit / miss" .-> CACHE[("data/scores/*.parquet<br/>content-keyed score cache")]
    AFR <-. "hit / miss" .-> CACHE
    LIDS <-. "hit / miss" .-> CACHE
    EMB <-. "hit / miss" .-> ECACHE[("data/scores/mpnet_en_embed.npy + _keys.parquet<br/>content-keyed vector cache")]

    %% ---------------- 4. pool ----------------
    PROCD --> POOL
    subgraph POOL["4 · pool &mdash; RUN_POOL · tiered cutoffs"]
        direction TB
        TIER["<b>curated</b> gezmu · afridoc_health · afridoc_tech · quran<br/>cosine 0.0 · africomet 0.0 → <i>disabled</i><br/><br/><b>mined</b> nllb · ccaligned<br/>cosine &ge; 0.7 · africomet &ge; 0.80<br/><br/><b>uniform, every source</b> LID &ge; 0.90 both sides"]
        TIER --> MRG["concat → cross-source dedupe on <i>am</i> → seeded shuffle"]
        MRG --> DEC["decontaminate — drop any row overlapping a benchmark"]
        DEC --> SPL["80 / 10 / 10 split, stratified by Amharic length bucket<br/><i>STRATIFY_SEMANTIC=True (optional):</i> also stratified by<br/>k-means cluster (k=16) over mean-pooled English embeddings"]
    end
    ECACHE -. "lookup (cache-only, no model)" .-> SPL

    SPL --> FIN[("data/final/train.csv<br/>data/final/validation.csv<br/>data/final/test.csv")]

    %% ---------------- benchmarks: parallel, never trained on ----------------
    subgraph BEN["1b · collect_benchmark &mdash; held out, never trained on"]
        direction TB
        FLO[("flores200_am_en.csv<br/>2,009 pairs · Wikimedia")]
        MAF[("mafand_en_amh.csv<br/>1,936 pairs · news")]
    end
    FLO --> DEC
    MAF --> DEC

    classDef store fill:#e8f1f5,stroke:#2b6a80,color:#0c3345;
    class CSVR,PROCD,CACHE,FIN,NLP,CCP,FLO,MAF store;
```

**The one thing to remember about this half:** scoring and filtering are
deliberately separate. `score_*` only *annotates* `data/processed/` and caches
every score in `data/scores/` keyed by a hash of the sentence pair. The cutoffs
are applied at the **pool** stage. So retuning a threshold is a re-pool (seconds),
not a re-embed (hours), and *lowering* a cutoff brings rows back rather than
losing them permanently.

**Live cutoff values** are the ones in `processing/process.py`'s CONFIG block —
`process.py` passes them into `pool.main()`, so the module-level constants in
`processing/utils/pool.py` are only defaults for a standalone call.

**Why the tiering exists:** the curated sources are professionally human-translated.
Applying a QE/alignment cutoff tuned for noisy mined bitext to them gutted Gezmu to
13,638 of 124,409 pairs and Quran to 446 of 37,380. See EXPERIMENTS.md → "Root-cause
check against Gezmu et al."

---

## B. Model track

`data/final/` is the handoff point. Everything below is one-off artifact building
plus per-run training; none of it is a `process.py` stage.

```mermaid
flowchart TD
    FIN[("data/final/train.csv<br/>validation.csv · test.csv")]

    %% ---------------- 5. tokenizers ----------------
    FIN --> TKEN["5 · processing.train_tokenizer<br/>ByteLevel BPE · vocab 8k"]
    FIN --> TKAM["5 · processing.train_tokenizer_am<br/>Unigram / SentencePiece · vocab 8k"]
    TKEN --> TOKD[("data/tokenizer/en/")]
    TKAM --> TOKA[("data/tokenizer/am/")]

    %% ---------------- 6. prepare ----------------
    FIN --> PREP["6 · model.data.prepare<br/>encode + BOS/EOS · drop pairs over MAX_LEN=150"]
    TOKD --> PREP
    TOKA --> PREP
    PREP --> PRD[("data/prepared/am-en/{train,validation,test}.pkl")]

    %% ---------------- 7. train ----------------
    PRD --> TRAIN["7 · model.train &lt;config&gt;"]
    CFG[/"model/configs/*.yaml<br/>base_v4 · base_v6<br/>(archive/ = superseded)"/] --> TRAIN
    TRAIN --> CKPT[("runs/&lt;run_name&gt;/checkpoints/<br/>best.pt · last.pt")]
    TRAIN -. "periodic eval on a fixed<br/>validation subset → best.pt" .-> TRAIN
    TRAIN --> TB[("runs/&lt;run_name&gt;/tensorboard/")]

    %% ---------------- 8. evaluate ----------------
    CKPT --> EVI["8a · model.evaluate<br/><b>in-distribution</b> · decode + sacrebleu"]
    PRD --> EVI
    EVI --> SC1["validation BLEU"]

    CKPT --> EVB["8b · model.evaluate_benchmark<br/><b>out-of-distribution</b> · reads the benchmark CSV directly"]
    BEN2[("data/benchmarks/*.csv")] --> EVB
    EVB --> SC2["FLORES / MAFAND BLEU + chrF++"]

    BEN2 --> GT["8c · baselines.google_translate<br/>Cloud Translation v2 · raw Amharic, no normalization"]
    GT --> SC3["baseline BLEU, same scorer"]

    SC1 --> LOG["EXPERIMENTS.md"]
    SC2 --> LOG
    SC3 --> LOG

    classDef store fill:#e8f1f5,stroke:#2b6a80,color:#0c3345;
    class FIN,TOKD,TOKA,PRD,CKPT,TB,BEN2 store;
```

**Why `evaluate_benchmark` is a separate entry point** rather than another split
of `model.evaluate` — all three reasons matter for the numbers being publishable:

1. `model.evaluate` reads `data/prepared/*.pkl`, which would mean routing a
   benchmark through `data/final/` — the one directory that must only ever hold
   training data.
2. Both `prepare.py` (`MAX_LEN`) and `TranslationDataset` (`max_src_len`) *silently
   drop* over-length pairs. Dropping rows from a fixed benchmark changes what
   "FLORES devtest" means. `evaluate_benchmark` truncates instead — count in equals
   count out.
3. `model.evaluate` rebuilds references by decoding target ids back to text, a lossy
   tokenizer round-trip. A benchmark reference is scored exactly as it ships.

The Amharic **source** *is* normalized in `evaluate_benchmark` (it's the distribution
the model trained on); the English **reference** is never touched. `google_translate`
deliberately gets *raw* Amharic — normalizing for a black-box system would handicap it.

---

## Cheat sheet

| I want to… | Run |
|---|---|
| Re-tune a quality cutoff | Edit `processing/process.py` CONFIG, `python -m processing.process` (scores come from cache; only pool re-runs) |
| A/B semantic stratification | Flip `STRATIFY_SEMANTIC` in `processing/process.py` CONFIG (needs `RUN_EMBED=True` at least once first) |
| Re-tune the NLLB LASER cutoff | `python -m collection.collect nllb` (~3s, no network), then `python -m processing.process` |
| Force a re-score | Delete the relevant `data/scores/*.parquet` — required after changing a model or `clean/normalize.py`, since keys are hashes of normalized text |
| Add a new source | Drop a CSV in `data/raw/csv_raw/`; `process.py` globs the directory |
| Change vocab size | Edit `VOCAB_SIZE` in both tokenizer scripts, retrain both, **re-run `model.data.prepare`**, and train fresh (checkpoints become shape-incompatible) |
| Train a new run | `python -m model.train model/configs/<cfg>.yaml` |
| Score in-distribution | `python -m model.evaluate <cfg> <ckpt> validation` |
| Score OOD | `python -m model.evaluate_benchmark <cfg> <ckpt> data/benchmarks/flores200_am_en.csv devtest` |

## Invariants worth not breaking

- `data/final/` holds **training data only**. Benchmarks live in `data/benchmarks/`
  and reach the corpus build only via `decontaminate`, which *removes* rows.
- `model.max_len` (config) **≥** `MAX_LEN` (`prepare.py`) — the positional-encoding
  table has to index the longest sequence that survives.
- Tokenizer vocab size is baked into every checkpoint. Changing it invalidates all
  of them.
- Cross-source dedupe on `am` happens *before* the split, so no Amharic sentence
  leaks across train/val/test.
- Length buckets come from `processing/dist/lengths.py` — one source of truth shared
  by the length report and the stratified split.
- Semantic stratification (`STRATIFY_SEMANTIC=True`) clusters the *English* side but
  still groups by `am` for leak prevention — `processing.utils.pool.am_cluster_ids`
  mean-pools every English reference sharing an `am` into one vector first, so a
  multi-reference `am` (e.g. Quran verses) can't land in different clusters and
  defeat the grouping.
