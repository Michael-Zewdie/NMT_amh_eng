# NMT_amh_eng — Amharic→English machine translation, from scratch

Builds a cleaned, decontaminated Amharic–English parallel corpus from several
sources (AfriDocMT, Gezmu, religious Bible corpora, NLLB mined bitext), then
trains and evaluates a from-scratch Transformer on it — no `transformers`, no
fairseq; only `torch`, `tokenizers` and `sacrebleu`.

- **[PIPELINE.md](PIPELINE.md)** — end-to-end dataflow diagrams, cheat sheet, invariants.
- **[EXPERIMENTS.md](EXPERIMENTS.md)** — every run, its data snapshot, and its scores.

## Layout

```
collect/          data acquisition → data/raw/csv_raw/ — one file per source
  __main__.py           registry + CLI (python -m collect); ties the sources below together
  afridoc.py            AfriDocMT health + tech, HuggingFace
  gezmu.py               Gezmu, local parallel files
  nllb.py                 NLLB mined bitext — download parquet + LASER_CUTOFF filter
  (quran.py / ccaligned.py were removed 2026-08-14 — see archive/collect/)

processing/       corpus build: clean → annotate → pool → split
  process.py            THE entry point; CONFIG block + stage toggles at the top
  clean/
    normalize.py        NFC + labialization / homophone / punctuation merges (amh)
    filters.py          length, script-purity, paren-balance, dedupe → clean()
  utils/
    paths.py            single source of truth for every data/ location
    pool.py             tiered quality cutoffs, cross-source dedupe, 80/10/10 split
    decontaminate.py    drop any pooled row overlapping a held-out benchmark
    score_cache.py      content-keyed score cache + annotate_processed() driver
    score_labse.py      annotate labse_score (LaBSE cosine)
    score_africomet.py  annotate africomet_score (AfriCOMET-QE)
    score_lid.py        annotate source_lid / target_lid (fastText lid218e)
    en_encoder.py        English-only sentence encoder (all-mpnet-base-v2), for semantic stratification
    embed_cache.py       content-keyed cache for raw embedding vectors (score_cache.py's scalar-only analogue)
    score_embed.py       cache English-side embeddings (optional; feeds split_semantic)
  dist/                 length/domain/semantic reporting
    lengths.py          LENGTH_CUTOFFS + bucketize() — shared by pool and the report
    length_dist.py      sentence-length distribution (also a process.py stage)
    semantics.py         N_CLUSTERS + cluster_ids() (k-means) — shared by pool and the report
    semantic_dist.py     semantic-cluster distribution (optional process.py stage)
    websites.py          domains(): map a mined pair to its registered source domain
    domain_dist.py       NLLB source-domain bars, raw vs processed (analysis only)
  random/               standalone manual tools (not pipeline stages)
    char_freq.py        labialized-syllable counts over raw data
    remove_domain.py    drop rows from named source domains

model/            the from-scratch Transformer
  common.py             token ids, tokenizers, device/seed, checkpoints, inference setup, corpus_scores
  architecture/
    transformer.py       Seq2SeqTransformer + mask construction
    layers/               attention, encoder, decoder, feedforward, embeddings (Pre-LN)
  configs/
    config.py            YAML → dotted-access config (code, not data — lives next to the *.yaml it loads)
    archive/              every *.yaml, including the currently-used ones (base_v4, base_v6, …)
  data/
    prepare.py          data/final/*.csv → data/prepared/am-en/*.pkl (ids + BOS/EOS)
    dataset.py          Dataset / Batch / dynamic-padding collate
  optim.py              AdamW + inverse-sqrt warmup schedule
  train.py              training loop
  search.py             greedy + beam search (Wu et al. length penalty)
  tokenize/
    train_tokenizer.py    English ByteLevel BPE  → data/tokenizer/en/
    train_tokenizer_am.py Amharic Unigram        → data/tokenizer/am/
  evaluate/
    collect_benchmark.py  FLORES-200 + MAFAND-MT → data/benchmarks/ (never trained on; feeds evaluate_OOD)
    evaluate_in_dist.py  in-distribution BLEU/chrF++ on a data/prepared/*.pkl split
    evaluate_OOD.py       out-of-distribution BLEU/chrF++ on a FLORES/MAFAND benchmark CSV, --show-worst for per-sentence errors
  translate.py          interactive stdin REPL

baselines/
  google_translate.py   Cloud Translation v2 on the same benchmarks, same scorer

data/
  raw/            csv_raw/ (per-source CSVs), local/ (corpora, PDFs),
                  nllb_full/ · flores_full/ (downloaded sources)
  processed/      per-source cleaned + annotated CSVs; figs/ for generated charts
  scores/         cached LaBSE / AfriCOMET / LID scores, keyed by sentence content
  final/          train / validation / test splits — TRAINING DATA ONLY
  benchmarks/     FLORES-200, MAFAND-MT — held out, never trained on
  tokenizer/      trained tokenizer artifacts (am/, en/)
  prepared/       tokenized id caches per language pair
runs/             <run_name>/{checkpoints,tensorboard,benchmarks}/ (gitignored)
```

## How to run

Everything runs **as a module, from the project root**, so the package imports resolve:

```bash
# 1. Collect — every training source into data/raw/csv_raw/
python -m collect              # skips sources whose CSV already exists
python -m collect nllb         # rebuild just one source (~3s, no network)
python -m collect --force      # rebuild everything

# 2. Process — clean → annotate → pool → split
python -m processing.process              # CONFIG block + stage toggles at the top of the file
```

Retuning a cutoff is cheap: edit `COSINE_CUTOFF` / `AFRICOMET_CUTOFF` / `*_LID_CUTOFF`
in `processing/process.py` and re-run. The score stages hit `data/scores/`, never load
a model, and only the pool re-runs. Delete a `data/scores/*.parquet` to force a
re-score — required after changing a model or `processing/clean/normalize.py`, since the cache
keys are hashes of the normalized text.

```bash
# 3. Tokenizers + prepared caches (one-off artifacts, not process.py stages)
python -m model.tokenize.train_tokenizer      # English → data/tokenizer/en/
python -m model.tokenize.train_tokenizer_am   # Amharic → data/tokenizer/am/
python -m model.data.prepare              # data/final/*.csv → data/prepared/am-en/*.pkl

# 4. Train
python -m model.train model/configs/archive/base_v6.yaml

# 5. Evaluate
python -m model.evaluate.collect_benchmark    # FLORES-200 + MAFAND-MT → data/benchmarks/ (once; cached after)
python -m model.evaluate.evaluate_in_dist model/configs/archive/base_v4.yaml runs/am-en-base-v4/checkpoints/best.pt validation
python -m model.evaluate.evaluate_OOD model/configs/archive/base_v4.yaml runs/am-en-base-v4/checkpoints/best.pt \
        data/benchmarks/flores200_am_en.csv devtest --show-worst 20   # OOD + worst sentences
# Google Translate baseline — same benchmarks, same scorer. Costs API quota, so it
# refuses to run without --confirm; needs a Cloud Translation key in the environment.
python -m baselines.google_translate --dry-run    # character count only, no API call
python -m baselines.google_translate data/benchmarks/flores200_am_en.csv devtest --confirm

# 6. Translate interactively
python -m model.translate model/configs/archive/base_v4.yaml runs/am-en-base-v4/checkpoints/best.pt

# Analysis / manual tools (read-only unless noted)
python -m processing.dist.length_dist              # short/medium/long buckets + pie
python -m processing.dist.domain_dist [top_n]      # NLLB domain bars, raw vs processed
python -m processing.random.char_freq              # Amharic labialized-syllable counts
python -m processing.random.remove_domain <domain> # WRITES: drops rows from nllb.csv
```

## Two things that shape everything

**Scoring and filtering are separate.** `score_*` only *annotates* `data/processed/`;
the cutoffs are applied at the pool stage. So retuning a threshold is a re-pool
(seconds), not a re-embed (hours), and *lowering* a cutoff brings rows back rather
than losing them permanently.

**Cutoffs are tiered, not uniform.** `gezmu`, `afridoc_health`, `afridoc_tech` and
`religious` are professionally human-translated; applying a QE cutoff tuned for noisy
mined bitext to them gutted Gezmu to 13,638 of 124,409 pairs. The mined source
(`nllb`) keeps the strict floors. **LID is tiered too as of 2026-08-14** — on curated
text a 0.90 floor is a length artifact, not an am/en sanity check, and it was costing
7,751 correctly-aligned pairs. See EXPERIMENTS.md → "Root-cause check against Gezmu
et al." and "v4 data audit".

Length buckets come from one global, `LENGTH_CUTOFFS` in `processing/dist/lengths.py`
(default `(40, 120)` → short `<40`, medium, long `≥120` Amharic chars) — shared by
`processing.dist.length_dist` (which reports the distribution) and
`processing.utils.pool` (which **stratifies** the split by it).

**Semantic stratification is optional, off by default.** `processing.utils.pool.
split_semantic()` (`STRATIFY_SEMANTIC=True` in `process.py`) nests a k-means
partition of English-side embeddings (`processing/dist/semantics.py`,
`N_CLUSTERS=16`) on top of the length buckets, so train/val/test also carry the
same topic/domain mix, not just the same length mix. Embeddings come from a
dedicated English encoder (`all-mpnet-base-v2`, not LaBSE — LaBSE is tuned for
cross-lingual alignment, not English semantic distinctions), cached separately
from the scalar quality scores (`processing.utils.embed_cache`, vector-valued,
unlike `score_cache.py`). Since the split still groups by `am` for leak
prevention, clustering the English side requires mean-pooling every English
reference sharing an `am` into one vector first — see
`processing.utils.pool.am_cluster_ids`. Off by default so the length-only split
(used to train every model in `EXPERIMENTS.md`) stays available to A/B against.

Paths are never hardcoded per-script — every script imports what it needs from
`processing.utils.paths`, which anchors to the project root regardless of where you
invoke from.

## Dependencies

```bash
python -m venv venv && . venv/bin/activate
pip install -r requirements.txt
```

Pinned to the versions everything here was trained and evaluated with, on Python
3.12.3 / Linux x86-64. **The `torch` pin is the CUDA 12.1 build** — see the note at
the top of `requirements.txt` for the CPU and Apple Silicon variants. Training needs
an NVIDIA GPU; translation and evaluation run on CPU, just slowly.

`torch`, `tokenizers`, `sacrebleu`, `polars`, `pandas`, `numpy`, `matplotlib`,
`sentence-transformers`, `scikit-learn` (k-means for semantic stratification),
`datasets`, `pyarrow`, `tldextract`, `fasttext`,
`huggingface_hub`, `unbabel-comet` (AfriCOMET-QE; pins `numpy<2.0`, which downgrades
numpy but hasn't broken fasttext in practice).
