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

process/          corpus build: clean → annotate → pool → split
  process.py            THE entry point; CONFIG block + stage toggles at the top
  pool.py                tiered quality cutoffs, cross-source dedupe, 80/10/10 split
  clean/
    normalize.py        NFC + labialization / homophone / punctuation merges (amh)
    filters.py          length, script-purity, paren-balance, dedupe → clean()
  scoring/
    score_labse.py      annotate labse_score (LaBSE cosine)
    score_africomet.py  annotate africomet_score (AfriCOMET-QE)
    score_lid.py        annotate source_lid / target_lid (fastText lid218e)
    score_embed.py       cache English-side embeddings (optional; feeds split_semantic)
  cache/
    score_cache.py      content-keyed score cache + annotate_processed() driver
    embed_cache.py       content-keyed cache for raw embedding vectors (score_cache.py's scalar-only analogue)
  dist/                  length/domain/semantic reporting — standalone analyses, not pipeline stages
    length/
      lengths.py          LENGTH_CUTOFFS + bucketize() — shared by pool and the report
      length_dist.py      sentence-length distribution (also a process.py stage)
    semantic/
      en_encoder.py        English-only sentence encoder (all-mpnet-base-v2), for semantic stratification
      semantics.py         N_CLUSTERS + cluster_ids() (k-means) — shared by pool and the report
      semantic_dist.py     semantic-cluster distribution (optional process.py stage)
    nllb_domain_dist.py   NLLB source-domain bars, raw vs processed
    domain_shift.py        is a benchmark drawn from a different distribution than a training corpus?
    diversity.py            spread WITHIN one corpus (companion to domain_shift, which measures BETWEEN two)
  random/               standalone manual tools (not pipeline stages)
    char_freq.py        labialized-syllable counts over raw data
    remove_domain.py    drop rows from named source domains
  utils/
    paths.py            single source of truth for every data/ location
    websites.py          domains(): map a mined pair to its registered source domain
    decontaminate.py    drop any pooled row overlapping a held-out benchmark

model/            the from-scratch Transformer
  common.py             token ids, tokenizers, device/seed, checkpoints, inference setup, corpus_scores
  architecture/
    transformer.py       Seq2SeqTransformer + mask construction
    layers/               attention, encoder, decoder, feedforward, embeddings (Pre-LN)
  configs/
    config.py            YAML → dotted-access config (code, not data — lives next to the *.yaml it loads)
    archive/              every *.yaml this project has trained from — see "How to run" below
  data/
    dataset.py           Dataset / Batch / dynamic-padding collate
  training/
    optim.py              AdamW + inverse-sqrt warmup schedule
    train.py               training loop
  search/
    greedy.py, beam.py, decode.py    greedy + beam search (Wu et al. length penalty)
  tokenize/
    preprocess.py          Moses tokenize/detokenize + AT4MT Amharic transliteration
    at4mt_transliteration.py   unmodified 3rd-party module preprocess.py wraps
  evaluate/
    collect_benchmark.py  FLORES-200 + MAFAND-MT → data/benchmarks/ (never trained on; feeds evaluate_OOD)
    evaluate_in_dist.py  in-distribution BLEU/chrF++ on a data/prepared*/*.pkl split
    evaluate_OOD.py       out-of-distribution BLEU/chrF++ on a FLORES/MAFAND benchmark CSV, --show-worst for per-sentence errors
  translate.py          interactive stdin REPL

experiments/       each package owns one experiment end to end — corpus, tokenizer,
                   training recipe, and eval — as `python -m experiments.<name> <subcommand>`
  domain_breadth/       does training-corpus domain breadth trade in-distribution BLEU
                        for OOD generalization? (arms.py, corpus.py, evaluate.py, train.py)
  clean_recipe/          the current best model's recipe (arms.py, corpus.py, evaluate.py)

baselines/
  google_translate.py   Cloud Translation v2 on the same benchmarks, same scorer

data/
  raw/            csv_raw/ (per-source CSVs), local/ (corpora, PDFs),
                  nllb_full/ · flores_full/ (downloaded sources)
  processed/      per-source cleaned + annotated CSVs; figs/ for generated charts
  scores/         cached LaBSE / AfriCOMET / LID scores, keyed by sentence content
  final*/         train / validation / test splits — TRAINING DATA ONLY. One per
                  experiment's own corpus (final_broad_v2/, final_clean/, …); the
                  bare final/ is process.py's generic pooled output
  benchmarks/     FLORES-200, MAFAND-MT — held out, never trained on
  tokenizer/      trained tokenizer artifacts, one shared vocab per experiment
                  (shared_translit*/); archive/ holds v1's separate am/ + en/
  prepared*/      tokenized id caches, one per experiment, matching tokenizer/
runs/             <run_name>/{checkpoints,tensorboard,config.yaml,manifest.json}/ (gitignored)

archive/          superseded runs, data and code — nothing deleted, see archive/README.md
```

Tokenizer fitting and `.pkl` encoding are no longer standalone scripts (that was
v1's `model/tokenize/train_tokenizer*.py` + `model/data/prepare.py`, both now in
`archive/model/` — see `archive/README.md`). Every experiment since the Gezmu-repro
recipe fits one shared 8k vocab on its own corpus and encodes inline
(`experiments/<name>/arms.py`'s `cmd_build`), because a shared vocab lets the same
tokenizer score either side without a src/tgt split.

## How to run

Everything runs **as a module, from the project root**, so the package imports resolve:

```bash
# 1. Collect — every training source into data/raw/csv_raw/
python -m collect              # skips sources whose CSV already exists
python -m collect nllb         # rebuild just one source (~3s, no network)
python -m collect --force      # rebuild everything

# 2. Process — clean → annotate → pool → split
python -m process.process                 # CONFIG block + stage toggles at the top of the file
```

Retuning a cutoff is cheap: edit `COSINE_CUTOFF` / `AFRICOMET_CUTOFF` / `*_LID_CUTOFF`
in `process/process.py` and re-run. The score stages hit `data/scores/`, never load
a model, and only the pool re-runs. Delete a `data/scores/*.parquet` to force a
re-score — required after changing a model or `process/clean/normalize.py`, since the cache
keys are hashes of the normalized text.

```bash
# 3. Each experiment owns its own corpus + tokenizer + training from here —
#    there is no longer one shared model/configs/*.yaml to hand-edit and run.
#    e.g. the current best model's recipe:
python -m experiments.clean_recipe corpus --write   # build data/final_clean
python -m experiments.clean_recipe build  --arm lower
python -m experiments.clean_recipe train  --arm lower [--resume]
python -m experiments.clean_recipe eval
# or the domain-breadth comparison (two arms, always evaluated together):
python -m experiments.domain_breadth corpus --write
python -m experiments.domain_breadth build --arm narrow|broad_v2
python -m experiments.domain_breadth train  --arm narrow|broad_v2 [--resume]
python -m experiments.domain_breadth eval
# model.training.train itself is generic — point it at any run's own config.yaml
# (e.g. to resume by hand, or to launch a config an experiment package wrote):
python -m model.training.train runs/am-en-clean-lower/config.yaml

# 4. Evaluate
python -m model.evaluate.collect_benchmark    # FLORES-200 + MAFAND-MT → data/benchmarks/ (once; cached after)
# Each run keeps its own config.yaml, so pass the run's copy rather than a shared one.
python -m model.evaluate.evaluate_in_dist runs/am-en-clean-lower/config.yaml runs/am-en-clean-lower/checkpoints/best.pt validation
python -m model.evaluate.evaluate_OOD runs/am-en-clean-lower/config.yaml runs/am-en-clean-lower/checkpoints/best.pt \
        data/benchmarks/flores200_am_en.csv devtest --show-worst 20   # OOD + worst sentences
# Archived runs still score from where they now live, e.g.:
#   python -m model.evaluate.evaluate_OOD archive/runs/am-en-base-v4/config.yaml \
#           archive/runs/am-en-base-v4/checkpoints/best.pt data/benchmarks/flores200_am_en.csv devtest
# Google Translate baseline — same benchmarks, same scorer. Costs API quota, so it
# refuses to run without --confirm; needs a Cloud Translation key in the environment.
python -m baselines.google_translate --dry-run    # character count only, no API call
python -m baselines.google_translate data/benchmarks/flores200_am_en.csv devtest --confirm

# 5. Translate interactively (defaults to am-en-clean-lower; --config/--checkpoint override)
python -m model.translate

# Analysis / manual tools (read-only unless noted)
python -m process.dist.length.length_dist          # short/medium/long buckets + pie
python -m process.dist.nllb_domain_dist [top_n]     # NLLB domain bars, raw vs processed
python -m process.dist.domain_shift --train <final*/train.csv> --benchmark <name:split> ...
python -m process.random.char_freq                  # Amharic labialized-syllable counts
python -m process.random.remove_domain <domain>      # WRITES: drops rows from nllb.csv
```

See each `experiments/<name>/__main__.py` docstring for that experiment's full
subcommand list and what it writes; `EXPERIMENTS.md` is the scored record of
every run either package has produced.

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

Length buckets come from one global, `LENGTH_CUTOFFS` in `process/dist/length/lengths.py`
(default `(40, 120)` → short `<40`, medium, long `≥120` Amharic chars) — shared by
`process.dist.length.length_dist` (which reports the distribution) and
`process.pool` (which **stratifies** the split by it).

**Semantic stratification is optional, off by default.** `process.pool.
split_semantic()` (`STRATIFY_SEMANTIC=True` in `process.py`) nests a k-means
partition of English-side embeddings (`process/dist/semantic/semantics.py`,
`N_CLUSTERS=16`) on top of the length buckets, so train/val/test also carry the
same topic/domain mix, not just the same length mix. Embeddings come from a
dedicated English encoder (`all-mpnet-base-v2`, not LaBSE — LaBSE is tuned for
cross-lingual alignment, not English semantic distinctions), cached separately
from the scalar quality scores (`process.cache.embed_cache`, vector-valued,
unlike `score_cache.py`). Since the split still groups by `am` for leak
prevention, clustering the English side requires mean-pooling every English
reference sharing an `am` into one vector first — see
`process.pool.am_cluster_ids`. Off by default so the length-only split
(used to train every model in `EXPERIMENTS.md`) stays available to A/B against.

Paths are never hardcoded per-script — every script imports what it needs from
`process.utils.paths`, which anchors to the project root regardless of where you
invoke from.

## Dependencies

```bash
python -m venv venv && . venv/bin/activate
pip install -r requirements.txt                        # run the models
pip install -r requirements-pipeline.txt               # + rebuild the data
```

`requirements.txt` is the small set needed to load a checkpoint and translate or
evaluate, and installs on Linux/CUDA and macOS alike. Everything for collection,
filtering and quality scoring lives in `requirements-pipeline.txt`, which is heavier
and less portable — `unbabel-comet` drags numpy back below 2.0, and `fasttext`
builds from source.

Developed on Python 3.12 / Linux x86-64 with torch 2.5.1+cu121. Python 3.13 works
but has no 2.5.1 wheels, so it resolves to torch 2.6+; the checkpoints load either
way. Training needs an NVIDIA GPU (it autocasts to bf16). Translation and evaluation
run anywhere — CUDA, Apple Silicon via MPS, or CPU.

`torch`, `tokenizers`, `sacrebleu`, `polars`, `pandas`, `numpy`, `matplotlib`,
`sentence-transformers`, `scikit-learn` (k-means for semantic stratification),
`datasets`, `pyarrow`, `tldextract`, `fasttext`,
`huggingface_hub`, `unbabel-comet` (AfriCOMET-QE; pins `numpy<2.0`, which downgrades
numpy but hasn't broken fasttext in practice).
