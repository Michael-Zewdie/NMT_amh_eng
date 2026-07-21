# SyntheticDataGen — Amharic–English NMT corpus pipeline

Builds a cleaned, split Amharic↔English parallel corpus from several sources
(AfriDocMT, Gezmu, Tanzil/Quran, and the full NLLB mined bitext).

## Layout

```
nmt/            shared kernel, importable everywhere (no side effects)
  paths.py      single source of truth for every data/ location
  lengths.py    short/medium/long bucket cutoffs (length report + stratified split)
  websites.py   domains(): map a mined pair to its registered source domain
collect.py      download / assemble every source (incl. NLLB) → data/raw/csv_raw/
collection/     data-acquisition helpers
  search_nllb.py    substring search over the raw NLLB parquet
  search_domain.py  sample pairs from one source domain
process.py      clean every CSV (NLLB first) → data/processed/, then pool → data/final/
processing/     stages process.py calls
  clean/        the text-cleaning kernel
    normalize.py    NFC + labialization/homophone/punctuation merges (amh)
    filters.py      length / script-purity / dedupe row drops + clean()
  pool.py       apply quality cutoffs, pool every source + 80/10/10 split
  score_labse.py    annotate LaBSE cosine (labse_score)
  score_africomet.py annotate AfriCOMET-QE (africomet_score)
  score_lid.py      annotate fastText LID (source_lid / target_lid)
  score_cache.py    content-keyed cache so scores survive a re-clean
  remove_domain.py  drop rows from named source domains
explore/        EDA / spot-checks (read-only)
  char_freq.py  labialized-syllable counts
  length_dist.py    sentence-length distributions
scratch/        throwaway experiments (not part of the pipeline)
data/
  raw/          csv_raw/ (per-source CSVs), local/ (parallel files, PDFs),
                nllb_full/ (the mined parquet)
  csv/          per-source cleaned CSVs, nllb.csv, lengthdist.json
    figs/       generated charts (length_buckets_pie.png, nllb_domains_bar.png)
  scores/       cached LaBSE/AfriCOMET/LID scores, keyed by sentence content
  final/        train / validation / test splits
```

## How to run

Run scripts **as modules, from the project root** (this repo directory), so the
shared `nmt` package is importable:

```bash
# 1. Collect — every source into csv_raw/ (NLLB parquet downloaded if missing, then laser-filtered)
python -m collection.collect       # → data/raw/csv_raw/{afridoc_*,gezmu,quran,nllb}.csv
# Skips any source whose CSV already exists (their inputs are fixed), so a repeat run
# is instant and touches no network. Tuning LASER_CUTOFF? Rebuild just that source:
python -m collection.collect nllb  # ~3s, reads the local parquet, never re-downloads
python -m collection.collect --force   # rebuild everything regardless

# 2. Process — one command cleans every source (NLLB first) → data/processed/ → data/final/
python process.py                     # config + toggles live at the top of the file
# (score_labse, score_africomet, score_lid, pool, length_dist run as toggled stages
#  inside process.py; remove_domain is a standalone manual tool:
#  python -m processing.remove_domain <domain> …)

# Retuning a cutoff? Just edit COSINE_CUTOFF / *_LID_CUTOFF and re-run `python process.py`.
# The score stages hit the data/scores/ cache, never load a model, and only pool re-runs.
# Delete a data/scores/*.parquet to force a re-score (after a model or normalize.py change).

# 3. Explore (read-only spot-checks)
python -m explore.char_freq                       # Amharic labialized-syllable counts
python -m explore.length_dist                     # short/medium/long buckets + pie (also a process.py stage)
python -m explore.domain_dist [top_n]             # NLLB domain bars, raw vs processed (standalone)
python -m collection.search_nllb <term> [n]       # substring search over raw NLLB
python -m collection.search_domain jw.org 5 raw   # sample pairs from one source domain
```

Length buckets are controlled by one global, `LENGTH_CUTOFFS` in `nmt/lengths.py`
(default `(40, 120)` → short `<40`, medium, long `≥120` Amharic chars). It is the
single source of truth shared by `explore.length_dist` (which reports the
distribution) and `processing.pool` (which **stratifies** the train/val/test split
by it, so every split carries the same length proportions). `domain_dist` is
analysis-only — it is **not** a `process.py` stage.

Paths are never hardcoded per-script — every script imports what it needs from
`nmt.paths`, which anchors to this directory regardless of where you invoke from.

## Dependencies

`polars`, `pandas`, `numpy`, `matplotlib`, `torch`, `sentence-transformers`,
`datasets`, `pyarrow`, `tldextract`, `unbabel-comet` (AfriCOMET-QE; pins
`numpy<2.0`, which downgrades numpy but hasn't broken fasttext in practice).
