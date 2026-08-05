"""
processing.utils.paths — single source of truth for every data location in the pipeline.

Anchored to the project root via ``parents[2]`` (this file is
``<root>/processing/utils/paths.py``), so scripts resolve data correctly no matter
which stage folder they live in or what the current working directory is. Import the
constants you need instead of rebuilding ``Path(__file__).parent / "data" / ...`` in
each script.
"""
from pathlib import Path

ROOT      = Path(__file__).resolve().parents[2]   # project root: <root>/processing/utils/paths.py
DATA      = ROOT / "data"

RAW       = DATA / "raw"
CSV_RAW   = RAW / "csv_raw"     # collected/merged per-source CSVs
LOCAL     = RAW / "local"       # local source material (parallel corpora, PDFs)
NLLB_FULL = RAW / "nllb_full" / "amh_Ethi-eng_Latn.parquet"
CCALIGNED_FULL = RAW / "ccaligned_full"
FLORES_FULL = RAW / "flores_full"   # extracted flores200_dataset/ (benchmark source, never trained on)

PROCESSED = DATA / "processed"  # per-source cleaned CSVs, NLLB nllb.csv, dist JSONs
FINAL     = DATA / "final"      # train/validation/test splits
BENCHMARKS = DATA / "benchmarks"  # held-out human-translated eval sets, kept out of the training pool
FIGS      = PROCESSED / "figs"  # generated charts
SCORES    = DATA / "scores"     # content-keyed model-score caches (see utils.score_cache)

TOKENIZER    = DATA / "tokenizer"   # trained tokenizer artifacts
TOKENIZER_EN = TOKENIZER / "en"     # English BPE tokenizer output
TOKENIZER_AM = TOKENIZER / "am"     # Amharic Unigram tokenizer output

PREPARED = DATA / "prepared"        # tokenized id caches per split, keyed by lang pair (model/data/prepare.py)
RUNS     = ROOT / "runs"            # training runs: <run_name>/{checkpoints,tensorboard}/ (gitignored)
