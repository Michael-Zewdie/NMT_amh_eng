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

PROCESSED = DATA / "processed"  # per-source cleaned CSVs, NLLB nllb.csv, dist JSONs
FINAL     = DATA / "final"      # train/validation/test splits
FIGS      = PROCESSED / "figs"  # generated charts
SCORES    = DATA / "scores"     # content-keyed model-score caches (see utils.score_cache)
EMBEDDINGS = DATA / "embeddings" # content-keyed embedding-vector caches (see utils.embed_cache)
MODELS     = DATA / "models"    # persisted clustering models (see dist.clusters)
