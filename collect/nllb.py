"""collect.nllb — download the NLLB mined parquet if missing, then
laser-filter it down to a tractable am/en CSV (csv_raw/nllb.csv).

Re-running this is only useful when LASER_CUTOFF (or the LID floors) change —
the parquet itself is a fixed Meta release, fetched once and re-read locally
after that. `python -m collect nllb` always rebuilds the CSV (naming a
source is a statement of intent); a default `collect` run skips it if the
CSV already exists.
"""
import csv as csvlib

import pandas as pd
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq

from process.utils.paths import CSV_RAW, NLLB_FULL

# NLLB laser_score threshold: 1.1 keeps the top ~1.5% (~425k of 16.1M pairs).
LASER_CUTOFF = 1.06
# fastText LID (lid218e) confidence floors, mirrored on the non-NLLB side by
# process.scoring.score_lid. source = Amharic (amh_Ethi), target = English
# (eng_Latn). NLLB pre-filtered target_lid >= 0.95; source_lid has a noisy tail
# down to ~0.50, so the source floor does the real work here.
SOURCE_LID_CUTOFF = 0.95
TARGET_LID_CUTOFF = 0.95

# allenai/nllb is script-backed (nllb.py) and unusable on datasets>=4.0 (script
# loading + trust_remote_code were removed). That script only downloads this same
# GCS bitext anyway, so we fetch it directly — no HF wrapper, no version pin.
NLLB_URL = "https://storage.googleapis.com/allennlp-data-bucket/nllb/amh_Ethi-eng_Latn.gz"
NLLB_COLUMNS = [
    "amh", "eng", "laser_score", "source_lid", "target_lid",
    "source_source", "source_url", "target_source", "target_url",
]


def _download_nllb() -> None:
    """Stream the ~8GB NLLB mined TSV from GCS → data/raw/nllb_full/*.parquet.
    No LASER/mining runs locally — Meta pre-scored it; we just fetch + repack."""
    NLLB_FULL.parent.mkdir(parents=True, exist_ok=True)
    chunks = pd.read_csv(
        NLLB_URL, sep="\t", header=None, names=NLLB_COLUMNS, dtype=str,
        quoting=csvlib.QUOTE_NONE, na_filter=False, compression="gzip", chunksize=500_000,
    )
    writer, total = None, 0
    for chunk in chunks:
        for col in ("laser_score", "source_lid", "target_lid"):
            chunk[col] = pd.to_numeric(chunk[col], errors="coerce").astype("float32")
        chunk = chunk.dropna(subset=["laser_score", "source_lid", "target_lid"])
        table = pa.Table.from_pandas(chunk, preserve_index=False)
        writer = writer or pq.ParquetWriter(NLLB_FULL, table.schema, compression="zstd")
        writer.write_table(table)
        total += len(chunk)
        print(f"\r  downloading nllb: {total:,} rows", end="", flush=True)
    writer.close()
    print(f"\nnllb_full: {total:,} pairs → {NLLB_FULL}")


def collect_nllb() -> None:
    """Download the NLLB parquet if missing, then laser-filter → csv_raw/nllb.csv
    (am/en, keeping laser_score + url columns for downstream domain analysis).

    The parquet is only fetched when absent — a cutoff change re-reads the local
    2.3GB copy, it never re-downloads. scan_parquet keeps that read lazy so polars
    pushes the cutoffs into the scan and skips row groups that can't match, instead
    of materializing all 16.1M rows to throw most away.
    """
    if not NLLB_FULL.exists():
        _download_nllb()
    total = pl.scan_parquet(NLLB_FULL).select(pl.len()).collect().item()
    kept = (
        pl.scan_parquet(NLLB_FULL)
        .filter(
              (pl.col("laser_score") > LASER_CUTOFF)
            & (pl.col("source_lid") > SOURCE_LID_CUTOFF)
            & (pl.col("target_lid") > TARGET_LID_CUTOFF)
        )
        .rename({"amh": "am", "eng": "en"})
        .collect()
    )
    out = CSV_RAW / "nllb.csv"
    kept.write_csv(out)
    print(
        f"nllb: {total:,} → {kept.height:,} "
        f"(laser>{LASER_CUTOFF}, source_lid>{SOURCE_LID_CUTOFF}, target_lid>{TARGET_LID_CUTOFF}) → {out}"
    )
