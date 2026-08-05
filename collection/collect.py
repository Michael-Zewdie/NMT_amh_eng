"""
collect.py — Gather every source into data/raw/csv_raw/ as CSVs.

  - AfriDocMT (health, tech) from HuggingFace
  - Gezmu + Quran/Tanzil from local parallel files in data/raw/local/
  - NLLB: download the mined parquet if missing, then laser-filter it down to a
    tractable am/en CSV (csv_raw/nllb.csv)
  - CCAligned: download the OPUS moses-format am-en release (~346k pairs, web-mined)

Everything downstream (process.py) just cleans the CSVs this produces, so the
NLLB parquet is handled here, not there. Run (from the project root):

    python -m collection.collect          # every source; skips any whose CSV already exists
    python -m collection.collect nllb     # just NLLB — the LASER_CUTOFF loop (~3s, no network)
    python -m collection.collect --force  # rebuild everything from scratch

Re-running a source is only useful when its *input* changed. Gezmu and Quran read
fixed local files, and AfriDoc is pinned to a HuggingFace release, so their CSVs
can't change between runs — a default run skips them once they exist. That also
keeps `load_dataset` from round-tripping to HuggingFace, which it does on every
call even with a warm cache (~5s, and the one thing here that stalls on a bad
connection). CCAligned is a fixed OPUS release too, same story.

NLLB is different: its CSV depends on the cutoffs in this file, so tuning one means
rebuilding it. Name it explicitly (`python -m collection.collect nllb`) — an explicitly
named source always rebuilds, skip-if-exists only applies to a default run.

Outputs: data/raw/csv_raw/{afridoc_health,afridoc_tech,gezmu,quran,nllb,ccaligned}.csv
"""
import argparse
import csv as csvlib
import sys
import zipfile

import pandas as pd
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq
import requests
from datasets import load_dataset

from processing.utils.paths import CCALIGNED_FULL, CSV_RAW, LOCAL, NLLB_FULL

# NLLB laser_score threshold: 1.1 keeps the top ~1.5% (~425k of 16.1M pairs).
LASER_CUTOFF = 1.06
# fastText LID (lid218e) confidence floors, mirrored on the non-NLLB side by
# processing.utils.score_lid. source = Amharic (amh_Ethi), target = English
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

# OPUS CCAligned am-en release (Moses format): web-mined via per-document LASER
# alignment (El-Kishky et al. 2020), no per-pair score shipped — unlike nllb, so
# it gets no laser_score exemption from the labse_score cutoff downstream.
CCALIGNED_URL = "https://object.pouta.csc.fi/OPUS-CCAligned/v1/moses/am-en.txt.zip"


def collect_afridoc() -> None:
    """AfriDocMT health + tech from HuggingFace → csv_raw/afridoc_<domain>.csv."""
    for domain in ("health", "tech"):
        ds = load_dataset("masakhane/AfriDocMT", domain)
        df = pd.concat(  # merge train/valid/test
            [pd.DataFrame({"am": s["am"], "en": s["en"]}) for s in ds.values()],
            ignore_index=True,
        )
        out = CSV_RAW / f"afridoc_{domain}.csv"
        df.to_csv(out, index=False)
        print(f"afridoc_{domain}: {len(df)} raw pairs → {out}")


def collect_gezmu() -> None:
    """Gezmu am-en parallel corpus (local) → csv_raw/gezmu.csv."""
    gezmu_dir = LOCAL / "Gezmu"
    gezmu_src = gezmu_dir / "gezmu.csv"
    parallel = list(gezmu_dir.glob("*.am-en.base.am")) if gezmu_dir.exists() else []
    if parallel:
        am_lines, en_lines = [], []
        for am_file in sorted(parallel):
            en_file = am_file.with_suffix(".en")  # dev.am-en.base.am → .en
            am_lines += am_file.read_text(encoding="utf-8").splitlines()
            en_lines += en_file.read_text(encoding="utf-8").splitlines()
        df = pd.DataFrame({"am": am_lines, "en": en_lines})
    elif gezmu_src.exists():
        df = pd.read_csv(gezmu_src, dtype=str)
    else:
        raise FileNotFoundError(f"Gezmu data not found in {gezmu_dir}.")
    out = CSV_RAW / "gezmu.csv"
    df.to_csv(out, index=False)
    print(f"gezmu: {len(df)} raw pairs → {out}")


def collect_quran() -> None:
    """OPUS Tanzil am-en parallel corpus (local, Moses format) → csv_raw/quran.csv."""
    quran_dir = LOCAL / "Quran"
    quran_am, quran_en = quran_dir / "Tanzil.am-en.am", quran_dir / "Tanzil.am-en.en"
    if not (quran_am.exists() and quran_en.exists()):
        raise FileNotFoundError(f"Quran data not found — place Tanzil.am-en.am/.en in {quran_dir}.")
    df = pd.DataFrame({
        "am": quran_am.read_text(encoding="utf-8").splitlines(),
        "en": quran_en.read_text(encoding="utf-8").splitlines(),
    })
    out = CSV_RAW / "quran.csv"
    df.to_csv(out, index=False)
    print(f"quran: {len(df)} raw pairs → {out}")


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


def collect_ccaligned() -> None:
    """OPUS CCAligned am-en (web-mined, Moses format) → csv_raw/ccaligned.csv."""
    zip_path = CCALIGNED_FULL / "am-en.txt.zip"
    am_path = CCALIGNED_FULL / "CCAligned.am-en.am"
    en_path = CCALIGNED_FULL / "CCAligned.am-en.en"

    if not (am_path.exists() and en_path.exists()):
        CCALIGNED_FULL.mkdir(parents=True, exist_ok=True)
        resp = requests.get(CCALIGNED_URL, timeout=120)
        resp.raise_for_status()
        zip_path.write_bytes(resp.content)
        with zipfile.ZipFile(zip_path) as zf:
            zf.extract("CCAligned.am-en.am", CCALIGNED_FULL)
            zf.extract("CCAligned.am-en.en", CCALIGNED_FULL)
        zip_path.unlink()

    df = pd.DataFrame({
        "am": am_path.read_text(encoding="utf-8").splitlines(),
        "en": en_path.read_text(encoding="utf-8").splitlines(),
    })
    out = CSV_RAW / "ccaligned.csv"
    df.to_csv(out, index=False)
    print(f"ccaligned: {len(df)} raw pairs → {out}")


# Each source, with the CSV(s) it produces — the outputs a default run checks for
# before deciding it has nothing to do.
SOURCES: dict[str, tuple] = {
    "afridoc":   (collect_afridoc,   ("afridoc_health.csv", "afridoc_tech.csv")),
    "gezmu":     (collect_gezmu,     ("gezmu.csv",)),
    "quran":     (collect_quran,     ("quran.csv",)),
    "nllb":      (collect_nllb,      ("nllb.csv",)),
    "ccaligned": (collect_ccaligned, ("ccaligned.csv",)),
}


def main(argv: list[str] | None = None) -> None:
    """Run the named sources, or every source if none are named.

    Naming a source is a statement of intent, so it always rebuilds — that is the
    LASER_CUTOFF loop: `python collect.py nllb`. A default run instead skips
    sources whose CSVs are already there, since their inputs are fixed; --force
    overrides that.
    """
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1].strip())
    parser.add_argument("sources", nargs="*", choices=list(SOURCES), metavar="SOURCE",
                        help=f"sources to collect ({', '.join(SOURCES)}); default: all")
    parser.add_argument("--force", action="store_true",
                        help="rebuild even if the output CSV already exists")
    args = parser.parse_args(argv)

    CSV_RAW.mkdir(parents=True, exist_ok=True)
    # An explicitly named source always rebuilds; only a default run skips.
    named = bool(args.sources)
    for name in args.sources or list(SOURCES):
        fn, outputs = SOURCES[name]
        if not (named or args.force) and all((CSV_RAW / o).exists() for o in outputs):
            print(f"{name}: up to date, skipping "
                  f"(`python -m collection.collect {name}` to rebuild)")
            continue
        fn()


if __name__ == "__main__":
    main(sys.argv[1:])
