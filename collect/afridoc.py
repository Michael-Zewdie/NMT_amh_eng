"""collect.afridoc — AfriDocMT health + tech from HuggingFace
→ csv_raw/afridoc_health.csv, csv_raw/afridoc_tech.csv."""
import pandas as pd
from datasets import load_dataset

from process.utils.paths import CSV_RAW


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
