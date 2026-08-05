"""
remove_domain.py — Drop every row of data/processed/nllb.csv whose source website is
one of the given registered domains. Edits the file in place (regenerable via
process.py). Standalone manual tool — not a process.py stage.

Usage (from the project root):
       python -m processing.random.remove_domain <domain> [more domains ...]
       python -m processing.random.remove_domain blogspot.com scribd.com
"""
import sys

import polars as pl

from processing.utils.paths import PROCESSED
from processing.dist.websites import domains  # rows -> registered_domain

NLLB_CSV = PROCESSED / "nllb.csv"


def remove(targets: list[str]) -> None:
    """Drop every row of nllb.csv whose source domain is in targets (in place)."""
    if not targets:
        return
    df = pl.read_csv(NLLB_CSV)
    keep = df.filter(~domains(df).is_in(targets))
    keep.write_csv(NLLB_CSV)
    print(f"{NLLB_CSV.name}: {df.height:,} -> {keep.height:,} "
          f"({keep.height - df.height:+,}) after removing {targets}")


if __name__ == "__main__":
    targets = sys.argv[1:]
    if not targets:
        sys.exit("usage: python -m processing.random.remove_domain <domain> [more domains ...]")
    remove(targets)
