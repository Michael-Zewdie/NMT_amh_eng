"""
char_freq.py — Count occurrences of specific Amharic characters in raw data.

Scans the `am` column of every CSV in data/raw/csv_raw/ and reports how many
times each target labialized (-wa) syllable appears, per source and in total.
"""
from collections import Counter
import pandas as pd

from process.utils.paths import CSV_RAW

TARGETS = ["ኳ", "ጓ", "ኋ", "ቋ", "ጧ", "ሏ", "ሟ", "ፏ", "ቧ", "ቷ", "ኗ", "ዟ", "ሯ"]

raw_paths = sorted(CSV_RAW.glob("*.csv"))
if not raw_paths:
    raise FileNotFoundError(f"No CSVs in {CSV_RAW}")

per_source = {}
totals = Counter()

for path in raw_paths:
    text = "".join(pd.read_csv(path, dtype=str)["am"].dropna())
    counts = Counter({ch: text.count(ch) for ch in TARGETS})
    per_source[path.stem] = counts
    totals.update(counts)

# ── Report ────────────────────────────────────────────────────────────────────
names = [p.stem for p in raw_paths]
col_w = max(len(n) for n in names + ["TOTAL"]) + 2
lbl_w = 12  # "char  U+XXXX"

header = f"{'char':<{lbl_w}}" + "".join(f"{n:>{col_w}}" for n in names) + f"{'TOTAL':>{col_w}}"
print(header)
print("-" * len(header))
for ch in TARGETS:
    label = f"{ch}  U+{ord(ch):04X}"
    row = f"{label:<{lbl_w}}" + "".join(f"{per_source[n][ch]:>{col_w}}" for n in names)
    row += f"{totals[ch]:>{col_w}}"
    print(row)
print("-" * len(header))
grand = sum(totals.values())
print(f"{'sum':<{lbl_w}}" + "".join(f"{sum(per_source[n].values()):>{col_w}}" for n in names)
      + f"{grand:>{col_w}}")
