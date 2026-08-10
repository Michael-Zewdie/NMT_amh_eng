"""collect.quran — OPUS Tanzil am-en parallel corpus (local, Moses format)
→ csv_raw/quran.csv."""
import pandas as pd

from process.utils.paths import CSV_RAW, LOCAL


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
