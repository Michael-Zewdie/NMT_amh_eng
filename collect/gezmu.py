"""collect.gezmu — Gezmu am-en parallel corpus (local) → csv_raw/gezmu.csv."""
import pandas as pd

from process.utils.paths import CSV_RAW, LOCAL


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
