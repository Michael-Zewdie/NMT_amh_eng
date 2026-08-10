"""collect.ccaligned — OPUS CCAligned am-en release (Moses format)
→ csv_raw/ccaligned.csv."""
import zipfile

import pandas as pd
import requests

from process.utils.paths import CCALIGNED_FULL, CSV_RAW

# OPUS CCAligned am-en release (Moses format): web-mined via per-document LASER
# alignment (El-Kishky et al. 2020), no per-pair score shipped — unlike nllb, so
# it gets no laser_score exemption from the labse_score cutoff downstream.
CCALIGNED_URL = "https://object.pouta.csc.fi/OPUS-CCAligned/v1/moses/am-en.txt.zip"


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
