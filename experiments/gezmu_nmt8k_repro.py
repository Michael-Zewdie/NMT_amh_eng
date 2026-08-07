"""
experiments.gezmu_nmt8k_repro — build data/final/{train,validation,test}.csv
directly from Gezmu et al.'s own released train/dev/test split, with NO
quality filtering of any kind (no AfriCOMET-QE, no LaBSE cosine, no LID, no
length-ratio cutoff). This is the v2 baseline: a faithful reproduction of
their NMT-8K system (arXiv:2104.03543v3, LREC 2022), Table 3, am->en 33.0
BLEU — see model/configs/gezmu_8k.yaml for the full hyperparameter match.

Run (from the project root): python -m experiments.gezmu_nmt8k_repro

Why data/raw/local/Gezmu/*.am-en.base.{am,en} and not data/processed/gezmu.csv
or processing.utils.pool
--------------------------------------------------------------------------
data/processed/gezmu.csv (124,410 rows) has already been through this
pipeline's clean() — dedup, script-purity checks, etc — before any threshold
cutoff is even applied. That is itself a filtering step the paper's authors
never performed. data/raw/local/Gezmu/, by contrast, has exactly Table 1's
count (145,364 = 16,491 Awake + 72,512 Watchtower + 48,651 Bible + 7,710 news)
split into exactly Table 2's train/dev/test sizes (140,000 / 2,864 / 2,500,
totaling 145,364) — almost certainly the authors' own released split (the
paper's own appendix example, "About that time, my parents asked me to come
back home.", is verbatim test.am-en.base.en's first line). Skipping
processing.utils.pool entirely means skipping the dedup, script-purity, and
threshold logic all at once, rather than trying to disable each one.

The one row dropped
--------------------
train.am-en.base.am line 5407 is an empty string paired with the English
target "www.jw.org" — a boilerplate/URL extraction artifact, not a
translation pair (there is nothing to translate). This is the one row this
script removes; every other row of all three splits is kept exactly as
released. Not a quality judgment call the way AfriCOMET/LaBSE were — an
empty source has no BLEU-scorable translation to produce, and pandas would
otherwise read it back as NaN and crash tokenization downstream.

Writes directly to data/final/ (not a side-staging dir like the archived
experiments/gezmu_only.py used) because this reproduction IS v2's starting
point going forward, not one arm among several — see EXPERIMENTS.md and
memory/v2_experiments_archive.md.
"""
import pandas as pd

from processing.utils.paths import DATA, FINAL
from processing.utils.manifest import write_manifest, git_info, now

SRC = DATA / "raw" / "local" / "Gezmu"
# Gezmu's own split naming -> this pipeline's naming.
SPLIT_MAP = {"train": "train", "dev": "validation", "test": "test"}


def _read_pair(split: str) -> pd.DataFrame:
    am = (SRC / f"{split}.am-en.base.am").read_text(encoding="utf-8").splitlines()
    en = (SRC / f"{split}.am-en.base.en").read_text(encoding="utf-8").splitlines()
    assert len(am) == len(en), f"{split}: {len(am)} am lines vs {len(en)} en lines"
    return pd.DataFrame({"am": am, "en": en})


def main() -> None:
    FINAL.mkdir(parents=True, exist_ok=True)
    counts = {}

    for gezmu_split, out_split in SPLIT_MAP.items():
        df = _read_pair(gezmu_split)
        before = len(df)
        empty_src = df["am"].str.strip() == ""
        dropped_rows = df[empty_src]
        df = df[~empty_src].reset_index(drop=True)
        if len(dropped_rows):
            print(f"[gezmu-repro] {gezmu_split}: dropped {len(dropped_rows)} empty-source "
                  f"row(s): {dropped_rows.to_dict('records')}")
        df.to_csv(FINAL / f"{out_split}.csv", index=False)
        print(f"[gezmu-repro] {out_split}.csv: {before} -> {len(df)} pairs")
        counts[out_split] = {"before": before, "after": len(df)}

    write_manifest(FINAL, {
        "source": "data/raw/local/Gezmu (Gezmu, Nurnberger & Bati, arXiv:2104.03543v3, "
                   "Table 1/2 — the authors' own released train/dev/test split)",
        "filtering": "none — no AfriCOMET-QE, no LaBSE cosine, no LID, no length-ratio "
                     "cutoff, no dedup. Only exception: one empty-source boilerplate row "
                     "dropped from train (see this script's docstring).",
        "counts": counts,
        "expected_paper_counts": {"train": 140000, "validation": 2864, "test": 2500},
        "git": git_info(),
        "built_at": now(),
        "built_by": "experiments/gezmu_nmt8k_repro.py",
    })
    print(f"[gezmu-repro] wrote {FINAL}/manifest.json")


if __name__ == "__main__":
    main()
