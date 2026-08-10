"""
sample.py — pull a random sample of rows from one processed source, filtered to a
score band, so a human can eyeball how good or bad that band actually is.

Every cutoff in this pipeline (processing.utils.pool.AFRICOMET_CUTOFFS,
COSINE_CUTOFF, ...) is ultimately decided by reading real examples at a candidate
threshold — see process.py's AFRICOMET_CUTOFFS comment ("Inspecting the curated
tail by hand: rows below ~0.10 are genuine off-by-one sentence misalignments...").
This formalizes that one-off inspection into a reusable function/CLI instead of a
throwaway `df[df.africomet_score.between(...)].sample(20)` in a notebook each time
a cutoff needs (re-)tuning, or a new source needs a first one.

Reads data/processed/<source>.csv directly — the scored-but-unfiltered output of
score_labse / score_africomet / score_lid, *before* pool.py's cutoffs run — so a
sample here always includes exactly the rows a cutoff change would gain or lose.
Sources vary in which score columns they carry (nllb.csv has laser_score, the
local corpora don't; every source has labse_score + africomet_score + LID) — see
each score_*.py module for which.

Run (from the project root), e.g. to inspect the band the curated AfriCOMET floor
sits in for quran, or the low tail of NLLB's LaBSE score:
    python -m processing.utils.sample quran --africomet 0.10 0.30
    python -m processing.utils.sample nllb --labse 0.0 0.5 -n 10
"""
import argparse

import pandas as pd

from processing.utils.paths import PROCESSED

# Every score column this pipeline can produce. A source only carries the ones its
# own score_*.py stage(s) wrote — see sample_by_score's error message for which
# sources currently have a given column.
SCORE_COLUMNS = ("africomet_score", "labse_score", "laser_score", "source_lid", "target_lid")


def sample_by_score(source: str, n: int = 20, seed: int = 42,
                     **score_ranges: tuple[float, float]) -> pd.DataFrame:
    """n random rows from data/processed/<source>.csv whose scores fall inside the
    given range(s).

    score_ranges are keyword args named after score columns — africomet_score=,
    labse_score=, laser_score=, source_lid=, target_lid= — each an inclusive
    (min, max) tuple. A column left out is not filtered on at all; a column passed
    that this source doesn't carry (e.g. laser_score on a non-NLLB source) raises,
    naming which sources do carry it, instead of silently returning everything or
    raising a bare KeyError.

    Returns fewer than n rows rather than erroring when the filtered band is
    smaller than n — that's the band being narrow, itself useful information about
    the source, not a bug in this function.
    """
    path = PROCESSED / f"{source}.csv"
    if not path.exists():
        known = sorted(p.stem for p in PROCESSED.glob("*.csv"))
        raise FileNotFoundError(f"{path} does not exist — known sources: {known}")
    df = pd.read_csv(path, dtype=str)

    for col, (lo, hi) in score_ranges.items():
        if col not in df.columns:
            carriers = sorted(p.stem for p in PROCESSED.glob("*.csv")
                               if col in pd.read_csv(p, dtype=str, nrows=0).columns)
            raise ValueError(f"{source}.csv has no {col} column — carried by: {carriers}")
        scores = pd.to_numeric(df[col], errors="coerce")
        df = df[scores.between(lo, hi)]

    cols = ["am", "en"] + [c for c in SCORE_COLUMNS if c in df.columns]
    return df[cols].sample(n=min(n, len(df)), random_state=seed).reset_index(drop=True)


def _flag_name(col: str) -> str:
    """africomet_score -> africomet, source_lid -> source-lid: drop the _score
    suffix where there is one, dashes for argparse, matched back to the same
    string (minus dashes) as the dest attribute name in main()."""
    return col.replace("_score", "").replace("_", "-")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sample rows from a processed source, filtered by score range.")
    parser.add_argument("source", help="stem of a data/processed/*.csv file, e.g. gezmu, nllb")
    parser.add_argument("-n", type=int, default=20, help="rows to sample (default 20)")
    parser.add_argument("--seed", type=int, default=42)
    for col in SCORE_COLUMNS:
        parser.add_argument(f"--{_flag_name(col)}", nargs=2, type=float, metavar=("MIN", "MAX"),
                             help=f"keep only rows with {col} in [MIN, MAX]")
    args = parser.parse_args()

    score_ranges = {}
    for col in SCORE_COLUMNS:
        val = getattr(args, col.replace("_score", ""))
        if val is not None:
            score_ranges[col] = tuple(val)

    sample = sample_by_score(args.source, n=args.n, seed=args.seed, **score_ranges)
    with pd.option_context("display.max_colwidth", 80, "display.width", 200, "display.max_rows", None):
        print(sample.to_string(index=False))
    print(f"\n{len(sample)} row(s) shown, source={args.source}, filters={score_ranges or 'none'}")


if __name__ == "__main__":
    main()
