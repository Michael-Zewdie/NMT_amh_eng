"""
process.py — Clean every CSV in data/raw/csv_raw/ into data/processed/, then pool + split.

Runs each source (including the laser-filtered nllb.csv that collect.py produced)
through the shared clean() pipeline (processing.clean.filters: normalize → length →
script_purity → dedupe), then pools everything into an 80/10/10 split.

Everything upstream — HuggingFace/local collection and the NLLB parquet →
csv_raw/nllb.csv laser-filter — lives in collect.py. This file only ever touches
CSVs that already sit in csv_raw.

Adjust the CONFIG block, then run (from the project root):

    python process.py

Stage order: clean csv_raw/*.csv → [score_labse] → [score_africomet] → [score_lid] → [embed_english] → pool → length_dist.

Scoring and filtering are separate. score_labse/score_africomet/score_lid only *annotate*
data/processed/, caching every score by sentence content; the cutoffs below are
applied at the pool stage. So retuning a threshold means re-running with
RUN_CLEAN/RUN_LABSE/RUN_LID left on — the models never load, every score comes from
the cache, and only the pool re-runs. Nothing is lost by lowering a cutoff.
"""
import polars as pl

from processing.utils.paths import CSV_RAW, PROCESSED
from processing.clean.filters import clean

# ── CONFIG ─────────────────────────────────────────────────────────────────────
# Quality floors — applied at the pool stage, so changing one is a cheap re-run
# (scores are cached; no model reloads). 0.0 disables a cutoff.
COSINE_CUTOFF   = .7               # non-NLLB LaBSE cosine threshold
AFRICOMET_CUTOFF = .65              # AfriCOMET-QE adequacy/fluency floor (placeholder — needs tuning)
SOURCE_LID_CUTOFF = .90            # Amharic LID confidence floor
TARGET_LID_CUTOFF = .90            # English LID confidence floor
SEED            = 42                # shuffle / split seed
AMH_LEN         = (5, 500)          # (min, max) chars kept, Amharic side
ENG_LEN         = (10, 500)         # (min, max) chars kept, English side
SPLIT           = (0.8, 0.1, 0.1)   # train / validation / test ratios
N_CLUSTERS      = 20                # semantic-cluster stratification axis (see dist.clusters)

# Every CSV to clean (collect.py put them all here, nllb.csv included).
CSV_SOURCES = sorted(CSV_RAW.glob("*.csv"))

# ── STAGE TOGGLES ──────────────────────────────────────────────────────────────
RUN_CLEAN         = True            # clean csv_raw/*.csv → data/processed/
RUN_LABSE         = True            # annotate labse_score (cached; slow only on unseen text)
RUN_AFRICOMET     = True            # annotate africomet_score (cached; slow only on unseen text)
RUN_LID           = True            # annotate source_lid/target_lid (cached; slow only on unseen text)
RUN_EMBED_ENGLISH = False            # warm the BGE-large-en-v1.5 embedding cache for semantic clustering (cached; slow only on unseen text)
RUN_POOL          = True            # apply the cutoffs, pool every source + split → data/final/ (+ always regenerates the length-dist chart)
# ────────────────────────────────────────────────────────────────────────────────


def banner(title: str) -> None:
    print(f"\n{'=' * 70}\n=== {title}\n{'=' * 70}")


def main() -> None:
    PROCESSED.mkdir(parents=True, exist_ok=True)

    if RUN_CLEAN:
        banner("clean — csv_raw/*.csv → data/processed/")
        if not CSV_SOURCES:
            raise FileNotFoundError(f"No CSVs in {CSV_RAW} — run `python collect.py` first.")
        for f in CSV_SOURCES:
            df = pl.read_csv(f, infer_schema_length=0)  # all-string, like pandas dtype=str
            print(f"[{f.stem}] input: {df.height}")
            # nllb.csv is noisy mined bitext — one Amharic sentence can turn up matched
            # to several English strings of varying mining quality, so am-only dedupe
            # (keep the first/only row) is the right call. The curated sources can carry
            # genuine multi-reference translations of the same sentence — quran.csv alone
            # has up to ~46 independent translator versions per verse — so dropping a row
            # there requires both sides to match, not just the Amharic.
            dedupe_keys = ("am",) if f.stem == "nllb" else ("am", "en")
            clean(df, f.stem, "am", "en", AMH_LEN, ENG_LEN, dedupe_keys=dedupe_keys).write_csv(PROCESSED / f.name)

    if RUN_LABSE:
        banner("score_labse — annotate LaBSE cosine (cached)")
        from processing.utils import score_labse
        score_labse.main()

    if RUN_AFRICOMET:
        banner("score_africomet — annotate AfriCOMET-QE africomet_score (cached)")
        from processing.utils import score_africomet
        score_africomet.main()

    if RUN_LID:
        banner("score_lid — annotate fastText LID source_lid/target_lid (cached)")
        from processing.utils import score_lid
        score_lid.main()

    if RUN_EMBED_ENGLISH:
        banner("embed_english — warm the BGE-large-en-v1.5 embedding cache (cached)")
        from processing.utils import embed_english
        embed_english.main()

    if RUN_POOL:
        banner(f"pool — cutoffs + merge all sources + {SPLIT} split")
        from processing.utils import pool
        pool.main(split_ratios=SPLIT, seed=SEED,
                  cosine_cutoff=COSINE_CUTOFF,
                  africomet_cutoff=AFRICOMET_CUTOFF,
                  source_lid_cutoff=SOURCE_LID_CUTOFF,
                  target_lid_cutoff=TARGET_LID_CUTOFF,
                  n_clusters=N_CLUSTERS)

        # A pool run always refreshes the length-distribution report + pie chart,
        # so the reported buckets match the split that just used them.
        banner("length_dist — sentence-length distributions")
        from processing.dist import length_dist
        length_dist.main()

    banner("done")


if __name__ == "__main__":
    main()
