"""
process.py — Clean every CSV in data/raw/csv_raw/ into data/processed/, then pool + split.

Runs each source (including the laser-filtered nllb.csv that collect.py produced)
through the shared clean() pipeline (processing.clean.filters: normalize → length →
script_purity → dedupe), then pools everything into an 80/10/10 split.

Everything upstream — HuggingFace/local collection and the NLLB parquet →
csv_raw/nllb.csv laser-filter — lives in collection/collect.py. This file only ever touches
CSVs that already sit in csv_raw.

Adjust the CONFIG block, then run (from the project root):

    python -m processing.process

Stage order: clean csv_raw/*.csv → [score_labse] → [score_africomet] → [score_lid]
→ [score_embed] → pool → length_dist [→ semantic_dist].

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
COSINE_CUTOFF   = .7               # non-NLLB LaBSE cosine threshold, mined sources
SOURCE_LID_CUTOFF = .90            # Amharic LID confidence floor (uniform — every source)
TARGET_LID_CUTOFF = .90            # English LID confidence floor (uniform — every source)
# gezmu/afridoc_*/quran are professionally human-translated, already-validated corpora —
# labse_score/africomet_score are automated re-checks of alignment/adequacy that make
# sense for mined data, not a substitute for that. See processing.utils.pool.CURATED_SOURCES.
CURATED_COSINE_CUTOFF    = 0.0
# AfriCOMET-QE floor, per source (see processing.utils.pool.AFRICOMET_CUTOFFS — this dict
# is that file's default, overridden here with the values this pipeline actually runs
# with). 0.15 for the curated sources is a garbage filter, NOT a quality gate —
# deliberately an order of magnitude below the mined 0.80. Inspecting the curated tail by
# hand (processing.utils.sample.sample_by_score pulls exactly this kind of band): rows
# below ~0.10 are genuine off-by-one sentence misalignments (an Amharic clause paired
# with the adjacent English one) and garbled cells, while everything from ~0.30 up is
# correct translation that AfriCOMET merely scores low because archaic/liturgical Amharic
# is out of its distribution. 0.15 sits in that gap: it drops ~310 rows of 173,638
# (0.18%). Anything at or above 0.6 would cut 44% of quran and 22% of gezmu — the
# uniform-cutoff mistake this tiering exists to prevent. See EXPERIMENTS.md. Every
# curated source shares 0.15 today because none has needed a different number yet — this
# is a dict, not a single CURATED_AFRICOMET_CUTOFF, precisely so one can be tuned on its
# own the moment it does (e.g. a newly added curated source with its own noise profile).
AFRICOMET_CUTOFFS = {
    "gezmu":          0.15,
    "afridoc_health": 0,
    "afridoc_tech":   0,
    "quran":          0.15,
    "religious":      0.15,
    "nllb":           0.80,
    "ccaligned":      0.80,
}
DEFAULT_AFRICOMET_CUTOFF = 0.80    # any processed source not named above (mined-tier default)
# CURATED_COSINE_CUTOFF stays 0.0 (and un-tiered, unlike AfriCOMET above): LaBSE's
# curated tail is NOT separable the same way — correct Quran verses score as low as real
# Gezmu misalignments, so no per-source threshold works. Catching those needs a
# neighbour-relative margin score, not a floor on raw cosine.
SEED            = 42                # shuffle / split seed
AMH_LEN         = (5, 500)          # (min, max) chars kept, Amharic side
ENG_LEN         = (10, 500)         # (min, max) chars kept, English side
SPLIT           = (0.8, 0.1, 0.1)   # train / validation / test ratios

# Semantic stratification (optional, off by default — see processing.utils.pool.
# split_semantic). N_CLUSTERS is the k-means k over English-side embeddings,
# STRATIFY_SEMANTIC nests it on top of the length stratification when on. Off
# by default so the plain length-only split stays the reproducible default;
# flip it on deliberately to A/B against every model trained so far.
N_CLUSTERS         = 16
STRATIFY_SEMANTIC  = True

# Every CSV to clean (collect.py put them all here, nllb.csv included).
CSV_SOURCES = sorted(CSV_RAW.glob("*.csv"))

# ── STAGE TOGGLES ──────────────────────────────────────────────────────────────
RUN_CLEAN         = False            # clean csv_raw/*.csv → data/processed/
RUN_LABSE         = True            # annotate labse_score (cached; slow only on unseen text)
RUN_AFRICOMET     = True            # annotate africomet_score (cached; slow only on unseen text)
RUN_LID           = True            # annotate source_lid/target_lid (cached; slow only on unseen text)
RUN_EMBED         = True            # cache English-side embeddings for semantic stratification (cached; slow only on unseen text; only consumed if STRATIFY_SEMANTIC=True)
RUN_POOL          = True            # apply the cutoffs, pool every source + split → data/final/ (+ always regenerates the length-dist chart, + semantic-dist chart when STRATIFY_SEMANTIC=True)
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
            # nllb.csv and ccaligned.csv are noisy mined bitext — one Amharic sentence can
            # turn up matched to several English strings of varying mining quality, so
            # am-only dedupe (keep the first/only row) is the right call. The curated
            # sources can carry genuine multi-reference translations of the same sentence —
            # quran.csv alone has up to ~46 independent translator versions per verse — so
            # dropping a row there requires both sides to match, not just the Amharic.
            dedupe_keys = ("am",) if f.stem in ("nllb", "ccaligned") else ("am", "en")
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

    if RUN_EMBED:
        banner("score_embed — cache English-side embeddings for semantic stratification (cached)")
        from processing.utils import score_embed
        score_embed.main()

    if RUN_POOL:
        banner(f"pool — cutoffs + merge all sources + {SPLIT} split"
               f"{' (semantic-stratified)' if STRATIFY_SEMANTIC else ''}")
        from processing.utils import pool
        pool.main(split_ratios=SPLIT, seed=SEED,
                  cosine_cutoff=COSINE_CUTOFF,
                  africomet_cutoffs=AFRICOMET_CUTOFFS,
                  default_africomet_cutoff=DEFAULT_AFRICOMET_CUTOFF,
                  source_lid_cutoff=SOURCE_LID_CUTOFF,
                  target_lid_cutoff=TARGET_LID_CUTOFF,
                  curated_cosine_cutoff=CURATED_COSINE_CUTOFF,
                  stratify_semantic=STRATIFY_SEMANTIC,
                  n_clusters=N_CLUSTERS)

        # A pool run always refreshes the length-distribution report + pie chart,
        # so the reported buckets match the split that just used them.
        banner("length_dist — sentence-length distributions")
        from processing.dist import length_dist
        length_dist.main()

        if STRATIFY_SEMANTIC:
            banner("semantic_dist — semantic-cluster distribution")
            from processing.dist import semantic_dist
            semantic_dist.main()

    banner("done")


if __name__ == "__main__":
    main()
