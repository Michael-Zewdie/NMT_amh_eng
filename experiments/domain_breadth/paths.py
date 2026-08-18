"""Every on-disk location this experiment reads or writes.

WARNING — the identifiers below are a live contract, not free-form naming.
Every arm here is a FULLY TRAINED run whose checkpoints, tokenized data and
vocabularies already sit on disk under exactly these names, and whose numbers
are published in EXPERIMENTS.md. Rename one and the run is orphaned silently:
eval reports a missing checkpoint, or a rebuild quietly retrains against a
different corpus than the one that produced the results. That holds for the
archived broad arm too — archiving moved its files, it did not free the names.

The asymmetric naming below is deliberate and frozen, not an oversight:
    narrow -> data/prepared_translit/        (built first, before a second arm existed)
    broad  -> data/prepared_broad_translit/  (named around it afterwards)
Regularising these would mean moving data that trained models point at.
"""
from process.utils.paths import DATA, LOCAL, ROOT

ARCHIVE_DATA = ROOT / "archive" / "data"

GEZMU = LOCAL / "Gezmu"                 # Gezmu et al.'s own release, read as raw text
# BROAD v1 was archived 2026-08-17 (superseded by v2 on every fixed benchmark), corpus
# and all. Its paths point into archive/ rather than being deleted from this table:
# `test_broad` is still a published column of the 3x3, so narrow and broad_v2 have to
# keep scoring on it. Renamed on the way in — archive/data/final_broad was already
# taken by am-en-broad-old — to match its `*_translit` siblings.
FINAL_BROAD = ARCHIVE_DATA / "final_broad_translit"   # v1, FROZEN — trained am-en-broad
FINAL_BROAD_V2 = DATA / "final_broad_v2"    # built by corpus.py today

# --- frozen: artifacts already on disk are named by these -------------------
TOK_DIRS = {
    "narrow": DATA / "tokenizer" / "shared_translit",
    "broad":  ARCHIVE_DATA / "tokenizer" / "shared_translit_broad",
    "broad_v2": DATA / "tokenizer" / "shared_translit_broad_v2",
}
PREPARED_DIRS = {          # the value model.training.train reads as data.prepared_dir
    "narrow": "data/prepared_translit",
    "broad":  "archive/data/prepared_broad_translit",
    "broad_v2": "data/prepared_broad_v2_translit",
}
RUN_NAMES = {"narrow": "am-en-narrow", "broad": "am-en-broad",
             "broad_v2": "am-en-broad-v2"}


def prepared_dir(arm: str):
    # Repo-relative, so an archived arm's path resolves the same way a live one's does.
    return ROOT / PREPARED_DIRS[arm] / "am-en"


RESULTS = ROOT / "experiments" / "domain_breadth" / "results.json"
