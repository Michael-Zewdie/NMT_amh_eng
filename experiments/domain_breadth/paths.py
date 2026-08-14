"""Every on-disk location this experiment reads or writes.

WARNING — the identifiers below are a live contract, not free-form naming.
am-en-narrow and am-en-broad are both FULLY TRAINED runs whose checkpoints,
tokenized data and vocabularies already sit on disk under exactly these names,
and whose numbers are published in EXPERIMENTS.md. Rename one and the run is
orphaned silently: eval reports a missing checkpoint, or a rebuild quietly
retrains against a different corpus than the one that produced the results.

The asymmetric naming below is deliberate and frozen, not an oversight:
    narrow -> data/prepared_translit/        (built first, before a second arm existed)
    broad  -> data/prepared_broad_translit/  (named around it afterwards)
Regularising these would mean moving data that trained models point at.
"""
from process.utils.paths import DATA, LOCAL, ROOT

GEZMU = LOCAL / "Gezmu"                 # Gezmu et al.'s own release, read as raw text
FINAL_BROAD = DATA / "final_broad"      # built by corpus.py

# --- frozen: artifacts already on disk are named by these -------------------
TOK_DIRS = {
    "narrow": DATA / "tokenizer" / "shared_translit",
    "broad":  DATA / "tokenizer" / "shared_translit_broad",
}
PREPARED_DIRS = {          # the value model.training.train reads as data.prepared_dir
    "narrow": "data/prepared_translit",
    "broad":  "data/prepared_broad_translit",
}
RUN_NAMES = {"narrow": "am-en-narrow", "broad": "am-en-broad"}


def prepared_dir(arm: str):
    return DATA / PREPARED_DIRS[arm].split("/", 1)[1] / "am-en"


RESULTS = ROOT / "experiments" / "domain_breadth" / "results.json"
