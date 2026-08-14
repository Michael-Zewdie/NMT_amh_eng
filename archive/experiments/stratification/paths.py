"""Every on-disk location this experiment reads or writes.

WARNING — the identifiers below are a live contract, not free-form naming.
am-en-splitstrat-length is a PARTIALLY TRAINED run (step 210,000 of 250,000,
~5 GPU-hours, interrupted 2026-08-12 19:36 UTC by a host reboot) whose
checkpoints, tokenized data and shared vocabulary already sit on disk under
exactly these names, and `train --arm length --resume` finds them by string.
The checkpoints themselves are plain tensors (model/common.py:53 saves
state_dicts + an int + a float, no pickled classes) so they survive any amount
of code reorganisation — but only for as long as these strings keep pointing
at them. Rename one and the run is orphaned, silently: the resume path just
reports a missing file, or worse, starts fresh.
"""
from process.utils.paths import DATA, ROOT

# --- frozen: artifacts already on disk are named by these ------------------
TOK_DIR = DATA / "tokenizer" / "shared_translit_splitstrat"


def run_name(arm: str) -> str:
    return f"am-en-splitstrat-{arm}"


def final_dir(arm: str):
    return DATA / f"final_splitstrat_{arm}"


def prepared_dir(arm: str):
    return DATA / f"prepared_splitstrat_{arm}" / "am-en"


# --- not frozen: eval has never run, so nothing has been written here yet ---
# Moved from experiments/split_strategy_results.json when this experiment
# became a package, following experiments/domain_breadth/results.json's
# results-sit-next-to-their-code convention.
RESULTS = ROOT / "experiments" / "stratification" / "results.json"
