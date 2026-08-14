"""The single arm, its build, and the recipe.

    LOWER   English target lowercased before the vocabulary is fit

Following Gezmu et al., whose released `*.base.*` files are lowercased (0.0% of
their English carries any uppercase, train/dev/test alike) and whose 33.0 BLEU
this project has been chasing.

Amharic is not lowercased and does not need to be: AT4MT transliteration already
emits all-lowercase Latin (`translit_am` output contains no uppercase character),
and Ge'ez has no case. "Lowercase everything" is therefore a target-side-only
change here — worth stating, because it means only the target half of the shared
vocabulary moves.

HOW CASING IS HANDLED — the same way experiments.domain_breadth already does it
------------------------------------------------------------------------------
By CASE-INSENSITIVE SCORING, not by restoring case. Nothing in this pipeline
recases: `detok_en` only rejoins punctuation, so a model trained on lowercase
emits lowercase forever. Rather than bolt on a truecaser, evaluate.py lowercases
both hypothesis and reference and scores that — exactly what
experiments.domain_breadth.evaluate's `_ci` keys do, for exactly the same reason
(its narrow arm trains on Gezmu's lowercased release and can never emit case).

That choice buys direct comparability: am-en-narrow and am-en-broad already have
`_ci` numbers on the same benchmarks (FLORES 6.22 / 12.46, MAFAND 4.23 / 5.95),
so this arm drops straight into that table with no new machinery.

The cased column is still reported as the floor, and should be read as measuring
missing capital letters rather than translation quality — a PERFECT but lowercase
translation caps at ~80.1 BLEU on FLORES and ~71.6 on MAFAND. Neither arm's cased
BLEU is comparable to EXPERIMENTS.md's main table; the `_ci` column is.

Hyperparameters are chained off am-en-narrow's manifest — the Gezmu recipe
(Moses + AT4MT + shared 8k Unigram + tied embeddings, beam 4 / lp 0.6) — read
fresh from disk on every launch. Only max_steps is overridden; see MAX_STEPS.
"""
import json
import pickle
import subprocess
import sys
import time
from dataclasses import dataclass

import pandas as pd
import yaml

from experiments.clean_recipe.corpus import FINAL_CLEAN
from model.common import BOS_ID, EOS_ID
from model.tokenize.preprocess import moses_en, translit_am
from process.utils.paths import DATA, ROOT, RUNS

SPLITS = ["train", "validation", "test"]
VOCAB_SIZE = 8000
SPECIALS = ["<pad>", "<unk>", "<s>", "</s>"]
MAX_LEN = 150

BASELINE = "am-en-narrow"    # the Gezmu recipe; see module docstring

# RAISED 60,000 -> 250,000 on 2026-08-14, mid-run, after the screening budget did
# its job: val_bleu was still climbing +0.7 per 5k steps at step 40,000 (23.38 @30k
# -> 24.45 @35k -> 25.14 @40k) with val_loss still falling, so 60k was plainly
# leaving BLEU on the table.
#
# Extending mid-run is safe here specifically because the schedule is Vaswani
# inverse-sqrt (model/training/optim.py:36):
#     d_model^-0.5 * min(step^-0.5, step * warmup_steps^-1.5)
# There is no max_steps term, so a resumed 250k run sees bit-identical learning
# rates to a native 250k run. Under a cosine or linear-decay schedule this would
# NOT hold — changing the horizon would reshape the whole trajectory — so do not
# copy this move to a run with a different scheduler without rechecking.
#
# The real payoff is comparability rather than the extra ~2 BLEU: am-en-narrow and
# am-en-broad are both 250,000-step runs, so this makes the FLORES/MAFAND
# comparison against them step-matched as well as recipe-matched.
MAX_STEPS = 250_000


@dataclass(frozen=True)
class Arm:
    name: str
    lowercase: bool

    @property
    def run(self) -> str:
        return f"am-en-clean-{self.name}"

    @property
    def tok_dir(self):
        return DATA / "tokenizer" / f"shared_translit_clean_{self.name}"

    @property
    def prepared_rel(self) -> str:
        return f"data/prepared_clean_{self.name}"

    @property
    def prepared(self):
        return DATA / f"prepared_clean_{self.name}" / "am-en"


# One arm by design. A cased control is one entry away — Arm("cased", lowercase=False)
# — but the project chose to follow Gezmu's lowercase-then-truecase setup directly
# rather than re-measure a decision that already has evidence behind it.
ARMS = {"lower": Arm("lower", lowercase=True)}
DEFAULT_ARM = "lower"


def read_split(split: str) -> tuple[list[str], list[str]]:
    df = pd.read_csv(FINAL_CLEAN / f"{split}.csv", usecols=["am", "en"], dtype=str).dropna()
    return df["am"].tolist(), df["en"].tolist()


def preprocess(arm: Arm, am: list[str], en: list[str]) -> tuple[list[str], list[str]]:
    """Moses + AT4MT on the source, Moses (+ optional lowercase) on the target."""
    src = [translit_am(s) for s in am]
    tgt = [moses_en(s) for s in en]
    if arm.lowercase:
        tgt = [s.lower() for s in tgt]
    return src, tgt


def cmd_build(args) -> None:
    from tokenizers.implementations import SentencePieceUnigramTokenizer

    arm = ARMS[args.arm]
    if not (FINAL_CLEAN / "train.csv").exists():
        sys.exit(f"no {FINAL_CLEAN}/train.csv — run `corpus --write` first")

    splits, raw_example = {}, None
    for split in SPLITS:
        am, en = read_split(split)
        splits[split] = preprocess(arm, am, en)
        if split == "train":
            raw_example = (am[0], en[0])
        print(f"[build] {split}: {len(am):,} pairs Moses-tokenized + transliterated"
              f"{' + lowercased' if arm.lowercase else ''}", flush=True)
    print(f"[build]   am: {raw_example[0][:60]}")
    print(f"[build]    -> {splits['train'][0][0][:60]}")
    print(f"[build]   en: {raw_example[1][:60]}")
    print(f"[build]    -> {splits['train'][1][0][:60]}")

    tr_am, tr_en = splits["train"]
    tok = SentencePieceUnigramTokenizer()
    tok.train_from_iterator(tr_am + tr_en, vocab_size=VOCAB_SIZE,
                            special_tokens=SPECIALS, unk_token="<unk>")
    arm.tok_dir.mkdir(parents=True, exist_ok=True)
    tok.save(str(arm.tok_dir / "tokenizer.json"))
    print(f"[build] fresh shared {VOCAB_SIZE} vocab (fit on THIS arm's train) -> {arm.tok_dir}")

    arm.prepared.mkdir(parents=True, exist_ok=True)
    for split, (am, en) in splits.items():
        s = [[BOS_ID, *e.ids, EOS_ID] for e in tok.encode_batch(am)]
        t = [[BOS_ID, *e.ids, EOS_ID] for e in tok.encode_batch(en)]
        keep = [i for i, (a, b) in enumerate(zip(s, t)) if len(a) <= MAX_LEN and len(b) <= MAX_LEN]
        with open(arm.prepared / f"{split}.pkl", "wb") as f:
            pickle.dump({"src": [s[i] for i in keep], "tgt": [t[i] for i in keep]}, f)
        print(f"[build] {split}: kept {len(keep):,} of {len(am):,}")
    print(f"[build] next: python -m experiments.clean_recipe train --arm {arm.name}")


def make_config(arm: Arm) -> dict:
    manifest = RUNS / BASELINE / "manifest.json"
    if not manifest.exists():
        manifest = ROOT / "archive" / "runs" / BASELINE / "manifest.json"
    if not manifest.exists():
        sys.exit(f"cannot find {BASELINE}'s manifest — needed to pin hyperparameters")
    cfg = json.loads(manifest.read_text())["config"]
    cfg["run_name"] = arm.run
    cfg["data"]["prepared_dir"] = arm.prepared_rel
    cfg["data"]["src_tokenizer"] = str(arm.tok_dir)
    cfg["data"]["tgt_tokenizer"] = str(arm.tok_dir)
    cfg["training"]["max_steps"] = MAX_STEPS
    cfg["training"]["resume_from"] = None
    return cfg


def cmd_train(args) -> None:
    arm = ARMS[args.arm]
    run_dir = RUNS / arm.run
    run_dir.mkdir(parents=True, exist_ok=True)
    cfg = make_config(arm)
    if getattr(args, "resume", False):
        ckpt = run_dir / "checkpoints" / "last.pt"
        if not ckpt.exists():
            sys.exit(f"--resume: no {ckpt}")
        cfg["training"]["resume_from"] = str(ckpt)
    config_path = run_dir / "config.yaml"
    config_path.write_text(yaml.safe_dump(cfg, sort_keys=False))
    log_path = run_dir / "train.log"
    print(f"[train] {arm.run}: {cfg['training']['max_steps']:,} steps -> {log_path}", flush=True)

    started = time.time()
    with open(log_path, "a" if cfg["training"].get("resume_from") else "w") as log:
        proc = subprocess.run([sys.executable, "-m", "model.training.train", str(config_path)],
                              stdout=log, stderr=subprocess.STDOUT, cwd=ROOT)
    elapsed = time.time() - started
    best = None
    for line in log_path.read_text().splitlines():
        if "new best val_bleu" in line:
            best = float(line.split("new best val_bleu")[1].split()[0])
    (run_dir / "manifest.json").write_text(json.dumps({
        "run_name": arm.run, "experiment": "clean_recipe", "arm": arm.name,
        "lowercase": arm.lowercase, "baseline": BASELINE, "config": cfg,
        "wall_clock_seconds": round(elapsed, 1), "returncode": proc.returncode,
        "best_val_bleu": best,
    }, indent=2))
    print(f"[train] {'ok' if proc.returncode == 0 else 'FAILED'} in {elapsed/60:.1f} min, "
          f"best val_bleu={best}", flush=True)
    if proc.returncode != 0:
        sys.exit(f"training failed — see {log_path}")
