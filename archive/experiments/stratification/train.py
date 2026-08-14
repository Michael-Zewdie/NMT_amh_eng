"""Training driver — one recipe, pinned to am-en-broad, for all arms.

Recipe, pinned to am-en-broad's manifest (REVISED 2026-08-12, see
below): architecture (512d/8h/6+6/dff2048, ~56.5M params), LR, warmup,
dropout, label smoothing, grad clip, 250,000 steps, tied embeddings, beam 4 /
lp 0.6 — the Moses + AT4MT-transliteration + ONE shared 8k Unigram vocabulary
recipe, matching experiments.domain_breadth's two arms.

make_config() is deliberately the single source of that recipe for every arm.
The arms differ ONLY in run_name and prepared_dir; every hyperparameter is
read fresh from the baseline manifest on each launch, so the three runs cannot
drift apart. Any BLEU gap between arms is then attributable to the split axis
under test rather than to a stale hyperparameter in one arm's copy.

Originally pinned to am-en-broad-8k instead (no transliteration, reusing the
Gezmu-fit data/tokenizer/{am,en} untouched). Two things changed that:

  1. On the identical "broad" corpus, am-en-broad beats am-en-broad-8k
     by +1.05 FLORES / +0.47 MAFAND (EXPERIMENTS.md) — a strictly better recipe
     available at zero extra confound cost (see point 2).
  2. Measuring the reused Gezmu-fit vocab against THIS corpus (health/tech/mined
     web text, not Gezmu's Watchtower/Bible register) found real fragmentation:
     English +22.9% more subwords/word, Amharic +12.2% more subwords/word plus a
     new 1.47% UNK rate (0% on Gezmu text). A floor-effect risk shared equally by
     all three arms (not a between-arm confound), but real — same shape as the
     diverse10k/health10k experiment's own floor-effect problem elsewhere in
     this project.

Architecture size was deliberately NOT scaled up despite this corpus being
~3.1x am-en-broad-8k/am-en-gezmu-8k's 140k. Investigated against this project's
own training logs before deciding, not assumed: am-en-broad-8k (0.96 of budget)
and am-en-broad (0.92) both plateaued and DECLINED after their peak;
am-en-base-v4 — same core architecture, MORE training data (409k pairs, more
than this experiment's ~346k/arm) — plateaued around step 86k and was manually
killed at 180k/224k. Nothing in this project's history shows this architecture
capacity-starved at or above ~350-400k pairs. If anything the opposite:
EXPERIMENTS.md's small-vs-base comparison found a 4x-smaller 256d/32M-param
model (am-en-small-moderate) matched/beat the 512d/93M-param base-v4 on
out-of-domain generalization (MAFAND 6.99 vs 6.30), "may generalize better
per-parameter... worth a real head-to-head" — the one relevant precedent in this
project points away from bigger being better, not toward it. No scaling-up
experiment is queued in EXPERIMENTS.md's backlog. Keeping size unchanged is
therefore both evidence-based AND the correct experimental control regardless
(changing model size would itself be a new confound).
"""
import json
import subprocess
import sys
import time

import yaml

from experiments.stratification.paths import TOK_DIR, run_name
from process.utils.paths import ROOT, RUNS

BASELINE = "am-en-broad"


def make_config(arm: str) -> dict:
    """am-en-broad's own config (architecture, LR schedule, tied
    embeddings, 250,000 steps — see module docstring for why this baseline and
    why the architecture is left unscaled), with only run name, tokenizer, and
    data source changed to this experiment's shared vocab / arm-specific split."""
    manifest = RUNS / BASELINE / "manifest.json"
    if not manifest.exists():
        manifest = ROOT / "archive" / "runs" / BASELINE / "manifest.json"
    if not manifest.exists():
        sys.exit(f"cannot find {BASELINE}'s manifest — needed to pin hyperparameters")
    cfg = json.loads(manifest.read_text())["config"]
    cfg["run_name"] = run_name(arm)
    cfg["data"]["prepared_dir"] = f"data/prepared_splitstrat_{arm}"
    cfg["data"]["src_tokenizer"] = str(TOK_DIR)
    cfg["data"]["tgt_tokenizer"] = str(TOK_DIR)
    cfg["training"]["resume_from"] = None
    return cfg


def cmd_train(args) -> None:
    arm = args.arm
    run_dir = RUNS / run_name(arm)
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
    print(f"[train] {run_name(arm)}: {cfg['training']['max_steps']:,} steps -> {log_path}", flush=True)

    started = time.time()
    # Append on resume: "w" truncated the pre-interruption history, which is how
    # the length arm's first 210,000 steps were lost from its log (the tensorboard
    # events survived, but the log is what gets grepped for best val_bleu below).
    with open(log_path, "a" if cfg["training"].get("resume_from") else "w") as log:
        proc = subprocess.run([sys.executable, "-m", "model.training.train", str(config_path)],
                              stdout=log, stderr=subprocess.STDOUT, cwd=ROOT)
    elapsed = time.time() - started
    best = None
    for line in log_path.read_text().splitlines():
        if "new best val_bleu" in line:
            best = float(line.split("new best val_bleu")[1].split()[0])
    (run_dir / "manifest.json").write_text(json.dumps({
        "run_name": run_name(arm), "experiment": "stratification", "arm": arm, "baseline": BASELINE,
        "config": cfg, "wall_clock_seconds": round(elapsed, 1),
        "returncode": proc.returncode, "best_val_bleu": best,
    }, indent=2))
    print(f"[train] {'ok' if proc.returncode == 0 else 'FAILED'} in {elapsed/60:.1f} min, "
          f"best val_bleu={best}", flush=True)
    if proc.returncode != 0:
        sys.exit(f"training failed — see {log_path}")
