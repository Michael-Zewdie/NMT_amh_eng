"""Training driver — one recipe, shared by both arms.

Held constant across arms: Unigram algorithm, 8k shared vocabulary, Moses + AT4MT
transliteration, tied embeddings, and every training hyperparameter, which
arms.make_config reads fresh from the source run's manifest on each launch so
they cannot drift. Varies: the corpus. That is the experiment.
"""
import json
import subprocess
import sys
import time

import yaml

from experiments.domain_breadth.arms import ARMS, make_config
from process.utils.paths import ROOT, RUNS


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
        print(f"[train] resuming from {ckpt}", flush=True)
    config_path = run_dir / "config.yaml"
    config_path.write_text(yaml.safe_dump(cfg, sort_keys=False))
    log_path = run_dir / "train.log"
    print(f"[train] {arm.run}: {cfg['training']['max_steps']:,} steps -> {log_path}", flush=True)

    started = time.time()
    with open(log_path, "w") as log:
        proc = subprocess.run([sys.executable, "-m", "model.training.train", str(config_path)],
                              stdout=log, stderr=subprocess.STDOUT, cwd=ROOT)
    elapsed = time.time() - started
    best = None
    for line in log_path.read_text().splitlines():
        if "new best val_bleu" in line:
            best = float(line.split("new best val_bleu")[1].split()[0])
    (run_dir / "manifest.json").write_text(json.dumps({
        "run_name": arm.run, "experiment": "domain_breadth", "arm": arm.name,
        "config_from": arm.config_from, "config": cfg,
        "wall_clock_seconds": round(elapsed, 1),
        "returncode": proc.returncode, "best_val_bleu": best,
    }, indent=2))
    print(f"[train] {'ok' if proc.returncode == 0 else 'FAILED'} in {elapsed/60:.1f} min, "
          f"best val_bleu={best}", flush=True)
    if proc.returncode != 0:
        sys.exit(f"training failed — see {log_path}")
