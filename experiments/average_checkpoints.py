"""
experiments.average_checkpoints — average the last N model-only snapshots
model/train.py keeps in runs/<name>/checkpoints/avg/ (when the config sets
training.checkpoint_avg_n) into a single checkpoint, decodable like any other
via model/common.py's load_checkpoint.

Why this exists: Gezmu et al. §4.2 (arXiv:2104.03543v3) decode from "a single
model obtained by averaging the last twelve checkpoints" — see
model/configs/gezmu_8k.yaml. Simple parameter averaging (Vaswani et al. 2017's
own practice, and t2t's `avg_checkpoints.py`), not an ensemble: one forward
pass at decode time, same cost as a single checkpoint.

Run (from the project root):
    python -m experiments.average_checkpoints <run_name> [n]

    python -m experiments.average_checkpoints am-en-gezmu-8k       # last 12 (default)
    python -m experiments.average_checkpoints am-en-gezmu-8k 8     # last 8

Writes runs/<run_name>/checkpoints/averaged.pt.
"""
import sys

import torch

from processing.utils.paths import RUNS

DEFAULT_N = 12


def average_state_dicts(paths: list) -> dict:
    """Plain arithmetic mean, tensor by tensor. Every checkpoint must share
    the same architecture (same keys, same shapes) — true by construction
    here since they're all snapshots of one training run."""
    state_dicts = [torch.load(p, map_location="cpu", weights_only=True)["model"] for p in paths]
    keys = state_dicts[0].keys()
    for i, sd in enumerate(state_dicts[1:], 1):
        if sd.keys() != keys:
            sys.exit(f"{paths[i]} has different keys than {paths[0]} — can't average")

    averaged = {}
    for key in keys:
        stacked = torch.stack([sd[key].float() for sd in state_dicts], dim=0)
        averaged[key] = stacked.mean(dim=0).to(state_dicts[0][key].dtype)
    return averaged


def main() -> None:
    if len(sys.argv) < 2:
        sys.exit("usage: python -m experiments.average_checkpoints <run_name> [n]")
    run_name = sys.argv[1]
    n = int(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_N

    avg_dir = RUNS / run_name / "checkpoints" / "avg"
    if not avg_dir.exists():
        sys.exit(f"{avg_dir} doesn't exist — was this run started with training.checkpoint_avg_n set?")

    paths = sorted(avg_dir.glob("step*.pt"), key=lambda p: int(p.stem.removeprefix("step")))
    if len(paths) < n:
        print(f"[average] WARNING: only {len(paths)} snapshots available, asked for {n} — averaging all of them")
    paths = paths[-n:]
    steps = [int(p.stem.removeprefix("step")) for p in paths]
    print(f"[average] averaging {len(paths)} checkpoints, steps {steps[0]}..{steps[-1]}")

    averaged = average_state_dicts(paths)
    out_path = RUNS / run_name / "checkpoints" / "averaged.pt"
    torch.save({"model": averaged, "step": steps[-1], "averaged_from_steps": steps}, out_path)
    print(f"[average] wrote {out_path}")


if __name__ == "__main__":
    main()
