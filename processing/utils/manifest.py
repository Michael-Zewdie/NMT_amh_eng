"""
processing.utils.manifest — tiny shared helper so every stage that makes a
decision (what data survived which threshold, what hyperparameters a run
used) writes that decision down as JSON next to its output, instead of only
printing it to whatever log happens to be capturing stdout that day.

Born from a concrete failure: reconstructing am-en-narrow's AfriCOMET/LID
cutoffs after the fact required grepping three separate .log files. See
EXPERIMENTS.md and the "experimental process" discussion around 2026-08-06.

Two call sites use this:
  - data manifests:  data/final*/manifest.json   (written by pool.py / domain_breadth.py)
                      copied forward to           data/prepared/<pair>/manifest.json (prepare.py)
  - run manifests:    runs/<name>/manifest.json   (written by model/train.py)
"""
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path


def git_info() -> dict:
    """{"sha": ..., "dirty": ...} for whatever commit produced this artifact.
    Never raises — a missing git binary or a non-repo checkout degrades to
    {"sha": None, "dirty": None} rather than aborting a training/data run."""
    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
        dirty = bool(subprocess.check_output(
            ["git", "status", "--porcelain"], stderr=subprocess.DEVNULL
        ).decode().strip())
        return {"sha": sha, "dirty": dirty}
    except Exception:
        return {"sha": None, "dirty": None}


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def write_manifest(out_dir: Path, data: dict) -> Path:
    """Write data as pretty JSON to out_dir/manifest.json, creating out_dir if
    needed. Returns the path written, so callers can log it."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "manifest.json"
    path.write_text(json.dumps(data, indent=2, default=str) + "\n")
    return path


def read_manifest(path: Path) -> dict | None:
    path = Path(path)
    if not path.exists():
        return None
    return json.loads(path.read_text())
