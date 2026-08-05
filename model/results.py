"""
model.results — the results table, grouped by experiment.

Run (from the project root):
    python -m model.results                  every experiment
    python -m model.results domain-breadth   one experiment
    python -m model.results --legacy         include unverifiable hyp-derived rows
    python -m model.results --list           experiment names only

Where the numbers come from
---------------------------
results/benchmarks.csv, written by model/rescore.py, which scores every
checkpoint on every benchmark in a single run through a single code path. Rows in
that file are mutually comparable by construction. Nothing here is
hand-maintained; results/experiments.yaml supplies only the grouping and the
question each experiment was asking.

Why not read the .hyp.txt files
-------------------------------
That is what this module used to do, and it was wrong. A .hyp.txt records neither
the decoding strategy nor the revision of the code that produced it, so files
written weeks apart were being compared as if they were commensurable.
am-en-base-v6 is the proof: its stored hypotheses score 17.35, while the same
checkpoint, config, benchmark file and references reproduce 15.57 today under
every decode setting from beam 1 to beam 6. Those legacy numbers are still
printed under --legacy, clearly separated, because deleting the evidence would
hide the discrepancy rather than resolve it.

The in-distribution column
--------------------------
Read from each run's TensorBoard scalars, and NOT comparable across runs: it is a
periodic eval-*subset* peak whose size differs per run, measured on a validation
split drawn from that run's own training pool. Narrow the pool and the exam
narrows with it — am-en-narrow scores 26.63 there and 0.68 on FLORES. The column
is printed with its subset size attached as a standing reminder.
"""
import argparse
import glob
from pathlib import Path

import pandas as pd
import yaml

from processing.utils.paths import BENCHMARKS, ROOT, RUNS

RESULTS = ROOT / "results"
RESULTS_CSV = RESULTS / "benchmarks.csv"
EXPERIMENTS = RESULTS / "experiments.yaml"

BENCH_COLS = [("flores200_am_en", "devtest", "FLORES devtest"),
              ("mafand_en_amh", "test", "MAFAND test")]
SPLIT_SUFFIXES = ("devtest", "dev", "test", "train")


def load_scores() -> pd.DataFrame:
    if not RESULTS_CSV.exists():
        raise SystemExit(f"{RESULTS_CSV} not found — run `python -m model.rescore` first")
    return pd.read_csv(RESULTS_CSV, dtype=str)


def subset_peak(run_dir: Path) -> tuple[float, int] | None:
    """Best periodic eval-subset BLEU and the step it happened at."""
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    except ImportError:
        return None
    points: dict[int, float] = {}
    for f in sorted(glob.glob(str(run_dir / "tensorboard" / "events*"))):
        ea = EventAccumulator(f)
        ea.Reload()
        if "val/bleu" in ea.Tags()["scalars"]:
            for x in ea.Scalars("val/bleu"):
                points[x.step] = x.value
    if not points:
        return None
    step, best = max(points.items(), key=lambda t: t[1])
    return best, step


def subset_size(run_name: str) -> int | None:
    for cfg in glob.glob(str(ROOT / "model" / "configs" / "**" / "*.yaml"), recursive=True):
        try:
            raw = yaml.safe_load(Path(cfg).read_text())
        except Exception:
            continue
        if raw and raw.get("run_name") == run_name:
            return raw.get("training", {}).get("eval_subset_size")
    return None


def legacy_scores(run_name: str) -> dict[str, str]:
    """Score the strategy-less <stem>_<split>.hyp.txt files still on disk."""
    from model.evaluate import corpus_scores
    out: dict[str, str] = {}
    for hyp in sorted((RUNS / run_name).glob("benchmarks/*.hyp.txt")):
        stem = hyp.stem.replace(".hyp", "")
        if "." in stem:                      # a re-scored file, not a legacy one
            continue
        for s in SPLIT_SUFFIXES:
            if stem.endswith(f"_{s}"):
                bench, split = stem[: -len(s) - 1], s
                break
        else:
            continue
        path = BENCHMARKS / f"{bench}.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path, dtype=str).fillna("")
        if split and "split" in df.columns:
            df = df[df["split"] == split]
        refs = df["en"].tolist()
        hyps = hyp.read_text(encoding="utf-8").splitlines()
        if len(hyps) != len(refs):
            continue
        sc = corpus_scores(hyps, refs)
        out[bench] = f"{sc['bleu']:5.2f}/{sc['chrf++']:5.2f}"
    return out


def cell(df: pd.DataFrame, run: str, bench: str, split: str, decode: str) -> str:
    r = df[(df.run == run) & (df.benchmark == bench) & (df.split == split) & (df.decode == decode)]
    if r.empty:
        return "—"
    return f"{float(r.iloc[-1].bleu):5.2f}/{float(r.iloc[-1]['chrf++']):5.2f}"


def print_experiment(name: str, spec: dict, scores: pd.DataFrame, show_legacy: bool) -> None:
    print(f"\n\033[1m{name}\033[0m")
    for key in ("question", "hypothesis", "verdict", "note"):
        if spec.get(key):
            body = " ".join(str(spec[key]).split())
            label = f"{key}:"
            print(f"  {label:12s}{body}")
    print()

    hdr = (f"  {'run':22s} {'in-dist (subset)':>18s} "
           + " ".join(f"{lbl + ' greedy':>20s} {lbl + ' beam4':>20s}" for _, _, lbl in BENCH_COLS))
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for run in spec.get("runs", []):
        ind = "—"
        peak = subset_peak(RUNS / run) if (RUNS / run).exists() else None
        if peak:
            n = subset_size(run)
            ind = f"{peak[0]:.2f}" + (f" @{n}ex" if n else "")
        cells = []
        for bench, split, _ in BENCH_COLS:
            cells.append(f"{cell(scores, run, bench, split, 'greedy'):>20s}")
            cells.append(f"{cell(scores, run, bench, split, 'beam4'):>20s}")
        print(f"  {run:22s} {ind:>18s} " + " ".join(cells))

        if show_legacy:
            leg = legacy_scores(run)
            if leg:
                pretty = "  ".join(f"{b.split('_')[0]} {v}" for b, v in leg.items())
                print(f"  {'':22s} {'legacy hyp:':>18s} {pretty}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1].strip())
    ap.add_argument("experiment", nargs="?", help="one experiment name (default: all)")
    ap.add_argument("--legacy", action="store_true",
                    help="also score the old strategy-less .hyp.txt files (not comparable)")
    ap.add_argument("--list", action="store_true", help="list experiment names and exit")
    args = ap.parse_args()

    spec = yaml.safe_load(EXPERIMENTS.read_text())
    if args.list:
        for k, v in spec.items():
            print(f"{k:24s} {len(v.get('runs', []))} run(s)")
        return

    scores = load_scores()
    wanted = [args.experiment] if args.experiment else list(spec)
    for name in wanted:
        if name not in spec:
            raise SystemExit(f"unknown experiment {name!r} — try --list")
        print_experiment(name, spec[name], scores, args.legacy)

    print("\nBLEU/chrF++ on fixed benchmark files — comparable across every re-scored run.")
    print("in-dist is a per-run eval SUBSET peak on that run's own pool -> NOT comparable.")
    print(f"source: {RESULTS_CSV.relative_to(ROOT)} (written by model/rescore.py)")


if __name__ == "__main__":
    main()
