"""
experiments.report — regenerate the numbers/params table from structured
sources (runs/*/manifest.json + results/benchmarks.csv + results/indist.csv)
instead of maintaining it by hand in EXPERIMENTS.md, so it can't silently drift
from what actually ran (see processing.utils.manifest's docstring for the
motivating incident).

Run (from the project root): python -m experiments.report
Writes experiments/report.md — a generated table, safe to regenerate any time
a run finishes or gets rescored. EXPERIMENTS.md's prose sections (the "why"
behind each experiment, the root-cause investigations) are NOT reproduced here
and stay hand-written; this covers only what a machine can state as fact.
"""
import json

import pandas as pd

from processing.utils.paths import ROOT, RUNS

BENCHMARKS_CSV = ROOT / "results" / "benchmarks.csv"
INDIST_CSV = ROOT / "results" / "indist.csv"
OUT = ROOT / "experiments" / "report.md"


def load_manifests() -> dict[str, dict]:
    out = {}
    for run_dir in sorted(RUNS.iterdir()):
        if not run_dir.is_dir():
            continue
        mpath = run_dir / "manifest.json"
        if mpath.exists():
            out[run_dir.name] = json.loads(mpath.read_text())
    return out


def best_benchmark_rows(bench: pd.DataFrame) -> pd.DataFrame:
    """One row per (run, benchmark), preferring beam4 over greedy."""
    bench = bench.sort_values("decode", ascending=True)  # "beam4" < "greedy" lexically
    return bench.drop_duplicates(subset=["run", "benchmark"], keep="first")


def _data_provenance(manifest: dict) -> dict | None:
    """Backfilled runs (experiments/backfill_manifests.py) store an
    already-summarized 'data_provenance' dict at the manifest's top level.
    Live runs (model/train.py) instead carry the full pool/build manifest
    under 'data_manifest' (whatever processing.utils.pool or a bespoke
    data-build script like experiments/gezmu_nmt8k_repro.py wrote to
    data/final/manifest.json) — different producers, different shapes, so
    normalize both into the {"cutoffs"/"filtering", "train_pairs"/"counts"}
    view this module reads, rather than silently returning "—" for every
    run that was never backfilled."""
    dp = manifest.get("data_provenance")
    if dp is not None:
        return dp
    return manifest.get("data_manifest") or None


def cutoffs_summary(manifest: dict) -> str:
    dp = _data_provenance(manifest)
    if dp is None:
        return "—"
    if "filtering" in dp:  # e.g. experiments/gezmu_nmt8k_repro.py's "none — ..." string
        return dp["filtering"]
    c = dp.get("cutoffs")
    if c is None:
        return "unrecoverable"
    if isinstance(c, str):
        return c
    if c.get("tiered") or ("curated" in c and "mined" in c):
        curated = c["curated"]
        mined = c["mined"]
        return (f"tiered: curated labse>{curated['labse_score']}/africomet>{curated['africomet_score']}, "
                f"mined labse>{mined['labse_score']}/africomet>{mined['africomet_score']}")
    if "africomet_score_uniform" in c:
        return f"uniform africomet>{c['africomet_score_uniform']}"
    parts = [f"{k}>{v}" for k, v in c.items() if v is not None and not isinstance(v, dict)]
    return ", ".join(parts) if parts else "—"


def train_pairs_summary(manifest: dict) -> str:
    dp = _data_provenance(manifest)
    if dp is None:
        return "—"
    if "counts" in dp:  # experiments/gezmu_nmt8k_repro.py's {"train": {"before":, "after":}, ...}
        n = dp["counts"].get("train", {}).get("after")
        return f"{n:,}" if isinstance(n, int) else "unrecoverable"
    n = dp.get("train_pairs")
    return f"{n:,}" if isinstance(n, int) else "unrecoverable"


def config_summary(manifest: dict) -> str:
    cfg = manifest.get("config") or {}
    model = cfg.get("model", {})
    training = cfg.get("training", {})
    d = model.get("d_model")
    layers = model.get("n_encoder_layers")
    steps = training.get("max_steps")
    if d is None or steps is None:
        return "—"
    return f"d={d} {layers}+{layers}L, {steps:,} steps"


def main() -> None:
    manifests = load_manifests()
    bench = best_benchmark_rows(pd.read_csv(BENCHMARKS_CSV)) if BENCHMARKS_CSV.exists() else pd.DataFrame()
    indist = pd.read_csv(INDIST_CSV) if INDIST_CSV.exists() else pd.DataFrame()

    rows = []
    for run_name, manifest in manifests.items():
        # Results columns go first — the wide provenance text (data cutoffs
        # especially) pushes later columns past the fold in narrow markdown
        # previews, so keep what people actually scan for up front.
        row = {"run": run_name}

        run_indist = indist[indist["run"] == run_name] if not indist.empty else pd.DataFrame()
        if not run_indist.empty:
            r = run_indist.iloc[0]
            row["in-distribution BLEU/chrF++"] = f"{r['bleu']:.2f} / {r['chrf++']:.2f} ({r['decode']})"
        else:
            row["in-distribution BLEU/chrF++"] = "—"

        run_bench = bench[bench["run"] == run_name] if not bench.empty else pd.DataFrame()
        for _, r in run_bench.iterrows():
            key = r["benchmark"].replace("flores200_am_en", "FLORES").replace("mafand_en_amh", "MAFAND")
            row[f"{key} BLEU/chrF++"] = f"{r['bleu']:.2f} / {r['chrf++']:.2f} ({r['decode']})"

        row["architecture/steps"] = config_summary(manifest)
        row["data cutoffs"] = cutoffs_summary(manifest)
        row["train pairs"] = train_pairs_summary(manifest)
        row["backfilled"] = manifest.get("backfilled", False)
        rows.append(row)

    df = pd.DataFrame(rows).fillna("—")
    lines = ["# Generated experiment report", "",
             f"Regenerate: `python -m experiments.report`. Source: `runs/*/manifest.json` + "
             f"`{BENCHMARKS_CSV.relative_to(ROOT)}` + `{INDIST_CSV.relative_to(ROOT)}`. "
             f"Narrative/rationale lives in EXPERIMENTS.md, not here — this file is numbers only, "
             f"so it can't drift from what actually ran. In-distribution rows are greedy-decoded "
             f"(model/rescore_indist.py); a run missing here either hasn't been scored yet or has "
             f"no recoverable validation cache (see that script's docstring) — not the same thing, "
             f"check the console output of the run that (didn't) produce its row.",
             "", df.to_markdown(index=False)]
    OUT.write_text("\n".join(lines) + "\n")
    print(f"[report] wrote {OUT} ({len(df)} runs)")


if __name__ == "__main__":
    main()
