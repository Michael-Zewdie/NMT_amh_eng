"""model.rescore — re-score EVERY checkpoint on EVERY benchmark through ONE code
path, in one run, and write the results to results/benchmarks.csv.

Run (from the project root):
    python -m model.rescore [--runs NAME[,NAME...]] [--beams 1,4] [--dry-run]

Why this exists
---------------
Every FLORES/MAFAND number in EXPERIMENTS.md came from a stored .hyp.txt written
at a different time by a different revision of the inference code. Those are only
comparable if the code was identical at each write, and it demonstrably was not:
am-en-base-v6's stored hypotheses score 17.35, while the same checkpoint, config,
benchmark file and references reproduce 15.57 today. That gap was chased through
the model (val_loss reproduces the training log to 4 decimals), the decoder, the
tokenizers, the benchmark CSV and the normalizer — all unchanged. The stored
hypotheses are simply not reproducible.

A table whose rows were measured by different code is not a table, so this
re-measures everything at once. Rows written here are mutually comparable by
construction; rows in the legacy table are not.

Output layout
-------------
    results/benchmarks.csv                 one row per (run, benchmark, split, decode)
    runs/<run>/benchmarks/<stem>_<split>.<decode>.hyp.txt

New hypothesis files carry the decode strategy in the NAME, so they can never be
silently compared against the old strategy-less <stem>_<split>.hyp.txt files.
The legacy files are left untouched — they are the evidence for the discrepancy
above, not something to overwrite.

Not covered
-----------
en->am runs are skipped. corpus_scores uses sacrebleu's 13a tokenizer, which is
correct for an English target and wrong for an Amharic one; scoring en-am-base-v1
here would produce a confidently meaningless number. That direction needs
tokenize="none" and its own benchmark orientation.
"""
import argparse
import csv
import time
from pathlib import Path

import torch

from model.common import describe_decoding, discover_runs, load_for_inference
from model.config import load_config
from model.evaluate import corpus_scores
from model.evaluate_benchmark import encode_sources, load_benchmark, translate
from processing.utils.paths import BENCHMARKS, RUNS

RESULTS = Path("results")
RESULTS_CSV = RESULTS / "benchmarks.csv"

# (benchmark stem, split) pairs every am->en run is measured on.
BENCH_SPLITS = [("flores200_am_en", "devtest"), ("mafand_en_amh", "test")]

FIELDS = ["run", "benchmark", "split", "decode", "beam_size", "length_penalty",
          "bleu", "chrf++", "n_sentences", "checkpoint", "scored_at"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1].strip())
    ap.add_argument("--runs", help="comma-separated run names (default: all discovered)")
    ap.add_argument("--beams", default="1,4",
                    help="beam sizes to score at (default 1,4 — greedy and the shipped setting)")
    ap.add_argument("--dry-run", action="store_true", help="list what would be scored, then exit")
    args = ap.parse_args()

    jobs = discover_runs()
    if args.runs:
        want = set(args.runs.split(","))
        jobs = [j for j in jobs if j[0] in want]
    beams = [int(b) for b in args.beams.split(",")]

    plan = []
    for run, cfg_path, ckpt in jobs:
        cfg = load_config(cfg_path)
        if cfg.data.src_lang != "am" or cfg.data.tgt_lang != "en":
            print(f"[skip] {run}: {cfg.data.src_lang}->{cfg.data.tgt_lang}, "
                  f"needs a non-13a tokenizer — see module docstring")
            continue
        plan.append((run, cfg_path, ckpt))

    print(f"\n{len(plan)} run(s) x {len(BENCH_SPLITS)} benchmark(s) x {len(beams)} decode(s) "
          f"= {len(plan) * len(BENCH_SPLITS) * len(beams)} scoring passes")
    for run, cfg_path, _ in plan:
        print(f"  {run:22s} {cfg_path}")
    if args.dry_run:
        return

    RESULTS.mkdir(parents=True, exist_ok=True)
    fresh = not RESULTS_CSV.exists()
    fh = RESULTS_CSV.open("a", newline="", encoding="utf-8")
    writer = csv.DictWriter(fh, fieldnames=FIELDS)
    if fresh:
        writer.writeheader()

    for run, cfg_path, ckpt in plan:
        print(f"\n{'=' * 72}\n=== {run}  ({cfg_path.name})\n{'=' * 72}", flush=True)
        try:
            cfg, model, tok_src, tok_tgt, device = load_for_inference(str(cfg_path), str(ckpt))
        except Exception as e:                       # vocab mismatch on pre-8k runs, etc.
            print(f"[skip] {run}: cannot load checkpoint — {type(e).__name__}: {e}")
            continue

        for stem, split in BENCH_SPLITS:
            path = BENCHMARKS / f"{stem}.csv"
            if not path.exists():
                print(f"[skip] {stem}: {path} missing")
                continue
            df = load_benchmark(path, split)
            ids, truncated = encode_sources(tok_src, df["am"].tolist(), cfg.data.max_src_len)
            refs = df["en"].tolist()

            for beam in beams:
                cfg["inference"]["beam_size"] = beam
                lp = cfg.inference.length_penalty
                hyps = translate(model, ids, tok_tgt, cfg, device)
                assert len(hyps) == len(refs), "row count changed during evaluation"
                s = corpus_scores(hyps, refs)

                tag = "greedy" if beam == 1 else f"beam{beam}"
                out_dir = RUNS / run / "benchmarks"
                out_dir.mkdir(parents=True, exist_ok=True)
                (out_dir / f"{stem}_{split}.{tag}.hyp.txt").write_text(
                    "\n".join(h.replace("\n", " ") for h in hyps), encoding="utf-8")

                writer.writerow({
                    "run": run, "benchmark": stem, "split": split, "decode": tag,
                    "beam_size": beam, "length_penalty": lp if beam > 1 else "",
                    "bleu": f"{s['bleu']:.2f}", "chrf++": f"{s['chrf++']:.2f}",
                    "n_sentences": len(refs), "checkpoint": str(ckpt),
                    "scored_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                })
                fh.flush()
                print(f"[{run}] {stem}:{split} {tag:6s} "
                      f"BLEU {s['bleu']:6.2f}  chrF++ {s['chrf++']:6.2f}"
                      f"{f'  ({truncated} truncated)' if truncated else ''}", flush=True)

        del model
        torch.cuda.empty_cache()

    fh.close()
    print(f"\nwrote {RESULTS_CSV}")
    print("render the table with:  python -m model.results")


if __name__ == "__main__":
    main()
