"""model.rescore_indist — score EVERY checkpoint's own validation split through
ONE code path, in one run, and write the results to results/indist.csv.

Run (from the project root):
    python -m model.rescore_indist [--runs NAME[,NAME...]] [--dry-run]

Why this exists
----------------
The in-distribution number is the other half of every entry in EXPERIMENTS.md
(FLORES/MAFAND is model.rescore's job; this is "how well does the model fit the
pool it actually trained on"), but until now it only existed as hand-typed prose.
Same rationale as model.rescore: a number that was computed by hand, once, at an
unknown revision of the code, isn't trustworthy next to ten others computed the
same way. This re-measures every run's validation BLEU/chrF++ through one path.

Decode strategy: greedy only (beam_size=1), for two reasons. First, best.pt was
itself selected on greedy periodic-eval scores (model/train.py forces beam_size=1
during training — see its docstring), so greedy is the comparison that matches
how the checkpoint was chosen. Second, validation splits run 5k-51k sentences and
decoding has no KV cache (model/search.py) — beam 4 on the largest splits here
would take substantially longer for a number nobody selected checkpoints on.

The landmine this works around
-------------------------------
model/data/dataset.py's TranslationDataset resolves its data from
cfg.data.prepared_dir, defaulting to the project's current data/prepared/ if the
key is absent. That default is correct for the current production runs
(base-v5, base-v6, ...) but WRONG for two other categories, both skipped here
rather than silently mismeasured:

  1. Runs with a non-default tokenizer (data.src_tokenizer / data.tgt_tokenizer
     set) but no prepared_dir override. These predate the 8k tokenizer retrain —
     their checkpoints only decode correctly against the 32k vocabulary named in
     their config, but data/prepared/am-en/ today holds ids from the CURRENT 8k
     tokenizer. Loading would not crash (8k-range ids are valid indices into a
     32k embedding table) — it would silently score gibberish. This is
     am-en-base, -baseline, -base-v2, -base-v3, -small, -small-moderate,
     -small-strict: their original prepared caches don't exist anymore
     (data/final/ has been overwritten many times since — see EXPERIMENTS.md),
     so there is no way to score them in-distribution honestly. Skipped, not
     faked.
  2. Runs whose config now names an explicit prepared_dir (narrow, broad,
     gezmu-only, mixed-93k, nllb-93k, base-v4, religious, nllb-146k) are exempt
     from the check above and scored normally — that key exists specifically so
     this script (and anyone else) reads the right data instead of whatever
     currently occupies data/prepared/am-en/.

Output layout
-------------
    results/indist.csv    one row per run: run, split, decode, beam_size, bleu,
                           chrf++, n_sentences, prepared_dir, checkpoint, scored_at
"""
import argparse
import csv
import time
from pathlib import Path

import torch

from model.common import discover_runs, load_for_inference
from model.config import load_config
from model.data.dataset import make_dataloader
from model.evaluate import evaluate_loader

RESULTS = Path("results")
RESULTS_CSV = RESULTS / "indist.csv"

SPLIT = "validation"

FIELDS = ["run", "split", "decode", "beam_size", "bleu", "chrf++",
          "n_sentences", "prepared_dir", "checkpoint", "scored_at"]


def needs_dedicated_prepared_dir(cfg) -> bool:
    """True if this config's tokenizer is non-default but it names no
    prepared_dir — the exact combination that silently mismatches ids against
    vocabulary (see module docstring, landmine 1)."""
    has_custom_tokenizer = bool(cfg.data.get("src_tokenizer") or cfg.data.get("tgt_tokenizer"))
    has_prepared_dir = bool(cfg.data.get("prepared_dir"))
    return has_custom_tokenizer and not has_prepared_dir


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1].strip())
    ap.add_argument("--runs", help="comma-separated run names (default: all discovered)")
    ap.add_argument("--dry-run", action="store_true", help="list what would be scored, then exit")
    args = ap.parse_args()

    jobs = discover_runs()
    if args.runs:
        want = set(args.runs.split(","))
        jobs = [j for j in jobs if j[0] in want]

    plan = []
    for run, cfg_path, ckpt in jobs:
        cfg = load_config(cfg_path)
        if cfg.data.src_lang != "am" or cfg.data.tgt_lang != "en":
            print(f"[skip] {run}: {cfg.data.src_lang}->{cfg.data.tgt_lang}, not this project's "
                  f"reporting direction")
            continue
        if needs_dedicated_prepared_dir(cfg):
            print(f"[skip] {run}: non-default tokenizer, no prepared_dir override — original "
                  f"validation cache is unrecoverable (see module docstring)")
            continue
        plan.append((run, cfg_path, ckpt))

    print(f"\n{len(plan)} run(s) to score on their own {SPLIT!r} split (greedy)")
    for run, cfg_path, ckpt in plan:
        print(f"  {run:22s} {cfg_path}  ({ckpt.name})")
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
        except Exception as e:
            print(f"[skip] {run}: cannot load checkpoint — {type(e).__name__}: {e}")
            continue

        try:
            loader = make_dataloader(SPLIT, cfg, shuffle=False)
        except FileNotFoundError as e:
            print(f"[skip] {run}: no {SPLIT} split found — {e}")
            del model
            torch.cuda.empty_cache()
            continue

        t0 = time.time()
        s = evaluate_loader(model, loader, tok_tgt, cfg, device, beam_size=1, return_all=True)
        dt = time.time() - t0

        writer.writerow({
            "run": run, "split": SPLIT, "decode": "greedy", "beam_size": 1,
            "bleu": f"{s['bleu']:.2f}", "chrf++": f"{s['chrf++']:.2f}", "n_sentences": s["n_sentences"],
            "prepared_dir": str(cfg.data.get("prepared_dir") or "data/prepared"),
            "checkpoint": str(ckpt), "scored_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        })
        fh.flush()
        print(f"[{run}] {SPLIT} greedy  BLEU {s['bleu']:6.2f}  chrF++ {s['chrf++']:6.2f}  "
              f"n={s['n_sentences']}  ({dt:.0f}s)", flush=True)

        del model
        torch.cuda.empty_cache()

    fh.close()
    print(f"\nwrote {RESULTS_CSV}")


if __name__ == "__main__":
    main()
