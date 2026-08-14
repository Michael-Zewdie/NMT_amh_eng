"""
experiments.domain_dist_10k — does the DOMAIN DISTRIBUTION of a small training
corpus trade in-distribution BLEU against out-of-distribution generalization?

Run (from the project root):
    python -m experiments.domain_dist_10k build     # build both data arms
    python -m experiments.domain_dist_10k train     # 2 arms x 5 seeds, sequential
    python -m experiments.domain_dist_10k eval      # score every finished run
    python -m experiments.domain_dist_10k report    # mean +/- std table

Hypothesis under test
---------------------
    single-domain training  -> HIGHER in-distribution BLEU, WORSE OOD (FLORES/MAFAND)
    multi-domain  training  -> LOWER  in-distribution BLEU, BETTER OOD

Two arms, matched at N=10,000 pooled pairs, differing only in domain distribution:

    HEALTH   AfriDocMT health — one domain, human-translated medical documents
    DIVERSE  NLLB mined bitext, top 10,000 by africomet_score — 1,116 distinct
             source web domains at this cut, head concentration essentially
             unchanged from the full mined pool (top-5 share 54.5% vs 56.3%)

Deliberate asymmetry, stated rather than hidden
-----------------------------------------------
The HEALTH arm is taken UNFILTERED — all 10,000 raw pairs, no AfriCOMET/LaBSE
quality floor — because AfriDocMT is human-translated and a QE model's opinion
of it is not evidence. The DIVERSE arm is quality-selected by construction (the
top 10,000 of 2.36M mined pairs). So the arms differ in provenance quality as
well as domain breadth. That is inherent to the comparison being asked for
(a curated human corpus vs. the best obtainable mined corpus), not an oversight:
it means a HEALTH win is confounded, while a DIVERSE win on OOD is the stronger
result because it happens *despite* the mined arm's noisier provenance.

What is NOT skipped even in the "unfiltered" arm, and why
---------------------------------------------------------
  * dropna            — a null side cannot be tokenized (0 rows affected here)
  * decontamination   — a FLORES/MAFAND sentence sitting in train destroys the
                        very OOD number this experiment exists to measure. It
                        caught 3 real rows in the DIVERSE arm, one of them a
                        MAFAND *test* sentence. Never optional.
  * am-grouped split  — duplicates are KEPT (no rows deleted), but every row
                        sharing an Amharic side lands in the same split, so a
                        pair cannot appear in both train and test. Leakage
                        control, not quality filtering.

Why 5 seeds: ~12M parameters against ~7.9k training pairs is a high-variance
regime. A single run's score is not a measurement; the seed distribution is.

Why the 2x2 eval: each arm's test split has its own intrinsic difficulty, so
"in-distribution BLEU" is NOT comparable between arms as a bare number. Scoring
every model on BOTH test splits turns that ambiguity into the actual quantity of
interest — how far each training distribution transfers.

Configs are written to runs/<run_name>/config.yaml (gitignored, co-located with
the checkpoints they produced) rather than into model/configs/, so this
experiment adds no tracked files beyond this module.

Outputs:
    data/final_{health10k,diverse10k}/, data/prepared_{health10k,diverse10k}/am-en/
    runs/dd-{arm}-s{seed}/{config.yaml,manifest.json,train.log,checkpoints/}
    experiments/domain_dist_10k_results.json
"""
import argparse
import json
import pickle
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import torch
import yaml

from model.common import (BOS_ID, EOS_ID, corpus_scores, load_for_inference,
                          load_tokenizer)
from process import pool as pool_mod
from process.utils.paths import BENCHMARKS, CSV_RAW, DATA, PROCESSED, ROOT, RUNS

N = 10_000
SEED = 42
MAX_LEN = 150
SPLITS = ["validation", "test", "train"]
RATIOS = (0.8, 0.1, 0.1)
LID_FLOOR = 0.90

ARMS = ["health10k", "diverse10k"]
SEEDS = [1, 2, 3, 4, 5]
RESULTS_PATH = ROOT / "experiments" / "domain_dist_10k_results.json"


# ----------------------------------------------------------------- build ----
def load_health() -> pd.DataFrame:
    """All 10,000 AfriDocMT health pairs, unfiltered — see module docstring."""
    df = pd.read_csv(CSV_RAW / "afridoc_health.csv", dtype=str)[["am", "en"]].dropna()
    print(f"[health] {len(df):,} raw pairs (no quality filtering applied)")
    return df


def load_diverse() -> pd.DataFrame:
    """Top N NLLB pairs by africomet_score, above the LID floor.

    LID is a language-identity check, not a quality score: rows below it are not
    Amharic/English at all, so they cannot be part of a domain-breadth question.
    """
    df = pd.read_csv(PROCESSED / "nllb.csv", dtype=str)
    for c in ("africomet_score", "source_lid", "target_lid"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    ok = df[(df.source_lid > LID_FLOOR) & (df.target_lid > LID_FLOOR)].dropna(
        subset=["am", "en", "africomet_score"])
    top = ok.nlargest(N, "africomet_score")
    print(f"[diverse] {len(ok):,} rows above LID floor -> top {len(top):,} by africomet "
          f"(range {top.africomet_score.min():.4f}–{top.africomet_score.max():.4f})")
    return top[["am", "en"]].reset_index(drop=True)


def build_arm(name: str, df: pd.DataFrame) -> dict:
    """Pool -> decontaminate -> match to N -> split -> tokenize. Returns a manifest."""
    print(f"\n=== building arm: {name} ===")
    pooled = pool_mod.decontaminate(pool_mod.pool([df], seed=SEED))
    if len(pooled) < N:
        print(f"[{name}] note: {len(pooled):,} pairs after dedup+decontam (target {N:,})")
    pooled = pooled.head(N).reset_index(drop=True)

    final_out = DATA / f"final_{name}"
    final_out.mkdir(parents=True, exist_ok=True)
    original, pool_mod.FINAL = pool_mod.FINAL, final_out
    try:
        pool_mod.split(pooled, ratios=RATIOS, seed=SEED)
    finally:
        pool_mod.FINAL = original

    src_tok, tgt_tok = load_tokenizer("am"), load_tokenizer("en")
    out_dir = DATA / f"prepared_{name}" / "am-en"
    out_dir.mkdir(parents=True, exist_ok=True)
    sizes = {}
    for sp in SPLITS:
        d = pd.read_csv(final_out / f"{sp}.csv", usecols=["am", "en"], dtype=str).dropna()
        s = [[BOS_ID, *e.ids, EOS_ID] for e in src_tok.encode_batch(d.am.tolist())]
        t = [[BOS_ID, *e.ids, EOS_ID] for e in tgt_tok.encode_batch(d.en.tolist())]
        keep = [i for i, (a, b) in enumerate(zip(s, t)) if len(a) <= MAX_LEN and len(b) <= MAX_LEN]
        with open(out_dir / f"{sp}.pkl", "wb") as f:
            pickle.dump({"src": [s[i] for i in keep], "tgt": [t[i] for i in keep]}, f)
        print(f"[{name}] {sp}: kept {len(keep):,} of {len(d):,}")
        sizes[sp] = len(keep)

    manifest = {
        "arm": name, "experiment": "domain_dist_10k", "n_pooled_used": len(pooled),
        "built_at": datetime.now(timezone.utc).isoformat(),
        "split_ratios": list(RATIOS), "prepared_sizes": sizes, "seed": SEED,
        "tokenizer_note": "reused data/tokenizer/{am,en} as-is (fit on Gezmu's train split) "
                          "rather than retraining per arm — holds vocabulary fixed across arms",
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def cmd_build(_args) -> None:
    ms = {"health10k": build_arm("health10k", load_health()),
          "diverse10k": build_arm("diverse10k", load_diverse())}
    print("\n=== matched arm sizes ===")
    for name, m in ms.items():
        s = m["prepared_sizes"]
        print(f"  {name:12s} train {s['train']:>6,}  val {s['validation']:>5,}  test {s['test']:>5,}")


# ----------------------------------------------------------------- train ----
def make_config(arm: str, seed: int) -> dict:
    """Identical across arms except data.prepared_dir; across seeds except training.seed."""
    return {
        "run_name": f"dd-{arm}-s{seed}",
        "model": {"d_model": 256, "n_heads": 4, "n_encoder_layers": 4, "n_decoder_layers": 4,
                  "d_ff": 1024, "dropout": 0.3, "max_len": 150, "tie_output_projection": True},
        "data": {"src_lang": "am", "tgt_lang": "en", "max_src_len": 128, "max_tgt_len": 128,
                 "batch_size": 128, "num_workers": 8, "prepared_dir": f"data/prepared_{arm}"},
        "training": {"warmup_steps": 500, "label_smoothing": 0.1, "grad_clip_norm": 1.0,
                     "accum_steps": 1, "max_steps": 10000, "amp_dtype": "bf16", "seed": seed,
                     "save_every_steps": 200, "eval_every_steps": 200, "eval_subset_size": 500,
                     "log_every_steps": 100, "resume_from": None},
        # Periodic in-training eval forces greedy regardless (see model.training.train);
        # eval reads its beam from the CLI, so this value is a default, not a commitment.
        "inference": {"beam_size": 1, "max_len": 150, "length_penalty": 0.6},
    }


def run_dir_for(arm: str, seed: int) -> Path:
    return RUNS / f"dd-{arm}-s{seed}"


def run_one(arm: str, seed: int, force: bool = False) -> bool:
    run_dir = run_dir_for(arm, seed)
    manifest_path = run_dir / "manifest.json"
    run_dir.mkdir(parents=True, exist_ok=True)

    # Written before the skip check, not after: make_config is deterministic, and a
    # run that finished under an earlier invocation still needs its config on disk
    # for `eval` to load the checkpoint against.
    cfg = make_config(arm, seed)
    config_path = run_dir / "config.yaml"
    config_path.write_text(yaml.safe_dump(cfg, sort_keys=False))

    if not force and (run_dir / "checkpoints" / "best.pt").exists() and manifest_path.exists():
        if json.loads(manifest_path.read_text()).get("returncode") == 0:
            print(f"[run] dd-{arm}-s{seed} already finished — skipping", flush=True)
            return True

    log_path = run_dir / "train.log"
    started = time.time()
    print(f"[run] dd-{arm}-s{seed} starting -> {log_path}", flush=True)

    with open(log_path, "w") as log:
        proc = subprocess.run([sys.executable, "-m", "model.training.train", str(config_path)],
                              stdout=log, stderr=subprocess.STDOUT, cwd=ROOT)
    elapsed = time.time() - started

    best = None
    for line in log_path.read_text().splitlines():
        if "new best val_bleu" in line:
            best = float(line.split("new best val_bleu")[1].split()[0])
    manifest_path.write_text(json.dumps({
        "run_name": f"dd-{arm}-s{seed}", "experiment": "domain_dist_10k", "arm": arm, "seed": seed,
        "config": cfg, "finished_at": datetime.now(timezone.utc).isoformat(),
        "wall_clock_seconds": round(elapsed, 1), "returncode": proc.returncode,
        "best_val_bleu": best,
    }, indent=2))
    ok = proc.returncode == 0
    print(f"[run] dd-{arm}-s{seed} {'ok' if ok else 'FAILED'} in {elapsed/60:.1f} min, "
          f"best val_bleu={best}", flush=True)
    return ok


def cmd_train(args) -> None:
    jobs = [(a, s) for a in ARMS for s in SEEDS]
    if args.arm:
        jobs = [j for j in jobs if j[0] == args.arm]
    if args.seed:
        jobs = [j for j in jobs if j[1] == args.seed]
    t0 = time.time()
    failed = [f"{a}-s{s}" for a, s in jobs if not run_one(a, s, force=args.force)]
    print(f"\n=== {len(jobs)-len(failed)}/{len(jobs)} runs ok in {(time.time()-t0)/60:.1f} min ===")
    if failed:
        sys.exit(f"failed runs: {', '.join(failed)}")


# ------------------------------------------------------------------ eval ----
def score_benchmark(model, cfg, src_tok, tgt_tok, device, csv_path: Path, split: str) -> dict:
    """BLEU/chrF++ on a fixed benchmark file (FLORES/MAFAND)."""
    from model.evaluate.evaluate_OOD import encode_sources, load_benchmark, translate
    df = load_benchmark(csv_path, split)
    src_ids, _ = encode_sources(src_tok, df["am"].tolist(), cfg.data.max_src_len)
    hyps = translate(model, src_ids, tgt_tok, cfg, device)
    return corpus_scores(hyps, df["en"].tolist())


def score_in_dist(model, cfg, device, arm: str, beam: int) -> float:
    """Corpus BLEU on `arm`'s held-out test split, whichever arm the model came from.

    Repoints cfg.data.prepared_dir at the requested arm so one model can be scored
    against both arms' test splits — the 2x2 that makes in-distribution numbers
    mean something (see module docstring).

    Delegates to model.evaluate.evaluate_in_dist.evaluate_split rather than
    re-walking the loader here, so in-distribution BLEU is computed by exactly the
    same code path as every other in-distribution number in this project. That
    returns BLEU only; chrF++ is reported for the fixed benchmarks below, where
    cross-system comparability actually matters.
    """
    from model.evaluate.evaluate_in_dist import evaluate_split
    original = cfg.data.get("prepared_dir")
    cfg.data["prepared_dir"] = f"data/prepared_{arm}"
    try:
        return evaluate_split(model, "test", cfg, device, beam_size=beam)
    finally:
        cfg.data["prepared_dir"] = original


def cmd_eval(args) -> None:
    results = json.loads(RESULTS_PATH.read_text()) if RESULTS_PATH.exists() and not args.force else {}
    for arm in ARMS:
        for seed in SEEDS:
            key = f"dd-{arm}-s{seed}"
            run_dir = run_dir_for(arm, seed)
            ckpt = run_dir / "checkpoints" / "best.pt"
            manifest = run_dir / "manifest.json"
            # best.pt existing is NOT enough: training rewrites it at every eval
            # interval, so a run still in progress has one sitting on disk. Scoring
            # that would silently cache a half-trained model as a final result.
            # The manifest is only written after the process exits, so it is the
            # signal that this run is actually done.
            if not ckpt.exists() or not manifest.exists():
                print(f"[eval] {key}: not finished yet — skipping")
                continue
            if json.loads(manifest.read_text()).get("returncode") != 0:
                print(f"[eval] {key}: training did not exit cleanly — skipping")
                continue
            if key in results:
                print(f"[eval] {key}: already scored — skipping")
                continue
            print(f"[eval] {key} ...", flush=True)
            cfg, model, src_tok, tgt_tok, device = load_for_inference(run_dir / "config.yaml", ckpt)
            cfg.inference["beam_size"] = args.beam
            r = {"arm": arm, "seed": seed, "beam": args.beam}
            for target in ARMS:
                label = "in_dist_own" if target == arm else "in_dist_cross"
                r[label] = {"bleu": score_in_dist(model, cfg, device, target, args.beam),
                            "test_split_of": target}
            r["flores"] = score_benchmark(model, cfg, src_tok, tgt_tok, device,
                                          BENCHMARKS / "flores200_am_en.csv", "devtest")
            r["mafand"] = score_benchmark(model, cfg, src_tok, tgt_tok, device,
                                          BENCHMARKS / "mafand_en_amh.csv", "test")
            results[key] = r
            RESULTS_PATH.write_text(json.dumps(results, indent=2))
            print(f"[eval] {key}: own {r['in_dist_own']['bleu']:.2f} | "
                  f"cross {r['in_dist_cross']['bleu']:.2f} | "
                  f"FLORES {r['flores']['bleu']:.2f} | MAFAND {r['mafand']['bleu']:.2f}", flush=True)
            del model
            torch.cuda.empty_cache()
    print(f"\n[eval] {len(results)} runs scored -> {RESULTS_PATH}")


# ---------------------------------------------------------------- report ----
def cmd_report(_args) -> None:
    if not RESULTS_PATH.exists():
        sys.exit("no results yet — run `eval` first")
    rows = list(json.loads(RESULTS_PATH.read_text()).values())
    if not rows:
        sys.exit("results file is empty")
    metrics = [("in_dist_own", "in-dist (own test)"), ("in_dist_cross", "cross-domain test"),
               ("flores", "FLORES devtest"), ("mafand", "MAFAND test")]
    print(f"\ndomain distribution vs generalization — {len(rows)} runs, "
          f"beam {rows[0]['beam']}, mean +/- std over seeds\n")
    header = f"{'arm':<12}{'n':>3}  " + "".join(f"{lbl:>24}" for _, lbl in metrics)
    print(header)
    print("-" * len(header))
    for arm in ARMS:
        sub = [r for r in rows if r["arm"] == arm]
        if not sub:
            continue
        cells = ""
        for key, _ in metrics:
            vals = pd.Series([r[key]["bleu"] for r in sub])
            cells += f"{vals.mean():>17.2f} +/-{vals.std():>4.2f}" if len(vals) > 1 \
                else f"{vals.mean():>17.2f}      "
        print(f"{arm:<12}{len(sub):>3}  {cells}")
    print("\nper-seed BLEU")
    for arm in ARMS:
        for r in sorted([r for r in rows if r["arm"] == arm], key=lambda x: x["seed"]):
            print(f"  {arm:<12} seed {r['seed']}  own {r['in_dist_own']['bleu']:>6.2f}  "
                  f"cross {r['in_dist_cross']['bleu']:>6.2f}  "
                  f"FLORES {r['flores']['bleu']:>6.2f}  MAFAND {r['mafand']['bleu']:>6.2f}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("build").set_defaults(func=cmd_build)
    t = sub.add_parser("train")
    t.add_argument("--arm", choices=ARMS)
    t.add_argument("--seed", type=int)
    t.add_argument("--force", action="store_true", help="retrain even if a run already finished")
    t.set_defaults(func=cmd_train)
    e = sub.add_parser("eval")
    e.add_argument("--beam", type=int, default=1,
                   help="1 = greedy (default), matching every other number in EXPERIMENTS.md")
    e.add_argument("--force", action="store_true", help="rescore runs already in the results file")
    e.set_defaults(func=cmd_eval)
    sub.add_parser("report").set_defaults(func=cmd_report)
    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
