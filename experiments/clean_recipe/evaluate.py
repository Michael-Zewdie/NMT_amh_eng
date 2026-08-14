"""Scoring — in-distribution plus the two fixed benchmarks, each scored twice.

Case is handled exactly as experiments.domain_breadth.evaluate handles it, and
for the same reason: this arm trains on lowercased English and can never emit
case, so a cased score measures missing capital letters rather than translation
quality. Both scorings are always computed — scoring is free next to decoding.

    bleu      cased references, lowercase output — the floor. A PERFECT but
              lowercase translation caps at ~80.1 BLEU on FLORES and ~71.6 on
              MAFAND, so read this as a casing measurement, not a quality one.

    bleu_ci   both sides lowercased — the number that counts here, and the one
              directly comparable to am-en-narrow's and am-en-broad's existing
              `_ci` results (FLORES 6.22 / 12.46, MAFAND 4.23 / 5.95).

The in-distribution column is measured on this experiment's OWN test split, which
is a different corpus from v4's — quran and ccaligned are gone and NLLB is gated
harder — so it is NOT comparable to v4's 26.39. FLORES and MAFAND are fixed
held-out sets and carry no such caveat; they are the columns to trust for any
comparison against another model.
"""
import json

import pandas as pd
import sacrebleu

from experiments.clean_recipe.arms import ARMS, read_split
from experiments.clean_recipe.corpus import RESULTS
from model.common import load_for_inference
from model.tokenize.preprocess import detok_en, translit_am
from process.utils.paths import BENCHMARKS, RUNS

BENCHMARK_SETS = [("flores", "flores200_am_en", "devtest"),
                  ("mafand", "mafand_en_amh", "test")]


def score_arm(arm_name: str, checkpoint: str) -> dict:
    from model.evaluate.evaluate_OOD import encode_sources, translate

    arm = ARMS[arm_name]
    run_dir = RUNS / arm.run
    cfg, model, src_tok, tgt_tok, device = load_for_inference(
        run_dir / "config.yaml", run_dir / "checkpoints" / checkpoint)
    out = {}

    def score(label: str, am: list[str], refs: list[str]) -> None:
        ids, _ = encode_sources(src_tok, [translit_am(s) for s in am], cfg.data.max_src_len)
        hyps = [detok_en(h) for h in translate(model, ids, tgt_tok, cfg, device)]
        lo_h, lo_r = [h.lower() for h in hyps], [r.lower() for r in refs]
        out[label] = {
            "bleu": sacrebleu.corpus_bleu(hyps, [refs]).score,
            "chrf++": sacrebleu.corpus_chrf(hyps, [refs], word_order=2).score,
            "bleu_ci": sacrebleu.corpus_bleu(lo_h, [lo_r]).score,
            "chrf++_ci": sacrebleu.corpus_chrf(lo_h, [lo_r], word_order=2).score,
        }
        o = out[label]
        print(f"  {arm.run:20s} {label:14s} {o['bleu']:6.2f} BLEU / {o['chrf++']:6.2f} chrF++"
              f"   |  case-insensitive {o['bleu_ci']:6.2f} / {o['chrf++_ci']:6.2f}", flush=True)

    am, en = read_split("test")
    score("in_dist_test", am, en)
    for name, f, sp in BENCHMARK_SETS:
        df = pd.read_csv(BENCHMARKS / f"{f}.csv", dtype=str).fillna("")
        df = df[df["split"] == sp].reset_index(drop=True)
        score(name, df["am"].tolist(), df["en"].tolist())
    return out


def cmd_eval(args) -> None:
    results = {}
    for name, arm in ARMS.items():
        if not (RUNS / arm.run / "checkpoints" / args.checkpoint).exists():
            print(f"[eval] {arm.run}: no {args.checkpoint} — skipping")
            continue
        results[arm.run] = score_arm(name, args.checkpoint)
    if not results:
        raise SystemExit("[eval] nothing to score")
    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    RESULTS.write_text(json.dumps(results, indent=2))

    print(f"\n{'':24}{'in-dist*':>11}{'FLORES':>10}{'MAFAND':>10}")
    print("-" * 55)
    for run, r in results.items():
        for key, lbl in [("bleu", "cased (floor)"), ("bleu_ci", "case-INSENSITIVE")]:
            print(f"{lbl:<24}{r['in_dist_test'][key]:>11.2f}"
                  f"{r['flores'][key]:>10.2f}{r['mafand'][key]:>10.2f}")
    print("\ncase-insensitive reference points:  am-en-broad  FLORES 12.46  MAFAND 5.95")
    print("                                    am-en-narrow FLORES  6.22  MAFAND 4.23")
    print("* in-dist is this corpus's own test split — NOT comparable to v4's 26.39.")
    print(f"\n-> {RESULTS}")
