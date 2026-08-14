"""Scoring, plus the diversity check that guards its main known weakness.

Known limitation, stated rather than hidden: in-distribution BLEU is NOT
directly comparable across the three arms, because each produces a genuinely
different test.csv (own-test-set BLEU already burned this project once, see
EXPERIMENTS.md's diverse10k/health10k writeup — "diverse's higher own-test
BLEU is not evidence it is the better model"). Mitigation: `eval` also runs
process.dist.diversity on each arm's own train/test split, to check whether a
BLEU gap tracks a real reduction in train/test distributional distance or
just a test set that landed easier.

FLORES and MAFAND are shared across arms and carry no such caveat — they are
the same held-out sentences for every arm, so those two columns are the ones
to trust for a straight between-arm comparison.
"""
import json

import pandas as pd
import sacrebleu

from experiments.stratification.paths import RESULTS, final_dir, run_name
from experiments.stratification.splits import ARMS
from model.common import load_for_inference
from model.tokenize.preprocess import detok_en, translit_am
from process.utils.paths import BENCHMARKS, RUNS


def score_arm(arm: str, checkpoint: str) -> dict:
    from model.evaluate.evaluate_OOD import encode_sources, translate

    run_dir = RUNS / run_name(arm)
    cfg, model, src_tok, tgt_tok, device = load_for_inference(
        run_dir / "config.yaml", run_dir / "checkpoints" / checkpoint)
    out = {}

    def score(label: str, am: list[str], refs: list[str]) -> None:
        """Transliterate source, decode, DETOKENIZE, score against raw references
        — matches experiments.domain_breadth.evaluate's score(): the model emits
        Moses-tokenized English (trained on Moses-tokenized targets), so detok_en
        is required before scoring against raw references."""
        ids, _ = encode_sources(src_tok, [translit_am(s) for s in am], cfg.data.max_src_len)
        hyps = [detok_en(h) for h in translate(model, ids, tgt_tok, cfg, device)]
        out[label] = {"bleu": sacrebleu.corpus_bleu(hyps, [refs]).score,
                      "chrf++": sacrebleu.corpus_chrf(hyps, [refs], word_order=2).score}
        print(f"  {arm:10s} {label:14s} {out[label]['bleu']:6.2f} BLEU / "
              f"{out[label]['chrf++']:6.2f} chrF++", flush=True)

    own = pd.read_csv(final_dir(arm) / "test.csv", usecols=["am", "en"], dtype=str).dropna()
    score("in_dist_test", own["am"].tolist(), own["en"].tolist())
    for name, f, sp in [("flores", "flores200_am_en", "devtest"), ("mafand", "mafand_en_amh", "test")]:
        df = pd.read_csv(BENCHMARKS / f"{f}.csv", dtype=str).fillna("")
        df = df[df["split"] == sp].reset_index(drop=True)
        score(name, df["am"].tolist(), df["en"].tolist())
    return out


def diversity_report(arm: str) -> dict:
    """Vendi score / mean pairwise distance for this arm's train vs test —
    a mechanistic check on whether a BLEU gap reflects genuine train/test
    alignment or an easier test set (see module docstring's "known limitation").
    Cache-only; skips silently if the embed cache doesn't cover this arm's
    text (run with --embed on process.dist.diversity manually if so)."""
    from process.dist.diversity import get_vectors, mean_pairwise_distance, vendi

    out = {}
    for split in ("train", "test"):
        df = pd.read_csv(final_dir(arm) / f"{split}.csv", usecols=["en"], dtype=str).dropna()
        try:
            V, coverage = get_vectors(df["en"].tolist(), embed=False)
        except SystemExit:
            print(f"[eval]   {arm}/{split}: no cached embeddings, skipping diversity report")
            continue
        out[split] = {"n": len(V), "coverage": coverage, "vendi": vendi(V),
                      "mean_pairwise_distance": mean_pairwise_distance(V)}
        print(f"[eval]   {arm}/{split}: n={len(V):,} vendi={out[split]['vendi']:.1f} "
              f"mean_pair_dist={out[split]['mean_pairwise_distance']:.4f} (coverage {coverage:.0%})")
    return out


def cmd_eval(args) -> None:
    results = {}
    for arm in ARMS:
        ckpt = RUNS / run_name(arm) / "checkpoints" / args.checkpoint
        if not ckpt.exists():
            print(f"[eval] {arm}: no {args.checkpoint} — skipping")
            continue
        results[arm] = {"bleu": score_arm(arm, args.checkpoint), "diversity": diversity_report(arm)}
    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    RESULTS.write_text(json.dumps(results, indent=2))

    print(f"\n{'arm':<12}{'in-dist test':>14}{'FLORES':>10}{'MAFAND':>10}")
    print("-" * 46)
    for arm, r in results.items():
        b = r["bleu"]
        print(f"{arm:<12}{b['in_dist_test']['bleu']:>14.2f}{b['flores']['bleu']:>10.2f}{b['mafand']['bleu']:>10.2f}")
    print(f"\n-> {RESULTS}")
