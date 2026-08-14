"""The 2x2: both models, both arms' test splits, both fixed benchmarks.

This lives outside either arm on purpose. It is the comparison the experiment
exists to make, so it belongs to neither half of it — previously it sat inside
broad.py, which left narrow.py documenting that its counterpart owned its
evaluation.

Every score is decoded from RAW TEXT through the scoring model's OWN tokenizer,
so a model can be scored against the other arm's test set even though the two
arms have different vocabularies (arms.py explains why they must). That is the
whole point of the 2x2, and it is not possible from the pre-tokenized .pkl
caches, which are each encoded with one arm's vocabulary only.

Both arms are scored with best.pt by default: am-en-narrow trained before
checkpoint averaging was restored and has no rolling snapshots, so averaging one
arm and not the other would compare a model against a handicapped opponent.

CASE IS A CONFOUND HERE, so every cell is scored twice.
Gezmu ships lowercased (0.0% of its English has any uppercase, train/dev/test
alike); the broad corpus does not (91% of sentences start capitalised), and so
do FLORES (98.7%) and MAFAND (92.1%). Nothing in the pipeline restores case --
detok_en only rejoins punctuation -- so the narrow model emits lowercase forever.
That hands narrow a free pass in-domain (lowercase output vs its own lowercase
references) and penalises it out-of-domain (same output vs cased benchmarks),
inflating the experiment's headline gap from BOTH ends. Measured ceiling: a
PERFECT translation that is merely lowercase scores 80.1 BLEU on FLORES and 71.6
on MAFAND, not 100.

The `_ci` keys are the confound-free comparison. Retraining narrow on cased text
is not an option -- Gezmu releases only the lowercased `*.base.*` files -- so
case-insensitive scoring is the fix. Scoring is free next to decoding, so both
are always computed; --lowercase only chooses which table is printed.
"""
import json

import pandas as pd
import sacrebleu

from experiments.domain_breadth.arms import ARMS
from experiments.domain_breadth.paths import RESULTS
from model.common import load_for_inference
from model.tokenize.preprocess import detok_en, translit_am
from process.utils.paths import BENCHMARKS, RUNS

# Result keys are frozen at the names the published numbers were written under
# (EXPERIMENTS.md, results.json): the narrow arm's own test set is Gezmu's.
TEST_LABELS = {"narrow": "test_gezmu", "broad": "test_broad"}
BENCHMARK_SETS = [("flores", "flores200_am_en", "devtest"),
                  ("mafand", "mafand_en_amh", "test")]


def score_model(arm_name: str, checkpoint: str) -> dict:
    """Score one arm's model on both arms' test splits and both benchmarks."""
    from model.evaluate.evaluate_OOD import encode_sources, translate

    arm = ARMS[arm_name]
    run_dir = RUNS / arm.run
    cfg, model, src_tok, tgt_tok, device = load_for_inference(
        run_dir / "config.yaml", run_dir / "checkpoints" / checkpoint)
    out = {}

    def score(label: str, am: list[str], refs: list[str]) -> None:
        """Transliterate source, decode, DETOKENIZE, score against raw references.

        Detokenizing is what keeps these numbers comparable to am-en-gezmu-8k's:
        that baseline was trained and scored on natural English, while these
        models emit Moses-tokenized English. Scoring tokenized hypotheses against
        raw references would produce a gap that is pure preprocessing artifact.
        """
        ids, _ = encode_sources(src_tok, [translit_am(s) for s in am], cfg.data.max_src_len)
        hyps = [detok_en(h) for h in translate(model, ids, tgt_tok, cfg, device)]
        lo_h, lo_r = [h.lower() for h in hyps], [r.lower() for r in refs]
        out[label] = {
            "bleu": sacrebleu.corpus_bleu(hyps, [refs]).score,
            "chrf++": sacrebleu.corpus_chrf(hyps, [refs], word_order=2).score,
            "bleu_ci": sacrebleu.corpus_bleu(lo_h, [lo_r]).score,
            "chrf++_ci": sacrebleu.corpus_chrf(lo_h, [lo_r], word_order=2).score,
        }
        print(f"  {arm.run:24s} {label:16s} {out[label]['bleu']:6.2f} BLEU / "
              f"{out[label]['chrf++']:6.2f} chrF++   |  case-insensitive "
              f"{out[label]['bleu_ci']:6.2f} / {out[label]['chrf++_ci']:6.2f}", flush=True)

    for other in ARMS.values():                       # own-domain and cross-domain
        am, en = other.read("test")
        score(TEST_LABELS[other.name], am, en)
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
        results[arm.run] = score_model(name, args.checkpoint)
    RESULTS.write_text(json.dumps(results, indent=2))

    key = "bleu_ci" if getattr(args, "lowercase", False) else "bleu"
    heading = ("case-INSENSITIVE (confound-free)" if key == "bleu_ci"
               else "as-scored, cased (narrow is confounded — see module docstring)")
    print(f"\n{heading}")
    print(f"{'':26}{'own-domain':>14}{'cross-domain':>14}{'FLORES':>10}{'MAFAND':>10}")
    print("-" * 74)
    for name, arm in ARMS.items():
        r = results.get(arm.run)
        if r is None:
            continue
        other = next(o for o in ARMS if o != name)
        label = "NARROW (gezmu)" if name == "narrow" else "BROAD (pooled)"
        print(f"{label:<26}{r[TEST_LABELS[name]][key]:>14.2f}"
              f"{r[TEST_LABELS[other]][key]:>14.2f}"
              f"{r['flores'][key]:>10.2f}{r['mafand'][key]:>10.2f}")
    if len(results) == len(ARMS):
        n, b = ARMS["narrow"].run, ARMS["broad"].run
        d_f = results[b]["flores"][key] - results[n]["flores"][key]
        d_m = results[b]["mafand"][key] - results[n]["mafand"][key]
        print(f"\nbroad - narrow:  FLORES {d_f:+.2f}   MAFAND {d_m:+.2f}"
              f"   (experiment #2 at 140k: +6.37 / +2.50, cased)")
        if key == "bleu":
            print("re-run with --lowercase for the comparison that removes the casing confound")
    print(f"\n-> {RESULTS}")
