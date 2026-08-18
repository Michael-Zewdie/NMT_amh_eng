"""am-en-narrow and am-en-broad-v2, each on its OWN test split + FLORES + MAFAND.

Six decodes, always run together — the two arms in one invocation is the point,
since the comparison between them is the experiment. Three numbers per model:
own-domain fit, then the two shared benchmarks.

Two things this deliberately no longer does. It does not score the cross-domain
cells (each model on the other's test split) that made this a 2x2: FLORES and
MAFAND are already a shared yardstick held out from both arms, so the
off-diagonal cells cost a decode pass each to answer a question the benchmark
columns answer better. And it does not score broad v1 — superseded by the
register-disjoint v2 rebuild and archived. Both sets of numbers stay in
results.json and EXPERIMENTS.md; they are simply not re-computed.

Every score is decoded from RAW TEXT through the model's own tokenizer rather
than from the pre-tokenized .pkl caches, because the hypotheses have to be
DETOKENIZED before scoring. That is what keeps these numbers comparable to
am-en-gezmu-8k's: that baseline was trained and scored on natural English, while
these models emit Moses-tokenized English, and scoring tokenized hypotheses
against raw references would produce a gap that is pure preprocessing artifact.

Both arms are scored with best.pt by default: am-en-narrow trained before
checkpoint averaging was restored and has no rolling snapshots, so averaging one
arm and not the other would compare a model against a handicapped opponent.

CASE no longer splits the two arms -- Gezmu ships lowercased (0.0% of its English
has any uppercase, train/dev/test alike) and broad_v2 is lowercased to match
(arms.py), which is exactly why v1 was replaced. It still costs both of them the
same way on the BENCHMARKS, which are cased (FLORES 98.7% capitalised, MAFAND
92.1%): nothing in the pipeline restores case -- detok_en only rejoins
punctuation -- so a PERFECT but lowercase translation tops out at 80.1 BLEU on
FLORES and 71.6 on MAFAND, not 100. The `_ci` keys are the uncapped numbers.
Scoring is free next to decoding, so both are always computed; --lowercase only
chooses which table is printed.
"""
import json

import pandas as pd
import sacrebleu

from experiments.domain_breadth.arms import ARMS
from experiments.domain_breadth.paths import RESULTS
from model.common import load_for_inference
from model.tokenize.preprocess import detok_en, translit_am
from process.utils.paths import BENCHMARKS, RUNS

# The comparison, and the whole of it. Broad v1 is not scored: its corpus and
# tokenizer now live under archive/ (paths.py), it is superseded by the
# register-disjoint v2 rebuild, and its published numbers are already in
# EXPERIMENTS.md and results.json. It stays defined in arms.py so v1 remains
# reproducible; it is simply not part of the comparison any more.
EVAL_ARMS = ["narrow", "broad_v2"]

# Result keys are frozen at the names the published numbers were written under
# (EXPERIMENTS.md, results.json): the narrow arm's own test set is Gezmu's.
TEST_LABELS = {"narrow": "test_gezmu", "broad_v2": "test_broad_v2"}
ARM_LABELS = {"narrow": "NARROW (gezmu)", "broad_v2": "BROAD v2 (disjoint)"}
BENCHMARK_SETS = [("flores", "flores200_am_en", "devtest"),
                  ("mafand", "mafand_en_amh", "test")]


def score_model(arm_name: str, checkpoint: str) -> dict:
    """Score one arm's model on its own test split and both benchmarks."""
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
        print(f"  {arm.run:24s} {label:16s} {out[label]['bleu']:6.2f} BLEU / "
              f"{out[label]['chrf++']:6.2f} chrF++   |  case-insensitive "
              f"{out[label]['bleu_ci']:6.2f} / {out[label]['chrf++_ci']:6.2f}", flush=True)

    am, en = arm.read("test")                          # own domain only
    score(TEST_LABELS[arm_name], am, en)
    for name, f, sp in BENCHMARK_SETS:
        df = pd.read_csv(BENCHMARKS / f"{f}.csv", dtype=str).fillna("")
        df = df[df["split"] == sp].reset_index(drop=True)
        score(name, df["am"].tolist(), df["en"].tolist())
    return out


def cmd_eval(args) -> None:
    results = {}
    for name in EVAL_ARMS:
        arm = ARMS[name]
        if not (RUNS / arm.run / "checkpoints" / args.checkpoint).exists():
            print(f"[eval] {arm.run}: no {args.checkpoint} — skipping")
            continue
        results[arm.run] = score_model(name, args.checkpoint)

    # Merged, not overwritten: results.json also holds the cross-domain cells and
    # the broad v1 arm this command no longer scores. Those numbers are published
    # in EXPERIMENTS.md and a re-run must not silently delete them.
    on_disk = json.loads(RESULTS.read_text()) if RESULTS.exists() else {}
    for run, scores in results.items():
        on_disk.setdefault(run, {}).update(scores)
    RESULTS.write_text(json.dumps(on_disk, indent=2))

    key = "bleu_ci" if getattr(args, "lowercase", False) else "bleu"
    heading = ("case-INSENSITIVE" if key == "bleu_ci"
               else "as-scored (both arms are lowercase; the cased benchmarks cap them "
                    "at ~80 FLORES / ~72 MAFAND — see module docstring)")
    print(f"\n{heading}")
    scored = [n for n in EVAL_ARMS if ARMS[n].run in results]
    head = ["own test", "FLORES", "MAFAND"]
    print(f"{'':24}" + "".join(f"{h:>14}" for h in head))
    print("-" * (24 + 14 * len(head)))
    for name in scored:
        r = results[ARMS[name].run]
        cols = [TEST_LABELS[name], "flores", "mafand"]
        row = "".join(f"{r[c][key]:>14.2f}" for c in cols)
        print(f"{ARM_LABELS.get(name, name):<24}{row}")
    print("\n(own test is each arm's OWN split — the column is not a shared set;"
          "\n FLORES and MAFAND are held out from every arm and ARE comparable)")

    if len(scored) == 2:
        n, v = results[ARMS["narrow"].run], results[ARMS["broad_v2"].run]
        print(f"\nbroad_v2 - narrow:  FLORES {v['flores'][key] - n['flores'][key]:+.2f}"
              f"   MAFAND {v['mafand'][key] - n['mafand'][key]:+.2f}"
              f"   <- the experiment, at fixed size and steps")
    if key == "bleu":
        print("\nre-run with --lowercase to lift the cased benchmarks' ceiling on both arms")
    print(f"\n-> {RESULTS}")
