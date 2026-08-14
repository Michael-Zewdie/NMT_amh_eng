"""experiments.stratification — does SEMANTIC train/val/test stratification beat
plain DOMAIN (source) stratification on in-distribution BLEU, and how much of
either's gain (if any) is just what a length-only stratified split already buys?

Run (from the project root):
    python -m experiments.stratification build
    python -m experiments.stratification train --arm length|domain|semantic
    python -m experiments.stratification eval

Background
----------
process.pool already ships two of the three arms as library functions:
split() (length-bucket only — the default used by every model in
EXPERIMENTS.md) and split_semantic() (length bucket x k-means cluster on
mean-pooled English embeddings, k=16 — built, never used in a trained model).
The third arm, split_domain() (length bucket x named source), is new, added
to process/pool.py alongside split_semantic().

Layout
------
    paths.py      where everything lands on disk — FROZEN, see its warning
    corpus.py     which corpora, which cutoffs, and why (build, stage 1)
    splits.py     the three arms + the one shared vocab (build, stages 2-3)
    train.py      the recipe, pinned to am-en-broad, shared by all arms
    evaluate.py   scoring + the train/test diversity check that guards it

Outputs:
    data/final_splitstrat_{length,domain,semantic}/
    data/prepared_splitstrat_{length,domain,semantic}/am-en/
    runs/am-en-splitstrat-{length,domain,semantic}/
    experiments/stratification/results.json
"""
import argparse

from experiments.stratification.evaluate import cmd_eval
from experiments.stratification.splits import ARMS, cmd_build
from experiments.stratification.train import cmd_train


def main() -> None:
    ap = argparse.ArgumentParser(prog="python -m experiments.stratification",
                                 description=__doc__.strip().splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("build").set_defaults(func=cmd_build)
    t = sub.add_parser("train")
    t.add_argument("--arm", required=True, choices=ARMS)
    t.add_argument("--resume", action="store_true")
    t.set_defaults(func=cmd_train)
    e = sub.add_parser("eval")
    e.add_argument("--checkpoint", default="best.pt")
    e.set_defaults(func=cmd_eval)
    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
