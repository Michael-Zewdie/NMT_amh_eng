"""experiments.domain_breadth — does the DOMAIN BREADTH of the training corpus
trade in-distribution BLEU against out-of-distribution generalization?

    NARROW  Gezmu et al.'s own 140k split — one register, ~83% Watchtower/Bible
    BROAD   the 6 other sources pooled and size-matched, gezmu excluded entirely

Held constant: Unigram algorithm, 8k shared vocabulary, Moses + AT4MT
transliteration, tied embeddings, and every training hyperparameter. 


Run (from the project root):
    python -m experiments.domain_breadth corpus [--write]      # build data/final_broad_v2
    python -m experiments.domain_breadth build --arm narrow|broad
    python -m experiments.domain_breadth train --arm narrow|broad [--resume]
    python -m experiments.domain_breadth eval                  # narrow + broad_v2, together
    python -m experiments.domain_breadth chart [--top N]       # source mix + NLLB by site

Layout
------
    paths.py        where everything lands on disk — FROZEN, see its warning
    corpus.py       the broad arm's corpus: which sources, which cutoffs, and why
    arms.py         the two arms + the shared build; the only place they differ
    train.py        the recipe, shared by both arms
    evaluate.py     narrow vs broad_v2: own test set + the shared benchmarks
    source_dist.py  the broad v1 corpus's source mix, with the NLLB wedge broken out by site

Outputs:
    data/final_broad_v2/
    data/tokenizer/shared_translit{,_broad_v2}/
    data/prepared{_broad_v2,}_translit/am-en/
    runs/am-en-{narrow,broad-v2}/
    experiments/domain_breadth/results.json
    data/figs/final_broad_sources_pie.png

The broad v1 arm (`am-en-broad`) was archived 2026-08-17 — its run, corpus,
vocabulary and tokenized cache all live under archive/, and paths.py still
points at them so `test_broad` stays scoreable. See archive/README.md.
"""
import argparse

from experiments.domain_breadth.arms import ARMS, cmd_build
from experiments.domain_breadth.corpus import cmd_corpus
from experiments.domain_breadth.evaluate import cmd_eval
from experiments.domain_breadth.source_dist import TOP_N, cmd_chart
from experiments.domain_breadth.train import cmd_train


def main() -> None:
    ap = argparse.ArgumentParser(prog="python -m experiments.domain_breadth",
                                 description=__doc__.strip().splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("corpus", help="build data/final_broad_v2 from data/processed/")
    c.add_argument("--write", action="store_true",
                   help="write data/final_broad_v2/; without it, only report counts. "
                        "am-en-broad-v2 was trained on the current contents, so "
                        "overwriting is deliberate, not the default.")
    c.set_defaults(func=cmd_corpus)

    b = sub.add_parser("build", help="tokenize one arm into its .pkl caches")
    b.add_argument("--arm", choices=list(ARMS), required=True)
    b.set_defaults(func=cmd_build)

    t = sub.add_parser("train", help="train one arm")
    t.add_argument("--arm", choices=list(ARMS), required=True)
    t.add_argument("--resume", action="store_true",
                   help="continue from checkpoints/last.pt (best_bleu is restored, "
                        "so a resumed run cannot overwrite a better best.pt)")
    t.set_defaults(func=cmd_train)

    e = sub.add_parser("eval", help="narrow + broad_v2, each on its OWN test set "
                                    "+ FLORES + MAFAND")
    e.add_argument("--checkpoint", default="best.pt",
                   help="checkpoint filename under checkpoints/ (default best.pt)")
    e.add_argument("--lowercase", action="store_true",
                   help="print the case-INSENSITIVE table. Both scorings are always "
                        "computed and saved; this only picks which one is printed. "
                        "Both arms are lowercase and the benchmarks are cased, which "
                        "caps the cased table — see evaluate.py's note.")
    e.set_defaults(func=cmd_eval)

    ch = sub.add_parser("chart", help="the broad v1 corpus's source mix, and NLLB's by site")
    ch.add_argument("--top", type=int, default=TOP_N,
                    help=f"NLLB sites to bar (default {TOP_N}); the tail is reported, not drawn")
    ch.set_defaults(func=cmd_chart)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
