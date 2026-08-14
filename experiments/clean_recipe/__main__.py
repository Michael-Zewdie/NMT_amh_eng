"""experiments.clean_recipe — does cleaning the mined tier, dropping the sources
that were never helping, and following Gezmu's lowercase setup beat am-en-base-v4?

Built on the v4 data audit (EXPERIMENTS.md). Four changes from v4, one arm:

    NLLB gated at laser_score > 1.08     halves mined misalignment (16% -> 7%)
    quran, ccaligned, religious dropped  quran scored 14.13 in-dist vs gezmu's 27.59
    English lowercased                   Gezmu's own setup; scored case-insensitively
    Gezmu recipe + 60k steps             Moses + AT4MT + shared 8k + tied embeddings

AfriCOMET deliberately stays at 0.80 — raising it is a pure loss (0.80 -> 0.92
discards 97.6% of mined rows to move misalignment 1.6 points). See corpus.py.

Run (from the project root):
    python -m experiments.clean_recipe corpus [--write]    # build data/final_clean
    python -m experiments.clean_recipe build  --arm lower
    python -m experiments.clean_recipe train  --arm lower [--resume]
    python -m experiments.clean_recipe eval

Layout
------
    corpus.py     which sources, which cutoffs, and why — plus every on-disk path
    arms.py       the arm, its tokenization, and the training recipe
    evaluate.py   in-dist + FLORES + MAFAND, scored cased and case-insensitively

Outputs:
    data/final_clean/
    data/tokenizer/shared_translit_clean_lower/
    data/prepared_clean_lower/am-en/
    runs/am-en-clean-lower/
    experiments/clean_recipe/results.json
"""
import argparse

from experiments.clean_recipe.arms import ARMS, DEFAULT_ARM, cmd_build, cmd_train
from experiments.clean_recipe.corpus import cmd_corpus
from experiments.clean_recipe.evaluate import cmd_eval


def main() -> None:
    ap = argparse.ArgumentParser(prog="python -m experiments.clean_recipe",
                                 description=__doc__.strip().splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("corpus", help="build data/final_clean from data/processed/")
    c.add_argument("--write", action="store_true",
                   help="write data/final_clean/; without it, only report counts")
    c.set_defaults(func=cmd_corpus)

    b = sub.add_parser("build", help="tokenize the arm into its .pkl caches")
    b.add_argument("--arm", choices=list(ARMS), default=DEFAULT_ARM)
    b.set_defaults(func=cmd_build)

    t = sub.add_parser("train", help="train the arm")
    t.add_argument("--arm", choices=list(ARMS), default=DEFAULT_ARM)
    t.add_argument("--resume", action="store_true",
                   help="continue from checkpoints/last.pt (best_bleu is restored)")
    t.set_defaults(func=cmd_train)

    e = sub.add_parser("eval", help="in-dist + FLORES + MAFAND, cased and case-insensitive")
    e.add_argument("--checkpoint", default="best.pt")
    e.set_defaults(func=cmd_eval)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
