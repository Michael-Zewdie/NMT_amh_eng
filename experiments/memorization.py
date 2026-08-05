"""
experiments.memorization — do v6's high-scoring FLORES sentences score well
because the model translated them, or because it had seen something very close?

Run (from the project root):
    python -m experiments.memorization [run] [benchmark]

The question
------------
Sentence-level BLEU on FLORES devtest is heavily skewed: median 12.55, but a
handful of sentences clear 65-85. Those outliers look suspicious — #937 ("The
capital of Moldova is Chişinău…") scores 84.92, which is not the kind of sentence
a 17-BLEU system should nail.

processing.utils.decontaminate already removed exact and fuzzy (difflib >= 0.70)
overlaps between the training pool and every benchmark — 38 rows total. So this
is NOT asking whether the benchmark leaked. It asks the weaker, more interesting
question: are the high scorers sentences whose *nearest training neighbour* is
unusually close, just below the decontamination threshold?

Method
------
TF-IDF over character 3-5 grams of the ENGLISH side of data/final/train.csv,
nearest-neighbour by cosine. English rather than Amharic because a near-copy of
the reference is what actually inflates BLEU.

The control is the point. A high similarity number means nothing on its own —
"The capital of X is Y" is a common construction and every FLORES sentence has
*some* nearest neighbour. So the same measurement runs on a random sample of
ordinary-scoring sentences, and the two distributions are compared. Memorization
predicts the high scorers sit clearly above the control; if the two distributions
overlap, the high scores are real translation of easy, formulaic sentences.
"""
import csv
import sys

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

from processing.utils.paths import FINAL, ROOT

TOP_THRESHOLD = 50.0     # "high scoring" cutoff, sentence BLEU
CONTROL_N = 120          # random ordinary sentences to compare against
SEED = 0


def main() -> None:
    run = sys.argv[1] if len(sys.argv) > 1 else "am-en-base-v6"
    bench = sys.argv[2] if len(sys.argv) > 2 else "flores200_am_en_devtest"
    tsv = ROOT / "results" / f"{run.replace('am-en-', '')}_{bench.replace('200_am_en', '')}.tsv"
    tsv = tsv if tsv.exists() else ROOT / "results" / "v6_flores_devtest.tsv"

    d = pd.read_csv(tsv, sep="\t", dtype=str, quoting=csv.QUOTE_NONE)
    d["sentence_bleu"] = d.sentence_bleu.astype(float)

    train = pd.read_csv(FINAL / "train.csv", usecols=["am", "en"], dtype=str).dropna()
    print(f"training corpus: {len(train):,} pairs")
    print(f"benchmark: {len(d):,} sentences from {tsv.name}\n")

    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2, max_features=400_000)
    X = vec.fit_transform(train.en.tolist())
    nn = NearestNeighbors(n_neighbors=1, metric="cosine").fit(X)

    high = d[d.sentence_bleu >= TOP_THRESHOLD]
    rest = d[d.sentence_bleu < TOP_THRESHOLD]
    control = rest.sample(min(CONTROL_N, len(rest)), random_state=SEED)

    def nearest(frame):
        q = vec.transform(frame.reference_en.tolist())
        dist, idx = nn.kneighbors(q)
        return 1.0 - dist.ravel(), idx.ravel()

    hs, hi = nearest(high)
    cs, _ = nearest(control)

    print(f"{'group':28s} {'n':>4s} {'mean sim':>9s} {'median':>8s} {'p90':>7s} {'max':>7s}")
    print("-" * 68)
    for name, s in ((f"BLEU >= {TOP_THRESHOLD:.0f}", hs), ("control (BLEU < 50, random)", cs)):
        print(f"{name:28s} {len(s):>4d} {s.mean():>9.3f} {np.median(s):>8.3f} "
              f"{np.quantile(s, 0.9):>7.3f} {s.max():>7.3f}")

    from scipy.stats import mannwhitneyu
    u, p = mannwhitneyu(hs, cs, alternative="greater")
    print(f"\nMann-Whitney U (high > control): p = {p:.4g}")
    print("p < 0.05 would mean the high scorers really do sit closer to training text.\n")

    print("=== nearest training neighbour for the top scorers ===")
    order = np.argsort(-high.sentence_bleu.values)
    for j in order[:10]:
        row = high.iloc[j]
        t = train.iloc[hi[j]]
        print(f"\n#{row['index']}  sentBLEU {row.sentence_bleu:.1f}  nearest-train cosine {hs[j]:.3f}")
        print(f"  BENCH REF   {row.reference_en[:110]}")
        print(f"  NEAREST EN  {str(t.en)[:110]}")


if __name__ == "__main__":
    main()
