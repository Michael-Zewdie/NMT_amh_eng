"""
model.compare — read a run's benchmark output sentence by sentence, next to the
source and the human reference.

Run (from the project root):
    python -m model.compare <run> [benchmark] [--decode beam4] [--sort worst]
                            [--n 40] [--tsv out.tsv]

    python -m model.compare am-en-base-v6
    python -m model.compare am-en-base-v6 mafand_en_amh --sort best
    python -m model.compare am-en-narrow --tsv narrow.tsv

Aggregate BLEU says a system is at 17.35; it does not say what it gets wrong.
This prints per-sentence sentenceBLEU alongside the text so failure modes are
visible — entity mangling, dropped clauses, repetition loops, hallucinated spans.

--sort worst (the default) puts the lowest-scoring sentences first, which is
where the informative errors live. --sort best shows what the model does when it
succeeds; --sort index keeps benchmark order.

Reads the hypothesis files written by model/rescore.py, which carry the decoding
strategy in the filename (…_devtest.beam4.hyp.txt). It will NOT read the older
strategy-less …_devtest.hyp.txt files: those do not record how they were decoded,
and mixing the two is exactly the confusion rescore.py exists to end.

Source text shown is the NORMALIZED Amharic — what the model actually receives.
Un-normalized input costs ~1.8 BLEU, so showing the raw CSV text here would
misrepresent the input the hypothesis came from.
"""
import argparse
import csv
import sys

import sacrebleu

from model.evaluate_benchmark import load_benchmark
from processing.utils.paths import BENCHMARKS, RUNS

DEFAULT_BENCH = "flores200_am_en"
SPLITS = {"flores200_am_en": "devtest", "mafand_en_amh": "test"}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1].strip())
    ap.add_argument("run")
    ap.add_argument("benchmark", nargs="?", default=DEFAULT_BENCH)
    ap.add_argument("--decode", default="beam4", help="beam4 (default) or greedy")
    ap.add_argument("--sort", default="worst", choices=["worst", "best", "index"])
    ap.add_argument("--n", type=int, default=40, help="how many to print (default 40)")
    ap.add_argument("--tsv", help="also write every sentence to this file; a .csv "
                                  "extension writes RFC-4180 quoted CSV, anything "
                                  "else writes tab-separated")
    args = ap.parse_args()

    split = SPLITS.get(args.benchmark, "test")
    hyp_path = RUNS / args.run / "benchmarks" / f"{args.benchmark}_{split}.{args.decode}.hyp.txt"
    if not hyp_path.exists():
        legacy = hyp_path.with_name(f"{args.benchmark}_{split}.hyp.txt")
        extra = (f"\n(a legacy {legacy.name} exists but does not record its decoding "
                 f"strategy — re-score with `python -m model.rescore --runs {args.run}`)"
                 if legacy.exists() else "")
        sys.exit(f"missing {hyp_path}{extra}")

    df = load_benchmark(BENCHMARKS / f"{args.benchmark}.csv", split)
    hyps = hyp_path.read_text(encoding="utf-8").splitlines()
    src, refs = df["am"].tolist(), df["en"].tolist()
    if not len(hyps) == len(refs) == len(src):
        sys.exit(f"length mismatch: {len(hyps)} hyps, {len(refs)} refs, {len(src)} sources")

    rows = [(i, sacrebleu.sentence_bleu(h, [r]).score, s, r, h)
            for i, (s, r, h) in enumerate(zip(src, refs, hyps))]
    if args.sort == "worst":
        rows.sort(key=lambda t: t[1])
    elif args.sort == "best":
        rows.sort(key=lambda t: -t[1])

    if args.tsv:
        # .csv gets real RFC-4180 quoting so any CSV reader (pandas defaults,
        # Excel, Rainbow CSV's "CSV" dialect) parses it. Anything else gets TSV,
        # where quotes are NOT special — so embedded tabs and newlines must be
        # stripped instead, or they silently add columns and rows.
        as_csv = args.tsv.lower().endswith(".csv")
        header = ["index", "sentence_bleu", "source_am", "reference_en", "hypothesis_en"]
        with open(args.tsv, "w", encoding="utf-8", newline="") as f:
            w = csv.writer(f, delimiter="," if as_csv else "\t",
                           quoting=csv.QUOTE_MINIMAL if as_csv else csv.QUOTE_NONE,
                           escapechar=None if as_csv else "\\")
            w.writerow(header)
            for i, b, s, r, h in rows:
                if not as_csv:
                    s, r, h = (x.replace("\t", " ").replace("\n", " ") for x in (s, r, h))
                w.writerow([i, f"{b:.2f}", s, r, h])
        print(f"wrote {len(rows)} rows -> {args.tsv} ({'CSV' if as_csv else 'TSV'})\n")

    corpus = sacrebleu.corpus_bleu(hyps, [refs]).score
    print(f"{args.run}  {args.benchmark}:{split}  decode={args.decode}  "
          f"corpus BLEU {corpus:.2f}  ({len(rows)} sentences, showing {min(args.n, len(rows))} "
          f"sorted {args.sort})\n")
    for i, b, s, r, h in rows[: args.n]:
        print(f"\033[1m#{i}  sentBLEU {b:.1f}\033[0m")
        print(f"  AM   {s}")
        print(f"  REF  {r}")
        print(f"  HYP  {h}\n")


if __name__ == "__main__":
    main()
