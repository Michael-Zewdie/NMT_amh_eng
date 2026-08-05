"""
experiments.singleref_eval — score a multi-reference corpus's validation split the
two ways that are actually meaningful, instead of the one way that is not.

Run (from the project root):
    python -m experiments.singleref_eval [config] [checkpoint]

The problem this fixes
----------------------
data/final_religious/validation.csv holds ~5.6 English translations of each
Amharic verse, as separate rows. The model emits ONE hypothesis per source, so
corpus BLEU scores that single hypothesis against Young's Literal, Geneva 1599,
Basic English and four others as independent segments — precision converges to the
MEAN match across mutually incompatible translations, not the best one. The score
is pinned below what any single-reference evaluation would give, and it is not
comparable to am-en-nllb-146k, whose validation has ~1 reference per source.

Same artifact explains quran scoring lowest (16.13) in experiments/testset_diversity
despite being the narrowest domain in the corpus: 13.9 references per verse.

Three numbers, all on the SAME model and the same source sentences:

  multi-row     every row scored separately — what the arm currently reports,
                and what is wrong with it
  single-ref    one English per Amharic verse (first occurrence, seeded) — the
                number that IS comparable to a single-reference arm
  multi-ref     sacrebleu with all N translations as parallel reference streams,
                clipping against the max across them — the statistically correct
                score for a corpus that genuinely has N references

single-ref is the one to quote against nllb-146k. multi-ref is the honest ceiling.
"""
import pickle
import sys

import pandas as pd
import sacrebleu
import torch

from model.common import BOS_ID, EOS_ID, PAD_ID, describe_decoding, load_for_inference, load_tokenizer
from model.config import load_config
from model.search import decode
from processing.utils.paths import DATA, RUNS

DEFAULT_CONFIG = "model/configs/religious.yaml"
MAX_LEN = 150
SEED = 0


@torch.no_grad()
def decode_all(model, cfg, tok_src, tok_tgt, device, sources: list[str]) -> list[str]:
    ids = [[BOS_ID, *e.ids, EOS_ID][: cfg.data.max_src_len] for e in tok_src.encode_batch(sources)]
    order = sorted(range(len(ids)), key=lambda i: len(ids[i]))
    out: list[str | None] = [None] * len(ids)
    for start in range(0, len(order), cfg.data.batch_size):
        idx = order[start: start + cfg.data.batch_size]
        w = max(len(ids[i]) for i in idx)
        src = torch.full((len(idx), w), PAD_ID, dtype=torch.long)
        for r, i in enumerate(idx):
            src[r, : len(ids[i])] = torch.tensor(ids[i])
        src = src.to(device)
        gen = decode(model, src, src == PAD_ID, cfg)
        for r, i in enumerate(idx):
            out[i] = tok_tgt.decode(gen[r].tolist(), skip_special_tokens=True)
        print(f"\r  decoding {min(start + len(idx), len(order))}/{len(order)}", end="", flush=True)
    print()
    return out


def main() -> None:
    cfg_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_CONFIG
    cfg = load_config(cfg_path)
    ckpt = sys.argv[2] if len(sys.argv) > 2 else str(RUNS / cfg.run_name / "checkpoints" / "best.pt")
    cfg, model, tok_src, tok_tgt, device = load_for_inference(cfg_path, ckpt)

    arm = str(cfg.data.prepared_dir).replace("data/prepared_", "")
    va = pd.read_csv(DATA / f"final_{arm}" / "validation.csv", usecols=["am", "en"],
                     dtype=str).dropna()
    print(f"{cfg.run_name}  {describe_decoding(cfg)}")
    print(f"validation: {len(va):,} rows, {va.am.nunique():,} unique sources "
          f"({len(va) / va.am.nunique():.2f} refs per source)\n")

    # One decode per UNIQUE source — the model's output cannot depend on which
    # reference a row happens to carry, so decoding each source once is both
    # correct and ~5.6x cheaper than decoding every row.
    grouped = va.groupby("am")["en"].apply(list)
    sources = grouped.index.tolist()
    hyps = decode_all(model, cfg, tok_src, tok_tgt, device, sources)

    refs_first = [r[0] for r in grouped.values]
    single = sacrebleu.corpus_bleu(hyps, [refs_first])
    single_chrf = sacrebleu.corpus_chrf(hyps, [refs_first], word_order=2)

    width = max(len(r) for r in grouped.values)
    streams = [[(r[i] if i < len(r) else r[0]) for r in grouped.values] for i in range(width)]
    multi = sacrebleu.corpus_bleu(hyps, streams)
    multi_chrf = sacrebleu.corpus_chrf(hyps, streams, word_order=2)

    # what the arm currently reports: every row its own segment
    expanded_h, expanded_r = [], []
    for h, rs in zip(hyps, grouped.values):
        expanded_h += [h] * len(rs)
        expanded_r += rs
    rowwise = sacrebleu.corpus_bleu(expanded_h, [expanded_r])

    print(f"{'scoring':34s} {'BLEU':>7s} {'chrF++':>8s}")
    print("-" * 52)
    print(f"{'multi-row (what was reported)':34s} {rowwise.score:>7.2f} {'':>8s}")
    print(f"{'single-ref (compare to nllb-146k)':34s} {single.score:>7.2f} {single_chrf.score:>8.2f}")
    print(f"{'multi-ref (' + str(width) + ' streams, correct)':34s} {multi.score:>7.2f} {multi_chrf.score:>8.2f}")


if __name__ == "__main__":
    main()
