"""
prepare.py — Tokenize data/final/{split}.csv into an id-sequence cache under
data/prepared/<src>-<tgt>/{split}.pkl, with BOS/EOS added and any pair whose
either side exceeds MAX_LEN dropped (not truncated, so EOS is never lost).

Standalone script, not a process.py stage — this cache is a one-off
data-pipeline artifact independent of any particular training run's
hyperparameters, reusable across experiments.

Run (from the project root): python -m model.data.prepare

Note: MAX_LEN here is the longest sequence that survives, and a config's
model.max_len is the positional-encoding table it has to index into, so the
invariant runs model.max_len >= MAX_LEN — not the reverse. (It was previously
documented backwards here; nothing broke only because model.data.dataset
independently filtered to max_src_len=128 < 150, masking it.) Keep in sync
with model/configs/*.yaml.
"""
import pickle

import pandas as pd

from model.common import BOS_ID, EOS_ID, load_tokenizer
from process.utils.paths import FINAL, PREPARED

SRC_LANG = "am"
TGT_LANG = "en"
MAX_LEN = 150
SPLITS = ["validation", "test", "train"]


def _tokenize_column(tokenizer, sentences: list[str]) -> list[list[int]]:
    encodings = tokenizer.encode_batch(sentences)
    return [[BOS_ID, *enc.ids, EOS_ID] for enc in encodings]


def main() -> None:
    src_tokenizer = load_tokenizer(SRC_LANG)
    tgt_tokenizer = load_tokenizer(TGT_LANG)

    out_dir = PREPARED / f"{SRC_LANG}-{TGT_LANG}"
    out_dir.mkdir(parents=True, exist_ok=True)

    for split in SPLITS:
        df = pd.read_csv(FINAL / f"{split}.csv", usecols=[SRC_LANG, TGT_LANG], dtype=str)
        src_ids = _tokenize_column(src_tokenizer, df[SRC_LANG].tolist())
        tgt_ids = _tokenize_column(tgt_tokenizer, df[TGT_LANG].tolist())

        kept_src, kept_tgt = [], []
        for s, t in zip(src_ids, tgt_ids):
            if len(s) <= MAX_LEN and len(t) <= MAX_LEN:
                kept_src.append(s)
                kept_tgt.append(t)

        dropped = len(src_ids) - len(kept_src)
        print(f"[prepare] {split}: {len(src_ids)} pairs, dropped {dropped} over MAX_LEN={MAX_LEN}, kept {len(kept_src)}")

        with open(out_dir / f"{split}.pkl", "wb") as f:
            pickle.dump({"src": kept_src, "tgt": kept_tgt}, f)

    print(f"[prepare] wrote caches to {out_dir}")


if __name__ == "__main__":
    main()
