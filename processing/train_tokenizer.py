"""
train_tokenizer.py — Train a byte-level BPE tokenizer on the English side of
data/final/train.csv (the pipeline's actual pooled + split output, not a demo
sample).

Standalone script, not a process.py stage — tokenizer training is a one-off
artifact-producing step, not a per-run data-cleaning pass.

Run (from the project root): python -m processing.train_tokenizer
"""
import pandas as pd
from tokenizers import ByteLevelBPETokenizer

from processing.utils.paths import FINAL, TOKENIZER_EN

VOCAB_SIZE = 32000
MIN_FREQUENCY = 2
SPECIAL_TOKENS = ["<pad>", "<unk>", "<s>", "</s>"]


def main() -> None:
    df = pd.read_csv(FINAL / "train.csv", usecols=["en"], dtype=str)
    sentences = df["en"].tolist()
    print(f"[tokenizer] training on {len(sentences)} English sentences from {FINAL / 'train.csv'}")

    tokenizer = ByteLevelBPETokenizer()
    tokenizer.train_from_iterator(
        sentences,
        vocab_size=VOCAB_SIZE,
        min_frequency=MIN_FREQUENCY,
        special_tokens=SPECIAL_TOKENS,
    )

    TOKENIZER_EN.mkdir(parents=True, exist_ok=True)
    tokenizer.save_model(str(TOKENIZER_EN))               # vocab.json + merges.txt
    tokenizer.save(str(TOKENIZER_EN / "tokenizer.json"))   # single-file fast-tokenizer format

    print(f"[tokenizer] vocab_size={tokenizer.get_vocab_size()} → {TOKENIZER_EN}")


if __name__ == "__main__":
    main()
