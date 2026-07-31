"""
train_tokenizer_am.py — Train a Unigram (SentencePiece-style) tokenizer on the
Amharic side of data/final/train.csv (the pipeline's actual pooled + split
output, not a demo sample).

Standalone script, not a process.py stage — tokenizer training is a one-off
artifact-producing step, not a per-run data-cleaning pass.

Run (from the project root): python -m processing.train_tokenizer_am
"""
import pandas as pd
from tokenizers.implementations import SentencePieceUnigramTokenizer

from processing.utils.paths import FINAL, TOKENIZER_AM

VOCAB_SIZE = 32000
SPECIAL_TOKENS = ["<pad>", "<unk>", "<s>", "</s>"]


def main() -> None:
    df = pd.read_csv(FINAL / "train.csv", usecols=["am"], dtype=str)
    sentences = df["am"].tolist()
    print(f"[tokenizer] training on {len(sentences)} Amharic sentences from {FINAL / 'train.csv'}")

    tokenizer = SentencePieceUnigramTokenizer()
    tokenizer.train_from_iterator(
        sentences,
        vocab_size=VOCAB_SIZE,
        special_tokens=SPECIAL_TOKENS,
        unk_token="<unk>",
    )

    TOKENIZER_AM.mkdir(parents=True, exist_ok=True)
    tokenizer.save_model(str(TOKENIZER_AM))               # unigram.json
    tokenizer.save(str(TOKENIZER_AM / "tokenizer.json"))   # single-file fast-tokenizer format

    print(f"[tokenizer] vocab_size={tokenizer.get_vocab_size()} → {TOKENIZER_AM}")


if __name__ == "__main__":
    main()
