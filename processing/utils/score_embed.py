"""
score_embed.py — cache raw English-side sentence embeddings for semantic-cluster
stratification (processing.utils.pool.split_semantic). Unlike score_labse.py
(which keeps only a scalar cosine score), this stage persists the actual
embedding vectors, since k-means clustering needs the vectors themselves, not a
similarity number.

Uses a dedicated English encoder (processing.utils.en_encoder), not LaBSE —
LaBSE is tuned for cross-lingual alignment, not fine-grained English semantic
distinctions. This stage is independent of score_labse.py's model and cache.

Embeddings are cached by content hash of the English text alone
(processing.utils.embed_cache), shared across sources — an English sentence
appearing in two corpora is embedded once, and re-running after a cutoff or pool
change never re-embeds anything already seen. A run where everything hits the
cache never loads the model at all.

Runs AFTER the clean step in process.py (any point relative to score_labse/
score_africomet/score_lid — independent of them), on the already-normalized
am/en CSVs in data/processed/.

Run (from the project root): python -m processing.utils.score_embed
"""
import numpy as np

from processing.utils.embed_cache import annotate_missing
from processing.utils.en_encoder import get_en_encoder

EMBED_DIM = 768
TAG = "mpnet_en"


def encode_en(texts: list[str]) -> np.ndarray:
    """Embed a list of distinct English sentences with the dedicated English
    encoder — the only function in this file allowed to touch the model."""
    model = get_en_encoder()
    return model.encode(texts, batch_size=128, normalize_embeddings=True, show_progress_bar=True)


def main() -> None:
    """Ensure every am/en CSV in data/processed/ has its English text embedded
    and cached — no columns are written back to the CSVs, unlike score_labse.py;
    the embeddings live only in the data/scores/ cache (see embed_cache.py),
    looked up later by processing.utils.pool.split_semantic()."""
    annotate_missing(TAG, key_cols=("en",), compute=encode_en, dim=EMBED_DIM)


if __name__ == "__main__":
    main()
