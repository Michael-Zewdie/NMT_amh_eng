"""
embed_english.py — warm the embedding cache with BGE-large-en-v1.5 vectors for
every source's English text, for use in semantic-cluster stratification.

This is a fully independent stage from score_labse.py: it doesn't touch
labse_score, laser_score, or any existing cutoff. Clustering only needs
same-language topic separation, not cross-lingual alignment, so it uses a
strong English-only encoder instead of LaBSE — see the plan's rationale for why
LaBSE (and the Amharic side generally) is the weaker choice for this task.

Embeds every row's English text, not deduped to one-per-am — simpler than
replicating pool.py's cross-source canonical-reference pick at this per-source
stage, and it guarantees whatever pool.split() later selects as canonical is
already warm in cache.

Run (from the project root): python -m processing.utils.embed_english
"""
import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer

from processing.utils.paths import PROCESSED
from processing.utils import embed_cache

_bge_model = None


def _get_bge_model() -> SentenceTransformer:
    """Load BGE-large-en-v1.5 once, on CUDA/MPS if available."""
    global _bge_model
    if _bge_model is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
        print(f"[embed_english] loading BAAI/bge-large-en-v1.5 on {device}...")
        _bge_model = SentenceTransformer("BAAI/bge-large-en-v1.5", device=device)
    return _bge_model


def encode_texts(texts: list[str]) -> np.ndarray:
    """BGE-large-en-v1.5 embeddings, L2-normalized (cosine-ready for k-means)."""
    model = _get_bge_model()
    return model.encode(texts, batch_size=128, normalize_embeddings=True, show_progress_bar=True)


def main() -> None:
    """Embed every am/en CSV's `en` column in data/processed/ into the shared
    "bge_en" embedding cache; skip non-corpus files (by columns)."""
    warmed = 0
    for p in sorted(PROCESSED.glob("*.csv")):
        df = pd.read_csv(p, dtype=str)
        if not {"am", "en"}.issubset(df.columns):
            print(f"[embed_english] skipping {p.name} (not am/en parallel text)")
            continue
        print(f"[embed_english] embedding {p.name}: {len(df)} rows")
        embed_cache.embedded(df["en"], "bge_en", encode_texts)
        warmed += 1
    print(f"\n[embed_english] warmed {warmed} source(s) → data/embeddings/bge_en.*")


if __name__ == "__main__":
    main()
