"""
en_encoder.py — shared loader for the English-only sentence encoder used to build
semantic-cluster embeddings (processing.utils.score_embed, processing.dist.semantics).

Deliberately a different model from score_labse.py's LaBSE: LaBSE is trained for
cross-lingual alignment across 100+ languages, which optimizes for translation
matching, not fine-grained English semantic distinctions. sentence-transformers/
all-mpnet-base-v2 is the flagship general-purpose English embedding model in the
same library — trained on 1B+ sentence pairs for semantic similarity/clustering,
same 768-dim output shape as LaBSE, no special query/passage prefix convention.

This loader is never shared with score_labse.py's model — the two stages embed
different things (alignment quality vs. English semantic content) with different
models, and each caches its own singleton.
"""
import torch
from sentence_transformers import SentenceTransformer

EN_ENCODER_MODEL = "sentence-transformers/all-mpnet-base-v2"

_en_encoder = None


def get_en_encoder() -> SentenceTransformer:
    """Load the English encoder once, on the best available device."""
    global _en_encoder
    if _en_encoder is None:
        device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        print(f"[en_encoder] loading {EN_ENCODER_MODEL} on {device}...")
        _en_encoder = SentenceTransformer(EN_ENCODER_MODEL, device=device)
    return _en_encoder
