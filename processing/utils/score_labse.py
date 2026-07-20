"""
score_labse.py — Annotate the non-NLLB processed CSVs with a LaBSE cosine-similarity
quality score. The local, non-mined analogue to NLLB's laser_score.

This stage only ever *adds* a labse_score column; it drops nothing. The cosine
cutoff is applied at pool time (processing.utils.pool), so retuning the threshold
is a cheap re-pool instead of a re-embed, and lowering it can bring rows back.

Scores are cached by sentence content (processing.utils.score_cache), so re-running
after a clean step, a new source, or a cutoff change only embeds text the model has
not seen. A run where everything hits the cache never loads the model at all.

Runs AFTER the clean step in process.py, on the already-normalized am/en CSVs in
data/processed/, so LaBSE always embeds normalized text.

Run (from the project root): python -m processing.utils.score_labse

nllb.csv (its own laser_score) is skipped.
"""
import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer

from processing.utils.paths import PROCESSED
from processing.utils.score_cache import scored

# LaBSE cosine ranges 0-1; aligned pairs typically score ~0.6-0.9. The cutoff that
# consumes this score lives in process.py / pool.py — this file does not filter.
_labse_model = None


def _get_labse_model() -> SentenceTransformer:
    """Load LaBSE once, on MPS if available (first call downloads ~1.8GB)."""
    global _labse_model
    if _labse_model is None:
        device = "mps" if torch.backends.mps.is_available() else "cpu"
        print(f"[labse] loading sentence-transformers/LaBSE on {device}...")
        _labse_model = SentenceTransformer("sentence-transformers/LaBSE", device=device)
    return _labse_model


def labse_similarity(df: pd.DataFrame) -> pd.DataFrame:
    """Cosine similarity between LaBSE sentence embeddings of am/en — a local,
    non-mined analogue to NLLB's laser_score for semantic alignment quality."""
    model = _get_labse_model()
    am_emb = model.encode(df["am"].tolist(), batch_size=128, normalize_embeddings=True, show_progress_bar=True)
    en_emb = model.encode(df["en"].tolist(), batch_size=128, normalize_embeddings=True, show_progress_bar=True)
    df["labse_score"] = np.sum(am_emb * en_emb, axis=1)  # dot product of unit vectors = cosine sim
    return df


def main() -> None:
    """Add labse_score to every am/en CSV in data/processed/, writing back in place;
    skip non-corpus files (nllb.csv)."""
    annotated = 0
    for p in sorted(PROCESSED.glob("*.csv")):
        # nllb.csv is scored by its own laser_score, not LaBSE; other non-am/en
        # files (e.g. website-stats) are skipped by their columns.
        if p.name == "nllb.csv":
            continue
        df = pd.read_csv(p, dtype=str)
        if not {"am", "en"}.issubset(df.columns):
            print(f"[labse] skipping {p.name} (not am/en parallel text)")
            continue
        print(f"[labse] scoring {p.name}: {len(df)} rows")
        df = scored(df, "labse", ["labse_score"], labse_similarity)
        df.to_csv(p, index=False)
        annotated += 1
    print(f"\n[labse] annotated {annotated} source(s) → {PROCESSED}")


if __name__ == "__main__":
    main()
