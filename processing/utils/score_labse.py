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

laser_score and labse_score are the same kind of thing — semantic alignment
quality — just produced by different pipelines (Meta's mining vs. a local LaBSE
embed). A row that already carries laser_score (nllb.csv) doesn't need labse_score
too, so score_cache.scored() skips those rows via skip_where rather than this file
hardcoding nllb.csv by name.
"""
import os

import numpy as np
import pandas as pd
import torch
from huggingface_hub import constants
from huggingface_hub.file_download import are_symlinks_supported, repo_folder_name
from sentence_transformers import SentenceTransformer

from processing.utils.paths import PROCESSED
from processing.utils.score_cache import scored

_LABSE_REPO = "sentence-transformers/LaBSE"

# LaBSE cosine ranges 0-1; aligned pairs typically score ~0.6-0.9. The cutoff that
# consumes this score lives in process.py / pool.py — this file does not filter.
_labse_model = None


def _get_labse_model() -> SentenceTransformer:
    """Load LaBSE once, on CUDA/MPS if available (first call downloads ~1.8GB)."""
    global _labse_model
    if _labse_model is None:
        # Single-threaded warm-up: HF's concurrent file download has a Windows symlink
        # race (see score_africomet.py's _get_africomet_model for detail). Must warm the
        # repo's own cache subfolder, not HF_HUB_CACHE itself — see that file for why.
        are_symlinks_supported(os.path.join(constants.HF_HUB_CACHE, repo_folder_name(repo_id=_LABSE_REPO, repo_type="model")))
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
        print(f"[labse] loading {_LABSE_REPO} on {device}...")
        _labse_model = SentenceTransformer(_LABSE_REPO, device=device)
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
    skip non-corpus files (by columns) and rows that already carry laser_score
    (by content, via score_cache's skip_where)."""
    annotated = 0
    for p in sorted(PROCESSED.glob("*.csv")):
        df = pd.read_csv(p, dtype=str)
        if not {"am", "en"}.issubset(df.columns):
            print(f"[labse] skipping {p.name} (not am/en parallel text)")
            continue
        print(f"[labse] scoring {p.name}: {len(df)} rows")
        df = scored(df, "labse", ["labse_score"], labse_similarity, skip_where="laser_score")
        df.to_csv(p, index=False)
        annotated += 1
    print(f"\n[labse] annotated {annotated} source(s) → {PROCESSED}")


if __name__ == "__main__":
    main()
