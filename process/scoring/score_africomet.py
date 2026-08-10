"""
Annotate every processed CSV with an
AfriCOMET-QE quality-estimation score.
"""
import pandas as pd
import torch
from comet import download_model, load_from_checkpoint

from process.cache.score_cache import annotate_processed

# Tensor Cores (Ampere+/Ada) speed up matmul substantially at a negligible precision
# cost that doesn't matter for QE scoring — ~2.4x throughput measured on RTX 4000 Ada.
torch.set_float32_matmul_precision("high")

# AfriCOMET-QE scores 0-1, 1 = perfect translation. The cutoff that consumes this
# score lives in process.py / pool.py — this file does not filter.
_MODEL_NAME = "masakhane/africomet-qe-stl-1.1"
_africomet_model = None


def _get_africomet_model():
    """Load AfriCOMET-QE once (first call downloads ~2.2GB from HuggingFace)."""
    global _africomet_model
    if _africomet_model is None:
        print(f"[africomet] loading {_MODEL_NAME} ...")
        model_path = download_model(_MODEL_NAME)
        _africomet_model = load_from_checkpoint(model_path)
    return _africomet_model


def _device_kwargs() -> dict:
    """cuda > mps > cpu. COMET's predict() takes PyTorch-Lightning-style gpus/
    accelerator args rather than a plain device string, unlike score_labse's
    SentenceTransformer(device=...)."""
    if torch.cuda.is_available():
        return {"gpus": 1}
    if torch.backends.mps.is_available():
        return {"gpus": 1, "accelerator": "mps"}
    return {"gpus": 0}


def africomet_quality(df: pd.DataFrame) -> pd.DataFrame:
    """Reference-free QE score: how well `en` renders `am` as a translation — the
    AfriCOMET analogue to labse_similarity's cosine score. am is the source side
    (src), en is the translation being judged (mt), matching this corpus's am→en
    direction."""
    model = _get_africomet_model()
    data = [{"src": am, "mt": en} for am, en in zip(df["am"], df["en"])]
    output = model.predict(data, batch_size=256, progress_bar=True, **_device_kwargs())
    df["africomet_score"] = output.scores
    return df


def main() -> None:
    """Add africomet_score to every am/en CSV in data/processed/, writing back in
    place. Every source is scored, including nllb.csv — see module docstring."""
    annotate_processed("africomet", ["africomet_score"], africomet_quality)


if __name__ == "__main__":
    main()
