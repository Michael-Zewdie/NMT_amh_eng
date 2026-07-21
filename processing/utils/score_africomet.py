"""
score_africomet.py — Annotate every processed CSV (including nllb.csv) with an
AfriCOMET-QE quality-estimation score. A second, independent quality gate
alongside score_labse's cosine similarity.

LaBSE is a general-purpose multilingual embedding model measuring semantic
similarity; masakhane/africomet-qe-stl-1.1 is a reference-free COMET model built
on afro-xlmr-large-76L, trained specifically on African-language MT judgments
(WMT 2024) to score translation adequacy/fluency directly. That's a different
failure mode than embedding similarity, so this stage runs *alongside*
score_labse rather than replacing it — both cutoffs must pass at pool time
(processing.utils.pool).

Unlike score_labse (which skips rows already carrying laser_score, since cosine
similarity and LASER's mining margin are the same kind of signal), this stage
scores every source, nllb.csv included: LASER's margin doesn't check adequacy or
fluency, and nllb.csv — the noisiest source — is exactly where that gap matters
most. No skip_where is used.

Scores are cached by sentence content (processing.utils.score_cache), so
re-running after a clean step, a new source, or a cutoff change only scores text
the model has not seen. A run where everything hits the cache never loads the
model at all.

Runs AFTER the clean step in process.py, on the already-normalized am/en CSVs in
data/processed/.

Run (from the project root): python -m processing.utils.score_africomet

Requires: pip install unbabel-comet (first call downloads the model from
HuggingFace, ~2.2GB for the 76L encoder).
"""
import pandas as pd
import torch
from comet import download_model, load_from_checkpoint

from processing.utils.paths import PROCESSED
from processing.utils.score_cache import scored

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
    output = model.predict(data, batch_size=128, progress_bar=True, **_device_kwargs())
    df["africomet_score"] = output.scores
    return df


def main() -> None:
    """Add africomet_score to every am/en CSV in data/processed/, writing back in
    place. Every source is scored, including nllb.csv — see module docstring."""
    annotated = 0
    for p in sorted(PROCESSED.glob("*.csv")):
        df = pd.read_csv(p, dtype=str)
        if not {"am", "en"}.issubset(df.columns):
            print(f"[africomet] skipping {p.name} (not am/en parallel text)")
            continue
        print(f"[africomet] scoring {p.name}: {len(df)} rows")
        df = scored(df, "africomet", ["africomet_score"], africomet_quality)
        df.to_csv(p, index=False)
        annotated += 1
    print(f"\n[africomet] annotated {annotated} source(s) → {PROCESSED}")


if __name__ == "__main__":
    main()
