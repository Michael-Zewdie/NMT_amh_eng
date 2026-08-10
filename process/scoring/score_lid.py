"""
score_lid.py — Annotate the non-NLLB processed CSVs with fastText language-ID
confidence scores (source_lid, target_lid). The local analogue to NLLB's own
source_lid/target_lid columns: the *same* model (Meta's lid218e, 217-language
fastText) and the *same* two labels (amh_Ethi, eng_Latn), so a Gezmu/AfriDoc/Quran
row carries LID scores directly comparable to an NLLB row.

This stage only ever *adds* columns; it drops nothing. The LID floors are applied
at pool time (processing.utils.pool), so retuning them is a cheap re-pool instead
of a re-score, and lowering one can bring rows back.

Scores are cached by sentence content (processing.utils.score_cache), so re-running
after a clean step, a new source, or a cutoff change only scores text the model has
not seen. A run where everything hits the cache never loads the model at all.

Runs AFTER the clean step in process.py, on the already-normalized am/en CSVs in
data/processed/, so LID always sees normalized text. First call downloads the
~1.2GB model from HuggingFace (facebook/fasttext-language-identification).

Run (from the project root): python -m processing.utils.score_lid

nllb.csv already ships these columns from Meta, so it is skipped.
"""
import fasttext
import pandas as pd
from huggingface_hub import hf_hub_download

from process.cache.score_cache import annotate_processed

# fastText LID (lid218e) is very confident on clean, in-script text; aligned rows
# sit ~0.95-1.0. The floors that consume these scores live in process.py / pool.py —
# this file does not filter.
_LID_REPO, _LID_FILE = "facebook/fasttext-language-identification", "model.bin"
_AMH_LABEL, _ENG_LABEL = "__label__amh_Ethi", "__label__eng_Latn"

_lid_model = None


def _get_lid_model() -> "fasttext.FastText._FastText":
    """Load lid218e once (first call downloads ~1.2GB from HuggingFace)."""
    global _lid_model
    if _lid_model is None:
        print(f"[lid] loading {_LID_REPO} ...")
        _lid_model = fasttext.load_model(hf_hub_download(_LID_REPO, _LID_FILE))
    return _lid_model


def _label_confidence(texts: list[str], label: str) -> list[float]:
    """For each text, the probability lid218e assigns to `label` — i.e. the
    confidence the text is that language. fastText rejects embedded newlines, so
    flatten each line first; empty text scores 0.0."""
    model = _get_lid_model()
    scores = []
    for text in texts:
        flat = " ".join(str(text).split())
        if not flat:
            scores.append(0.0)
            continue
        labels, probs = model.predict(flat, k=-1)  # full 217-way distribution
        # clamp: fastText can return probs a hair above 1.0; NLLB's columns are in [0, 1]
        scores.append(min(1.0, float(dict(zip(labels, probs)).get(label, 0.0))))
    return scores


def lid_scores(df: pd.DataFrame, am_col: str = "am", en_col: str = "en") -> pd.DataFrame:
    """Add source_lid (P(amh_Ethi) on am) and target_lid (P(eng_Latn) on en),
    the local analogue of NLLB's own per-sentence LID confidence columns."""
    df["source_lid"] = _label_confidence(df[am_col].tolist(), _AMH_LABEL)
    df["target_lid"] = _label_confidence(df[en_col].tolist(), _ENG_LABEL)
    return df


def main() -> None:
    """Add source_lid/target_lid to every am/en CSV in data/processed/, writing back
    in place; skip nllb.csv (already ships these columns from Meta) and non-am/en files."""
    annotate_processed("lid", ["source_lid", "target_lid"], lid_scores, skip_files=("nllb.csv",))


if __name__ == "__main__":
    main()
