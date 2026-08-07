"""
model.translate — interactive REPL: read source-language text from stdin,
print the translation, using a trained checkpoint. Reuses model.search's
decode dispatcher rather than a second decoding implementation, so the REPL
uses whatever strategy the config's inference.beam_size asks for.

Run (from the project root): python -m model.translate [config_path] [checkpoint_path]
Defaults: model/configs/base_v6.yaml, <run_dir>/checkpoints/last.pt.
"""
import sys

import polars as pl
import torch

from model.common import BOS_ID, EOS_ID, load_for_inference
from model.config import Config
from model.search import decode
from processing.clean.normalize import normalize

DEFAULT_CONFIG = "model/configs/base_v6.yaml"  # stale, see model/train.py's
# DEFAULT_CONFIG comment — moved to archive/ in the 2026-08-07 v2 archive pass.


def normalize_source(text: str) -> str:
    """Apply the same Amharic normalization the training corpus went through.

    Non-optional: the model only ever saw normalized text, so skipping this
    measures a preprocessing mismatch rather than translation quality. Doing so
    costs ~1.8 BLEU on FLORES devtest (17.35 -> 15.57 for am-en-base-v6), which
    is larger than the entire gain from beam search. model.evaluate_benchmark
    does this inside load_benchmark(); the REPL has to do it explicitly, since
    its input arrives as a bare line of stdin.
    """
    df = normalize(pl.DataFrame({"am": [text], "en": [""]}), "am", "en", name=None)
    return df["am"][0]


def translate(model, text: str, src_tokenizer, tgt_tokenizer, cfg: Config, device: torch.device) -> str:
    ids = [BOS_ID, *src_tokenizer.encode(normalize_source(text)).ids, EOS_ID]
    src = torch.tensor([ids], device=device)
    src_pad_mask = torch.zeros_like(src, dtype=torch.bool)
    generated = decode(model, src, src_pad_mask, cfg)
    return tgt_tokenizer.decode(generated[0].tolist(), skip_special_tokens=True)


def main() -> None:
    config_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_CONFIG
    checkpoint_path = sys.argv[2] if len(sys.argv) > 2 else None
    cfg, model, src_tokenizer, tgt_tokenizer, device = load_for_inference(config_path, checkpoint_path)

    print(f"[translate] loaded {cfg.run_name} — type {cfg.data.src_lang} text, Ctrl+D to exit", file=sys.stderr)
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        print(translate(model, line, src_tokenizer, tgt_tokenizer, cfg, device))


if __name__ == "__main__":
    main()
