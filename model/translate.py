"""
model.translate — interactive REPL: read source-language text from stdin,
print the translation, using a trained checkpoint. Reuses model.evaluate's
greedy_decode rather than a second decoding implementation.

Run (from the project root): python -m model.translate [config_path] [checkpoint_path]
Defaults: model/configs/base.yaml, <run_dir>/checkpoints/last.pt.
"""
import sys

import torch

from model.checkpoint import load_checkpoint
from model.data.tokenizer import BOS_ID, EOS_ID, PAD_ID, load_tokenizer
from model.evaluate import greedy_decode
from model.transformer import Seq2SeqTransformer
from model.utils.common import get_device
from model.utils.config import Config, load_config
from processing.utils.paths import RUNS

DEFAULT_CONFIG = "model/configs/base.yaml"


def translate(model, text: str, src_tokenizer, tgt_tokenizer, cfg: Config, device: torch.device) -> str:
    ids = [BOS_ID, *src_tokenizer.encode(text).ids, EOS_ID]
    src = torch.tensor([ids], device=device)
    src_pad_mask = torch.zeros_like(src, dtype=torch.bool)
    generated = greedy_decode(model, src, src_pad_mask, cfg.inference.max_len)
    return tgt_tokenizer.decode(generated[0].tolist(), skip_special_tokens=True)


def main() -> None:
    config_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_CONFIG
    cfg = load_config(config_path)
    checkpoint_path = sys.argv[2] if len(sys.argv) > 2 else str(RUNS / cfg.run_name / "checkpoints" / "last.pt")

    device = get_device()
    src_tokenizer = load_tokenizer(cfg.data.src_lang)
    tgt_tokenizer = load_tokenizer(cfg.data.tgt_lang)
    model = Seq2SeqTransformer.from_config(cfg, src_tokenizer.get_vocab_size(), tgt_tokenizer.get_vocab_size(), PAD_ID, device)
    load_checkpoint(checkpoint_path, model, map_location=device)
    model.eval()

    print(f"[translate] loaded {checkpoint_path} — type {cfg.data.src_lang} text, Ctrl+D to exit", file=sys.stderr)
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        print(translate(model, line, src_tokenizer, tgt_tokenizer, cfg, device))


if __name__ == "__main__":
    main()
