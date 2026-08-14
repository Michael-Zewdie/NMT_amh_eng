"""
model.translate — translate Amharic on the command line, or from stdin.

Run (from the project root):
    python -m model.translate "የኢትዮጵያ መንግሥት አዲስ ፖሊሲ አስታወቀ"   # one-shot
    python -m model.translate                                    # REPL, Ctrl+D to exit
    cat lines.txt | python -m model.translate                    # a file, one line each

The text is positional; the model is chosen with --config / --checkpoint.
Defaults: DEFAULT_CONFIG below, and that run's checkpoints/last.pt.

SOURCE PREPROCESSING IS NOT OPTIONAL AND IS NOT ONE FIXED CHAIN.
A model only understands text preprocessed the way its training data was, and
this project has two incompatible recipes in play:

  normalize   Ethiopic kept as-is, run through process.clean.normalize.
              The base*/small* runs. Skipping it costs ~1.8 BLEU on FLORES
              (17.35 -> 15.57 for am-en-base-v6), more than beam search gains.

  translit    Moses-tokenized, then Ethiopic -> Latin via AT4MT, and the
              generated English detokenized on the way back out. Every run
              since the Gezmu reproduction: am-en-narrow, am-en-broad, and
              anything else built on a data/tokenizer/shared_translit* vocab.

Feeding raw Ethiopic to a translit model is not a small mismatch -- its
vocabulary has no Ethiopic in it at all, so the input arrives 50% <unk> and
decodes to whitespace. --preprocess defaults to `auto`, which probes the
tokenizer with a known Ethiopic string and picks the chain that vocabulary can
actually read, so the right thing happens without the caller having to know
which recipe a checkpoint came from.
"""
import argparse
import sys

import polars as pl
import torch

from model.common import BOS_ID, EOS_ID, UNK_ID, load_for_inference
from model.configs.config import Config
from model.search.decode import decode
from model.tokenize.preprocess import detok_en, translit_am
from process.clean.normalize import normalize

DEFAULT_CONFIG = "runs/am-en-broad/config.yaml"

# Any Amharic sentence works; this one is only ever fed to the tokenizer.
_PROBE = "የኢትዮጵያ መንግሥት አዲስ ፖሊሲ አስታወቀ ።"


def normalize_source(text: str) -> str:
    """Apply the same Amharic normalization the training corpus went through."""
    df = normalize(pl.DataFrame({"am": [text], "en": [""]}), "am", "en", name=None)
    return df["am"][0]


def detect_recipe(src_tokenizer) -> str:
    """'translit' or 'normalize', decided by what the vocabulary can encode.

    A transliterated vocabulary is Latin-only, so raw Ethiopic comes back mostly
    <unk> (measured: 50% on data/tokenizer/shared_translit, versus 0% once
    transliterated). A normalize-recipe vocabulary encodes Ethiopic natively.
    The gap is wide enough that any threshold in between works.
    """
    ids = src_tokenizer.encode(_PROBE).ids
    unk_rate = sum(1 for i in ids if i == UNK_ID) / max(len(ids), 1)
    return "translit" if unk_rate > 0.2 else "normalize"


def translate(model, text: str, src_tokenizer, tgt_tokenizer, cfg: Config,
              device: torch.device, recipe: str = "normalize") -> str:
    prepared = translit_am(text) if recipe == "translit" else normalize_source(text)
    ids = [BOS_ID, *src_tokenizer.encode(prepared).ids, EOS_ID]
    src = torch.tensor([ids], device=device)
    src_pad_mask = torch.zeros_like(src, dtype=torch.bool)
    generated = decode(model, src, src_pad_mask, cfg)
    out = tgt_tokenizer.decode(generated[0].tolist(), skip_special_tokens=True)
    # A translit model emits Moses-tokenized English ("do n't", " ."); undo it.
    return detok_en(out) if recipe == "translit" else out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.strip().splitlines()[1],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("text", nargs="*", help="Amharic text; omit to read stdin")
    ap.add_argument("--config", default=DEFAULT_CONFIG, help=f"default {DEFAULT_CONFIG}")
    ap.add_argument("--checkpoint", help="default: that run's checkpoints/last.pt")
    ap.add_argument("--preprocess", choices=["auto", "translit", "normalize"], default="auto",
                    help="source chain; 'auto' probes the tokenizer (default)")
    args = ap.parse_args()

    cfg, model, src_tok, tgt_tok, device = load_for_inference(args.config, args.checkpoint)
    recipe = detect_recipe(src_tok) if args.preprocess == "auto" else args.preprocess
    how = f"{recipe} (auto-detected)" if args.preprocess == "auto" else recipe
    print(f"[translate] {cfg.run_name} — {how} preprocessing, "
          f"beam {cfg.inference.beam_size}", file=sys.stderr)

    if args.text:
        print(translate(model, " ".join(args.text), src_tok, tgt_tok, cfg, device, recipe))
        return

    if sys.stdin.isatty():
        print(f"[translate] type {cfg.data.src_lang} text, Ctrl+D to exit", file=sys.stderr)
    for line in sys.stdin:
        line = line.strip()
        if line:
            print(translate(model, line, src_tok, tgt_tok, cfg, device, recipe), flush=True)


if __name__ == "__main__":
    main()
