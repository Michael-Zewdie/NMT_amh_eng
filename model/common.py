"""
model.common — shared runtime pieces: special-token ids, tokenizer loading,
device/seed setup, checkpoint save/load, and the "load a trained model" preamble
that every inference entry point needs.

Special-token ids match data/tokenizer/{en,am}/tokenizer.json exactly (see
processing/train_tokenizer.py and processing/train_tokenizer_am.py) — import
them from here rather than hardcoding. Neither tokenizer has a post_processor
configured, so encode(...).ids does NOT include BOS/EOS; callers add them.
"""
import random
from pathlib import Path

import numpy as np
import torch
from tokenizers import Tokenizer

from model.config import Config, load_config
from model.transformer import Seq2SeqTransformer
from processing.utils.paths import RUNS, TOKENIZER_AM, TOKENIZER_EN

PAD_ID, UNK_ID, BOS_ID, EOS_ID = 0, 1, 2, 3

_TOKENIZER_PATHS = {"am": TOKENIZER_AM, "en": TOKENIZER_EN}


def load_tokenizer(lang: str, tokenizer_dir: str | Path | None = None) -> Tokenizer:
    """Load a tokenizer, defaulting to the current data/tokenizer/{am,en}.

    `tokenizer_dir` exists because a checkpoint is only loadable with the exact
    tokenizer it was trained against — vocabulary size is baked into the
    embedding and (tied) output projection. am-en-base-v4 was trained on the 32k
    tokenizers and will not load against today's 8k ones, so its config names its
    own directory via data.src_tokenizer / data.tgt_tokenizer. Runs that omit
    those keys get the default, which is every run since the 8k retrain.
    """
    path = Path(tokenizer_dir) if tokenizer_dir else _TOKENIZER_PATHS[lang]
    return Tokenizer.from_file(str(path / "tokenizer.json"))


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def save_checkpoint(path, model, optimizer, scheduler, step: int) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "step": step,
        },
        path,
    )


def load_checkpoint(path, model, optimizer=None, scheduler=None, map_location=None) -> int:
    ckpt = torch.load(path, map_location=map_location, weights_only=True)
    model.load_state_dict(ckpt["model"])
    if optimizer is not None:
        optimizer.load_state_dict(ckpt["optimizer"])
    if scheduler is not None:
        scheduler.load_state_dict(ckpt["scheduler"])
    return ckpt["step"]


def load_for_inference(config_path, checkpoint_path=None) -> tuple:
    """Config + eval-mode model + both tokenizers + device, ready to decode.

    Shared by model.evaluate, model.evaluate_benchmark and model.translate,
    which otherwise repeat this preamble verbatim. checkpoint_path defaults to
    the run's last.pt — pass best.pt explicitly when that's the one you want.
    """
    cfg = load_config(config_path)
    if checkpoint_path is None:
        checkpoint_path = RUNS / cfg.run_name / "checkpoints" / "last.pt"
    device = get_device()
    src_tokenizer = load_tokenizer(cfg.data.src_lang, cfg.data.get("src_tokenizer"))
    tgt_tokenizer = load_tokenizer(cfg.data.tgt_lang, cfg.data.get("tgt_tokenizer"))
    model = Seq2SeqTransformer.from_config(
        cfg, src_tokenizer.get_vocab_size(), tgt_tokenizer.get_vocab_size(), PAD_ID, device
    )
    load_checkpoint(checkpoint_path, model, map_location=device)
    model.eval()
    return cfg, model, src_tokenizer, tgt_tokenizer, device


def describe_decoding(cfg: Config) -> str:
    """One-line summary of what cfg.inference asks for, for eval banners."""
    k = cfg.inference.beam_size
    return "greedy" if k <= 1 else f"beam {k}, length penalty {cfg.inference.length_penalty}"
