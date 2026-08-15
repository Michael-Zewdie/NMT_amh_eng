"""
model.common — shared runtime pieces: special-token ids, tokenizer loading,
device/seed setup, checkpoint save/load, and the "load a trained model" preamble
that every inference entry point needs.

Special-token ids match data/tokenizer/{en,am}/tokenizer.json exactly (see
model/tokenize/train_tokenizer.py and model/tokenize/train_tokenizer_am.py) —
import them from here rather than hardcoding. Neither tokenizer has a
post_processor configured, so encode(...).ids does NOT include BOS/EOS; callers add them.
"""
import random
from pathlib import Path

import numpy as np
import sacrebleu
import torch
from tokenizers import Tokenizer

from model.configs.config import Config, load_config
from model.architecture.transformer import Seq2SeqTransformer
from process.utils.paths import RUNS, TOKENIZER_AM, TOKENIZER_EN

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
    f = path / "tokenizer.json"
    if not f.exists():
        raise FileNotFoundError(
            f"no tokenizer at {f}" + ("" if tokenizer_dir else
            f" — this is the DEFAULT for lang={lang!r}, used because the config named no "
            f"data.{'src' if lang == 'am' else 'tgt'}_tokenizer. Point the config at the "
            f"tokenizer this run's prepared/*.pkl was built with."))
    return Tokenizer.from_file(str(f))


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    """CUDA, else Apple Silicon's MPS, else CPU.

    MPS is worth several times CPU for decoding on a Mac. It is not used for
    training: model.training.train autocasts to bf16, which MPS does not
    support, and no run in this project was trained on one.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def save_checkpoint(path, model, optimizer, scheduler, step: int, best_bleu: float | None = None) -> None:
    """`best_bleu` is persisted so a resumed run can restore it — see
    load_best_bleu. Optional, and omitted for weight-only snapshots."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "step": step,
            "best_bleu": best_bleu,
        },
        path,
    )


def save_weights_only(path, model) -> None:
    """Weights alone, for the rolling snapshots that average_checkpoints reads.

    Optimizer + scheduler state roughly triples a checkpoint's size and is
    meaningless for averaging, so keeping 12 rolling snapshots costs ~12 x 100MB
    rather than ~12 x 600MB.
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict()}, path)


def load_checkpoint(path, model, optimizer=None, scheduler=None, map_location=None) -> int:
    ckpt = torch.load(path, map_location=map_location, weights_only=True)
    model.load_state_dict(ckpt["model"])
    if optimizer is not None:
        optimizer.load_state_dict(ckpt["optimizer"])
    if scheduler is not None:
        scheduler.load_state_dict(ckpt["scheduler"])
    return ckpt["step"]


def load_best_bleu(path, map_location=None) -> float:
    """The best val BLEU recorded in a checkpoint, or -1.0 if absent.

    Restoring this on resume is not cosmetic: model.training.train compares each
    eval against it to decide whether to overwrite best.pt. Starting a resumed
    run at -1.0 makes the first eval unconditionally "best", silently replacing a
    genuinely better pre-interruption checkpoint with a worse one. Checkpoints
    written before this field existed return -1.0, preserving the old behaviour
    rather than crashing.
    """
    ckpt = torch.load(path, map_location=map_location, weights_only=True)
    value = ckpt.get("best_bleu")
    return -1.0 if value is None else float(value)


def average_checkpoints(paths, model, map_location=None) -> None:
    """Load the arithmetic mean of several checkpoints' weights into `model`.

    Gezmu et al. (and Vaswani et al. before them) decode from an average of the
    last N checkpoints rather than a single one; measured at +0.47 BLEU on
    am-en-gezmu-8k. Averaging happens in float64 to avoid drift across a dozen
    float32 tensors, then casts back to each parameter's own dtype.
    """
    paths = list(paths)
    if not paths:
        raise ValueError("no checkpoints to average")
    acc: dict[str, torch.Tensor] = {}
    for p in paths:
        state = torch.load(p, map_location=map_location, weights_only=True)["model"]
        for k, v in state.items():
            acc[k] = v.to(torch.float64) if k not in acc else acc[k] + v.to(torch.float64)
    target = model.state_dict()
    model.load_state_dict({k: (v / len(paths)).to(target[k].dtype) for k, v in acc.items()})


def load_for_inference(config_path, checkpoint_path=None) -> tuple:
    """Config + eval-mode model + both tokenizers + device, ready to decode.

    Shared by model.evaluate.evaluate_in_dist, model.evaluate.evaluate_OOD and
    model.translate, which otherwise repeat this preamble verbatim. checkpoint_path defaults to
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


def corpus_scores(hyps: list[str], refs: list[str]) -> dict[str, float]:
    """Corpus BLEU + chrF++ for a hypothesis/reference list pair.

    The single scoring function for the whole project, so model.evaluate.evaluate_in_dist
    and model.evaluate.evaluate_OOD are never compared across different metric
    settings — the usual way MT numbers end up quietly incomparable.

    Default sacrebleu tokenization (13a) is correct for an English target. A
    future en->am direction will need tokenize="none" — no Amharic-aware
    tokenizer exists in sacrebleu.

    chrF++ (chrF with word_order=2) is reported alongside BLEU because it is what
    FLORES-200 tables use, and because character n-grams degrade more gracefully
    than word n-grams when a system is weak — a low-BLEU model can still show
    real signal in chrF++.
    """
    return {
        "bleu": sacrebleu.corpus_bleu(hyps, [refs]).score,
        "chrf++": sacrebleu.corpus_chrf(hyps, [refs], word_order=2).score,
    }
