"""
model.evaluate — decode + sacrebleu corpus BLEU on a data split.

Run (from the project root): python -m model.evaluate [config_path] [checkpoint_path] [split]
Defaults: model/configs/base_v6.yaml, <run_dir>/checkpoints/last.pt, "validation".

Decoding strategy comes from `inference.beam_size` in the config (see
model.search): 1 is greedy, >1 is beam search with `inference.length_penalty`.
Neither uses a KV cache, so both re-run the decoder over the full prefix each
step — correct, not fast. Batch-level early stopping: the loop exits once every
sequence has produced EOS, rather than always running to inference.max_len.
"""
import sys

import sacrebleu
import torch

from model.common import describe_decoding, load_for_inference, load_tokenizer
from model.config import Config
from model.data.dataset import make_dataloader
from model.search import decode

DEFAULT_CONFIG = "model/configs/base_v6.yaml"


def corpus_scores(hyps: list[str], refs: list[str]) -> dict[str, float]:
    """Corpus BLEU + chrF++ for a hypothesis/reference list pair.

    The single scoring function for the whole project, so our model and any
    external baseline (model.evaluate_benchmark, baselines.google_translate) are
    never compared across different metric settings — the usual way MT numbers
    end up quietly incomparable.

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


def evaluate_loader(model, loader, tgt_tokenizer, cfg: Config, device: torch.device,
                    beam_size: int | None = None, return_all: bool = False):
    """Corpus BLEU over an already-built DataLoader. Shared by evaluate_split
    (a full data split, e.g. for a standalone post-training run) and
    model.train's periodic in-training eval (a small fixed subset, since decode
    has no KV cache and is too slow to run on a full split every
    eval_every_steps).

    `beam_size` overrides `cfg.inference.beam_size`. model.train passes 1 so
    in-training eval stays greedy no matter how the run will finally be scored:
    beam search costs ~beam_size times as much, and a config asking for beam 4
    would otherwise quadruple every periodic eval — turning a 250k-step run into
    mostly eval. Checkpoint selection therefore compares greedy scores against
    greedy scores, which is the comparison that matters for picking best.pt.

    Returns BLEU alone by default (what every existing caller wants); pass
    `return_all=True` for the full {"bleu", "chrf++"} dict plus sentence count,
    e.g. model.rescore_indist reporting both metrics like model.rescore does.
    """
    hyps: list[str] = []
    refs: list[str] = []
    model.eval()
    for batch in loader:
        batch = batch.to(device)
        generated = decode(model, batch.src, batch.src_pad_mask, cfg, beam_size=beam_size)
        hyps.extend(tgt_tokenizer.decode(ids, skip_special_tokens=True) for ids in generated.tolist())

        full_tgt = torch.cat([batch.tgt_in[:, :1], batch.tgt_out], dim=1)  # reconstruct [BOS, ..., EOS]
        refs.extend(tgt_tokenizer.decode(ids, skip_special_tokens=True) for ids in full_tgt.tolist())

    scores = corpus_scores(hyps, refs)
    if return_all:
        return {**scores, "n_sentences": len(refs)}
    return scores["bleu"]


def evaluate_split(model, split: str, cfg: Config, device: torch.device,
                   beam_size: int | None = None, return_all: bool = False):
    tgt_tokenizer = load_tokenizer(cfg.data.tgt_lang)
    loader = make_dataloader(split, cfg, shuffle=False)
    return evaluate_loader(model, loader, tgt_tokenizer, cfg, device, beam_size=beam_size,
                            return_all=return_all)


def main() -> None:
    config_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_CONFIG
    checkpoint_path = sys.argv[2] if len(sys.argv) > 2 else None
    split = sys.argv[3] if len(sys.argv) > 3 else "validation"

    cfg, model, _, _, device = load_for_inference(config_path, checkpoint_path)
    print(f"[evaluate] decoding: {describe_decoding(cfg)}")
    print(f"[evaluate] {split} BLEU: {evaluate_split(model, split, cfg, device):.2f}")


if __name__ == "__main__":
    main()
