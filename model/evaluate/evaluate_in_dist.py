"""
model.evaluate.evaluate_in_dist — decode + sacrebleu corpus BLEU on a data split
drawn from the model's own training pool (data/prepared/*.pkl).

Run (from the project root): python -m model.evaluate.evaluate_in_dist [config_path] [checkpoint_path] [split]
Defaults: model/configs/archive/base_v6.yaml, <run_dir>/checkpoints/last.pt, "validation".

Decoding strategy comes from `inference.beam_size` in the config (see
model.search): 1 is greedy, >1 is beam search with `inference.length_penalty`.
Neither uses a KV cache, so both re-run the decoder over the full prefix each
step — correct, not fast. Batch-level early stopping: the loop exits once every
sequence has produced EOS, rather than always running to inference.max_len.

For scoring against the held-out FLORES/MAFAND benchmarks instead — out of
distribution, never trained on — see model.evaluate.evaluate_OOD.
"""
import sys

import torch

from model.common import corpus_scores, describe_decoding, load_for_inference, load_tokenizer
from model.configs.config import Config
from model.data.dataset import make_dataloader
from model.search.decode import decode

DEFAULT_CONFIG = "model/configs/archive/base_v6.yaml"


def evaluate_loader(model, loader, tgt_tokenizer, cfg: Config, device: torch.device,
                    beam_size: int | None = None) -> float:
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

    return corpus_scores(hyps, refs)["bleu"]


def evaluate_split(model, split: str, cfg: Config, device: torch.device,
                   beam_size: int | None = None) -> float:
    tgt_tokenizer = load_tokenizer(cfg.data.tgt_lang)
    loader = make_dataloader(split, cfg, shuffle=False)
    return evaluate_loader(model, loader, tgt_tokenizer, cfg, device, beam_size=beam_size)


def main() -> None:
    config_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_CONFIG
    checkpoint_path = sys.argv[2] if len(sys.argv) > 2 else None
    split = sys.argv[3] if len(sys.argv) > 3 else "validation"

    cfg, model, _, _, device = load_for_inference(config_path, checkpoint_path)
    print(f"[evaluate] decoding: {describe_decoding(cfg)}")
    print(f"[evaluate] {split} BLEU: {evaluate_split(model, split, cfg, device):.2f}")


if __name__ == "__main__":
    main()
