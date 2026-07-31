"""
model.train — training loop entrypoint.

Run (from the project root): python -m model.train [config_path]
config_path defaults to model/configs/base.yaml (no argparse, matching the
repo's existing plain-sys.argv precedent, e.g. collection/collect.py).
"""
import sys
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter

from model.checkpoint import load_checkpoint, save_checkpoint
from model.data.dataset import TranslationDataset, collate_fn, make_dataloader
from model.data.tokenizer import PAD_ID, load_tokenizer
from model.evaluate import evaluate_loader
from model.optim import build_optimizer, build_scheduler
from model.transformer import Seq2SeqTransformer
from model.utils.common import get_device, set_seed
from model.utils.config import Config, load_config
from processing.utils.paths import RUNS

DEFAULT_CONFIG = "model/configs/base.yaml"


def make_eval_subset_loader(cfg: Config) -> DataLoader:
    """A small fixed slice of validation, used for cheap periodic in-training
    eval (loss + BLEU) — see model.evaluate.evaluate_loader's docstring for
    why this stays small rather than covering the whole validation split."""
    dataset = TranslationDataset("validation", cfg)
    subset = Subset(dataset, range(min(cfg.training.eval_subset_size, len(dataset))))
    return DataLoader(subset, batch_size=cfg.data.batch_size, shuffle=False, collate_fn=collate_fn)


def compute_loss(model, batch, criterion, amp_dtype, device) -> torch.Tensor:
    with torch.autocast(device_type=device.type, dtype=amp_dtype):
        logits = model(batch.src, batch.src_pad_mask, batch.tgt_in, batch.tgt_pad_mask)
        return criterion(logits.reshape(-1, logits.size(-1)), batch.tgt_out.reshape(-1))


@torch.no_grad()
def run_eval(model, eval_loader, tgt_tokenizer, criterion, cfg, device, amp_dtype) -> tuple[float, float]:
    model.eval()
    total_loss, n_batches = 0.0, 0
    for batch in eval_loader:
        batch = batch.to(device)
        total_loss += compute_loss(model, batch, criterion, amp_dtype, device).item()
        n_batches += 1
    val_loss = total_loss / n_batches

    bleu = evaluate_loader(model, eval_loader, tgt_tokenizer, cfg, device)
    model.train()
    return val_loss, bleu


def main() -> None:
    config_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_CONFIG
    cfg = load_config(config_path)
    set_seed(cfg.training.seed)
    device = get_device()
    print(f"[train] device={device}  run_name={cfg.run_name}")

    src_tokenizer = load_tokenizer(cfg.data.src_lang)
    tgt_tokenizer = load_tokenizer(cfg.data.tgt_lang)
    train_loader = make_dataloader("train", cfg)
    eval_loader = make_eval_subset_loader(cfg)
    steps_per_epoch = len(train_loader)
    print(f"[train] {steps_per_epoch} steps/epoch ({len(train_loader.dataset)} pairs, batch_size={cfg.data.batch_size})")

    model = Seq2SeqTransformer.from_config(cfg, src_tokenizer.get_vocab_size(), tgt_tokenizer.get_vocab_size(), PAD_ID, device)
    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, cfg)
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_ID, label_smoothing=cfg.training.label_smoothing)

    step = 0
    if cfg.training.resume_from:
        step = load_checkpoint(cfg.training.resume_from, model, optimizer, scheduler, map_location=device)
        print(f"[train] resumed from {cfg.training.resume_from} at step {step}")

    run_dir = RUNS / cfg.run_name
    ckpt_dir = run_dir / "checkpoints"
    writer = SummaryWriter(log_dir=str(run_dir / "tensorboard"))
    amp_dtype = torch.bfloat16 if cfg.training.amp_dtype == "bf16" else torch.float16

    model.train()
    running_loss, running_count, running_tokens = 0.0, 0, 0
    start_time = time.time()
    data_iter = iter(train_loader)

    while step < cfg.training.max_steps:
        optimizer.zero_grad()
        accum_loss = 0.0

        for _ in range(cfg.training.accum_steps):
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(train_loader)
                batch = next(data_iter)
            batch = batch.to(device)

            loss = compute_loss(model, batch, criterion, amp_dtype, device) / cfg.training.accum_steps
            loss.backward()
            accum_loss += loss.item()
            running_tokens += (batch.tgt_out != PAD_ID).sum().item()

        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.training.grad_clip_norm)
        optimizer.step()
        scheduler.step()
        step += 1

        running_loss += accum_loss
        running_count += 1

        if step % cfg.training.log_every_steps == 0:
            avg_loss = running_loss / running_count
            lr = scheduler.get_last_lr()[0]
            elapsed = time.time() - start_time
            tokens_per_sec = running_tokens / elapsed
            epoch = step / steps_per_epoch
            print(f"[train] step {step}/{cfg.training.max_steps}  epoch {epoch:.2f}  loss {avg_loss:.4f}  lr {lr:.6f}  "
                  f"{tokens_per_sec:.0f} tok/s  {elapsed:.1f}s")
            writer.add_scalar("train/loss", avg_loss, step)
            writer.add_scalar("train/lr", lr, step)
            writer.add_scalar("train/tokens_per_sec", tokens_per_sec, step)
            writer.add_scalar("train/epoch", epoch, step)
            running_loss, running_count, running_tokens = 0.0, 0, 0
            start_time = time.time()

        if step % cfg.training.eval_every_steps == 0:
            val_loss, bleu = run_eval(model, eval_loader, tgt_tokenizer, criterion, cfg, device, amp_dtype)
            print(f"[train] step {step}  val_loss {val_loss:.4f}  val_bleu {bleu:.2f}")
            writer.add_scalar("val/loss", val_loss, step)
            writer.add_scalar("val/bleu", bleu, step)

        if step % cfg.training.save_every_steps == 0:
            save_checkpoint(ckpt_dir / "last.pt", model, optimizer, scheduler, step)
            print(f"[train] checkpoint saved at step {step} -> {ckpt_dir / 'last.pt'}")

    save_checkpoint(ckpt_dir / "last.pt", model, optimizer, scheduler, step)
    writer.close()
    print(f"[train] finished at step {step}")


if __name__ == "__main__":
    main()
