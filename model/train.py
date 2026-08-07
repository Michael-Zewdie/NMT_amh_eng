"""
model.train — training loop entrypoint.

Run (from the project root): python -m model.train [config_path]
config_path defaults to model/configs/gezmu_8k.yaml (no argparse, matching the
repo's existing plain-sys.argv precedent, e.g. collection/collect.py).
Superseded configs live in model/configs/archive/ and archive/model/configs/.
"""
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter

from model.common import (
    PAD_ID, get_device, load_checkpoint, load_tokenizer, save_checkpoint,
    save_checkpoint_weights_only, set_seed,
)
from model.config import Config, load_config
from model.data.dataset import TranslationDataset, collate_fn, make_dataloader
from model.evaluate import evaluate_loader
from model.optim import build_optimizer, build_scheduler
from model.transformer import Seq2SeqTransformer
from processing.utils.paths import RUNS, PREPARED
from processing.utils.manifest import write_manifest, read_manifest, git_info, now

DEFAULT_CONFIG = "model/configs/gezmu_8k.yaml"  # v2's first config — see its
# header for what this run is and how it relates to archive/model/configs/base_v5.yaml.


def save_rolling_checkpoint(ckpt_dir: Path, model, step: int, avg_n: int) -> None:
    """Keep a rolling window of the last `avg_n` model-only snapshots in
    ckpt_dir/avg/, named by step, for experiments.average_checkpoints to
    average post-hoc (Gezmu et al. §4.2 decode from an average of the last
    twelve). No-op when avg_n <= 0 — every existing config omits
    training.checkpoint_avg_n and is unaffected.
    """
    if avg_n <= 0:
        return
    avg_dir = ckpt_dir / "avg"
    save_checkpoint_weights_only(avg_dir / f"step{step}.pt", model, step)
    kept = sorted(avg_dir.glob("step*.pt"), key=lambda p: int(p.stem.removeprefix("step")))
    for stale in kept[:-avg_n]:
        stale.unlink()


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

    # beam_size=1 regardless of cfg.inference.beam_size — periodic eval stays
    # greedy so a beam-4 config doesn't quadruple its cost. See evaluate_loader.
    bleu = evaluate_loader(model, eval_loader, tgt_tokenizer, cfg, device, beam_size=1)
    model.train()
    return val_loss, bleu


def main() -> None:
    main_start_time = time.time()
    config_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_CONFIG
    cfg = load_config(config_path)
    set_seed(cfg.training.seed)
    device = get_device()
    print(f"[train] device={device}  run_name={cfg.run_name}")

    # Write runs/<name>/manifest.json now, before any training happens, so the
    # exact config + data provenance is captured even if the run later crashes
    # or gets killed. It's updated with final results at the end of main().
    # See processing.utils.manifest for why this exists — reconstructing a past
    # run's hyperparameters/data cutoffs from stdout logs alone doesn't scale.
    run_dir = RUNS / cfg.run_name
    # Same prepared_dir resolution as TranslationDataset (model/data/dataset.py) —
    # an experiment arm with its own data.prepared_dir must get ITS manifest, not
    # production's, or this ends up recording the wrong data provenance for the
    # exact same reason prepared_dir itself exists (see that file's docstring).
    prepared_root = Path(cfg.data.get("prepared_dir") or PREPARED)
    data_manifest_path = prepared_root / f"{cfg.data.src_lang}-{cfg.data.tgt_lang}" / "manifest.json"
    data_manifest = read_manifest(data_manifest_path)
    if data_manifest is None:
        print(f"[train] WARNING: no data manifest at {data_manifest_path} — "
              f"this run's data provenance (thresholds/sources) won't be recorded")
    write_manifest(run_dir, {
        "run_name": cfg.run_name,
        "config_path": config_path,
        "config": dict(cfg),
        "git": git_info(),
        "started_at": now(),
        "data_manifest_path": str(data_manifest_path),
        "data_manifest": data_manifest,
        "status": "running",
    })

    src_tokenizer = load_tokenizer(cfg.data.src_lang)
    tgt_tokenizer = load_tokenizer(cfg.data.tgt_lang)
    train_loader = make_dataloader("train", cfg)
    eval_loader = make_eval_subset_loader(cfg)
    # len(train_loader) counts micro-batches; an "epoch" is in OPTIMIZER steps, of
    # which each consumes accum_steps micro-batches — without the divisor the epoch
    # figure logged below is understated by exactly accum_steps.
    steps_per_epoch = max(1, len(train_loader) // cfg.training.accum_steps)
    print(f"[train] {steps_per_epoch} steps/epoch ({len(train_loader.dataset)} pairs, "
          f"batch_size={cfg.data.batch_size} x accum_steps={cfg.training.accum_steps} "
          f"= {cfg.data.batch_size * cfg.training.accum_steps} sentences/step)")

    model = Seq2SeqTransformer.from_config(cfg, src_tokenizer.get_vocab_size(), tgt_tokenizer.get_vocab_size(), PAD_ID, device)
    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, cfg)
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_ID, label_smoothing=cfg.training.label_smoothing)

    step = 0
    best_bleu = -1.0
    if cfg.training.resume_from:
        step, best_bleu = load_checkpoint(cfg.training.resume_from, model, optimizer, scheduler, map_location=device)
        print(f"[train] resumed from {cfg.training.resume_from} at step {step}, best_bleu={best_bleu:.2f}")

    ckpt_dir = run_dir / "checkpoints"
    avg_n = cfg.training.get("checkpoint_avg_n", 0)
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
            if bleu > best_bleu:
                best_bleu = bleu
                save_checkpoint(ckpt_dir / "best.pt", model, optimizer, scheduler, step, best_bleu)
                print(f"[train] new best val_bleu {bleu:.2f} at step {step} -> {ckpt_dir / 'best.pt'}")

        if step % cfg.training.save_every_steps == 0:
            save_checkpoint(ckpt_dir / "last.pt", model, optimizer, scheduler, step, best_bleu)
            print(f"[train] checkpoint saved at step {step} -> {ckpt_dir / 'last.pt'}")
            save_rolling_checkpoint(ckpt_dir, model, step, avg_n)

    save_checkpoint(ckpt_dir / "last.pt", model, optimizer, scheduler, step, best_bleu)
    save_rolling_checkpoint(ckpt_dir, model, step, avg_n)
    writer.close()
    print(f"[train] finished at step {step}")

    manifest = read_manifest(run_dir / "manifest.json") or {}
    manifest.update({
        "status": "finished",
        "finished_at": now(),
        "final_step": step,
        "best_val_bleu": best_bleu,
        "wall_clock_seconds": round(time.time() - main_start_time, 1),
    })
    write_manifest(run_dir, manifest)


if __name__ == "__main__":
    main()
