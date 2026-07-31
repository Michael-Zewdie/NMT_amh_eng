from pathlib import Path

import torch


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
