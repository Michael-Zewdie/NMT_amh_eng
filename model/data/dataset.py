"""
model.data.dataset — loads a prepared id-sequence split (see
model/data/prepare.py) into a torch Dataset/DataLoader with dynamic padding.

Mask convention used throughout this project: boolean mask where True means
"block this position" (matches torch's additive-mask-friendly convention and
is documented once here rather than re-derived per file).
"""
from __future__ import annotations

import pickle
from dataclasses import dataclass

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

from model.data.tokenizer import PAD_ID
from model.utils.config import Config
from processing.utils.paths import PREPARED


@dataclass
class Batch:
    src: torch.Tensor          # [B, S]
    src_pad_mask: torch.Tensor  # [B, S] bool, True = pad
    tgt_in: torch.Tensor       # [B, T-1]
    tgt_out: torch.Tensor      # [B, T-1]
    tgt_pad_mask: torch.Tensor  # [B, T-1] bool, True = pad

    def to(self, device: torch.device) -> "Batch":
        return Batch(
            src=self.src.to(device),
            src_pad_mask=self.src_pad_mask.to(device),
            tgt_in=self.tgt_in.to(device),
            tgt_out=self.tgt_out.to(device),
            tgt_pad_mask=self.tgt_pad_mask.to(device),
        )


class TranslationDataset(Dataset):
    def __init__(self, split: str, cfg: Config):
        pair_dir = PREPARED / f"{cfg.data.src_lang}-{cfg.data.tgt_lang}"
        with open(pair_dir / f"{split}.pkl", "rb") as f:
            raw = pickle.load(f)

        self.src: list[list[int]] = []
        self.tgt: list[list[int]] = []
        for s, t in zip(raw["src"], raw["tgt"]):
            if len(s) <= cfg.data.max_src_len and len(t) <= cfg.data.max_tgt_len:
                self.src.append(s)
                self.tgt.append(t)

        self.lengths = [len(s) for s in self.src]

    def __len__(self) -> int:
        return len(self.src)

    def __getitem__(self, idx: int) -> tuple[list[int], list[int]]:
        return self.src[idx], self.tgt[idx]


def collate_fn(batch: list[tuple[list[int], list[int]]]) -> Batch:
    src_list, tgt_list = zip(*batch)

    src = pad_sequence([torch.tensor(s) for s in src_list], batch_first=True, padding_value=PAD_ID)
    tgt = pad_sequence([torch.tensor(t) for t in tgt_list], batch_first=True, padding_value=PAD_ID)

    tgt_in = tgt[:, :-1]
    tgt_out = tgt[:, 1:]

    return Batch(
        src=src,
        src_pad_mask=src == PAD_ID,
        tgt_in=tgt_in,
        tgt_out=tgt_out,
        tgt_pad_mask=tgt_in == PAD_ID,
    )


def make_dataloader(split: str, cfg: Config, shuffle: bool | None = None) -> DataLoader:
    dataset = TranslationDataset(split, cfg)
    if shuffle is None:
        shuffle = split == "train"
    return DataLoader(
        dataset,
        batch_size=cfg.data.batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn,
        num_workers=cfg.data.num_workers,
        pin_memory=True,
    )
