"""
model.utils.config — dataclass schema + YAML loader for training configs.

A missing or extra key in the YAML fails loudly via TypeError (dataclasses
reject unknown/missing kwargs) rather than silently defaulting — no schema
validation library needed for a config this small.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import yaml


@dataclass
class ModelConfig:
    d_model: int
    n_heads: int
    n_encoder_layers: int
    n_decoder_layers: int
    d_ff: int
    dropout: float
    max_len: int
    tie_output_projection: bool


@dataclass
class DataConfig:
    src_lang: str
    tgt_lang: str
    max_src_len: int
    max_tgt_len: int
    batch_size: int
    num_workers: int


@dataclass
class TrainingConfig:
    warmup_steps: int
    label_smoothing: float
    grad_clip_norm: float
    accum_steps: int
    max_steps: int
    amp_dtype: str
    seed: int
    save_every_steps: int
    eval_every_steps: int
    eval_subset_size: int
    log_every_steps: int
    resume_from: Optional[str]


@dataclass
class InferenceConfig:
    beam_size: int
    max_len: int
    length_penalty: float


@dataclass
class Config:
    run_name: str
    model: ModelConfig
    data: DataConfig
    training: TrainingConfig
    inference: InferenceConfig


def load_config(path: str | Path) -> Config:
    raw = yaml.safe_load(Path(path).read_text())
    return Config(
        run_name=raw["run_name"],
        model=ModelConfig(**raw["model"]),
        data=DataConfig(**raw["data"]),
        training=TrainingConfig(**raw["training"]),
        inference=InferenceConfig(**raw["inference"]),
    )
