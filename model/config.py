"""
model.config — YAML → dotted-access config object.

Every key becomes an attribute and nested mappings nest, so `cfg.model.d_model`
and `cfg.training.accum_steps` read exactly as they did under the previous
dataclass schema. Config is a dict subclass, so `dict(cfg)`, `.keys()` and
`in` all still work.

Tradeoff vs. the dataclass version this replaced: there is no schema, so a
missing or misspelled key surfaces as an AttributeError at the point of use
rather than a TypeError at load time. The error names the key and lists what
the section does contain, which is usually enough to spot a typo immediately.
"""
from pathlib import Path

import yaml


class Config(dict):
    def __init__(self, mapping: dict):
        super().__init__({k: Config(v) if isinstance(v, dict) else v for k, v in mapping.items()})

    def __getattr__(self, key: str):
        try:
            return self[key]
        except KeyError:
            raise AttributeError(f"no config key {key!r} — this section has: {', '.join(self)}") from None


def load_config(path: str | Path) -> Config:
    return Config(yaml.safe_load(Path(path).read_text()))
