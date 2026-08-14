"""The two arms, and the shared build that tokenizes either one.

ARMS below is the whole of what differs between them: where the text comes from,
and which finished run's manifest supplies the hyperparameters. Everything else
in this package takes `arm` only as a string it looks up here.

    NARROW  Gezmu et al.'s own 140k split — one register, ~83% Watchtower/Bible
    BROAD   the 6 other sources pooled and size-matched (corpus.py), gezmu excluded

Vocabulary is fit FRESH PER ARM rather than shared. That is deliberate and is the
opposite of experiments.stratification, which fits one vocab for all three of its
arms: those are partitions of a single pool, so any vocab is equally native to all
of them, whereas these two arms are genuinely different corpora. A shared vocab
would be native to whichever arm it was fit on and foreign to the other, which is
an asymmetry pointing straight at the variable under test. Fitting per arm removes
it — at the cost that a model can only be scored on the other arm's test set from
raw text, never from the .pkl caches (see evaluate.py).

Config is chained rather than duplicated: narrow clones am-en-gezmu-8k's
hyperparameters, and broad clones NARROW'S OWN run manifest. So every training
knob is read fresh from disk on each launch and the two arms cannot drift apart —
any BLEU gap is attributable to the corpus rather than to a stale copy of a
learning rate.
"""
import json
import pickle
import sys
from dataclasses import dataclass, field
from typing import Callable

import pandas as pd

from experiments.domain_breadth.paths import (
    FINAL_BROAD, GEZMU, PREPARED_DIRS, RUN_NAMES, TOK_DIRS, prepared_dir,
)
from model.common import BOS_ID, EOS_ID
from model.tokenize.preprocess import moses_en, translit_am
from process.utils.paths import ROOT, RUNS

SPLITS = ["train", "validation", "test"]
VOCAB_SIZE = 8000
SPECIALS = ["<pad>", "<unk>", "<s>", "</s>"]
MAX_LEN = 150

# Gezmu ships its splits as raw text files under these stems; "dev" is what this
# package calls "validation" everywhere else.
GEZMU_STEMS = {"train": "train", "validation": "dev", "test": "test"}


def read_narrow(split: str) -> tuple[list[str], list[str]]:
    stem = GEZMU_STEMS[split]
    am = (GEZMU / f"{stem}.am-en.base.am").read_text(encoding="utf-8").splitlines()
    en = (GEZMU / f"{stem}.am-en.base.en").read_text(encoding="utf-8").splitlines()
    assert len(am) == len(en), f"{split}: {len(am)} am vs {len(en)} en"
    return am, en


def read_broad(split: str) -> tuple[list[str], list[str]]:
    df = pd.read_csv(FINAL_BROAD / f"{split}.csv", usecols=["am", "en"], dtype=str).dropna()
    return df["am"].tolist(), df["en"].tolist()


@dataclass(frozen=True)
class Arm:
    name: str
    read: Callable[[str], tuple[list[str], list[str]]]
    config_from: str                          # run whose manifest pins the hyperparameters
    overrides: dict = field(default_factory=dict)   # cfg tweaks applied on top

    @property
    def run(self) -> str:
        return RUN_NAMES[self.name]

    @property
    def tok_dir(self):
        return TOK_DIRS[self.name]

    @property
    def prepared(self):
        return prepared_dir(self.name)


ARMS = {
    "narrow": Arm("narrow", read_narrow, config_from="am-en-gezmu-8k",
                  # The Gezmu baseline predates tied embeddings; the paper's recipe
                  # uses them, and broad inherits this through narrow's manifest.
                  overrides={"model": {"tie_embeddings": True}}),
    "broad":  Arm("broad", read_broad, config_from=RUN_NAMES["narrow"]),
}


def make_config(arm: Arm) -> dict:
    """arm.config_from's own config, with only tokenization and data paths changed."""
    # A source run may have been archived (checkpoints are ~4.6GB); its manifest is
    # the only part needed here, so fall back to archive/runs/.
    manifest = RUNS / arm.config_from / "manifest.json"
    if not manifest.exists():
        manifest = ROOT / "archive" / "runs" / arm.config_from / "manifest.json"
    if not manifest.exists():
        sys.exit(f"cannot find {arm.config_from}'s manifest in runs/ or archive/runs/ — "
                 f"needed to pin {arm.name}'s hyperparameters")
    cfg = json.loads(manifest.read_text())["config"]
    cfg["run_name"] = arm.run
    cfg["data"]["prepared_dir"] = PREPARED_DIRS[arm.name]
    cfg["data"]["src_tokenizer"] = str(arm.tok_dir)
    cfg["data"]["tgt_tokenizer"] = str(arm.tok_dir)
    cfg["training"]["resume_from"] = None      # a source run may have been resumed; this starts clean
    for section, values in arm.overrides.items():
        cfg[section].update(values)
    return cfg


def cmd_build(args) -> None:
    """Moses + AT4MT-transliterate, fit this arm's shared vocab on its own train
    split, encode every split, drop over-length pairs, cache as .pkl."""
    from tokenizers.implementations import SentencePieceUnigramTokenizer

    arm = ARMS[args.arm]
    splits, example = {}, None
    for split in SPLITS:
        am, en = arm.read(split)
        splits[split] = ([translit_am(s) for s in am], [moses_en(s) for s in en])
        if split == "train":
            example = am[0]        # kept from this read; re-reading train to print
                                   # one line costs a full re-parse of the corpus
        print(f"[build] {split}: {len(am):,} pairs Moses-tokenized + transliterated", flush=True)
    print(f"[build]   example: {example[:55]}")
    print(f"[build]        -> {splits['train'][0][0][:55]}")

    tr_am, tr_en = splits["train"]
    tok = SentencePieceUnigramTokenizer()
    tok.train_from_iterator(tr_am + tr_en, vocab_size=VOCAB_SIZE,
                            special_tokens=SPECIALS, unk_token="<unk>")
    arm.tok_dir.mkdir(parents=True, exist_ok=True)
    tok.save(str(arm.tok_dir / "tokenizer.json"))
    print(f"[build] fresh shared {VOCAB_SIZE} vocab (fit on THIS arm's train) -> {arm.tok_dir}")

    arm.prepared.mkdir(parents=True, exist_ok=True)
    for split, (am, en) in splits.items():
        s = [[BOS_ID, *e.ids, EOS_ID] for e in tok.encode_batch(am)]
        t = [[BOS_ID, *e.ids, EOS_ID] for e in tok.encode_batch(en)]
        keep = [i for i, (a, b) in enumerate(zip(s, t)) if len(a) <= MAX_LEN and len(b) <= MAX_LEN]
        with open(arm.prepared / f"{split}.pkl", "wb") as f:
            pickle.dump({"src": [s[i] for i in keep], "tgt": [t[i] for i in keep]}, f)
        print(f"[build] {split}: kept {len(keep):,} of {len(am):,}")
