"""The three arms, and the ONE shared vocabulary they are all tokenized with.

ARM_SPLITTERS below is the whole of what differs between arms — three calls
into process.pool, one per stratification axis. Everything else in this
package takes `arm` only as a string it interpolates into a path.

Vocabulary is fit ONCE and shared across all three arms (see fit_shared_vocab),
not per-arm the way experiments.domain_breadth.arms fits fresh per arm. That
experiment fits per-arm because it compares two genuinely DIFFERENT corpora (gezmu vs
broad), where a shared vocab is native to one and foreign to the other — fitting
fresh per arm removes that asymmetry. That asymmetry does not exist here: all
three split-strategy arms are partitions of the SAME 432,781-row pool, so any
vocab fit on any one arm's train text is equally native/foreign to the others.
Fitting once (on the length arm's train.csv, the split least entangled with
either axis under test) and sharing it keeps vocabulary a true constant instead
of introducing arm-to-arm wobble from fitting three near-identical-but-not-
identical vocabularies — the same "one shared vocab as a controlled constant"
principle already used by am-en-gezmu-8k/am-en-broad-8k sharing one Gezmu-fit
vocab by design (EXPERIMENTS.md: "Broad wins anyway, which makes the result
stronger, not weaker").

Note the ordering constraint this creates, and why build() is one sequence
rather than three independent per-arm pipelines: the shared vocab is fit on
VOCAB_FIT_ARM's train.csv, so EVERY arm's tokenization depends on the length
arm's split already existing on disk. Splits are therefore written for all
three arms, then the vocab is fit, then all three are tokenized.
"""
import pickle

import pandas as pd

from experiments.stratification.corpus import assemble_pool
from experiments.stratification.paths import TOK_DIR, final_dir, prepared_dir
from model.common import BOS_ID, EOS_ID
from model.tokenize.preprocess import moses_en, translit_am
from process import pool as pool_mod

# The only arm-specific code in this package. Adding a fourth stratification
# axis means adding one entry here — ARMS, the CLI's --arm choices and eval's
# comparison table all derive from it.
ARM_SPLITTERS = {
    "length":   lambda pooled: pool_mod.split(pooled[["am", "en"]]),
    "domain":   lambda pooled: pool_mod.split_domain(pooled[["am", "en", "_domain"]],
                                                     domain_col="_domain"),
    "semantic": lambda pooled: pool_mod.split_semantic(pooled[["am", "en"]]),
}
ARMS = list(ARM_SPLITTERS)

SPLITS = ["train", "validation", "test"]
VOCAB_SIZE = 8000
VOCAB_FIT_ARM = "length"   # neutral reference split the shared vocab is fit on — see module docstring
SPECIALS = ["<pad>", "<unk>", "<s>", "</s>"]
MAX_LEN = 150


def write_split(arm: str, pooled: pd.DataFrame) -> None:
    """Run one arm's splitter with process.pool's module-level FINAL output dir
    temporarily pointed at this arm's own directory — pool's split functions
    write to data/final/ by default, and all three arms must land side by side
    instead of overwriting each other."""
    out = final_dir(arm)
    out.mkdir(parents=True, exist_ok=True)
    original, pool_mod.FINAL = pool_mod.FINAL, out
    try:
        ARM_SPLITTERS[arm](pooled)
    finally:
        pool_mod.FINAL = original


def translit_pairs(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    """(am, en) columns run through the same Moses + AT4MT transform training
    will see: translit_am Moses-tokenizes THEN transliterates Amharic to Latin,
    moses_en Moses-tokenizes English. See model/tokenize/preprocess.py."""
    return [translit_am(s) for s in df["am"]], [moses_en(s) for s in df["en"]]


def fit_shared_vocab() -> None:
    """Fit ONE shared Unigram vocabulary on VOCAB_FIT_ARM's train split only
    (never val/test — matches experiments.domain_breadth.arms' train-only fit),
    shared by all three arms. See module docstring for why one shared vocab is
    the right choice here (unlike domain_breadth's per-arm fit)."""
    from tokenizers.implementations import SentencePieceUnigramTokenizer

    df = pd.read_csv(final_dir(VOCAB_FIT_ARM) / "train.csv", usecols=["am", "en"], dtype=str).dropna()
    am, en = translit_pairs(df)
    print(f"[build] fitting shared {VOCAB_SIZE} vocab on {VOCAB_FIT_ARM}/train.csv ({len(df):,} pairs)")
    print(f"[build]   example: {df['am'].iloc[0][:55]}")
    print(f"[build]        -> {am[0][:55]}")

    tok = SentencePieceUnigramTokenizer()
    tok.train_from_iterator(am + en, vocab_size=VOCAB_SIZE, special_tokens=SPECIALS, unk_token="<unk>")
    TOK_DIR.mkdir(parents=True, exist_ok=True)
    tok.save(str(TOK_DIR / "tokenizer.json"))
    print(f"[build] shared vocab -> {TOK_DIR}")


def tokenize_arm(arm: str, tok) -> None:
    """Translit + tokenize this arm's final/*.csv with the ONE shared vocab
    fit_shared_vocab() produced — same vocab object for every arm, by design
    (see module docstring)."""
    out_dir = prepared_dir(arm)
    out_dir.mkdir(parents=True, exist_ok=True)
    for split in SPLITS:
        df = pd.read_csv(final_dir(arm) / f"{split}.csv", usecols=["am", "en"], dtype=str).dropna()
        am, en = translit_pairs(df)
        s = [[BOS_ID, *e.ids, EOS_ID] for e in tok.encode_batch(am)]
        t = [[BOS_ID, *e.ids, EOS_ID] for e in tok.encode_batch(en)]
        keep = [i for i, (a, b) in enumerate(zip(s, t)) if len(a) <= MAX_LEN and len(b) <= MAX_LEN]
        with open(out_dir / f"{split}.pkl", "wb") as f:
            pickle.dump({"src": [s[i] for i in keep], "tgt": [t[i] for i in keep]}, f)
        print(f"[build]   {arm}/{split}: kept {len(keep):,} of {len(df):,}")


def cmd_build(_args) -> None:
    from tokenizers import Tokenizer

    pooled = assemble_pool()
    for arm in ARMS:
        print(f"\n[build] === {arm} split ===")
        write_split(arm, pooled)

    fit_shared_vocab()
    tok = Tokenizer.from_file(str(TOK_DIR / "tokenizer.json"))
    for arm in ARMS:
        print(f"\n[build] === {arm} tokenize ===")
        tokenize_arm(arm, tok)
