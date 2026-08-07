"""
experiments.fixed_size_corpus — build a corpus of a FIXED size from a chosen set
of sources, so data volume can be held constant while composition varies.

Run (from the project root):
    python -m experiments.fixed_size_corpus <name> <source[,source...]|all>

    python -m experiments.fixed_size_corpus nllb    nllb
    python -m experiments.fixed_size_corpus mixed   all

Why
---
am-en-gezmu-only changed TWO things at once against am-en-base-v6: it narrowed
the domain (one source instead of six) AND cut volume 4.4x (93,649 vs 408,826
train pairs). Its FLORES collapsed 17.35 -> 3.14, but that experiment cannot say
how much of the collapse was diversity and how much was simply less data.

Holding volume fixed at gezmu-only's size separates them:

    arm A  gezmu   @ 117,049 pooled   narrow, curated      (already run)
    arm B  nllb    @ 117,049 pooled   diverse, mined
    arm C  all     @ 117,049 pooled   diverse, mixed sources

    B or C vs A  -> effect of DIVERSITY at fixed volume
    C vs am-en-base-v6 (408,826) -> effect of VOLUME at fixed composition

POOL_TARGET is gezmu-only's post-decontamination pooled count, so every arm
splits from an identically sized pool and lands on ~93.6k train pairs.

Subsampling happens AFTER pool() has already deduped and seed-shuffled, so a
head slice is a uniform random draw that preserves each arm's natural source
mix. Cutoffs, tokenizers, and the split are identical to v6 — tokenizers are
deliberately NOT retrained, so vocabulary is held fixed and composition is the
only variable.

Outputs to staging dirs so data/final/ and data/prepared/ (what am-en-base-v6
trained on) are never touched:
    data/final_<name>/{train,validation,test}.csv
    data/prepared_<name>/am-en/{train,validation,test}.pkl
"""
import pickle
import sys

import pandas as pd

from model.common import BOS_ID, EOS_ID, load_tokenizer
from processing.utils import pool as pool_mod
from processing.utils.paths import DATA, PROCESSED
from processing.utils.manifest import write_manifest, git_info, now

POOL_TARGET = 117_049      # gezmu-only's pooled size after decontamination
MAX_LEN = 150              # keep in sync with model/data/prepare.py
SPLITS = ["validation", "test", "train"]
SEED = 42

# Same cutoffs processing/process.py passes, per tier.
MINED = dict(cosine=0.7, africomet=0.80)
CURATED = dict(cosine=0.0, africomet=0.15)
SOURCE_LID_CUTOFF = TARGET_LID_CUTOFF = 0.90


def build(name: str, sources: list[str]) -> None:
    frames = []
    source_tiers = {}
    source_counts = {}
    for stem in sources:
        path = PROCESSED / f"{stem}.csv"
        if not path.exists():
            sys.exit(f"missing {path}")
        curated = stem in pool_mod.CURATED_SOURCES
        tier = CURATED if curated else MINED
        df = pool_mod.load_pairs(path, tier["cosine"], tier["africomet"],
                                 SOURCE_LID_CUTOFF, TARGET_LID_CUTOFF)
        if df is None:
            print(f"[{name}] skipping {path.name} (not am/en parallel text)")
            continue
        print(f"[{name}]   {stem}: {len(df):,} kept")
        source_tiers[stem] = {"curated": curated, **tier}
        source_counts[stem] = len(df)
        frames.append(df)
    if not frames:
        sys.exit("no usable sources")

    pooled_full = pool_mod.pool(frames, seed=SEED)
    pooled = pool_mod.decontaminate(pooled_full)
    if len(pooled) < POOL_TARGET:
        sys.exit(f"[{name}] only {len(pooled):,} pooled pairs, need {POOL_TARGET:,}")
    # pool() already deduped + seed-shuffled, so a head slice is a uniform draw.
    pooled = pooled.head(POOL_TARGET).reset_index(drop=True)
    print(f"[{name}] subsampled to {len(pooled):,} pooled pairs")

    final_out = DATA / f"final_{name}"
    final_out.mkdir(parents=True, exist_ok=True)
    original = pool_mod.FINAL
    pool_mod.FINAL = final_out
    try:
        splits = pool_mod.split(pooled)
    finally:
        pool_mod.FINAL = original

    src_tok, tgt_tok = load_tokenizer("am"), load_tokenizer("en")
    out_dir = DATA / f"prepared_{name}" / "am-en"
    out_dir.mkdir(parents=True, exist_ok=True)
    prepared_sizes = {}
    for sp in SPLITS:
        df = pd.read_csv(final_out / f"{sp}.csv", usecols=["am", "en"], dtype=str).dropna()
        s = [[BOS_ID, *e.ids, EOS_ID] for e in src_tok.encode_batch(df["am"].tolist())]
        t = [[BOS_ID, *e.ids, EOS_ID] for e in tgt_tok.encode_batch(df["en"].tolist())]
        keep = [i for i, (a, b) in enumerate(zip(s, t)) if len(a) <= MAX_LEN and len(b) <= MAX_LEN]
        with open(out_dir / f"{sp}.pkl", "wb") as f:
            pickle.dump({"src": [s[i] for i in keep], "tgt": [t[i] for i in keep]}, f)
        print(f"[{name}] {sp}: kept {len(keep):,} of {len(df):,}")
        prepared_sizes[sp] = len(keep)
    print(f"[{name}] wrote caches to {out_dir}")

    manifest = {
        "arm": name,
        "experiment": "fixed_size_corpus",
        "built_at": now(),
        "git": git_info(),
        "sources": sources,
        "source_tiers": source_tiers,  # per-source {curated, cosine, africomet} actually applied
        "source_survivor_counts": source_counts,
        "lid_cutoffs": {"source_lid": SOURCE_LID_CUTOFF, "target_lid": TARGET_LID_CUTOFF},
        "pool_target": POOL_TARGET,
        "pooled_after_dedup": len(pooled_full),
        "pooled_after_decontam_and_subsample": len(pooled),
        "split_sizes": {k: len(v) for k, v in splits.items()},
        "prepared_sizes": prepared_sizes,
    }
    write_manifest(final_out, manifest)
    write_manifest(out_dir, manifest)


if __name__ == "__main__":
    if len(sys.argv) < 3:
        sys.exit(__doc__.strip())
    name, spec = sys.argv[1], sys.argv[2]
    if spec == "all":
        srcs = sorted(p.stem for p in PROCESSED.glob("*.csv"))
    else:
        srcs = spec.split(",")
    print(f"[{name}] sources: {srcs}")
    build(name, srcs)
