"""model.rescore — re-score EVERY surviving checkpoint on EVERY benchmark through
ONE code path, in one run, and write the results to results/benchmarks.csv.

Run (from the project root):
    python -m model.rescore [--runs NAME[,NAME...]] [--beams 4] [--dry-run]

Restored 2026-08-17 from commit 3b6683c^ (it was deleted in the "Refactor"
commit, which is why results/experiments.yaml still names a module that was not
on disk). Three things changed on the way back in, all of them corrections:

    BEAM IS THE DEFAULT      --beams now defaults to 4, not "1,4". Greedy is a
                             training-loop economy (model/training/train.py
                             forces beam_size=1 so periodic eval is not 4x the
                             cost); it was never the strategy any run ships with.
                             Every current config says beam 4 / lp 0.6.
    PREPROCESSING IS PROBED  the original fed raw Ethiopic to every model. That
                             is correct only for the base*/small* runs. Every run
                             since the Gezmu reproduction has a Latin-only
                             transliterated vocabulary, which sees raw Ethiopic
                             as ~50% <unk> and decodes to whitespace. The recipe
                             is detected per model from the vocabulary itself
                             (model.translate.detect_recipe) and the matching
                             chain applied on the way in and out.
    ARCHIVED RUNS INCLUDED   discovery covers archive/runs/ as well as runs/, and
                             falls back to last.pt for the runs that predate
                             best.pt tracking. Archiving a run is not supposed to
                             make it unscoreable.

Why this exists
---------------
Every FLORES/MAFAND number in EXPERIMENTS.md's cross-model table came from a
stored .hyp.txt written at a different time by a different revision of the
inference code. Those are only comparable if the code was identical at each
write, and it demonstrably was not: am-en-base-v6's stored hypotheses score
17.35, while the same checkpoint, config, benchmark file and references
reproduce 15.57 today. A table whose rows were measured by different code is not
a table, so this re-measures everything at once. Rows written here are mutually
comparable by construction; rows in the legacy table are not.

Case is scored both ways. `bleu`/`chrf++` are cased; `bleu_ci`/`chrf++_ci`
lowercase both sides. The distinction is not cosmetic — am-en-narrow,
am-en-broad-v2 and am-en-clean-lower train on lowercased English and
structurally cannot emit a capital, so their cased score against a cased
benchmark measures missing capitals rather than translation quality. The 36
legacy rows predate these two columns and carry them empty.

Output layout
-------------
    results/benchmarks.csv                 one row per (run, benchmark, split, decode)
    <run_dir>/benchmarks/<stem>_<split>.<decode>.hyp.txt

Hypothesis files carry the decode strategy in the NAME, so they can never be
silently compared against the old strategy-less <stem>_<split>.hyp.txt files.
The legacy files are left untouched — they are the evidence for the discrepancy
above, not something to overwrite.

Not covered
-----------
en->am runs are skipped. corpus_scores uses sacrebleu's 13a tokenizer, which is
correct for an English target and wrong for an Amharic one; scoring
en-am-base-v1 here would produce a confidently meaningless number.
"""
import argparse
import csv
import re
import time
from pathlib import Path

import polars as pl
import torch

from model.common import corpus_scores, load_for_inference
from model.configs.config import Config, load_config
from model.evaluate.evaluate_OOD import encode_sources, load_benchmark, translate
from model.tokenize.preprocess import detok_en, translit_am
from model.translate import detect_recipe
from process.clean.normalize import normalize
from process.utils.paths import BENCHMARKS, DATA, ROOT, RUNS

CONFIG_DIRS = [ROOT / "model" / "configs", ROOT / "model" / "configs" / "archive"]
RUN_DIRS = [RUNS, ROOT / "archive" / "runs"]
RESULTS_CSV = ROOT / "results" / "benchmarks.csv"

# (benchmark stem, split) pairs every am->en run is measured on.
BENCH_SPLITS = [("flores200_am_en", "devtest"), ("mafand_en_amh", "test")]

# See sanity_rate(): correctly-paired models measure 83-90%, mismatched ones 40-45%.
SANITY_FLOOR = 0.65


class VocabularyMismatch(RuntimeError):
    """Checkpoint loaded, but decoded against a vocabulary it was not trained on."""

FIELDS = ["run", "benchmark", "split", "decode", "beam_size", "length_penalty",
          "bleu", "chrf++", "bleu_ci", "chrf++_ci", "n_sentences", "recipe",
          "checkpoint", "scored_at"]


def find_config(run_dir: Path) -> Path | Config | None:
    """The run's own config.yaml, else the model/configs yaml naming this run.

    A run directory's own copy wins because it is the one the archiving pass
    repointed at archived data — the model/configs/archive/ copy of the same run
    may still name a path that moved.
    """
    own = run_dir / "config.yaml"
    if own.exists():
        return own
    for d in CONFIG_DIRS:
        for p in sorted(d.glob("*.yaml")):
            try:
                if load_config(p).get("run_name") == run_dir.name:
                    return p
            except Exception:
                continue
    return None


def checkpoint_vocab(ckpt: Path) -> dict[str, int]:
    """The src/tgt vocabulary sizes baked into a checkpoint's embedding tables.

    This is the ground truth for which tokenizer a run needs. Reading it off the
    checkpoint beats trusting the config, because several configs name a
    tokenizer directory whose CONTENTS were later replaced (the 32k -> 8k
    retrain reused data/tokenizer/{am,en}) or moved.
    """
    sd = torch.load(ckpt, map_location="cpu", weights_only=True)["model"]
    return {"am": sd["src_embed.embedding.weight"].shape[0],
            "en": sd["tgt_embed.embedding.weight"].shape[0]}


def resolve_tokenizers(cfg: Config, ckpt: Path) -> Config:
    """Point each side at a tokenizer dir whose vocab size matches the checkpoint.

    Three things break the configs' own tokenizer paths, and all three are the
    project's own history rather than anything wrong with the checkpoints:

      * the pre-v4 runs (base, base-v2, base-v3, baseline, small*) named no
        tokenizer at all and took the default data/tokenizer/{am,en}, which held
        32k vocabularies at the time and holds 8k ones now. Same path, different
        contents — which is why they fail with a size mismatch rather than a
        missing file. Verified: base-v3 scores 14.36 FLORES greedy against
        {am,en}_32k_v4 against a recorded 14.25, and emits fluent English.
      * data/tokenizer/{am,en} then moved to data/tokenizer/archive/.
      * the splitstrat and broad vocabularies moved under archive/data/.

    Matching on vocab size makes all three self-correcting. It is NOT sufficient
    on its own: several 8k vocabularies have existed at this path over the
    project's life, so a size match can still pair a checkpoint with the wrong
    map — which loads cleanly and decodes fluent nonsense. sanity_rate() is the
    check that catches that; this function only narrows the candidates.
    """
    from tokenizers import Tokenizer

    search = [DATA / "tokenizer", DATA / "tokenizer" / "archive",
              ROOT / "archive" / "data" / "tokenizer"]
    cands = [d for root in search if root.is_dir()
             for d in sorted(root.iterdir()) if (d / "tokenizer.json").exists()]
    want = checkpoint_vocab(ckpt)

    for lang, key in (("am", "src_tokenizer"), ("en", "tgt_tokenizer")):
        named = cfg.data.get(key)
        fits = [d for d in cands
                if Tokenizer.from_file(str(d / "tokenizer.json")).get_vocab_size() == want[lang]]
        if named and Path(named) in [Path(d) for d in fits]:
            continue                                   # config is already right
        if not fits:
            raise FileNotFoundError(
                f"no tokenizer anywhere with vocab_size={want[lang]} for {lang} "
                f"(checkpoint {ckpt}); searched {[str(s) for s in search]}")
        # Prefer the directory the config named (it only moved), then one named
        # for this language — `am`/`en` themselves, or the `am_32k_v4` style the
        # pre-v4 defaults were archived under — then a unique match.
        stem = Path(named).name if named else lang
        pick = (next((d for d in fits if d.name == stem), None)
                or next((d for d in fits if d.name == lang or d.name.startswith(f"{lang}_")), None)
                or (fits[0] if len(fits) == 1 else None))
        if pick is None:
            raise FileNotFoundError(
                f"{len(fits)} tokenizers have vocab_size={want[lang]} and none is named "
                f"{stem!r} or {lang!r} — cannot tell which one {ckpt} was trained on: "
                f"{[d.name for d in fits]}")
        cfg["data"][key] = str(pick)
    return cfg


def discover(names: set[str] | None) -> list[tuple[str, Path, Path]]:
    """(run name, config, checkpoint) for every scoreable run under RUN_DIRS."""
    out, seen = [], set()
    for root in RUN_DIRS:
        for run_dir in sorted(p for p in root.glob("*/") if p.is_dir()):
            name = run_dir.name
            if name in seen or (names and name not in names):
                continue
            ckpt = next((run_dir / "checkpoints" / c for c in ("best.pt", "last.pt")
                         if (run_dir / "checkpoints" / c).exists()), None)
            if ckpt is None:
                continue
            cfg_path = find_config(run_dir)
            if cfg_path is None:
                print(f"[skip] {name}: no config.yaml in the run dir and no "
                      f"model/configs yaml with that run_name")
                continue
            seen.add(name)
            out.append((name, cfg_path, ckpt))
    return out


def sanity_rate(hyps: list[str], refs: list[str]) -> float:
    """Fraction of decoded words that appear anywhere in the benchmark's references.

    A checkpoint loaded against a same-size but DIFFERENT vocabulary loads
    cleanly and decodes fluent-looking token salad — "af the hopese re with
    accgr the hope" — because the id->token map is a consistent relabelling of
    the one the model learned. Nothing upstream catches it: shapes match, no
    <unk> spike, and sacrebleu simply reports a low score, which is
    indistinguishable from a genuinely bad model.

    Measured on this project's own checkpoints, the two populations do not
    overlap: every correctly-paired model scores 83-90% (including
    dd-health10k-s1 at 1.46 BLEU, near the architecture's floor), while every
    mismatched one scores 40-45%. SANITY_FLOOR sits between them.
    """
    vocab = set()
    for r in refs:
        vocab |= set(re.findall(r"[a-z']+", r.lower()))
    words = [w for h in hyps for w in re.findall(r"[a-z']+", h.lower())]
    return sum(1 for w in words if w in vocab) / max(len(words), 1)


def prepare_sources(sentences: list[str], recipe: str) -> list[str]:
    """The source-side chain this model's vocabulary was trained under."""
    if recipe == "translit":
        return [translit_am(s) for s in sentences]
    df = normalize(pl.DataFrame({"am": sentences, "en": [""] * len(sentences)}),
                   "am", "en", name=None)
    assert len(df) == len(sentences), "normalize changed the row count"
    return df["am"].to_list()


def already_scored() -> set[tuple[str, str, str, str]]:
    """(run, benchmark, split, decode) keys already in the CSV.

    This sweep is hours long over dozens of checkpoints and a shared GPU, so it
    has to be resumable: a re-run picks up where it stopped instead of decoding
    everything again. --force overrides.

    Only rows written by THIS version count as done, identified by a non-empty
    `recipe` column. The 36 legacy rows carry a beam4 pass for nine runs, but
    those are the rows the module docstring calls not-a-table — they predate the
    preprocessing probe and fed raw Ethiopic to transliterated vocabularies.
    Treating them as done would skip precisely the runs most in need of redoing.
    """
    if not RESULTS_CSV.exists():
        return set()
    with RESULTS_CSV.open(newline="", encoding="utf-8") as f:
        return {(r["run"], r["benchmark"], r["split"], r["decode"])
                for r in csv.DictReader(f) if r.get("recipe")}


def score_run(name: str, cfg_path, ckpt: Path, beams: list[int], writer, fh,
              done: set[tuple[str, str, str, str]]) -> None:
    todo = [(stem, split, beam) for stem, split in BENCH_SPLITS for beam in beams
            if (name, stem, split, "greedy" if beam == 1 else f"beam{beam}") not in done]
    if not todo:
        print(f"[{name}] every pass already in {RESULTS_CSV.name} — skipping "
              f"(--force to re-decode)", flush=True)
        return

    cfg = resolve_tokenizers(load_config(cfg_path), ckpt)
    cfg, model, tok_src, tok_tgt, device = load_for_inference(cfg, str(ckpt))
    recipe = detect_recipe(tok_src)
    print(f"[{name}] {recipe} recipe, {tok_src.get_vocab_size()} src vocab, {ckpt.name}",
          flush=True)

    for stem, split in BENCH_SPLITS:
        wanted = [b for s, sp, b in todo if (s, sp) == (stem, split)]
        if not wanted:
            continue
        path = BENCHMARKS / f"{stem}.csv"
        if not path.exists():
            print(f"[skip] {stem}: {path} missing")
            continue
        df = load_benchmark(path, split)
        refs = df["en"].tolist()
        src = prepare_sources(df["am"].tolist(), recipe)
        ids, truncated = encode_sources(tok_src, src, cfg.data.max_src_len)

        for beam in wanted:
            cfg["inference"]["beam_size"] = beam
            hyps = translate(model, ids, tok_tgt, cfg, device)
            if recipe == "translit":
                # A translit model emits Moses-tokenized English ("do n't", " .").
                hyps = [detok_en(h) for h in hyps]
            assert len(hyps) == len(refs), "row count changed during evaluation"
            rate = sanity_rate(hyps, refs)
            if rate < SANITY_FLOOR:
                raise VocabularyMismatch(
                    f"{name} decoded {rate:.0%} in-reference words on {stem} (floor "
                    f"{SANITY_FLOOR:.0%}) — the checkpoint loaded, but against the wrong "
                    f"vocabulary. src={cfg.data.src_tokenizer} tgt={cfg.data.tgt_tokenizer}. "
                    f"No row written; the vocabulary this run was trained on is either "
                    f"misidentified or no longer on disk.")
            s = corpus_scores(hyps, refs)
            ci = corpus_scores([h.lower() for h in hyps], [r.lower() for r in refs])

            tag = "greedy" if beam == 1 else f"beam{beam}"
            out_dir = ckpt.parent.parent / "benchmarks"
            out_dir.mkdir(parents=True, exist_ok=True)
            (out_dir / f"{stem}_{split}.{tag}.hyp.txt").write_text(
                "\n".join(h.replace("\n", " ") for h in hyps), encoding="utf-8")

            writer.writerow({
                "run": name, "benchmark": stem, "split": split, "decode": tag,
                "beam_size": beam,
                "length_penalty": cfg.inference.length_penalty if beam > 1 else "",
                "bleu": f"{s['bleu']:.2f}", "chrf++": f"{s['chrf++']:.2f}",
                "bleu_ci": f"{ci['bleu']:.2f}", "chrf++_ci": f"{ci['chrf++']:.2f}",
                "n_sentences": len(refs), "recipe": recipe,
                "checkpoint": str(ckpt), "scored_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            })
            fh.flush()
            print(f"[{name}] {stem}:{split} {tag:6s} BLEU {s['bleu']:6.2f} / "
                  f"chrF++ {s['chrf++']:6.2f}   |  ci {ci['bleu']:6.2f} / {ci['chrf++']:6.2f}"
                  f"{f'  ({truncated} truncated)' if truncated else ''}", flush=True)

    del model
    torch.cuda.empty_cache()


def open_csv():
    """Append handle on results/benchmarks.csv, migrating the header if needed.

    The legacy file has no bleu_ci/chrf++_ci/recipe columns. Appending wider rows
    under a narrower header would silently misalign every future read, so the
    existing rows are rewritten once under the full header with the new columns
    empty. Their numbers are untouched.
    """
    RESULTS_CSV.parent.mkdir(parents=True, exist_ok=True)
    rows, header = [], None
    if RESULTS_CSV.exists():
        with RESULTS_CSV.open(newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            header, rows = reader.fieldnames, list(reader)
    if header != FIELDS:
        with RESULTS_CSV.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=FIELDS)
            w.writeheader()
            for r in rows:
                w.writerow({k: r.get(k, "") for k in FIELDS})
        if rows:
            print(f"[csv] migrated {len(rows)} legacy row(s) to the {len(FIELDS)}-column header")
    fh = RESULTS_CSV.open("a", newline="", encoding="utf-8")
    return fh, csv.DictWriter(fh, fieldnames=FIELDS)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0].strip())
    ap.add_argument("--runs", help="comma-separated run names (default: all discovered)")
    ap.add_argument("--beams", default="4",
                    help="beam sizes to score at (default 4, the strategy every "
                         "current config ships; pass 1 for greedy, or 1,4 for both)")
    ap.add_argument("--dry-run", action="store_true", help="list what would be scored, then exit")
    ap.add_argument("--force", action="store_true",
                    help="re-decode passes already present in results/benchmarks.csv "
                         "(default: skip them, so an interrupted sweep resumes)")
    args = ap.parse_args()

    beams = [int(b) for b in args.beams.split(",")]
    plan = []
    for run, cfg_path, ckpt in discover(set(args.runs.split(",")) if args.runs else None):
        cfg = load_config(cfg_path)
        if cfg.data.src_lang != "am" or cfg.data.tgt_lang != "en":
            print(f"[skip] {run}: {cfg.data.src_lang}->{cfg.data.tgt_lang}, "
                  f"needs a non-13a tokenizer — see module docstring")
            continue
        plan.append((run, cfg_path, ckpt))

    print(f"\n{len(plan)} run(s) x {len(BENCH_SPLITS)} benchmark(s) x {len(beams)} decode(s) "
          f"= {len(plan) * len(BENCH_SPLITS) * len(beams)} scoring passes")
    for run, cfg_path, ckpt in plan:
        print(f"  {run:24s} {ckpt.name:8s} {cfg_path}")
    if args.dry_run:
        return

    done = set() if args.force else already_scored()
    fh, writer = open_csv()
    failed = []
    for run, cfg_path, ckpt in plan:
        print(f"\n{'=' * 78}\n=== {run}\n{'=' * 78}", flush=True)
        try:
            score_run(run, cfg_path, ckpt, beams, writer, fh, done)
        except Exception as e:            # vocab mismatch on pre-8k runs, missing data, etc.
            failed.append((run, f"{type(e).__name__}: {e}"))
            print(f"[skip] {run}: {type(e).__name__}: {e}", flush=True)
            torch.cuda.empty_cache()
    fh.close()

    print(f"\nwrote {RESULTS_CSV}")
    if failed:
        print(f"\n{len(failed)} run(s) could not be scored:")
        for run, why in failed:
            print(f"  {run:24s} {why.splitlines()[0][:110]}")


if __name__ == "__main__":
    main()
