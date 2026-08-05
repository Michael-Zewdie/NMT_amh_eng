"""
collect_benchmark.py — Pull held-out, human-translated eval sets that must stay
out of the training pool (process.py only ever globs csv_raw/*.csv, so keeping
these in data/benchmarks/ instead is what keeps them out).

  - MAFAND-MT (masakhane-io/lafand-mt) en-amh dev+test: 1,936 professionally
    translated news-domain pairs, curated by Masakhane. No train split exists
    for Amharic — this corpus is eval-only by construction.
  - FLORES-200 (Meta) amh_Ethi-eng_Latn dev+devtest: 2,009 professionally
    translated Wikimedia-domain sentences (wikinews/wikijunior/wikivoyage).
    The standard low-resource MT benchmark — reporting on it makes our numbers
    directly comparable to published systems (NLLB, Google Translate) instead of
    only to our own past runs. `devtest` is the split everyone reports; keep
    `dev` for tuning so devtest stays untouched.

Run (from the project root):

    python -m collection.collect_benchmark            # every benchmark
    python -m collection.collect_benchmark flores     # just one

Downloads are cached under data/raw/ and skipped when already present, so a
re-run is free. Deliberately no HuggingFace route for FLORES: facebook/flores is
script-backed and unusable on datasets>=4.0 (same breakage as allenai/nllb, see
collect.py), and openlanguagedata/flores_plus is gated behind terms acceptance.
The tarball needs neither an HF token nor a dataset script.

These CSVs are benchmarks, not training data — they are never cleaned, filtered
or deduped. processing.utils.decontaminate reads them back to strip any pooled
training row that overlaps them.

Outputs: data/benchmarks/{mafand_en_amh,flores200_am_en}.csv
"""
import argparse
import io
import json
import sys
import tarfile

import pandas as pd
import requests

from processing.utils.paths import BENCHMARKS, FLORES_FULL

MAFAND_URL = "https://raw.githubusercontent.com/masakhane-io/lafand-mt/main/data/json_files/en-amh/"

FLORES_URL = "https://dl.fbaipublicfiles.com/nllb/flores200_dataset.tar.gz"
FLORES_LANGS = {"am": "amh_Ethi", "en": "eng_Latn"}
FLORES_SPLITS = ("dev", "devtest")


def collect_mafand() -> None:
    rows = []
    for split, filename in (("dev", "dev.json"), ("test", "test.json")):
        resp = requests.get(MAFAND_URL + filename, timeout=60)
        resp.raise_for_status()
        for line in resp.text.splitlines():
            if not line.strip():
                continue
            pair = json.loads(line)["translation"]
            rows.append({"am": pair["amh"], "en": pair["en"], "split": split})

    df = pd.DataFrame(rows)
    BENCHMARKS.mkdir(parents=True, exist_ok=True)
    out = BENCHMARKS / "mafand_en_amh.csv"
    df.to_csv(out, index=False)
    print(f"mafand: {len(df)} pairs ({(df['split'] == 'dev').sum()} dev, "
          f"{(df['split'] == 'test').sum()} test) → {out}")


def _download_flores() -> None:
    """Fetch + extract the 25MB FLORES-200 tarball into data/raw/flores_full/.

    Extracts only the four amh/eng files we need rather than all 204 languages.
    The archive's top-level directory name is not assumed — members are matched
    on their trailing '<split>/<lang>.<split>' path so a repackaging upstream
    doesn't silently produce an empty result.
    """
    wanted = {f"{split}/{lang}.{split}" for split in FLORES_SPLITS for lang in FLORES_LANGS.values()}
    FLORES_FULL.mkdir(parents=True, exist_ok=True)

    print(f"  downloading flores200 ({FLORES_URL}) …")
    resp = requests.get(FLORES_URL, timeout=300)
    resp.raise_for_status()

    found = set()
    with tarfile.open(fileobj=io.BytesIO(resp.content), mode="r:gz") as tf:
        for member in tf.getmembers():
            suffix = "/".join(member.name.split("/")[-2:])
            if not member.isfile() or suffix not in wanted:
                continue
            src = tf.extractfile(member)
            dest = FLORES_FULL / suffix
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(src.read())
            found.add(suffix)

    missing = wanted - found
    if missing:
        raise RuntimeError(f"flores200 archive did not contain: {sorted(missing)}")
    print(f"  extracted {len(found)} files → {FLORES_FULL}")


def collect_flores() -> None:
    """FLORES-200 amh_Ethi/eng_Latn dev+devtest → benchmarks/flores200_am_en.csv.

    The per-language files are plain text, one sentence per line, aligned across
    every language by line number — so the pairing is a positional zip(), and a
    line-count mismatch between the two sides means the archive is corrupt.
    """
    paths = {(lang, split): FLORES_FULL / split / f"{code}.{split}"
             for lang, code in FLORES_LANGS.items() for split in FLORES_SPLITS}
    if not all(p.exists() for p in paths.values()):
        _download_flores()

    rows = []
    for split in FLORES_SPLITS:
        am = paths[("am", split)].read_text(encoding="utf-8").splitlines()
        en = paths[("en", split)].read_text(encoding="utf-8").splitlines()
        if len(am) != len(en):
            raise RuntimeError(f"flores200 {split}: {len(am)} am lines vs {len(en)} en lines — not aligned")
        rows += [{"am": a, "en": e, "split": split} for a, e in zip(am, en)]

    df = pd.DataFrame(rows)
    BENCHMARKS.mkdir(parents=True, exist_ok=True)
    out = BENCHMARKS / "flores200_am_en.csv"
    df.to_csv(out, index=False)
    counts = ", ".join(f"{n} {s}" for s, n in df["split"].value_counts().sort_index().items())
    print(f"flores200: {len(df)} pairs ({counts}) → {out}")


BENCHMARK_SOURCES = {"mafand": collect_mafand, "flores": collect_flores}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1].strip())
    parser.add_argument("benchmarks", nargs="*", choices=list(BENCHMARK_SOURCES), metavar="BENCHMARK",
                        help=f"benchmarks to collect ({', '.join(BENCHMARK_SOURCES)}); default: all")
    args = parser.parse_args(argv)

    for name in args.benchmarks or list(BENCHMARK_SOURCES):
        BENCHMARK_SOURCES[name]()


if __name__ == "__main__":
    main(sys.argv[1:])
