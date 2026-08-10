"""
collect — Gather every source into data/raw/csv_raw/ as CSVs.

  - AfriDocMT (health, tech) from HuggingFace           → collect.afridoc
  - Gezmu from local parallel files in data/raw/local/  → collect.gezmu
  - Quran/Tanzil from local parallel files              → collect.quran
  - NLLB: download the mined parquet if missing, then laser-filter it down to a
    tractable am/en CSV (csv_raw/nllb.csv)              → collect.nllb
  - CCAligned: download the OPUS moses-format am-en release (~346k pairs, web-mined)
                                                          → collect.ccaligned

One file per source (this file is just the registry + CLI that ties them
together) — each is independently runnable/importable, e.g.
`from collect.nllb import collect_nllb`.

Everything downstream (process.py) just cleans the CSVs this produces, so the
NLLB parquet is handled here, not there. Run (from the project root):

    python -m collect          # every source; skips any whose CSV already exists
    python -m collect nllb     # just NLLB — the LASER_CUTOFF loop (~3s, no network)
    python -m collect --force  # rebuild everything from scratch

Re-running a source is only useful when its *input* changed. Gezmu and Quran read
fixed local files, and AfriDoc is pinned to a HuggingFace release, so their CSVs
can't change between runs — a default run skips them once they exist. That also
keeps `load_dataset` from round-tripping to HuggingFace, which it does on every
call even with a warm cache (~5s, and the one thing here that stalls on a bad
connection). CCAligned is a fixed OPUS release too, same story.

NLLB is different: its CSV depends on the cutoffs in collect/nllb.py, so
tuning one means rebuilding it. Name it explicitly (`python -m collect nllb`)
— an explicitly named source always rebuilds, skip-if-exists only applies to a
default run.

Outputs: data/raw/csv_raw/{afridoc_health,afridoc_tech,gezmu,quran,nllb,ccaligned}.csv
"""
import argparse
import sys

from collect.afridoc import collect_afridoc
from collect.ccaligned import collect_ccaligned
from collect.gezmu import collect_gezmu
from collect.nllb import collect_nllb
from collect.quran import collect_quran
from process.utils.paths import CSV_RAW

# Each source, with the CSV(s) it produces — the outputs a default run checks for
# before deciding it has nothing to do.
SOURCES: dict[str, tuple] = {
    "afridoc":   (collect_afridoc,   ("afridoc_health.csv", "afridoc_tech.csv")),
    "gezmu":     (collect_gezmu,     ("gezmu.csv",)),
    "quran":     (collect_quran,     ("quran.csv",)),
    "nllb":      (collect_nllb,      ("nllb.csv",)),
    "ccaligned": (collect_ccaligned, ("ccaligned.csv",)),
}


def main(argv: list[str] | None = None) -> None:
    """Run the named sources, or every source if none are named.

    Naming a source is a statement of intent, so it always rebuilds — that is the
    LASER_CUTOFF loop: `python -m collect nllb`. A default run instead skips
    sources whose CSVs are already there, since their inputs are fixed; --force
    overrides that.
    """
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1].strip())
    parser.add_argument("sources", nargs="*", choices=list(SOURCES), metavar="SOURCE",
                        help=f"sources to collect ({', '.join(SOURCES)}); default: all")
    parser.add_argument("--force", action="store_true",
                        help="rebuild even if the output CSV already exists")
    args = parser.parse_args(argv)

    CSV_RAW.mkdir(parents=True, exist_ok=True)
    # An explicitly named source always rebuilds; only a default run skips.
    named = bool(args.sources)
    for name in args.sources or list(SOURCES):
        fn, outputs = SOURCES[name]
        if not (named or args.force) and all((CSV_RAW / o).exists() for o in outputs):
            print(f"{name}: up to date, skipping "
                  f"(`python -m collect {name}` to rebuild)")
            continue
        fn()


if __name__ == "__main__":
    main(sys.argv[1:])
