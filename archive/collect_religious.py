"""collect_religious.py — build a large single-domain RELIGIOUS am-en corpus by
joining one Amharic Bible against many English Bibles on canonical verse IDs.

Run (from the project root):
    python -m collection.collect_religious [--translations N] [--list]

Why this exists
---------------
OPUS carries exactly two religious am-en corpora: bible-uedin (61k pairs) and
Tanzil (94k). JW300 — the one large Watchtower set — was retracted for copyright.
So there is no 100-200k religious corpus to download, and the only honest route to
that size is to multiply the English side.

    OPUS bible-uedin ships Amharic as verse-ID XML   <seg id="b.GEN.1.1">
    BibleNLP/ebible ships 47 English Bibles aligned  line i <-> vref.txt line i

Both key to the same USFM verse reference, so the join is exact rather than
mined: no LASER, no cosine, no alignment guesswork. Every pair is a professional
human translation of a known verse.

    30,580 Amharic verses x N English translations

What this corpus is and is not
------------------------------
It is genuinely single-domain and genuinely human-translated. It is NOT 200k
independent sentences: unique Amharic is capped at 30,580 regardless of N, and
volume comes from English redundancy. Read any BLEU it produces with that in mind
— see EXPERIMENTS.md, where am-en-narrow scored 26.63 in-distribution and 0.68 on
FLORES off the same kind of data.

The redundancy has one real upside: N English renderings of every verse make
MULTI-REFERENCE BLEU possible on the validation split, which is the standard fix
for single-reference BLEU punishing correct-but-differently-worded output.

Text source: OPUS `/raw/` XML, NOT `/xml/`. The latter is word-tokenized (`<w>`
elements) and joining those tokens reintroduces the detokenization artifact that
already depresses quran's scores — sacrebleu warns "you forgot to detokenize your
test data" on it. `/raw/` carries the original sentence text.

Output: data/raw/csv_raw/religious.csv, ready for `python -m processing.process`,
which cleans it with dedupe_keys=("am","en") — so distinct English translations of
the same verse are preserved rather than collapsed.
"""
import argparse
import io
import re
import xml.etree.ElementTree as ET
import zipfile

import pandas as pd
import requests

from process.utils.paths import CSV_RAW, RAW

BIBLE_AM_XML_URL = "https://object.pouta.csc.fi/OPUS-bible-uedin/v1/raw/am.zip"
EBIBLE_RAW = "https://raw.githubusercontent.com/BibleNLP/ebible/main"
VREF_URL = f"{EBIBLE_RAW}/metadata/vref.txt"
CORPUS_URL = EBIBLE_RAW + "/corpus/{tid}.txt"

RELIGIOUS_DIR = RAW / "religious"

# English Bibles ordered by how much NEW phrasing each one adds, not by fame.
# All are public domain or explicitly redistributable, and all are whole-Bible
# (not New-Testament-only) so they actually cover the Amharic's 30,580 verses.
# The spread of register is deliberate: a model trained on only KJV-style English
# learns one idiolect, which would confound "religious domain" with "archaic style".
ENGLISH_TRANSLATIONS = [
    "eng-engwebp",     # World English Bible — modern, public domain
    "eng-eng_kjv",     # King James — archaic, formal
    "eng-engBBE",      # Bible in Basic English — ~1,000 word vocabulary
    "eng-engbsb",      # Berean Standard — modern literal
    "eng-engylt",      # Young's Literal — hyper-literal, odd syntax
    "eng-engDBY",      # Darby
    "eng-eng_asv",     # American Standard 1901
    "eng-englsv",      # Literal Standard Version
    "eng-engerv",      # Easy-to-Read Version
    "eng-enggnv",      # Geneva Bible 1599
]

VERSE_ID = re.compile(r"^b\.([A-Z0-9]{3})\.(\d+)\.(\d+)$")


def fetch_amharic_verses() -> dict[str, str]:
    """OPUS bible-uedin raw XML -> {"GEN 1:1": "amharic text"}.

    A <seg> can hold several <s> sentences; they are joined, since the English
    side is one line per verse and the pairing must be verse-to-verse.
    """
    RELIGIOUS_DIR.mkdir(parents=True, exist_ok=True)
    local = RELIGIOUS_DIR / "bible_uedin_am_raw.zip"
    if not local.exists():
        print(f"downloading Amharic Bible XML -> {local}")
        r = requests.get(BIBLE_AM_XML_URL, timeout=300)
        r.raise_for_status()
        local.write_bytes(r.content)

    with zipfile.ZipFile(local) as zf:
        name = next(n for n in zf.namelist() if n.endswith(".xml"))
        root = ET.parse(io.BytesIO(zf.read(name))).getroot()

    verses: dict[str, str] = {}
    for seg in root.iter("seg"):
        m = VERSE_ID.match(seg.get("id") or "")
        if not m:
            continue
        book, chapter, verse = m.groups()
        text = " ".join(" ".join("".join(s.itertext()).split()) for s in seg.iter("s"))
        text = " ".join(text.split())
        if text:
            verses[f"{book} {int(chapter)}:{int(verse)}"] = text
    print(f"amharic: {len(verses):,} verses parsed")
    return verses


def fetch_lines(url: str, cache: str) -> list[str]:
    path = RELIGIOUS_DIR / cache
    if not path.exists():
        r = requests.get(url, timeout=300)
        r.raise_for_status()
        path.write_bytes(r.content)
    return path.read_text(encoding="utf-8").splitlines()


def fetch_english(tid: str, vref: list[str]) -> dict[str, str] | None:
    """One BibleNLP translation -> {vref: text}. Blank lines mean the verse is
    absent from that translation and are skipped."""
    try:
        lines = fetch_lines(CORPUS_URL.format(tid=tid), f"{tid}.txt")
    except requests.HTTPError as e:
        print(f"  {tid}: unavailable ({e.response.status_code}) — skipping")
        return None
    if len(lines) != len(vref):
        print(f"  {tid}: {len(lines):,} lines != {len(vref):,} vref lines — skipping")
        return None
    return {v: t.strip() for v, t in zip(vref, lines) if t.strip() and t.strip() != "<range>"}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1].strip())
    ap.add_argument("--translations", type=int, default=6,
                    help="how many English Bibles to join (default 6, ~180k pairs)")
    ap.add_argument("--list", action="store_true",
                    help="report per-translation verse overlap and exit, without writing")
    args = ap.parse_args()

    amharic = fetch_amharic_verses()
    vref = fetch_lines(VREF_URL, "vref.txt")
    print(f"vref: {len(vref):,} canonical verse slots\n")

    RELIGIOUS_DIR.mkdir(parents=True, exist_ok=True)
    rows, used = [], []
    print(f"{'translation':16s} {'verses':>9s} {'paired':>9s}")
    print("-" * 38)
    for tid in ENGLISH_TRANSLATIONS:
        if not args.list and len(used) >= args.translations:
            break
        english = fetch_english(tid, vref)
        if english is None:
            continue
        shared = amharic.keys() & english.keys()
        print(f"{tid:16s} {len(english):>9,} {len(shared):>9,}")
        if len(shared) < 10_000:
            print(f"  {tid}: only {len(shared):,} shared verses — skipping (likely NT-only)")
            continue
        used.append(tid)
        rows.extend({"am": amharic[v], "en": english[v], "translation": tid, "vref": v}
                    for v in sorted(shared))

    if args.list:
        print("\n--list: nothing written")
        return
    if not rows:
        raise SystemExit("no usable English translations — check network access")

    df = pd.DataFrame(rows)
    # csv_raw sources carry am/en only — every other source does, and process.py's
    # clean() + the scorers assume that shape. Verse ref and translation id go to a
    # sidecar instead, so multi-reference grouping stays possible without making
    # this source structurally different from the ones it will be compared against.
    meta = RELIGIOUS_DIR / "religious_meta.csv"
    df.to_csv(meta, index=False)
    out = CSV_RAW / "religious.csv"
    df[["am", "en"]].to_csv(out, index=False)
    print(f"\nreligious: {len(df):,} raw pairs from {len(used)} translations "
          f"x {df.vref.nunique():,} verses -> {out}")
    print(f"  metadata (vref, translation) -> {meta}")
    print(f"  unique am: {df.am.nunique():,}   unique en: {df.en.nunique():,}")
    print(f"  translations: {', '.join(used)}")
    print("\nnext: python -m processing.process   (cleans + scores; keeps multi-reference "
          "rows via dedupe_keys=('am','en'))")


if __name__ == "__main__":
    main()
