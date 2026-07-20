#!/usr/bin/env python3
"""
Amharic–English tokenization strategy benchmark.

Compares tokenization strategies on an am–en parallel corpus and reports:

  * token fertility (pieces per whitespace word), per language side, and the
    am/en fertility ratio (batching / length-parity signal for NMT)
  * compression (chars per token) and mean sequence length
  * morphological boundary alignment: precision / recall / F1 of tokenizer cut
    points against gold morpheme boundaries (fidel-level)
  * token semantic purity (char-weighted cluster purity of tokens vs morphs)
  * Root Integrity Index (RII): how often adjacent radicals of a consonantal
    root end up inside the same token (pairwise), and the strict all-radicals-
    in-one-token rate
  * TSP (token semantic purity score): harmonic mean of boundary F1 and
    pairwise RII — the single scalar proposed for pre-training model selection

Strategies:
  sp-bpe        SentencePiece BPE, joint am+en vocab
  sp-unigram    SentencePiece Unigram LM, joint am+en vocab
  morph-unigram morphological pre-segmentation of the Amharic side
                (HornMorpho if importable, else a rule-based affix stripper)
                followed by SentencePiece Unigram
  byte          byte-level (ByT5-style), no training

Usage:
  python benchmark.py [--corpus sample_corpus.tsv] [--eval eval_words.tsv]
                      [--vocab-size 500] [--report report.md] [--json report.json]
                      [--strategies sp-bpe,sp-unigram,morph-unigram,byte]
                      [--no-normalize]

Corpus format: one sentence pair per line, "amharic<TAB>english".
Eval format:   see header of eval_words.tsv.
"""

import argparse
import io
import json
import sys
import unicodedata
from functools import lru_cache
from pathlib import Path

HERE = Path(__file__).resolve().parent

# --------------------------------------------------------------------------
# Ethiopic script helpers
# --------------------------------------------------------------------------
# The Ethiopic block encodes each consonant family as a run of 8 codepoints
# (orders 1–8: ä u i a e ə o wa), aligned to multiples of 8 from U+1200.
# The consonant "family base" is therefore recoverable arithmetically, which
# gives us the consonantal skeleton of a surface form without a morphological
# analyzer.

def fidel_family(ch: str) -> str:
    cp = ord(ch)
    if 0x1200 <= cp <= 0x135A:
        return chr(cp - ((cp - 0x1200) % 8))
    return ch


# Homophone fidel families collapsed for root matching and (optionally) for
# corpus normalization: ሐ/ኀ → ሀ, ሠ → ሰ, ዐ → አ, ፀ → ጸ.
FAMILY_MERGE = {"ሐ": "ሀ", "ኀ": "ሀ", "ሠ": "ሰ", "ዐ": "አ", "ፀ": "ጸ"}


def root_family(ch: str) -> str:
    fam = fidel_family(ch)
    return FAMILY_MERGE.get(fam, fam)


def normalize_fidel_variants(text: str) -> str:
    """Fold homophone fidel variants onto a canonical family, preserving the
    vowel order (e.g. ሑ → ሁ). Cheap, reversible-enough for tokenizer training."""
    out = []
    for ch in text:
        fam = fidel_family(ch)
        tgt = FAMILY_MERGE.get(fam)
        if tgt is not None:
            order = ord(ch) - ord(fam)
            mapped = ord(tgt) + order
            out.append(chr(mapped) if 0x1200 <= mapped <= 0x135A else ch)
        else:
            out.append(ch)
    return "".join(out)


def is_ethiopic_word(word: str) -> bool:
    return any(0x1200 <= ord(c) <= 0x137F for c in word)


# --------------------------------------------------------------------------
# Morphological pre-segmentation
# --------------------------------------------------------------------------
# Preferred: HornMorpho (pip install HornMorpho; `import hm`). Its API differs
# across versions, so we probe a few entry points and validate that the output
# is concatenation-preserving. Fallback: a deliberately conservative rule-based
# affix stripper. NOTE: the pypi package `amseg` is a sentence/word-level
# segmenter, not a morphological one — it does not fill this slot.

RULE_PREFIXES = [
    "እንደ", "ስለ", "እስከ", "አል",
    "የ", "በ", "ለ", "ከ", "ይ", "ት", "እ", "ተ", "መ",
]
RULE_SUFFIXES = [
    "አችሁ", "አቸው", "ላቸው", "ዎች", "ኦች",
    "ም", "ን", "ው", "ኩ", "ኝ", "ች", "ህ", "ሽ", "ዋ",
]
MIN_STEM = 2
MAX_PREFIXES = 2
MAX_SUFFIXES = 2


def rule_segment(word: str) -> list[str]:
    """Longest-match affix stripping in fidel space. A stand-in for a real
    analyzer: it cannot recover morphs fused into a single fidel (abugida
    vowel fusion) and it will over-strip occasionally. Concatenation of the
    returned morphs always equals the input."""
    if not is_ethiopic_word(word) or len(word) <= MIN_STEM:
        return [word]
    prefixes, suffixes = [], []
    stem = word
    for _ in range(MAX_PREFIXES):
        for p in sorted(RULE_PREFIXES, key=len, reverse=True):
            if stem.startswith(p) and len(stem) - len(p) >= MIN_STEM:
                if len(p) == 1 and len(stem) <= 3:
                    continue
                prefixes.append(p)
                stem = stem[len(p):]
                break
        else:
            break
    for _ in range(MAX_SUFFIXES):
        for s in sorted(RULE_SUFFIXES, key=len, reverse=True):
            if stem.endswith(s) and len(stem) - len(s) >= MIN_STEM:
                suffixes.insert(0, s)
                stem = stem[: len(stem) - len(s)]
                break
        else:
            break
    return prefixes + [stem] + suffixes


def make_segmenter():
    """Return (segment_fn, name). Tries HornMorpho, falls back to rules."""
    try:
        import hm  # type: ignore
    except ImportError:
        return rule_segment, "rule-based affix stripper"

    candidates = []
    for attr in ("seg_word", "seg", "segment"):
        if hasattr(hm, attr):
            candidates.append(getattr(hm, attr))

    def hm_segment(word: str) -> list[str]:
        for fn in candidates:
            try:
                res = fn("amh", word)
            except Exception:
                continue
            if isinstance(res, str):
                parts = [p for p in res.replace("-", " ").split() if p]
            elif isinstance(res, (list, tuple)):
                parts = [str(p) for p in res if p]
            else:
                continue
            if parts and "".join(parts) == word:
                return parts
        return rule_segment(word)

    # smoke-test: if HornMorpho never produces a usable segmentation, use rules
    probe = hm_segment("ተማሪዎች")
    if len(probe) > 1:
        return lru_cache(maxsize=1 << 20)(hm_segment), "HornMorpho"
    return rule_segment, "rule-based affix stripper (HornMorpho probe failed)"


# --------------------------------------------------------------------------
# Tokenizers
# --------------------------------------------------------------------------

class SPTokenizer:
    """SentencePiece wrapper trained in-memory on the given lines."""

    def __init__(self, name: str, lines: list[str], model_type: str, vocab_size: int):
        import sentencepiece as spm

        charset = {c for line in lines for c in line if not c.isspace()}
        floor = len(charset) + 16
        size = max(vocab_size, floor)
        last_err = None
        for _ in range(8):
            model = io.BytesIO()
            try:
                spm.SentencePieceTrainer.train(
                    sentence_iterator=iter(lines),
                    model_writer=model,
                    model_type=model_type,
                    vocab_size=size,
                    character_coverage=1.0,
                    minloglevel=2,
                )
                break
            except Exception as e:  # vocab too large for tiny corpora
                last_err = e
                size = max(floor, int(size * 0.75))
        else:
            raise RuntimeError(f"SentencePiece training failed for {name}: {last_err}")
        self.sp = spm.SentencePieceProcessor(model_proto=model.getvalue())
        self.name = name
        self.sp_model_type = model_type
        self.vocab_size = size

    def sentence_pieces(self, text: str) -> list[str]:
        return self.sp.encode(text, out_type=str)

    def word_pieces(self, word: str) -> list[str] | None:
        pieces = [p.replace("▁", "") for p in self.sp.encode(word, out_type=str)]
        pieces = [p for p in pieces if p]
        if "".join(pieces) != word:  # normalization/unk broke char alignment
            return None
        return pieces

    def word_boundaries(self, word: str):
        pieces = self.word_pieces(word)
        if pieces is None:
            return None
        cuts, acc = set(), 0
        for p in pieces[:-1]:
            acc += len(p)
            cuts.add(acc)
        return cuts, 0


class MorphTokenizer:
    """Morphological pre-segmentation (am side) + SentencePiece Unigram."""

    def __init__(self, name, am_lines, en_lines, segment, vocab_size):
        self.segment = segment
        self.name = name
        seg_am = [" ".join(m for w in line.split() for m in segment(w)) for line in am_lines]
        self.inner = SPTokenizer(name, seg_am + en_lines, "unigram", vocab_size)
        self.vocab_size = self.inner.vocab_size

    def sentence_pieces(self, text: str, lang: str = "am") -> list[str]:
        if lang == "am":
            text = " ".join(m for w in text.split() for m in self.segment(w))
        return self.inner.sentence_pieces(text)

    def word_pieces(self, word: str) -> list[str] | None:
        pieces = []
        for morph in self.segment(word):
            mp = self.inner.word_pieces(morph)
            if mp is None:
                return None
            pieces.extend(mp)
        return pieces

    def word_boundaries(self, word: str):
        pieces = self.word_pieces(word)
        if pieces is None:
            return None
        cuts, acc = set(), 0
        for p in pieces[:-1]:
            acc += len(p)
            cuts.add(acc)
        return cuts, 0


class ByteTokenizer:
    """ByT5-style: one token per UTF-8 byte. No training, no OOV."""

    name = "byte"
    vocab_size = 256

    def sentence_pieces(self, text: str) -> list[str]:
        return [f"{b:02x}" for b in text.encode("utf-8")]

    def word_pieces(self, word: str) -> list[str]:
        return [f"{b:02x}" for b in word.encode("utf-8")]

    def word_boundaries(self, word: str):
        cuts = set(range(1, len(word)))  # every fidel junction is a cut …
        intra = sum(len(c.encode("utf-8")) - 1 for c in word)  # … plus intra-fidel cuts
        return cuts, intra


# --------------------------------------------------------------------------
# Evaluation data
# --------------------------------------------------------------------------

class EvalItem:
    __slots__ = ("word", "gold_cuts", "complete", "root")

    def __init__(self, word, gold_cuts, complete, root):
        self.word = word
        self.gold_cuts = gold_cuts
        self.complete = complete
        self.root = root


def load_eval(path: Path) -> list[EvalItem]:
    items = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        cols = line.split("\t")
        if len(cols) != 4:
            continue
        word, seg, complete, root = cols
        cuts = set()
        if seg != "-":
            morphs = seg.split("|")
            if "".join(morphs) != word:
                print(f"  [warn] eval item skipped, seg mismatch: {word}", file=sys.stderr)
                continue
            acc = 0
            for m in morphs[:-1]:
                acc += len(m)
                cuts.add(acc)
        items.append(EvalItem(word, cuts, complete == "1", root if root != "-" else None))
    return items


def radical_positions(word: str, root: str) -> list[int] | None:
    """Greedy leftmost match of the root's consonant families against the
    word's fidel-family sequence. Returns char positions of the radicals."""
    fams = [root_family(c) for c in word]
    positions, i = [], 0
    for r in (root_family(c) for c in root):
        while i < len(fams) and fams[i] != r:
            i += 1
        if i == len(fams):
            return None
        positions.append(i)
        i += 1
    return positions


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------

def spans_from_cuts(cuts: set, n: int) -> list[tuple[int, int]]:
    edges = sorted(cuts | {0, n})
    return list(zip(edges, edges[1:]))


def evaluate_words(tok, items: list[EvalItem]) -> dict:
    rec_num = rec_den = prec_num = prec_den = 0
    pur_num = pur_den = 0
    rii_num = rii_den = 0
    strict_num = strict_den = 0
    skipped = 0

    for it in items:
        wb = tok.word_boundaries(it.word)
        if wb is None:
            skipped += 1
            continue
        cuts, _intra = wb
        n = len(it.word)

        if it.gold_cuts:
            rec_num += len(cuts & it.gold_cuts)
            rec_den += len(it.gold_cuts)
            if it.complete:
                prec_num += len(cuts & it.gold_cuts)
                prec_den += len(cuts)
                # char-weighted token purity vs gold morph spans
                morph_spans = spans_from_cuts(it.gold_cuts, n)
                for ts, te in spans_from_cuts(cuts, n):
                    pur_num += max(min(te, me) - max(ts, ms)
                                   for ms, me in morph_spans)
                pur_den += n

        if it.root:
            pos = radical_positions(it.word, it.root)
            if pos and len(pos) >= 2:
                tok_of = lambda p: sum(1 for c in cuts if c <= p)
                pairs = list(zip(pos, pos[1:]))
                same = sum(1 for a, b in pairs if tok_of(a) == tok_of(b))
                rii_num += same
                rii_den += len(pairs)
                strict_num += 1 if same == len(pairs) else 0
                strict_den += 1

    prec = prec_num / prec_den if prec_den else 0.0
    rec = rec_num / rec_den if rec_den else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    rii = rii_num / rii_den if rii_den else 0.0
    tsp = 2 * f1 * rii / (f1 + rii) if f1 + rii else 0.0
    return {
        "boundary_precision": prec,
        "boundary_recall": rec,
        "boundary_f1": f1,
        "token_purity": pur_num / pur_den if pur_den else 0.0,
        "rii_pairwise": rii,
        "rii_strict": strict_num / strict_den if strict_den else 0.0,
        "tsp": tsp,
        "eval_words_skipped": skipped,
    }


def evaluate_corpus(tok, pairs: list[tuple[str, str]]) -> dict:
    stats = {}
    for lang, idx in (("am", 0), ("en", 1)):
        tok_total = word_total = char_total = 0
        seq_lens = []
        for pair in pairs:
            line = pair[idx]
            if isinstance(tok, MorphTokenizer):
                pieces = tok.sentence_pieces(line, lang=lang)
            else:
                pieces = tok.sentence_pieces(line)
            tok_total += len(pieces)
            word_total += len(line.split())
            char_total += len(line.replace(" ", ""))
            seq_lens.append(len(pieces))
        stats[f"fertility_{lang}"] = tok_total / word_total
        stats[f"chars_per_token_{lang}"] = char_total / tok_total
        stats[f"mean_seq_len_{lang}"] = sum(seq_lens) / len(seq_lens)
    stats["fertility_ratio_am_en"] = stats["fertility_am"] / stats["fertility_en"]
    return stats


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------

ROWS = [
    ("vocab size", "vocab", "{:d}"),
    ("fertility am (tok/word)", "fertility_am", "{:.2f}"),
    ("fertility en (tok/word)", "fertility_en", "{:.2f}"),
    ("fertility ratio am/en", "fertility_ratio_am_en", "{:.2f}"),
    ("chars/token am", "chars_per_token_am", "{:.2f}"),
    ("mean seq len am (tokens)", "mean_seq_len_am", "{:.1f}"),
    ("mean seq len en (tokens)", "mean_seq_len_en", "{:.1f}"),
    ("boundary precision", "boundary_precision", "{:.3f}"),
    ("boundary recall", "boundary_recall", "{:.3f}"),
    ("boundary F1", "boundary_f1", "{:.3f}"),
    ("token purity", "token_purity", "{:.3f}"),
    ("RII pairwise", "rii_pairwise", "{:.3f}"),
    ("RII strict (root intact)", "rii_strict", "{:.3f}"),
    ("TSP = H(F1, RII)", "tsp", "{:.3f}"),
]


def render_table(results: dict[str, dict]) -> str:
    names = list(results)
    lines = ["| metric | " + " | ".join(names) + " |",
             "|---|" + "---|" * len(names)]
    for label, key, fmt in ROWS:
        cells = []
        for name in names:
            val = results[name].get(key)
            cells.append(fmt.format(val) if val is not None else "—")
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--corpus", type=Path, default=HERE / "sample_corpus.tsv")
    ap.add_argument("--eval", dest="eval_path", type=Path, default=HERE / "eval_words.tsv")
    ap.add_argument("--vocab-size", type=int, default=500)
    ap.add_argument("--strategies", default="sp-bpe,sp-unigram,morph-unigram,byte")
    ap.add_argument("--report", type=Path, default=HERE / "report.md")
    ap.add_argument("--json", dest="json_path", type=Path, default=None)
    ap.add_argument("--no-normalize", action="store_true",
                    help="skip homophone-fidel normalization of the Amharic side")
    args = ap.parse_args()

    pairs = []
    for raw in args.corpus.read_text(encoding="utf-8").splitlines():
        if "\t" not in raw:
            continue
        am, en = raw.split("\t", 1)
        am = unicodedata.normalize("NFC", am.strip())
        if not args.no_normalize:
            am = normalize_fidel_variants(am)
        pairs.append((am, en.strip()))
    if not pairs:
        sys.exit(f"no sentence pairs found in {args.corpus}")

    items = load_eval(args.eval_path)
    if not args.no_normalize:
        for it in items:
            it.word = normalize_fidel_variants(unicodedata.normalize("NFC", it.word))

    am_lines = [p[0] for p in pairs]
    en_lines = [p[1] for p in pairs]
    joint = am_lines + en_lines

    print(f"corpus: {len(pairs)} sentence pairs from {args.corpus.name}")
    print(f"eval:   {len(items)} gold words "
          f"({sum(1 for i in items if i.gold_cuts)} segmented, "
          f"{sum(1 for i in items if i.root)} with roots)")

    segment, seg_name = make_segmenter()
    print(f"morph pre-segmentation backend: {seg_name}\n")

    wanted = [s.strip() for s in args.strategies.split(",") if s.strip()]
    tokenizers = {}
    for strat in wanted:
        print(f"building {strat} ...")
        if strat == "sp-bpe":
            tokenizers[strat] = SPTokenizer(strat, joint, "bpe", args.vocab_size)
        elif strat == "sp-unigram":
            tokenizers[strat] = SPTokenizer(strat, joint, "unigram", args.vocab_size)
        elif strat == "morph-unigram":
            tokenizers[strat] = MorphTokenizer(strat, am_lines, en_lines, segment, args.vocab_size)
        elif strat == "byte":
            tokenizers[strat] = ByteTokenizer()
        else:
            sys.exit(f"unknown strategy: {strat}")

    # Fairness: the retry loop can leave strategies with different achieved
    # vocab sizes on small corpora, which dominates every metric. Rebuild the
    # larger ones down to the common minimum.
    sp_based = {n: t for n, t in tokenizers.items() if not isinstance(t, ByteTokenizer)}
    if len(sp_based) > 1:
        target = min(t.vocab_size for t in sp_based.values())
        for name, tok in sp_based.items():
            if tok.vocab_size > target:
                print(f"rebuilding {name} at shared vocab size {target} for fairness")
                if isinstance(tok, MorphTokenizer):
                    tokenizers[name] = MorphTokenizer(name, am_lines, en_lines, segment, target)
                else:
                    tokenizers[name] = SPTokenizer(name, joint, tok.sp_model_type, target)

    results = {}
    for name, tok in tokenizers.items():
        res = {"vocab": tok.vocab_size}
        res.update(evaluate_corpus(tok, pairs))
        res.update(evaluate_words(tok, items))
        results[name] = res

    table = render_table(results)
    print("\n" + table + "\n")
    print("notes:")
    print(" - byte-level: boundary/RII/purity numbers are metric artifacts (it cuts")
    print("   everywhere, inside every fidel); judge it on fertility/seq-len only.")
    print(" - token purity is maximized by over-segmentation; read it next to fertility.")
    print(" - TSP is the model-selection scalar: harmonic mean of boundary F1 and")
    print("   pairwise Root Integrity. Higher is better at comparable fertility.")

    report = (
        "# Tokenization benchmark report\n\n"
        f"- corpus: `{args.corpus.name}` ({len(pairs)} pairs)\n"
        f"- eval set: `{args.eval_path.name}` ({len(items)} words)\n"
        f"- morph backend: {seg_name}\n"
        f"- fidel normalization: {'off' if args.no_normalize else 'on'}\n\n"
        + table + "\n"
    )
    args.report.write_text(report, encoding="utf-8")
    print(f"\nreport written to {args.report}")

    if args.json_path:
        args.json_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"json written to {args.json_path}")


if __name__ == "__main__":
    main()
