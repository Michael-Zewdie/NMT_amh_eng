"""
model.tokenize.preprocess — the text preprocessing chain that runs BEFORE subword
tokenization: Moses tokenization, and Amharic Ethiopic -> Latin transliteration.

Lives here, next to train_tokenizer{,_am}.py, because it is one pipeline with
them — latinize, then learn/apply subwords. Kept out of process/clean/ because
these steps exist to serve the *vocabulary*, not to clean the text: transliteration
is applied specifically so a single shared subword vocabulary can cover both
languages, which is impossible while Amharic is in Ethiopic and English in Latin.

Order matters and follows Gezmu, Nurnberger & Bati (LREC 2022) §4.1:

    "We tokenized the English datasets with Moses' tokenizer script; we modified
     Moses' script to tokenize the Amharic datasets. Next, to share named-entities
     between the languages, the Amharic datasets were transliterated with a
     transliteration scheme, Amharic transliteration for machine translation"

i.e. Moses FIRST, transliterate SECOND. Transliteration is per whitespace token,
so tokenizing first is what gets punctuation split off into its own tokens.

Why each step earns its place (both measured on Gezmu's 140k train split):

  Moses tokenization
      SentencePiece Unigram does NOT pre-split punctuation, so without this
      21.4% of punctuation ends up glued to its word and 376 vocabulary entries
      are duplicates like "congregation." vs "congregation". With it: 0.05%.
      (English ByteLevel BPE splits punctuation natively and does not need this —
      it measured 0.00% glued either way. The need is specific to Unigram.)
      Training with the glue cost ~11 BLEU at matched steps.

  Transliteration
      Ethiopic is an abugida: each glyph fuses a consonant AND a vowel, so the
      consonantal root of a Semitic language is unreachable to any subword model
      -- "sbrat" and "sabara" share no Ethiopic character despite sharing a root.
      It also makes a shared vocabulary meaningful at all: cross-lingual shared
      subword types rise 180 -> 1,003 (2.3% -> 12.5% of types in use), and the
      shared pieces change from digit strings ('2012', '144,000') to named
      entities and loanwords ('ethiopia', 'protestant', 'benjamin', 'hospital').

Generated English must be run back through `detok_en` before scoring — a model
trained on Moses-tokenized targets emits Moses-tokenized text, and sacrebleu's
13a tokenizer cannot undo it (it splits punctuation on both sides, but cannot
rejoin "do n't" -> "don't").

The transliteration itself is the authors' own released implementation, vendored
unmodified at model/tokenize/at4mt_transliteration.py (MIT).
"""
from model.tokenize import at4mt_transliteration as at4mt

_EN_TOK = _AM_TOK = _EN_DETOK = None


def _moses():
    """Lazily built Moses tokenizers/detokenizer, so importing this module stays
    cheap and does not require sacremoses unless a caller actually preprocesses.

    escape=False is REQUIRED throughout: sacremoses XML-escapes by default,
    turning "don't" into "don &apos;t" and '"' into '&quot;'. That would be baked
    into the training text and emitted at inference, wrecking BLEU against raw
    references.

    Amharic uses the English tokenizer: the paper used a *modified* Moses script
    for Amharic, which is not released. On Ethiopic (and on transliterated Latin)
    Moses' English rules do little beyond splitting punctuation off, which is the
    part that matters here — but this is a deviation, not a match.
    """
    global _EN_TOK, _AM_TOK, _EN_DETOK
    if _EN_TOK is None:
        from sacremoses import MosesDetokenizer, MosesTokenizer
        _EN_TOK = MosesTokenizer(lang="en")
        _AM_TOK = MosesTokenizer(lang="en")
        _EN_DETOK = MosesDetokenizer(lang="en")
    return _EN_TOK, _AM_TOK, _EN_DETOK


def moses_en(sentence: str) -> str:
    """Moses-tokenize an English sentence."""
    return " ".join(_moses()[0].tokenize(sentence, escape=False))


def detok_en(sentence: str) -> str:
    """Undo Moses tokenization on generated English, before scoring."""
    return _moses()[2].detokenize(sentence.split())


def translit_am(sentence: str) -> str:
    """Moses-tokenize Ethiopic Amharic, then transliterate each token to Latin."""
    return " ".join(at4mt.ethiopic2latin(w) for w in _moses()[1].tokenize(sentence, escape=False))
