"""
model.data.tokenizer — loads the pretrained en/am tokenizer artifacts.

Both data/tokenizer/{en,am}/tokenizer.json were trained with the identical
special-token layout below (see processing/train_tokenizer.py and
processing/train_tokenizer_am.py) — every other file should import these
ids rather than hardcoding them. Neither tokenizer has a post_processor
configured, so encode(...).ids does NOT include BOS/EOS automatically.
"""
from tokenizers import Tokenizer

from processing.utils.paths import TOKENIZER_AM, TOKENIZER_EN

PAD_ID = 0
UNK_ID = 1
BOS_ID = 2
EOS_ID = 3

_TOKENIZER_PATHS = {
    "am": TOKENIZER_AM,
    "en": TOKENIZER_EN,
}


def load_tokenizer(lang: str) -> Tokenizer:
    path = _TOKENIZER_PATHS[lang] / "tokenizer.json"
    return Tokenizer.from_file(str(path))
