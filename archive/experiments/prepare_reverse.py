"""
experiments.prepare_reverse — tokenize data/final/ in the en->am direction.

Run: python -m experiments.prepare_reverse

Backtranslation for an am->en system needs monolingual ENGLISH translated INTO
Amharic, so the decoder always trains on real target text. That requires a
reverse en->am model first. Gezmu et al. measured +1.1 BLEU from exactly this
(Table 4: NMT 26.7 -> NMT+CACO 27.8) on this language pair.

Writes data/prepared/en-am/ — a DIFFERENT directory from the am-en cache, keyed
off cfg.data.{src_lang,tgt_lang}, so no swapping is needed and the forward
direction is untouched.
"""
import pickle

import pandas as pd

from model.common import BOS_ID, EOS_ID, load_tokenizer
from processing.utils.paths import FINAL, PREPARED

SRC, TGT, MAX_LEN = "en", "am", 150

def main() -> None:
    st, tt = load_tokenizer(SRC), load_tokenizer(TGT)
    out = PREPARED / f"{SRC}-{TGT}"; out.mkdir(parents=True, exist_ok=True)
    for sp in ("validation", "test", "train"):
        d = pd.read_csv(FINAL / f"{sp}.csv", usecols=["am", "en"], dtype=str).dropna()
        s = [[BOS_ID, *e.ids, EOS_ID] for e in st.encode_batch(d[SRC].tolist())]
        t = [[BOS_ID, *e.ids, EOS_ID] for e in tt.encode_batch(d[TGT].tolist())]
        keep = [i for i, (a, b) in enumerate(zip(s, t)) if len(a) <= MAX_LEN and len(b) <= MAX_LEN]
        with open(out / f"{sp}.pkl", "wb") as f:
            pickle.dump({"src": [s[i] for i in keep], "tgt": [t[i] for i in keep]}, f)
        print(f"[reverse] {sp}: kept {len(keep):,} of {len(d):,}")
    print(f"[reverse] wrote {out}")

if __name__ == "__main__":
    main()
