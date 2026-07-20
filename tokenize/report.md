# Tokenization benchmark report

- corpus: `sample_corpus.tsv` (30 pairs)
- eval set: `eval_words.tsv` (34 words)
- morph backend: rule-based affix stripper
- fidel normalization: on

| metric | sp-bpe | sp-unigram | morph-unigram | byte |
|---|---|---|---|---|
| vocab size | 210 | 210 | 210 | 256 |
| fertility am (tok/word) | 3.24 | 2.67 | 2.97 | 11.75 |
| fertility en (tok/word) | 2.53 | 2.25 | 2.31 | 4.71 |
| fertility ratio am/en | 1.28 | 1.18 | 1.29 | 2.49 |
| chars/token am | 1.13 | 1.38 | 1.24 | 0.31 |
| mean seq len am (tokens) | 11.1 | 9.2 | 10.2 | 40.3 |
| mean seq len en (tokens) | 12.7 | 11.3 | 11.5 | 23.6 |
| boundary precision | 0.423 | 0.391 | 0.453 | 0.375 |
| boundary recall | 0.938 | 0.781 | 1.000 | 1.000 |
| boundary F1 | 0.583 | 0.521 | 0.623 | 0.545 |
| token purity | 0.976 | 0.918 | 1.000 | 1.000 |
| RII pairwise | 0.235 | 0.176 | 0.176 | 0.000 |
| RII strict (root intact) | 0.190 | 0.095 | 0.095 | 0.000 |
| TSP = H(F1, RII) | 0.335 | 0.264 | 0.275 | 0.000 |
