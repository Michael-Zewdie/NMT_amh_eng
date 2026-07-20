# Amharic–English NMT tokenization pipeline

Deliverables in this folder:

| File | What it is |
|---|---|
| `README.md` | Engineering trade-off analysis (§1) and the evaluation metric (§2) |
| `benchmark.py` | Comparative benchmark of 4 tokenization strategies (§3) |
| `sample_corpus.tsv` | 30 am–en sentence pairs (demo input; swap in your real corpus) |
| `eval_words.tsv` | Gold morpheme segmentations + consonantal roots (demo scale) |
| `report.md` / `report.json` | Output of the last benchmark run |

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python benchmark.py --corpus your_corpus.tsv --vocab-size 16000
```

---

## 1. Engineering trade-offs: SRE vs. morph pre-segmentation vs. byte-level

### 1a. Semitic Root Encoding (root + pattern factorization)

Encode each analyzable word as a root token plus a pattern/template token
(plus affix tokens), e.g. `ይሰብራል → <ROOT:sbr> <TPL:yə12a3al>`. The Amharic
verbal lexicon collapses onto ~1.5–2k productive roots × a small closed set
of templates, so the embedding table does real morphological parameter
sharing instead of hoping BPE co-locates `ሰበረ / ይሰብራል / ተሰበረ / መስበር`.

Costs, in the order they will actually hurt you:

- **Analyzer dependency.** You need root+template analyses, which in practice
  means HornMorpho. Throughput is tens of words/second unanalyzed-path; you
  must precompute over the corpus vocabulary and cache (Zipf means a ~100k-type
  cache covers >99% of tokens). Budget for coverage holes: proper nouns,
  loanwords, and web-text misspellings need a **dual-route** design — analyzable
  words go through SRE, everything else falls back to a subword vocab. That
  fallback path is a second tokenizer you now maintain and whose interaction
  with the first you must test.
- **Ambiguity.** Amharic surface forms are morphologically ambiguous; HornMorpho
  routinely returns multiple analyses. You need a disambiguation policy
  (first-analysis, frequency prior, or a small tagger). Whatever you pick
  becomes a silent noise floor in training data.
- **The en→am direction is the trap.** For decoding into Amharic the model must
  emit valid (root, template) pairs and you must **synthesize** the surface form.
  Generation of ill-formed combinations is not an edge case — you need either
  constrained decoding over the template inventory or a robust generator
  (HornMorpho generation mode) plus a fallback when synthesis fails. If you only
  translate am→en, SRE's cost drops by roughly half; if you need both directions,
  this is where the engineering budget goes.
- **Train/serve skew.** The analyzer version, cache, and disambiguation policy
  are all part of the model artifact now. Pin and ship them together.

**Verdict:** highest ceiling for morphological generalization, highest
integration cost. Only justified if analyzer coverage on *your* corpus
vocabulary measures >95% (measure it — it's a one-afternoon script) and you
primarily decode into English.

### 1b. Morphological pre-segmentation + Unigram

Segment affixes off in a preprocessing pass, then train SentencePiece Unigram
on the pre-segmented text. The subword model then spends its vocabulary budget
inside morphs instead of across morph boundaries.

- **Tooling reality check:** the pypi package `amseg` is a *sentence/word*
  segmenter (normalization + word boundary handling), not a morphological
  segmenter — it does not fill this slot. The real options are HornMorpho's
  segmentation mode or training a supervised segmenter on the Amharic morpheme
  segmentation datasets. `benchmark.py` auto-uses HornMorpho if importable and
  otherwise falls back to a rule-based affix stripper (a stand-in; treat its
  absolute numbers accordingly).
- **Reversibility.** You must be able to detokenize. Mark segmentation joints
  (e.g. `የ+ ቤት` or a dedicated joiner symbol registered as
  `user_defined_symbols` in SentencePiece) so the inverse map is trivial and
  lossless. Test round-tripping on the full corpus as a CI check — silent
  irreversibility is the classic failure here.
- **Hard commitment.** Unlike SRE's dual route, segmenter errors are baked into
  the training data with no recovery path. Keep the segmenter conservative:
  high-precision affix stripping beats aggressive full analysis.
- **The abugida ceiling.** Fidels fuse consonant+vowel, so morph junctions like
  `ቤቱ = ቤት + ኡ` are *not expressible* in grapheme space. A grapheme-level
  pre-segmenter can only recover boundaries that fall between fidels. The lever
  that removes this ceiling: **transliterate to a segmental romanization (SERA
  or similar) → segment → SentencePiece → detransliterate**. Costs ~1.9×
  character length pre-tokenization and one more reversible mapping to test,
  but it makes the fused morphology reachable by any downstream tokenizer.
- **Cheap and composable.** Runs offline once per corpus; no decode-time
  dependency at all if you also train the SP model to be the *only* runtime
  artifact (the segmenter only shapes the training distribution — at inference
  you can feed raw text, at a small consistency penalty, or ship the cached
  segmenter).

**Verdict:** best cost/benefit for a small team. One preprocessing script +
standard SentencePiece runtime; degrades gracefully to plain Unigram where the
segmenter abstains.

### 1c. Byte-level (ByT5-style)

- **Zero pipeline cost, zero OOV, and — underrated for Amharic — robustness to
  the homophone-fidel spelling chaos** (ሀ/ሃ/ኀ/ሐ, ሰ/ሠ, አ/ዐ, ጸ/ፀ variation is
  rampant in web-scraped text). Subword models fragment on unnormalized
  variants; bytes don't care.
- **The cost is quadratic, literally.** Ethiopic is 3 bytes/char in UTF-8, so
  Amharic sentences become ~11–12 tokens/word (measured below) vs ~2–3 for
  subwords: 4–5× longer sequences on the source side, with attention cost
  scaling quadratically and autoregressive decoding scaling linearly on the
  target side. For the same wall-clock budget you train on a fraction of the
  data. ByT5's own results need deeper encoders to re-compose characters —
  budget architecture changes, not just a tokenizer swap.
- **Fertility asymmetry** (am 3 bytes/char vs en 1) skews batching and
  effective learning rates per side; see the `fertility ratio` row the
  benchmark reports.
- A middle ground worth one experiment: **byte-level BPE** (GPT-style) over
  NFC-normalized, homophone-folded text — no OOV, subword-length sequences.

**Verdict:** the strongest *baseline* per engineering-hour, and plausibly
competitive at low resource. Benchmark it before building anything clever; if
your data is <500k pairs, don't be surprised if it ties morph+Unigram.

### Recommendation

1. Always-on preprocessing: NFC + homophone-fidel folding (implemented in
   `benchmark.py`, `--no-normalize` to disable).
2. Primary candidate: morph pre-segmentation (HornMorpho, cached, conservative)
   + Unigram with joiner symbols; consider the transliteration variant if the
   metric in §2 shows fused boundaries dominating your error mass.
3. Baseline to beat: byte-level (or byte-BPE) — cheap to run, hard to dismiss.
4. SRE only after measuring analyzer coverage and only if am→en is the
   dominant direction.
5. Decide with §2's metrics *plus* a small NMT probe (transformer-base, ~50k
   steps) on the top two — intrinsic metrics rank candidates, they don't
   certify BLEU/chrF deltas.

---

## 2. Metric: Morphological Boundary Alignment + Root Integrity (TSP)

Let a word `w` of length `n` (in fidels) have gold morph boundary set
`G(w) ⊂ {1,…,n−1}` (cut positions) and let tokenizer `T` induce cut set
`T(w)`. Annotation is *partial by necessity* (fused abugida junctions are
inexpressible in grapheme space), so each gold item carries a completeness
flag and the estimators respect it:

**Boundary precision / recall / F1** (micro-averaged over the eval set):

```
P  =  Σ_w∈complete |T(w) ∩ G(w)|  /  Σ_w∈complete |T(w)|
R  =  Σ_w          |T(w) ∩ G(w)|  /  Σ_w          |G(w)|
F1 =  2PR / (P + R)
```

Recall uses every annotated boundary; precision only words whose annotation is
complete (otherwise unannotated-but-real boundaries would be scored as false
positives).

**Token semantic purity** (char-weighted cluster purity of tokens against
morph spans, complete items only):

```
Purity = Σ_w Σ_{t ∈ tokens(w)} max_{m ∈ morphs(w)} |t ∩ m|   /   Σ_w |w|
```

Purity is *gamed by over-segmentation* (character tokens score 1.0 by
construction), so it is reported but never used alone — always read it next to
fertility.

**Root Integrity Index.** For a word with root radicals `ρ₁…ρ_r` located at
fidel positions `p₁…p_r` (matched by consonant-family skeleton, robust to
vowel order and homophone variants), with `tok(p)` the token containing
position `p`:

```
RII_pair   = E_w [ (1/(r−1)) Σ_i 1{ tok(p_i) = tok(p_{i+1}) } ]
RII_strict = E_w [ 1{ tok(p_1) = … = tok(p_r) } ]
```

This is the metric that directly operationalizes "BPE fractures consonant
roots": it measures whether the discontinuous root skeleton survives inside
single embedding units.

**Headline scalar** — Token Semantic Purity score:

```
TSP = 2 · F1 · RII_pair / (F1 + RII_pair)
```

The harmonic mean forces both properties: a tokenizer cannot buy TSP by
shattering words (F1 precision collapses, and RII → 0 since radicals land in
different tokens), nor by whole-word memorization on a big vocab (recall → 0
on affix boundaries). **Decision rule:** maximize TSP subject to a fertility
parity constraint, e.g. `fertility_am / fertility_en ≤ 1.6`, at a fixed shared
vocab size.

Statistical hygiene before trusting a comparison: ≥2–5k gold words (the
Amharic segmentation shared-task data gets you there), and bootstrap-resample
the eval set to get a CI on TSP — differences under ~0.02 are noise at that
scale.

---

## 3. Benchmark script

```
.venv/bin/python benchmark.py \
    --corpus sample_corpus.tsv \      # am<TAB>en, one pair per line
    --eval eval_words.tsv \           # gold segs + roots (format in file header)
    --vocab-size 500 \                # shared target; auto-shrinks on tiny corpora
    --strategies sp-bpe,sp-unigram,morph-unigram,byte \
    --report report.md --json report.json
```

Implementation notes:

- SentencePiece models are trained **in-memory on the joint am+en corpus**
  (shared vocab, as you'd do for the NMT model). If the retry loop lands
  different achieved vocab sizes, all SP-based strategies are rebuilt at the
  common minimum so no strategy wins on vocabulary budget.
- Root matching maps every fidel to its consonant family arithmetically
  (`cp − ((cp − 0x1200) mod 8)`), folds homophone families, and greedily
  matches the root skeleton left-to-right — no analyzer needed at eval time.
- `morph-unigram` uses HornMorpho when `import hm` succeeds (with an API probe
  and LRU cache), else the rule-based stripper. Segmentations are validated to
  be concatenation-preserving; anything else falls back to the whole word.
- Byte-level rows for boundary/RII/purity are metric artifacts (it cuts inside
  every fidel); judge byte on fertility and sequence length, which is its real
  trade-off axis.
- Words whose SentencePiece pieces don't reconstruct the surface form
  (normalization or UNK) are excluded from word-level metrics and counted in
  `eval_words_skipped`.

**Demo-scale caveat:** with the bundled 30-pair corpus everything trains at
vocab ≈210 and over-segments; the run demonstrates the *metrics*, not a
ranking. Point `--corpus` at your real parallel data and `--eval` at a few
thousand gold words before drawing conclusions.
