# Pipeline dataflow

End-to-end dataflow for the Amharic–English corpus build. Run the whole thing
with `python process.py` (toggles + config at the top of that file).

```mermaid
flowchart TD
    %% ---------------- sources ----------------
    SRC["HuggingFace AfriDocMT<br/>+ local Gezmu / Quran"]
    NP[("data/raw/nllb_full/*.parquet<br/>16.1M mined pairs<br/>(amh, eng, laser_score, urls)")]

    SRC --> COL["collect.py"]
    NP --> COL
    COL --> A[("data/raw/csv_raw/*.csv<br/>gezmu · afridoc_health/tech · quran<br/>(am, en)")]
    COL --> B[("data/raw/csv_raw/nllb.csv<br/>laser_score &gt; 1.09<br/>(am, en, laser_score, urls)")]

    %% ---------------- process.py · RUN_CLEAN: one clean() loop over every csv_raw file ----------------
    A --> CL
    B --> CL
    subgraph PROC["process.py · RUN_CLEAN — shared clean() per CSV"]
        direction TB
        CL["normalize"] --> C2["length_normalization · script_purity"]
        C2 --> C3["dedupe (am)<br/>one English per Amharic"]
    end
    CL:::shared
    C3 --> PN[("data/processed/*.csv<br/>cleaned per-source<br/>(nllb.csv keeps laser_score + urls)")]

    %% ---------------- optional in-place annotate passes (never drop rows) ----------------
    PN -. "RUN_LABSE (non-NLLB)" .-> LAB["score_labse<br/>annotate labse_score"]
    LAB -. rewrite .-> PN
    PN -. "RUN_LID (non-NLLB)" .-> LID["score_lid<br/>annotate source_lid / target_lid"]
    LID -. rewrite .-> PN
    LAB <-. hit/miss .-> SC[("data/scores/*.parquet<br/>content-keyed score cache")]
    LID <-. hit/miss .-> SC
    PN -. "manual / standalone" .-> RD["remove_domain<br/>drop NLLB source domains"]
    RD -. rewrite .-> PN

    %% ---------------- pool + stratified split ----------------
    PN --> POOL["pool · RUN_POOL<br/>apply COSINE_CUTOFF + LID floors<br/>concat → cross-source dedupe (am) → shuffle<br/>→ length-stratified 80/10/10"]
    POOL --> TR[("data/final/train.csv")]
    POOL --> VA[("data/final/validation.csv")]
    POOL --> TE[("data/final/test.csv")]

    %% ---------------- reporting ----------------
    PN --> LD["length_dist · RUN_LENGTH_DIST<br/>short/medium/long buckets"]
    LD --> LJ[("data/processed/lengthdist.json")]
    LD --> LP[("data/processed/figs/length_buckets_pie.png")]

    %% ---------------- standalone analysis (not part of process.py) ----------------
    subgraph STAND["standalone analysis · read-only unless noted"]
        direction TB
        DD["domain_dist"] --> DDP[("figs/nllb_domains_bar.png")]
        CF["char_freq → stdout"]
        SN["search_nllb → stdout"]
        SD["search_domain → stdout"]
    end
    NP --> DD
    PN --> DD
    A --> CF
    NP --> SN
    NP --> SD
    PN --> SD

    classDef store fill:#e3f2fd,stroke:#1565c0,color:#0d47a1;
    classDef shared fill:#fff3e0,stroke:#e65100,color:#bf360c;
    class NP,A,B,PN,TR,VA,TE,LJ,LP,DDP store;
```

Length buckets (`length_dist`) and the stratified split (`pool`) both read their
cutoffs from `nmt/lengths.py`, so the reported distribution and the split match.

## Notes

- **`collect.py` gathers everything into `csv_raw/`.** HuggingFace AfriDocMT +
  local Gezmu/Quran, plus the NLLB parquet laser-filtered down to `csv_raw/nllb.csv`
  (16.1M → ~425k; the LASER margin score plus per-sentence `source_lid`/`target_lid`
  LID confidences ship with the data and are immutable, so filtering them up front is
  a cheap, lossless reduction — the filter keeps `laser_score`, `source_lid`, and
  `target_lid` all above their cutoffs). `nllb.csv` is standardized to `am/en` but
  keeps `laser_score` + LID + url columns.
- **Collection is per-source and incremental.** The ~2.3GB NLLB parquet is downloaded
  only when absent, so a `LASER_CUTOFF` change re-reads it locally (`scan_parquet`,
  cutoffs pushed into the scan) and never re-downloads. AfriDoc/Gezmu/Quran have fixed
  inputs, so a default run skips them once their CSVs exist — which also avoids
  `load_dataset`'s HuggingFace round-trip on every call. Tuning the NLLB cutoff is
  therefore `python -m collection.collect nllb`: ~3s, no network. A named source always
  rebuilds; `--force` rebuilds everything.
- **`process.py` cleans every `csv_raw/*.csv` uniformly** through one shared
  `clean()` (normalize → length → script_purity → dedupe-on-`am`), so the whole
  corpus — including NLLB — is normalized identically.
- **`pool`** applies the quality cutoffs (`COSINE_CUTOFF`, the two LID floors — each
  on the sources that carry that score column), concatenates, dedups on `am` (one
  English per Amharic sentence), then splits 80/10/10 **stratified by Amharic length
  bucket**, so every split carries the same short/medium/long proportions. Each `am`
  is unique after dedup, so no Amharic sentence leaks across train/val/test.
- **`score_labse`** and **`score_lid`** are optional in-place passes (dashed) that
  only *annotate* the non-NLLB CSVs — `labse_score` from LaBSE cosine, and
  `source_lid`/`target_lid` from the *same* fastText model (Meta's lid218e) NLLB
  scored its own pairs with, so a non-NLLB row carries LID confidences directly
  comparable to an NLLB row. Neither drops a row; the cutoffs live at the pool stage.
- **Scoring is separate from filtering, on purpose.** The models cost hours; a cutoff
  costs a comparison. Every score is cached in `data/scores/*.parquet` keyed by a
  hash of the sentence pair, so it survives a re-clean and is shared across sources.
  Retuning a threshold is a re-pool, not a re-embed, and *lowering* one recovers rows
  rather than losing them permanently. Delete a cache file to force a re-score — do
  this after changing a model or `processing/clean/normalize.py`, since the keys are
  hashes of normalized text.
- **`remove_domain`** is a standalone manual tool (not a `process.py` stage) that
  prunes NLLB source domains from `data/processed/nllb.csv` in place.
