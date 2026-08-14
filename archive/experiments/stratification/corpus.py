"""Corpus assembly — chosen specifically because its per-row domain is knowable.

Gezmu alone has NO per-row domain field (checked: data/processed/gezmu.csv is
just am/en + quality scores) — "domain-stratify Gezmu" is undefined, not just
weak, since there is exactly one stratum. Real per-row domain only exists for:

  - afridoc_health, afridoc_tech: each corpus IS one domain (curated,
    single-purpose), so together they contribute exactly 2 named domains,
    loaded straight from data/raw/csv_raw/ (see load_curated, revised
    2026-08-12 twice over): first to disable every process.pool quality gate
    including LID (which was silently dropping 165 health/264 tech rows),
    then discovered LID wasn't the real story — data/processed/afridoc_*.csv
    was both stale (pre-dates a clean() code change) AND, more importantly,
    process.clean.filters.script_purity was dropping ~26% of this corpus by
    rejecting rows with a legitimate bilingual-glossed term embedded in the
    Amharic column ("(Haemoglobin)", "B9(folate)") — a structural feature of
    this corpus, not noise. load_curated now bypasses data/processed/
    entirely and applies only normalize + paren_balance + dedupe(am,en),
    landing at 9,907/9,724 of the raw 10,000/10,000 each.
  - nllb: carries source_url/target_url per row, so process.utils.websites
    .domains() gives a REAL per-row registered-domain label. Filtered at
    NLLB_AFRICOMET_CUTOFF=0.8 (raised from process.pool's production default
    0.5, 2026-08-12 — per-user call that mined text specifically needs a
    stricter quality bar; 338,334 of 2,361,879 rows survive vs 1,819,989 at
    0.5), then collapsed to its top TOP_N_NLLB_DOMAINS=20 (raised from 15 —
    at the 0.8 bar the domain distribution is EVEN MORE Zipfian than at 0.5:
    even the top 100 domains cover only 45% of survivors, so N=20 is a
    deliberate compromise, chosen to stay close to the semantic arm's k=16
    rather than chased toward maximum coverage — see NLLB_AFRICOMET_CUTOFF's
    comment for the exact tradeoff) so domain-label cardinality (~22 total)
    stays in the same ballpark as the semantic arm's k=16 — otherwise a BLEU
    gap could just mean "more strata", not "domain vs semantic" as the KIND
    of signal. Rows outside the top 20 (a long Zipfian tail of one-off
    sites, plus unparseable "unknown" urls) are DROPPED, not lumped into a
    catch-all "other" bucket — the whole point of this corpus is that every
    stratum is a domain you can actually name.

quran/religious/ccaligned are excluded too, not just gezmu — this is a new,
deliberately-scoped corpus for this question, distinct from "broad" (which is
everything except gezmu). At NLLB_AFRICOMET_CUTOFF=0.8 / top-20 domains this
lands around ~102k pooled rows pre-dedup (90,185 NLLB + ~11,849 unfiltered
health/tech) — smaller than the original 0.5-cutoff/top-15 design's ~430k, by
deliberate tradeoff for cleaner mined text (2026-08-12), but still comfortably
above every floor-effect threshold this project has found (~50k+).
"""
import numpy as np
import pandas as pd
import polars as pl

from process import pool as pool_mod
from process.utils.paths import PROCESSED
from process.utils.websites import domains as registered_domains

TOP_N_NLLB_DOMAINS = 20   # raised from 15 (2026-08-12) — see NLLB_AFRICOMET_CUTOFF
NLLB_AFRICOMET_CUTOFF = 0.8   # raised from process.pool's default 0.5 (2026-08-12):
# per-user call that NLLB specifically needs a stricter quality bar than the
# production pipeline's default. At >0.8 only 338,334 of 2,361,879 rows survive
# (vs 1,819,989 at >0.5), and coverage got MORE Zipfian, not less — even the top
# 100 named domains cover only 45% of survivors. TOP_N_NLLB_DOMAINS raised
# 15->20 to partly compensate, chosen to stay close to the semantic arm's k=16
# (not chased up toward higher coverage) so domain-label cardinality doesn't
# drift from what's needed for a fair domain-vs-semantic comparison — a pure
# scale-maximizing choice would have picked N=100+ instead. This is a real,
# accepted tradeoff: less NLLB volume than the original 0.5-cutoff design, in
# exchange for a materially cleaner mined corpus.


def load_curated(name: str, domain_label: str) -> pd.DataFrame:
    """A whole AfriDocMT source, loaded from RAW (data/raw/csv_raw/, NOT
    data/processed/) and only lightly cleaned: normalize (text hygiene, drops
    nothing) + paren_balance (drops genuinely truncated fragments) +
    dedupe(am,en) (drops exact-duplicate pairs). Deliberately SKIPS
    length_normalization and script_purity (2026-08-12, per-user
    investigation into "there should be 10k of both"): script_purity alone
    was dropping ~26% of this corpus, and sampling the dropped rows showed
    it isn't catching noise — it's catching legitimate bilingual-glossed
    medical/tech terminology this source routinely embeds in the Amharic
    column ("(Haemoglobin)", "B9(folate)", "(biological theraphysitics)"),
    a structural feature of this corpus, not an error. data/processed/ was
    ALSO stale on top of that (built from cleaning code since changed,
    RUN_CLEAN never re-run) — moot now since this bypasses it entirely.
    Lands at 9,907/9,724 of the raw 10,000/10,000 each — essentially the
    whole corpus, with only genuine fragments/duplicates removed. No LID/
    cosine/AfriCOMET cutoffs apply here at all (this never touches
    process.pool.load_pairs); LID stays enabled for every other source."""
    from process.clean.filters import dedupe, normalize, paren_balance
    from process.utils.paths import CSV_RAW

    raw = pl.read_csv(CSV_RAW / f"{name}.csv", infer_schema_length=0)
    cleaned = dedupe(paren_balance(normalize(raw, "am", "en", name), "am", "en"), ("am", "en"))
    df = cleaned.select(["am", "en"]).to_pandas()
    df["_domain"] = domain_label
    print(f"[build]   {name}: {len(df)} pairs (of {raw.height} raw), domain={domain_label!r}")
    return df


def load_nllb_top_domains(top_n: int = TOP_N_NLLB_DOMAINS) -> pd.DataFrame:
    """nllb.csv rows surviving the standard mined cutoffs, restricted to the
    top_n most common registered domains (excluding "unknown"), each row
    tagged with its domain.

    Domains are ranked on POST-cutoff survivors, not raw crawl volume — a
    site that mines a lot of low-quality text shouldn't outrank one that
    mines less but cleaner text, for a corpus meant to represent "domains
    that actually contributed usable training data".
    """
    raw = pl.read_csv(PROCESSED / "nllb.csv",
                      columns=["am", "en", "source_url", "target_url", "laser_score",
                               "labse_score", "africomet_score", "source_lid", "target_lid"])
    df = raw.to_pandas()
    df = pool_mod.apply_cutoffs(df, "nllb", pool_mod.COSINE_CUTOFF, NLLB_AFRICOMET_CUTOFF,
                                pool_mod.SOURCE_LID_CUTOFF, pool_mod.TARGET_LID_CUTOFF)

    dom = registered_domains(pl.from_pandas(df[["source_url", "target_url"]])).to_pandas()
    df["_domain"] = dom.to_numpy()

    counts = df.loc[df["_domain"] != "unknown", "_domain"].value_counts()
    top = set(counts.head(top_n).index)
    print(f"[build]   nllb: top {top_n} named domains -> "
          f"{', '.join(f'{d}({c:,})' for d, c in counts.head(top_n).items())}")

    kept = df[df["_domain"].isin(top)][["am", "en", "_domain"]].reset_index(drop=True)
    print(f"[build]   nllb: {len(df)} survived mined cutoffs, "
          f"{len(kept)} kept ({len(kept) / len(df):.1%}) after restricting to top {top_n} named domains")
    return kept


def ensure_embedded(pooled: pd.DataFrame) -> None:
    """Make sure every pooled row's English text has a cached mpnet_en embedding
    before split_semantic() needs it (process.pool.split_semantic is cache-only
    and raises otherwise). Necessary because load_curated() now reads health/tech
    straight from raw (skipping script_purity/length_normalization, 2026-08-12)
    — ~7,600 rows that used to be filtered out of data/processed/ before ever
    reaching score_embed's cache now show up here for the first time. Computes
    only the gap (embedded() is a cache-diff, same mechanism process.dist.
    diversity's --embed path uses); a full hit costs nothing."""
    from process.cache.embed_cache import embedded
    from process.dist.semantic.en_encoder import get_en_encoder

    def compute(texts: list[str]) -> np.ndarray:
        return get_en_encoder().encode(texts, batch_size=128, normalize_embeddings=True,
                                       show_progress_bar=True)

    embedded(pooled, "mpnet_en", ("en",), compute, dim=768)


def assemble_pool() -> pd.DataFrame:
    frames = [
        load_curated("afridoc_health", "health"),
        load_curated("afridoc_tech", "tech"),
        load_nllb_top_domains(),
    ]
    pooled = pool_mod.decontaminate(pool_mod.pool(frames, seed=pool_mod.SEED))
    print(f"[build] domain composition after pool+decontam:")
    print(pooled["_domain"].value_counts().to_string())
    ensure_embedded(pooled)
    return pooled
