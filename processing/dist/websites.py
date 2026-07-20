"""
nmt.websites — map NLLB mined pairs to the registered domain they were crawled from.

Each pair carries the page it came from (source_url, falling back to target_url).
``domains()`` pulls the host out of that URL and collapses it to its registered
domain (am.econologie.com -> econologie.com) via tldextract, so a site's language/www
subdomains don't split into separate entries.

Shared by collection/search_domain.py and processing/remove_domain.py.
"""
import polars as pl
import tldextract

_HOST_RE = r"^https?://([^/?#]+)"

# Offline: use tldextract's bundled public-suffix snapshot, no network fetch.
_extract = tldextract.TLDExtract(suffix_list_urls=())


def _registered_domain(host: str) -> str:
    """Collapse a host to its registered domain (am.econologie.com -> econologie.com)."""
    if host == "unknown":
        return "unknown"
    return _extract(host).registered_domain or "unknown"


def domains(df: pl.DataFrame) -> pl.Series:
    """Registered domain each row was crawled from (source_url, else target_url)."""
    hosts = df.select(
        pl.coalesce(
            pl.col("source_url").str.extract(_HOST_RE, 1),
            pl.col("target_url").str.extract(_HOST_RE, 1),
        ).fill_null("unknown").alias("host")
    )["host"]

    # tldextract is a per-string Python call, so map only the distinct hosts.
    mapping = {host: _registered_domain(host) for host in hosts.unique().to_list()}
    return hosts.replace(mapping)
