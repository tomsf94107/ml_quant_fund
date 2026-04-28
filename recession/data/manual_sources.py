"""
Manual data sources — non-FRED ingestion.

Some series in our feature set aren't available via the FRED API but are
published elsewhere as direct downloads. This module wraps each one in a
fetch function that returns observations in the same shape as fred_client.

Currently implemented:
- EBP (Excess Bond Premium, Gilchrist-Zakrajšek): Fed Board CSV

Each fetch function:
- Caches the raw download in recession/cache/manual/
- Parses to a list of {date, value, vintage_date} dicts (matches fred_client)
- Stamps vintage_date = observation_month + publication_lag_days

Per-source caching is keyed on the URL+date so re-runs don't re-download
unchanged data, but does refresh when the upstream file changes.
"""
from __future__ import annotations

import csv
import io
import logging
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Optional

import requests

logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------

DEFAULT_CACHE_DIR = Path(__file__).resolve().parent.parent / "cache" / "manual"

EBP_CSV_URL = (
    "https://www.federalreserve.gov/econres/notes/feds-notes/ebp_csv.csv"
)
EBP_PUBLICATION_LAG_DAYS = 14   # mid-month update for prior month

# The Fed CSV header row: "date,gz_spread,ebp,est_prob"
# We want column 'ebp'. Other columns: gz_spread (raw GZ spread),
# est_prob (estimated default probability — distinct concept).
EBP_VALUE_COLUMN = "ebp"


# -----------------------------------------------------------------------------
# HTTP helper with cache
# -----------------------------------------------------------------------------

def _cached_download(
    url: str,
    cache_dir: Path,
    cache_filename: str,
    force_refresh: bool = False,
    timeout: int = 30,
) -> str:
    """
    Download a URL with simple file-based caching.

    Cache invalidation: by default we trust the cache for 7 days, then re-pull.
    For monthly data this is plenty — the EBP CSV updates monthly anyway.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / cache_filename

    if not force_refresh and cache_path.exists():
        age_days = (datetime.now() - datetime.fromtimestamp(cache_path.stat().st_mtime)).days
        if age_days < 7:
            logger.debug("Cache hit (%d days old): %s", age_days, cache_filename)
            return cache_path.read_text()

    logger.info("Downloading %s ...", url)
    resp = requests.get(url, timeout=timeout)
    resp.raise_for_status()
    cache_path.write_text(resp.text)
    return resp.text


# -----------------------------------------------------------------------------
# EBP (Gilchrist-Zakrajšek Excess Bond Premium)
# -----------------------------------------------------------------------------

def fetch_ebp(
    cache_dir: Optional[Path] = None,
    force_refresh: bool = False,
) -> list[dict]:
    """
    Pull EBP from the Federal Reserve Board CSV.

    The CSV format (as of April 2026):
        date, gz_spread, ebp, est_prob
        1973-01-01, 1.86, 0.42, 0.0167
        ...

    Returns:
        [{"date": "YYYY-MM-01", "value": float, "vintage_date": "YYYY-MM-DD"}, ...]
        Date is normalized to first of month; vintage_date = observation
        month-end + EBP_PUBLICATION_LAG_DAYS.
    """
    cache_dir = cache_dir or DEFAULT_CACHE_DIR

    try:
        csv_text = _cached_download(
            EBP_CSV_URL, cache_dir, "ebp_csv.csv",
            force_refresh=force_refresh,
        )
    except requests.RequestException as e:
        logger.error("Could not download EBP CSV: %s", e)
        return []

    out = []
    reader = csv.DictReader(io.StringIO(csv_text))
    if EBP_VALUE_COLUMN not in (reader.fieldnames or []):
        logger.error(
            "EBP CSV missing expected column '%s'; got %s",
            EBP_VALUE_COLUMN, reader.fieldnames,
        )
        return []

    for row in reader:
        # CSV date is typically 'YYYY-MM-DD' or 'M/D/YYYY'
        raw_date = (row.get("date") or "").strip()
        if not raw_date:
            continue

        try:
            d = _parse_date(raw_date)
        except ValueError:
            logger.warning("Skipping unparseable EBP date: %s", raw_date)
            continue

        raw_value = (row.get(EBP_VALUE_COLUMN) or "").strip()
        if raw_value in ("", ".", "NA", "N/A"):
            continue

        try:
            value = float(raw_value)
        except ValueError:
            logger.warning("Skipping non-numeric EBP value at %s: %r", raw_date, raw_value)
            continue

        # Normalize to first of month, stamp vintage
        observation_month = d.replace(day=1).isoformat()
        vintage_date = (d.replace(day=1) + _month_end_offset(d)
                        + timedelta(days=EBP_PUBLICATION_LAG_DAYS)).isoformat()

        out.append({
            "date":         observation_month,
            "value":        value,
            "vintage_date": vintage_date,
        })

    logger.info("Parsed %d EBP observations from CSV", len(out))
    return out


# -----------------------------------------------------------------------------
# Date helpers
# -----------------------------------------------------------------------------

def _parse_date(s: str) -> date:
    """Try a few common formats and return a date."""
    for fmt in ("%Y-%m-%d", "%m/%d/%Y", "%Y/%m/%d", "%d-%b-%Y"):
        try:
            return datetime.strptime(s, fmt).date()
        except ValueError:
            continue
    raise ValueError(f"Unrecognized date format: {s!r}")


def _month_end_offset(d: date) -> timedelta:
    """Days from the first of d's month to the last of d's month."""
    if d.month == 12:
        next_first = date(d.year + 1, 1, 1)
    else:
        next_first = date(d.year, d.month + 1, 1)
    last = next_first - timedelta(days=1)
    return last - d.replace(day=1)


# -----------------------------------------------------------------------------
# Registry
# -----------------------------------------------------------------------------

# Maps feature_name -> fetch function. Used by ingest.py to route manual
# fetches.
MANUAL_FETCHERS: dict = {
    "EBP": fetch_ebp,
}
