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
# SP500 — historical CSV + FRED top-up
# -----------------------------------------------------------------------------
# Why this is split:
#   FRED's SP500 series only has ~10 years of history (starts ~2015). For T2
#   (15% drawdown target) we need 1960+ history to have enough events for any
#   meaningful backtest. None of our paid sources cover deep SPX history:
#     - Massive (Polygon) Indices API: launched 2023, ~3 years history
#     - Unusual Whales: focuses on options flow, not deep index history
#     - FRED: ~10 years
#   Yahoo (^GSPC) has full daily history back to 1928 but the yfinance
#   library is occasionally flaky.
#
# Strategy: pull from Yahoo ONCE via yfinance, save the result to a permanent,
# version-controlled CSV at recession/data/sp500_history.csv, then never
# touch yfinance again. Going forward, this CSV serves as the historical
# anchor; FRED handles ongoing monthly top-ups via fred_client. The CSV
# is committed to git so the historical data is fully reproducible.
#
# Operational scripts:
#   - bootstrap_sp500_history.py: one-time backfill from Yahoo (~660 rows)
#   - manual_sources.fetch_sp500: the ongoing reader, used by ingest.py

YAHOO_SP500_TICKER = "^GSPC"
SP500_PUBLICATION_LAG_DAYS = 1

# Permanent committed location — NOT a cache. Lives with the code.
SP500_HISTORY_CSV = Path(__file__).resolve().parent / "sp500_history.csv"


def fetch_sp500(
    cache_dir: Optional[Path] = None,        # accepted for API compatibility, unused
    force_refresh: bool = False,             # accepted for API compatibility, unused
) -> list[dict]:
    """
    Read SP500 monthly history from the committed CSV.

    The CSV is created by `python -m recession.data.bootstrap_sp500_history`
    (one-time, requires yfinance). After that, this function is the only
    way the ingest pipeline reads SP500 historical data.

    Returns a list of monthly end-of-period observations:
        [{"date": "YYYY-MM-01", "value": float, "vintage_date": "YYYY-MM-DD"}]

    SP500 is market-priced, not revisable; vintage_date = month-end + 1 day.

    If the CSV doesn't exist yet, returns [] with an instructive log line
    pointing the user to the bootstrap script.
    """
    if not SP500_HISTORY_CSV.exists():
        logger.error(
            "SP500 history CSV not found at %s. "
            "Run the one-time bootstrap: "
            "python -m recession.data.bootstrap_sp500_history",
            SP500_HISTORY_CSV,
        )
        return []

    rows: list[dict] = []
    with open(SP500_HISTORY_CSV) as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                # CSV is already monthly — schema: month,close,vintage_date
                month_str = row["month"].strip()
                if len(month_str) == 7:                # "YYYY-MM"
                    obs_date = f"{month_str}-01"
                elif len(month_str) == 10:             # "YYYY-MM-DD"
                    obs_date = month_str
                else:
                    continue
                rows.append({
                    "date":         obs_date,
                    "value":        float(row["close"]),
                    "vintage_date": row.get("vintage_date", obs_date),
                })
            except (ValueError, KeyError):
                continue

    rows.sort(key=lambda r: r["date"])
    logger.info(
        "Loaded %d monthly SP500 observations from %s",
        len(rows), SP500_HISTORY_CSV.name,
    )
    return rows


# Backward-compat alias — the registry still references the old name.
fetch_sp500_yahoo = fetch_sp500


# -----------------------------------------------------------------------------
# Registry
# -----------------------------------------------------------------------------

# Maps feature_name -> fetch function. Used by ingest.py to route manual
# fetches.
MANUAL_FETCHERS: dict = {
    "EBP":   fetch_ebp,
    "SP500": fetch_sp500,
}
