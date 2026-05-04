"""
features/fred_client.py
─────────────────────────────────────────────────────────────────────────────
FRED (Federal Reserve Economic Data) API client.

Free, official US government source for macro indicators.
Replaces yfinance for VIX, TNX, DXY (more reliable than yfinance).

Usage:
  from features.fred_client import fred_get
  vix = fred_get('VIXCLS', start='2024-01-01', end='2026-04-30')
  tnx = fred_get('DGS10', start='2024-01-01', end='2026-04-30')
  dxy = fred_get('DTWEXBGS', start='2024-01-01', end='2026-04-30')

Series IDs we use:
  VIXCLS    — CBOE Volatility Index (VIX), daily close
  DGS10     — 10-Year Treasury Constant Maturity Rate (in %)
  DTWEXBGS  — Trade Weighted U.S. Dollar Index: Broad, Goods and Services
              (Fed's official broad dollar index, daily)

Returns a pandas DataFrame with 'date' and 'value' columns, or None on failure.
"""
from __future__ import annotations

import logging
import os
import time
from typing import Optional
from datetime import datetime

import pandas as pd
import requests

# Module-level Session for connection reuse (DNS pool conservation).
# Added May 4 2026 to prevent DNS thread exhaustion in Pipeline B/C.
_session = requests.Session()

log = logging.getLogger(__name__)

FRED_API_KEY = os.getenv("FRED_API_KEY", "")
BASE_URL = "https://api.stlouisfed.org/fred/series/observations"

# Retry config
MAX_RETRIES = 3
RETRY_BACKOFF = 1.0  # 1s, 2s, 4s


def fred_get(series_id: str,
             start: Optional[str] = None,
             end: Optional[str] = None,
             timeout: int = 10) -> Optional[pd.DataFrame]:
    """
    Fetch a FRED series.

    Args:
        series_id: FRED series ID (e.g. 'VIXCLS', 'DGS10', 'DTWEXBGS')
        start: 'YYYY-MM-DD' start date (inclusive)
        end:   'YYYY-MM-DD' end date (inclusive)
        timeout: HTTP request timeout in seconds

    Returns:
        DataFrame with columns ['date' (datetime64), 'value' (float)],
        or None if request failed.
        Missing/withheld values (FRED uses '.') are dropped.
    """
    if not FRED_API_KEY:
        log.warning("FRED_API_KEY not set — fred_get returning None")
        return None

    params = {
        "series_id": series_id,
        "api_key": FRED_API_KEY,
        "file_type": "json",
    }
    if start:
        params["observation_start"] = start
    if end:
        params["observation_end"] = end

    last_err = None
    for attempt in range(MAX_RETRIES):
        try:
            r = _session.get(BASE_URL, params=params, timeout=timeout)
            if r.status_code == 200:
                data = r.json()
                obs = data.get("observations", [])
                if not obs:
                    log.warning(f"FRED {series_id}: empty observations")
                    return pd.DataFrame(columns=["date", "value"])

                # FRED uses '.' for missing values
                rows = []
                for o in obs:
                    v = o.get("value", ".")
                    if v == "." or v is None:
                        continue
                    try:
                        rows.append({
                            "date": pd.to_datetime(o["date"]),
                            "value": float(v),
                        })
                    except (ValueError, KeyError):
                        continue

                df = pd.DataFrame(rows)
                if df.empty:
                    log.warning(f"FRED {series_id}: no valid observations after filtering")
                return df

            else:
                last_err = f"HTTP {r.status_code}: {r.text[:200]}"
                log.warning(f"FRED {series_id}: attempt {attempt+1}/{MAX_RETRIES} {last_err}")

        except (requests.exceptions.Timeout,
                requests.exceptions.ConnectionError,
                requests.exceptions.RequestException) as e:
            last_err = str(e)
            log.warning(f"FRED {series_id}: attempt {attempt+1}/{MAX_RETRIES} {last_err}")

        if attempt < MAX_RETRIES - 1:
            time.sleep(RETRY_BACKOFF * (2 ** attempt))

    log.error(f"FRED {series_id}: all {MAX_RETRIES} retries failed: {last_err}")
    return None


# Module-level cache for fred_get_as_series, keyed by series_id only.
# FRED historical data is immutable for past dates, so we fetch the
# WIDEST range on first call (start through today), cache it, and slice
# in memory for subsequent calls.
#
# Walk-forward calls build_feature_dataframe 3,739 times with varying
# end_date. Without this cache, that produced 11,217 FRED API calls
# triggering server-side rate limits (HTTP 500). Now produces 3 FRED
# calls total per process (one per series_id used).
# Added May 4 2026.
_FRED_FULL_SERIES_CACHE: dict = {}


def fred_get_as_series(series_id: str,
                       start: Optional[str] = None,
                       end: Optional[str] = None) -> Optional[pd.Series]:
    """
    Convenience wrapper: returns indexed pd.Series (index=date, values=value).
    Returns None on failure, empty Series if no data.

    May 4 2026: Caches the FULL series_id history at module level on first
    call, then slices in memory for the requested [start, end] range.
    Eliminates rate-limit issues during walk-forward where the same series
    is requested with different end_dates thousands of times.
    """
    if series_id not in _FRED_FULL_SERIES_CACHE:
        # First call for this series — fetch widest possible range.
        # FRED's default range is full history; passing start=None gets it.
        full_series = fred_get(series_id, start=None, end=None)
        if full_series is None:
            _FRED_FULL_SERIES_CACHE[series_id] = None
        elif full_series.empty:
            _FRED_FULL_SERIES_CACHE[series_id] = pd.Series(dtype=float)
        else:
            s = full_series.set_index("date")["value"]
            s.index = pd.to_datetime(s.index).normalize()
            _FRED_FULL_SERIES_CACHE[series_id] = s

    cached = _FRED_FULL_SERIES_CACHE[series_id]
    if cached is None:
        return None
    if cached.empty:
        return pd.Series(dtype=float)

    # Slice to requested range
    result = cached
    if start is not None:
        start_ts = pd.to_datetime(start).normalize()
        result = result[result.index >= start_ts]
    if end is not None:
        end_ts = pd.to_datetime(end).normalize()
        result = result[result.index <= end_ts]

    return result.copy()


# Quick test (run as script)
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    if not FRED_API_KEY:
        print("ERROR: FRED_API_KEY not set in environment")
        print("Set it with: export FRED_API_KEY='your_key'")
        exit(1)

    print(f"Testing FRED API with key prefix: {FRED_API_KEY[:8]}...")

    for series in ["VIXCLS", "DGS10", "DTWEXBGS"]:
        print(f"\nFetching {series}...")
        df = fred_get(series, start="2026-04-01", end="2026-04-30")
        if df is None:
            print(f"  FAIL: returned None")
        elif df.empty:
            print(f"  EMPTY: no observations")
        else:
            print(f"  OK: {len(df)} rows")
            print(f"  Latest: {df.iloc[-1]['date'].date()} = {df.iloc[-1]['value']}")
