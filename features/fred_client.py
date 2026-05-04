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


def fred_get_as_series(series_id: str,
                       start: Optional[str] = None,
                       end: Optional[str] = None) -> Optional[pd.Series]:
    """
    Convenience wrapper: returns indexed pd.Series (index=date, values=value).
    Returns None on failure, empty Series if no data.
    """
    df = fred_get(series_id, start=start, end=end)
    if df is None:
        return None
    if df.empty:
        return pd.Series(dtype=float)

    s = df.set_index("date")["value"]
    s.index = pd.to_datetime(s.index).normalize()
    return s


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
