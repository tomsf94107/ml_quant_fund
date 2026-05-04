"""
features/uw_options.py
─────────────────────────────────────────────────────────────────────────────
Unusual Whales options data client.

Provides 25-delta skew calculation using UW's Greeks endpoint.

Replaces:
  - Massive Greeks (returns empty on Starter tier — confirmed 2026-05-01)
  - yfinance options chain (flaky, can segfault Python via curl_cffi)

Endpoints used (in order):
  /api/stock/{ticker}/stock-state                — for spot price (1 call)
  /api/stock/{ticker}/option-contracts           — to enumerate expiries (1 call)
  /api/stock/{ticker}/greeks?expiry=YYYY-MM-DD   — per-strike Greeks/IV (1 call)

Total: 3 calls per ticker (or 2 if spot price is supplied).

Rate limits (UW Basic Annual):
  120 req/min
  40,000 req/day

Pipeline B impact: 125 tickers x 3 calls = 375 calls = 0.94% of daily budget.

Note: stock-state's "close" returns the regular-session close even during
pre/post-market hours. This is fine for end-of-day skew calculation but
caller should be aware if doing intraday work.
"""
from __future__ import annotations

import logging
import os
import re
import time
from datetime import datetime, timedelta, date
from typing import Optional

import requests

log = logging.getLogger(__name__)

UW_BASE_URL = "https://api.unusualwhales.com"
UW_API_KEY = os.getenv("UW_API_KEY", "")
UW_CLIENT_ID = "100001"

DEFAULT_TIMEOUT = 15
MAX_RETRIES = 3
RETRY_BACKOFF = 1.0

# Skew thresholds (match massive_options.py for consistency)
BEARISH_THRESHOLD = 0.03
BULLISH_THRESHOLD = -0.02

# Rate-limit warning thresholds (Level 2 monitoring)
PER_MIN_WARN_THRESHOLD = 20      # warn if remaining < 20/120
DAILY_WARN_THRESHOLD = 5000      # warn if remaining < 5000/40000
DAILY_LIMIT = 40000

# OCC option symbol format: TICKER + YYMMDD + (C/P) + 8-digit strike
# Tickers are variable length (1-6 chars).
_OCC_RE = re.compile(r"^([A-Z]+)(\d{6})([CP])(\d{8})$")


def _uw_headers() -> dict:
    return {
        "Authorization": f"Bearer {UW_API_KEY}",
        "UW-CLIENT-API-ID": UW_CLIENT_ID,
        "Accept": "application/json",
    }


def _check_rate_limits(headers: dict, path: str) -> None:
    """Log warnings if approaching per-minute or daily UW rate limits."""
    try:
        rpm = int(headers.get("x-uw-req-per-minute-remaining", 999))
        if rpm < PER_MIN_WARN_THRESHOLD:
            log.warning(
                f"UW per-minute budget low: {rpm}/120 remaining (path={path})"
            )
    except (ValueError, TypeError):
        pass
    try:
        used = int(headers.get("x-uw-daily-req-count", 0))
        remaining = DAILY_LIMIT - used
        if remaining < DAILY_WARN_THRESHOLD:
            log.warning(
                f"UW daily budget low: {remaining}/{DAILY_LIMIT} remaining (path={path})"
            )
    except (ValueError, TypeError):
        pass


def _uw_get(path: str, params: Optional[dict] = None) -> Optional[dict]:
    """Generic UW API GET, routed through centralized uw_client.

    FIXED May 4 2026: previously used raw requests.get() per call — no
    Session reuse, fresh DNS lookup every time. After ~80 tickers in
    Pipeline B Stage 3, curl_cffi DNS thread pool exhausted, crashing
    daily_runner mid-run (e.g. May 4 crash at USAR ticker 86). Now
    delegates to uw_client.uw_get which has a module-level
    requests.Session() reused across all UW calls.

    options endpoints are HISTORICAL data so we pass
    allow_outside_market=True to bypass the market-hours gate.
    """
    from features.uw_client import uw_get
    return uw_get(path, params=params, allow_outside_market=True,
                  max_retries=MAX_RETRIES, timeout=DEFAULT_TIMEOUT)


def _parse_expiry_from_symbol(symbol: str) -> Optional[str]:
    """Extract YYYY-MM-DD expiry from OCC option symbol (e.g. AAPL260501C00275000)."""
    if not symbol:
        return None
    m = _OCC_RE.match(symbol)
    if not m:
        return None
    yymmdd = m.group(2)
    return f"20{yymmdd[:2]}-{yymmdd[2:4]}-{yymmdd[4:6]}"


def get_spot_price_uw(ticker: str) -> Optional[float]:
    """Return current spot price from UW stock-state endpoint."""
    data = _uw_get(f"/api/stock/{ticker}/stock-state")
    if not data:
        return None
    state = data.get("data", {})
    close = state.get("close")
    try:
        return float(close) if close is not None else None
    except (TypeError, ValueError):
        return None


def get_target_expiry_uw(ticker: str, target_days: int = 30) -> Optional[str]:
    """
    Find the option expiry closest to target_days out.
    Returns expiry as 'YYYY-MM-DD' string, or None if none available.
    """
    data = _uw_get(f"/api/stock/{ticker}/option-contracts")
    if not data:
        return None
    contracts = data.get("data", []) if isinstance(data, dict) else data
    if not contracts:
        return None

    today = date.today()
    target_date = today + timedelta(days=target_days)

    expiries = set()
    for c in contracts:
        if not isinstance(c, dict):
            continue
        sym = c.get("option_symbol")
        exp = _parse_expiry_from_symbol(sym)
        if exp:
            expiries.add(exp)

    if not expiries:
        return None

    # Find closest to target_date that's in the future
    valid = []
    for e in expiries:
        try:
            d = datetime.strptime(e, "%Y-%m-%d").date()
            if d > today:
                valid.append((abs((d - target_date).days), e))
        except ValueError:
            continue

    if not valid:
        return None
    valid.sort()
    return valid[0][1]


def get_25delta_skew_uw(ticker: str,
                        current_price: Optional[float] = None,
                        target_days: int = 30) -> dict:
    """
    Compute 25-delta skew using UW's Greeks endpoint.

    25-delta skew = IV(25-delta put) - IV(25-delta call)
      Positive  -> puts more expensive -> bearish
      Negative  -> calls more expensive -> bullish

    Args:
        ticker:        equity symbol (e.g. 'AAPL')
        current_price: optional spot price; fetched from UW if None
        target_days:   target days-to-expiry for skew (default 30)

    Returns:
        dict matching get_25delta_skew_massive() shape:
        {
            "ticker", "skew_25d", "iv_25d_put", "iv_25d_call",
            "iv_rank", "skew_signal", "source", "error"
        }
    """
    result = {
        "ticker":       ticker,
        "skew_25d":     None,
        "iv_25d_put":   None,
        "iv_25d_call":  None,
        "iv_rank":      None,
        "skew_signal":  "NEUTRAL",
        "source":       "unusual_whales",
        "error":        None,
    }

    if not UW_API_KEY:
        result["error"] = "UW_API_KEY not set"
        return result

    if current_price is None:
        current_price = get_spot_price_uw(ticker)
        if current_price is None:
            result["error"] = "Could not get spot price from UW"
            return result

    expiry = get_target_expiry_uw(ticker, target_days=target_days)
    if expiry is None:
        result["error"] = "No suitable expiry found"
        return result

    data = _uw_get(f"/api/stock/{ticker}/greeks", params={"expiry": expiry})
    if not data:
        result["error"] = f"UW greeks fetch failed for expiry {expiry}"
        return result

    rows = data.get("data", []) if isinstance(data, dict) else data
    if not rows:
        result["error"] = f"No greeks data for expiry {expiry}"
        return result

    # Find strikes closest to call_delta=+0.25 and put_delta=-0.25
    best_call = None  # (abs_delta_diff, call_volatility, strike)
    best_put = None
    for r in rows:
        try:
            cd = float(r.get("call_delta", "nan"))
            pd_ = float(r.get("put_delta", "nan"))
            cv = float(r.get("call_volatility", "nan"))
            pv = float(r.get("put_volatility", "nan"))
            strike = float(r.get("strike", "nan"))
        except (TypeError, ValueError):
            continue
        if any(x != x for x in (cd, pd_, cv, pv, strike)):
            continue

        cdiff = abs(cd - 0.25)
        if best_call is None or cdiff < best_call[0]:
            best_call = (cdiff, cv, strike)

        pdiff = abs(pd_ - (-0.25))
        if best_put is None or pdiff < best_put[0]:
            best_put = (pdiff, pv, strike)

    if best_call is None or best_put is None:
        result["error"] = "Could not find 25-delta strikes"
        return result

    if best_call[0] > 0.20:
        result["error"] = f"No call near 25-delta (off by {best_call[0]:.2f})"
        return result
    if best_put[0] > 0.20:
        result["error"] = f"No put near 25-delta (off by {best_put[0]:.2f})"
        return result

    iv_call = best_call[1]
    iv_put = best_put[1]
    skew = iv_put - iv_call

    result["iv_25d_call"] = round(iv_call, 4)
    result["iv_25d_put"] = round(iv_put, 4)
    result["skew_25d"] = round(skew, 4)
    result["expiry"] = expiry
    result["call_strike"] = round(best_call[2], 2)
    result["put_strike"] = round(best_put[2], 2)

    if skew > BEARISH_THRESHOLD:
        result["skew_signal"] = "BEARISH"
    elif skew < BULLISH_THRESHOLD:
        result["skew_signal"] = "BULLISH"
    else:
        result["skew_signal"] = "NEUTRAL"

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Quick CLI test
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    import json

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s %(message)s")

    tickers = sys.argv[1:] if len(sys.argv) > 1 else ["AAPL", "NVDA", "SPY"]
    for t in tickers:
        print(f"\n=== {t} ===")
        r = get_25delta_skew_uw(t)
        print(json.dumps(r, indent=2))


def get_pc_ratio_uw(ticker: str) -> dict:
    """
    Get put/call volume ratio from UW options-volume endpoint.

    Returns dict with:
        pc_ratio:    put_volume / call_volume (None on failure)
        call_volume: total call volume today
        put_volume:  total put volume today
        source:      'unusual_whales'
        error:       None or string
    """
    result = {
        "ticker":      ticker,
        "pc_ratio":    None,
        "call_volume": None,
        "put_volume":  None,
        "source":      "unusual_whales",
        "error":       None,
    }
    if not UW_API_KEY:
        result["error"] = "UW_API_KEY not set"
        return result

    data = _uw_get(f"/api/stock/{ticker}/options-volume")
    if not data:
        result["error"] = "UW options-volume fetch failed"
        return result

    rows = data.get("data", []) if isinstance(data, dict) else data
    if not rows or not isinstance(rows, list):
        result["error"] = "No options-volume data"
        return result

    row = rows[0]
    try:
        call_vol = float(row.get("call_volume", 0))
        put_vol = float(row.get("put_volume", 0))
    except (TypeError, ValueError):
        result["error"] = "Could not parse volumes"
        return result

    result["call_volume"] = int(call_vol)
    result["put_volume"] = int(put_vol)
    if call_vol > 0:
        result["pc_ratio"] = round(put_vol / call_vol, 4)
    else:
        result["error"] = "Zero call volume"

    return result
