# features/dark_pool.py
# ─────────────────────────────────────────────────────────────────────────────
# Dark pool ratio via Polygon.io tick data.
# Requires Polygon Stocks Starter plan ($29/mo) — exchange==4 is FINRA/OTC.
# Falls back to 0.0 if Polygon plan doesn't support tick data.
#
# Usage:
#   from features.dark_pool import get_dark_pool_ratio
#   ratio = get_dark_pool_ratio("AAPL", "2026-04-17")
# ─────────────────────────────────────────────────────────────────────────────

from __future__ import annotations
import os
import requests
from datetime import date, timedelta
from typing import Optional

POLYGON_API_KEY = os.getenv("POLYGON_API_KEY", "pvpkxx6PRbgfvepY33Ao_bi4iNMY1pPz")
BASE_URL = "https://api.polygon.io"

# Exchange ID 4 = FINRA/OTC (off-exchange / dark pool)
DARK_POOL_EXCHANGE_ID = 4


def get_dark_pool_ratio(
    ticker: str,
    trade_date: Optional[str] = None,
) -> dict:
    """
    Compute dark pool ratio for a ticker on a given date.

    Dark pool ratio = off-exchange volume / total volume
    High ratio (>0.50) = institutions trading out of sight
    Spike + flat price = likely accumulation
    Spike + falling price = likely distribution

    Returns dict with:
        dp_ratio        : 0.0 to 1.0 (fraction of volume off-exchange)
        dp_volume       : total off-exchange shares traded
        total_volume    : total shares traded
        dp_signal       : "HIGH" | "NORMAL" | "LOW"
        error           : error message if failed
    """
    ticker = ticker.upper().strip()
    if trade_date is None:
        trade_date = str(date.today() - timedelta(days=1))

    result = {
        "ticker":       ticker,
        "trade_date":   trade_date,
        "dp_ratio":     0.0,
        "dp_volume":    0,
        "total_volume": 0,
        "dp_signal":    "NORMAL",
        "error":        None,
    }

    try:
        url = (
            f"{BASE_URL}/v3/trades/{ticker}"
            f"?timestamp.gte={trade_date}T09:30:00Z"
            f"&timestamp.lte={trade_date}T16:00:00Z"
            f"&limit=50000"
            f"&apiKey={POLYGON_API_KEY}"
        )
        r = requests.get(url, timeout=10)

        if r.status_code == 403:
            result["error"] = "Polygon Stocks Starter plan required for tick data"
            return result
        if r.status_code != 200:
            result["error"] = f"Polygon API error: {r.status_code}"
            return result

        trades = r.json().get("results", [])
        if not trades:
            result["error"] = "No trades returned"
            return result

        total_vol = sum(t.get("size", 0) for t in trades)
        dp_vol    = sum(t.get("size", 0) for t in trades
                        if t.get("exchange") == DARK_POOL_EXCHANGE_ID)

        if total_vol == 0:
            result["error"] = "Zero total volume"
            return result

        dp_ratio = dp_vol / total_vol
        result["dp_ratio"]     = round(dp_ratio, 4)
        result["dp_volume"]    = dp_vol
        result["total_volume"] = total_vol

        if dp_ratio > 0.50:
            result["dp_signal"] = "HIGH"
        elif dp_ratio < 0.25:
            result["dp_signal"] = "LOW"
        else:
            result["dp_signal"] = "NORMAL"

    except Exception as e:
        result["error"] = str(e)

    return result


def get_dark_pool_ratio_batch(
    tickers: list[str],
    trade_date: Optional[str] = None,
) -> dict[str, float]:
    """
    Get dark pool ratios for multiple tickers.
    Returns {ticker: dp_ratio} dict.
    """
    return {
        t: get_dark_pool_ratio(t, trade_date).get("dp_ratio", 0.0)
        for t in tickers
    }


def dark_pool_to_multiplier(dp_ratio: float) -> float:
    """
    Convert dark pool ratio to a signal multiplier.
    High DP ratio is ambiguous (could be accumulation or distribution)
    so we use a mild boost only when combined with other signals.

    dp_ratio > 0.60 → 1.05 (mild boost — institutions are active)
    dp_ratio > 0.40 → 1.02
    dp_ratio < 0.20 → 0.98 (low institutional activity)
    else            → 1.00
    """
    if dp_ratio > 0.60:   return 1.05
    if dp_ratio > 0.40:   return 1.02
    if dp_ratio < 0.20:   return 0.98
    return 1.00


if __name__ == "__main__":
    import sys
    ticker = sys.argv[1] if len(sys.argv) > 1 else "AAPL"
    result = get_dark_pool_ratio(ticker)
    print(f"{ticker}: dp_ratio={result['dp_ratio']:.2%} signal={result['dp_signal']} error={result['error']}")
