# features/dark_pool.py
# Dark pool ratio via Unusual Whales API.
# Endpoint: GET /api/darkpool/{ticker}
# Requires: Unusual Whales API Basic ($125/mo)

from __future__ import annotations
import os
import requests
from datetime import date, timedelta
from typing import Optional

UW_API_KEY = os.getenv("UW_API_KEY", "")
BASE_URL   = "https://api.unusualwhales.com"
HEADERS    = {"Authorization": f"Bearer {UW_API_KEY}"}


def get_dark_pool_ratio(
    ticker: str,
    trade_date: Optional[str] = None,
) -> dict:
    """
    Get dark pool ratio for a ticker via Unusual Whales.

    Returns dict with:
        dp_ratio   : 0.0 to 1.0
        dp_volume  : off-exchange volume
        dp_signal  : HIGH | NORMAL | LOW
        error      : error message if failed
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
        url = f"{BASE_URL}/api/darkpool/{ticker}"
        params = {"date": trade_date}
        r = requests.get(url, headers=HEADERS, params=params, timeout=10)

        if r.status_code == 401:
            result["error"] = "Invalid UW API key"
            return result
        if r.status_code == 403:
            result["error"] = "UW API plan does not include dark pool"
            return result
        if r.status_code != 200:
            result["error"] = f"UW API error: {r.status_code}"
            return result

        data = r.json()
        trades = data.get("data", [])

        if not trades:
            result["error"] = "No dark pool trades returned"
            return result

        # All trades returned are already dark pool trades
        # volume field = total market volume for the day
        dp_vol    = sum(float(t.get("size", 0)) for t in trades)
        total_vol = float(trades[0].get("volume", 0)) if trades else 0

        if total_vol == 0:
            result["error"] = "Zero total volume"
            return result

        dp_ratio = dp_vol / total_vol
        result["dp_ratio"]     = round(dp_ratio, 4)
        result["dp_volume"]    = int(dp_vol)
        result["total_volume"] = int(total_vol)

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
    return {
        t: get_dark_pool_ratio(t, trade_date).get("dp_ratio", 0.0)
        for t in tickers
    }


def dark_pool_to_multiplier(dp_ratio: float) -> float:
    if dp_ratio > 0.60:   return 1.05
    if dp_ratio > 0.40:   return 1.02
    if dp_ratio < 0.20:   return 0.98
    return 1.00


if __name__ == "__main__":
    import sys
    ticker = sys.argv[1] if len(sys.argv) > 1 else "AAPL"
    result = get_dark_pool_ratio(ticker)
    print(f"{ticker}: dp_ratio={result['dp_ratio']:.2%} signal={result['dp_signal']} error={result['error']}")
