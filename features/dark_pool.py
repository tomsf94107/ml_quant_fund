# features/dark_pool.py
# Dark pool ratio via Unusual Whales API.
# All UW access routed through features.uw_client (market-hours gated,
# daily-budget tracked, 429/5xx retry). On gate-closed / rate-limited /
# failed, falls back to dark_pool_history SQLite table.

from __future__ import annotations
from datetime import date, timedelta
from typing import Optional

from features.uw_client import uw_get


def _get_dark_pool_from_db(ticker: str, trade_date: str, result: dict) -> dict:
    """Fallback to DB when UW API is unavailable (market closed, rate limited, or error)."""
    try:
        import sqlite3
        from pathlib import Path
        db = Path(__file__).parent.parent / "accuracy.db"
        cutoff = str((date.today() - timedelta(days=3)).isoformat())
        with sqlite3.connect(db, timeout=30) as conn:
            row = conn.execute("""
                SELECT dp_ratio, dp_volume, total_volume, dp_signal
                FROM dark_pool_history
                WHERE ticker=? AND date>=?
                ORDER BY date DESC LIMIT 1
            """, (ticker, cutoff)).fetchone()
        if row:
            result["dp_ratio"]     = row[0]
            result["dp_volume"]    = row[1]
            result["total_volume"] = row[2]
            result["dp_signal"]    = row[3]
            result["error"]        = None
            result["source"]       = "DB_FALLBACK"
        else:
            result["error"] = "UW skipped + no DB data"
    except Exception as e:
        result["error"] = f"DB fallback failed: {e}"
    return result


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
        # uw_client enforces market-hours gate, daily budget, and 429/5xx retry.
        # Returns None when gate is closed, budget exhausted, or all retries fail.
        data = uw_get(
            f"/api/darkpool/{ticker}",
        )
        if data is None:
            return _get_dark_pool_from_db(ticker, trade_date, result)

        trades = data.get("data", [])
        if not trades:
            result["error"] = "No dark pool trades returned"
            return result

        # All trades returned are already dark pool trades.
        # volume field = total market volume for the day.
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
