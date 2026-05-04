# features/absorption.py
# ─────────────────────────────────────────────────────────────────────────────
# Absorption signal via Polygon.io 1-minute bars.
# Requires Polygon Stocks Starter plan ($29/mo).
#
# Absorption = candle with volume > 2x average BUT price range < 0.3%
# Meaning: huge order flow absorbed without price moving = institutional activity
# Precedes reversals — someone is absorbing all supply (or demand).
#
# Usage:
#   from features.absorption import get_absorption_signal
#   sig = get_absorption_signal("AAPL", "2026-04-17")
# ─────────────────────────────────────────────────────────────────────────────

from __future__ import annotations
import os
import requests
import pandas as pd
import numpy as np
from datetime import date, timedelta
from typing import Optional

# Module-level Session for connection reuse (DNS pool conservation).
# Added May 4 2026 to prevent DNS thread exhaustion in Pipeline B/C.
_session = requests.Session()

POLYGON_API_KEY = os.getenv("POLYGON_API_KEY", "pvpkxx6PRbgfvepY33Ao_bi4iNMY1pPz")
BASE_URL = "https://api.polygon.io"

# Absorption thresholds
VOL_MULTIPLIER   = 2.0    # volume must be 2x 20-period avg
RANGE_THRESHOLD  = 0.003  # price range must be < 0.3% of close
ABS_WINDOW       = 30     # rolling window for absorption count
ABS_SIGNAL_MIN   = 3      # min absorption candles in window to trigger signal


def get_minute_bars(
    ticker: str,
    from_date: str,
    to_date: str,
) -> pd.DataFrame:
    """
    Fetch 1-minute OHLCV bars from Polygon.
    Returns DataFrame with columns: open, high, low, close, volume.
    """
    url = (
        f"{BASE_URL}/v2/aggs/ticker/{ticker}/range/1/minute/"
        f"{from_date}/{to_date}"
        f"?adjusted=true&sort=asc&limit=50000"
        f"&apiKey={POLYGON_API_KEY}"
    )
    r = _session.get(url, timeout=15)
    if r.status_code == 403:
        raise PermissionError("Polygon Stocks Starter plan required for minute bars")
    if r.status_code != 200:
        raise Exception(f"Polygon API error: {r.status_code}")

    results = r.json().get("results", [])
    if not results:
        return pd.DataFrame()

    df = pd.DataFrame(results)
    df["t"] = pd.to_datetime(df["t"], unit="ms")
    df.set_index("t", inplace=True)
    df.rename(columns={"o":"open","h":"high","l":"low","c":"close","v":"volume"}, inplace=True)
    return df[["open","high","low","close","volume"]]


def compute_absorption(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute absorption signal on 1-minute bar DataFrame.
    Adds columns: vol_avg_20, range_pct, absorption, abs_count_30m, abs_signal
    """
    df = df.copy()
    df["vol_avg_20"]   = df["volume"].rolling(20).mean()
    df["range_pct"]    = (df["high"] - df["low"]) / df["close"].replace(0, np.nan)

    df["absorption"]   = (
        (df["volume"] > VOL_MULTIPLIER * df["vol_avg_20"]) &
        (df["range_pct"] < RANGE_THRESHOLD)
    )
    df["abs_count_30m"] = df["absorption"].rolling(ABS_WINDOW).sum()
    df["abs_signal"]    = df["abs_count_30m"] >= ABS_SIGNAL_MIN
    return df


def backtest_absorption(df: pd.DataFrame) -> pd.DataFrame:
    """
    Backtest: after each absorption signal, what happens at 5/15/30/60 min?
    Returns DataFrame with forward returns at each horizon.
    """
    adf = compute_absorption(df)
    signal_idx = adf.index[adf["abs_signal"]]
    rows = []

    for ts in signal_idx:
        try:
            pos = adf.index.get_loc(ts)
            entry = adf.iloc[pos]["close"]
            for h in [5, 15, 30, 60]:
                if pos + h < len(adf):
                    future = adf.iloc[pos + h]["close"]
                    ret    = (future - entry) / entry
                    rows.append({"signal_time": ts, "horizon_min": h, "return": ret})
        except Exception:
            continue

    return pd.DataFrame(rows)


def absorption_summary(df: pd.DataFrame) -> dict:
    """
    Compute win rate and mean return at each horizon after absorption signals.
    """
    bt = backtest_absorption(df)
    if bt.empty:
        return {}

    summary = {}
    for h in [5, 15, 30, 60]:
        hdf = bt[bt["horizon_min"] == h]["return"]
        if len(hdf) > 0:
            summary[f"win_rate_{h}m"]  = round(float((hdf > 0).mean()), 3)
            summary[f"mean_ret_{h}m"]  = round(float(hdf.mean()), 4)
            summary[f"n_signals_{h}m"] = len(hdf)
    return summary


def get_absorption_signal(
    ticker: str,
    trade_date: Optional[str] = None,
    lookback_days: int = 5,
) -> dict:
    """
    Get today's absorption signal for a ticker.

    Returns dict with:
        abs_signal      : True if 3+ absorption candles in any 30-min window today
        abs_count       : total absorption candles today
        win_rate_30m    : historical win rate at 30-min horizon (from backtest)
        mean_ret_30m    : historical mean return at 30-min horizon
        error           : error message if failed
    """
    ticker = ticker.upper().strip()
    if trade_date is None:
        trade_date = str(date.today())

    from_date = str(
        date.fromisoformat(trade_date) - timedelta(days=lookback_days)
    )

    result = {
        "ticker":       ticker,
        "trade_date":   trade_date,
        "abs_signal":   False,
        "abs_count":    0,
        "win_rate_30m": None,
        "mean_ret_30m": None,
        "error":        None,
    }

    try:
        df = get_minute_bars(ticker, from_date, trade_date)
        if df.empty:
            result["error"] = "No minute bar data"
            return result

        adf = compute_absorption(df)

        # Today's signal
        today_str = trade_date
        today_data = adf[adf.index.date == date.fromisoformat(today_str)]
        if not today_data.empty:
            result["abs_signal"] = bool(today_data["abs_signal"].any())
            result["abs_count"]  = int(today_data["absorption"].sum())

        # Historical win rates
        summary = absorption_summary(df)
        result["win_rate_30m"] = summary.get("win_rate_30m")
        result["mean_ret_30m"] = summary.get("mean_ret_30m")

    except PermissionError as e:
        result["error"] = str(e)
    except Exception as e:
        result["error"] = str(e)

    return result


def absorption_to_multiplier(abs_signal: bool, win_rate_30m: Optional[float]) -> float:
    """
    Convert absorption signal to a probability multiplier.
    Only boosts when signal is True AND historical win rate > 55%.
    """
    if not abs_signal:
        return 1.00
    if win_rate_30m and win_rate_30m > 0.60:
        return 1.08   # strong historical backing
    if win_rate_30m and win_rate_30m > 0.55:
        return 1.04   # mild backing
    return 1.02       # signal present but unvalidated


if __name__ == "__main__":
    import sys
    ticker = sys.argv[1] if len(sys.argv) > 1 else "AAPL"
    result = get_absorption_signal(ticker)
    print(f"{ticker}: abs_signal={result['abs_signal']} count={result['abs_count']} win_rate_30m={result['win_rate_30m']} error={result['error']}")
