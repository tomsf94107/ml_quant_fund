# data/price_loader.py
# ─────────────────────────────────────────────────────────────────────────────
# Thin yfinance wrapper. Single responsibility: download price data and
# return a clean, normalized DataFrame.
#
# Used by features/builder.py and any UI page that needs raw price data.
# Zero Streamlit imports. Backend only.
# ─────────────────────────────────────────────────────────────────────────────

from __future__ import annotations

from datetime import date, datetime, timedelta
from typing import Optional

import pandas as pd
import yfinance as yf

# ── Default lookback ──────────────────────────────────────────────────────────
DEFAULT_START = "2018-01-01"


def load_prices(
    ticker: str,
    start_date: str | date = DEFAULT_START,
    end_date: str | date | None = None,
) -> pd.DataFrame:
    """
    Download OHLCV for `ticker` and return a clean DataFrame.

    Output columns (all lowercase):
        date, ticker, open, high, low, close, volume

    Rules:
        - MultiIndex columns from yfinance are flattened immediately
        - date column is python date objects (not Timestamp)
        - No NaN rows in close or volume
        - Returns empty DataFrame (not raises) if download fails
    """
    ticker = ticker.upper().strip()
    if end_date is None:
        end_date = datetime.today().strftime("%Y-%m-%d")

    try:
        raw = yf.download(
            ticker,
            start=str(start_date),
            end=str(end_date),
            auto_adjust=True,
            progress=False,
        )
    except Exception as e:
        print(f"  ⚠ yfinance download failed for {ticker}: {e}")
        return pd.DataFrame()

    if raw.empty:
        return pd.DataFrame()

    # Flatten MultiIndex columns (yfinance quirk with single ticker)
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)

    raw = raw.reset_index()

    # Normalize column names
    raw.columns = [c.strip().lower().replace(" ", "_") for c in raw.columns]

    # Handle date column aliases
    for alias in ("datetime", "index", "date"):
        if alias in raw.columns:
            raw = raw.rename(columns={alias: "date"})
            break

    raw["date"]   = pd.to_datetime(raw["date"]).dt.date
    raw["ticker"] = ticker

    # Keep only what we need
    keep = [c for c in ["date", "ticker", "open", "high", "low", "close", "volume"]
            if c in raw.columns]
    raw = raw[keep].dropna(subset=["close", "volume"]).reset_index(drop=True)

    return raw


def load_prices_multi(
    tickers: list[str],
    start_date: str | date = DEFAULT_START,
    end_date: str | date | None = None,
) -> dict[str, pd.DataFrame]:
    """
    Download prices for multiple tickers.
    Returns dict: {ticker: DataFrame}
    """
    return {
        t: load_prices(t, start_date, end_date)
        for t in tickers
    }


def get_latest_close(ticker: str) -> Optional[float]:
    """Return the most recent closing price for `ticker`."""
    df = load_prices(ticker, start_date=(date.today() - timedelta(days=7)))
    if df.empty:
        return None
    return float(df["close"].iloc[-1])
