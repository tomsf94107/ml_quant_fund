# features/builder.py
# ─────────────────────────────────────────────────────────────────────────────
# THE canonical feature pipeline. One function. One output schema. No compat
# wrappers anywhere else in the codebase.
#
# Output schema (all lowercase, snake_case):
#   date, ticker, close, volume,
#   return_1d, return_3d, return_5d,
#   ma_5, ma_10, ma_20,
#   volatility_5d, volatility_10d,
#   rsi_14, macd, macd_signal,
#   bb_upper, bb_lower, bb_width,
#   volume_zscore, volume_spike,
#   vwap, obv, atr,
#   spy_ret, xlk_ret,
#   sentiment_score,          ← 0.0 placeholder until sentiment pipeline runs
#   insider_net_shares, insider_7d, insider_21d,
#   congress_net_shares,
#   risk_today, risk_next_1d, risk_next_3d, risk_prev_1d,
#   is_pandemic
#
# Rules:
#   - All optional signals (sentiment, insider, congress, risk) default to 0.0
#     silently. They NEVER crash the pipeline.
#   - yfinance MultiIndex columns are flattened immediately on download.
#   - Output always has a clean RangeIndex (not DatetimeIndex) so it plays
#     nicely with both sklearn and Streamlit dataframes.
#   - This file has zero Streamlit imports. Zero UI code. Backend only.
# ─────────────────────────────────────────────────────────────────────────────

from __future__ import annotations

import sqlite3
import os
from datetime import datetime, date
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import yfinance as yf

# ── Constants ─────────────────────────────────────────────────────────────────
SPY_TICKER      = "SPY"
SECTOR_ETF      = "XLK"
VOL_LOOKBACK    = 20        # sessions for volume z-score
INSIDER_DB      = os.getenv("INSIDER_DB_PATH", "insider_trades.db")
CONGRESS_DB     = os.getenv("CONGRESS_DB_PATH", "congress_trades.db")

PANDEMIC_START  = pd.Timestamp("2020-03-01")
PANDEMIC_END    = pd.Timestamp("2023-12-31")

# ── Output schema (enforced at the end) ──────────────────────────────────────
OUTPUT_COLUMNS = [
    "date", "ticker",
    "close", "volume",
    "return_1d", "return_3d", "return_5d",
    "ma_5", "ma_10", "ma_20",
    "volatility_5d", "volatility_10d",
    "rsi_14", "macd", "macd_signal",
    "bb_upper", "bb_lower", "bb_width",
    "volume_zscore", "volume_spike",
    "vwap", "obv", "atr",
    "spy_ret", "xlk_ret",
    "sentiment_score",
    "insider_net_shares", "insider_7d", "insider_21d",
    "congress_net_shares",
    "risk_today", "risk_next_1d", "risk_next_3d", "risk_prev_1d",
    "is_pandemic",
]


# ══════════════════════════════════════════════════════════════════════════════
#  PRIVATE HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _download(ticker: str, start: str, end: str) -> pd.DataFrame:
    """Download OHLCV from yfinance and flatten any MultiIndex columns."""
    df = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
    if df.empty:
        raise ValueError(f"yfinance returned no data for {ticker} ({start} → {end})")

    # Flatten MultiIndex (yfinance sometimes returns ('Close', 'AAPL') style)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df.reset_index()

    # Normalise column names to lowercase
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    # Rename 'date' if yfinance called it something else
    for alias in ("datetime", "index"):
        if alias in df.columns and "date" not in df.columns:
            df = df.rename(columns={alias: "date"})

    df["date"] = pd.to_datetime(df["date"]).dt.date   # keep as date, not timestamp
    df["ticker"] = ticker.upper()
    return df


def _market_return(etf: str, start: str, end: str,
                   index: pd.Index) -> pd.Series:
    """Fetch ETF daily return, reindexed to match the main df's date index."""
    try:
        tmp = yf.download(etf, start=start, end=end,
                          auto_adjust=True, progress=False)
        if isinstance(tmp.columns, pd.MultiIndex):
            tmp.columns = tmp.columns.get_level_values(0)
        tmp = tmp.reset_index()
        tmp.columns = [c.strip().lower() for c in tmp.columns]
        tmp["date"] = pd.to_datetime(tmp["date"]).dt.date
        ret = tmp.set_index("date")["close"].pct_change()
        return ret.reindex(index).rename(f"{etf.lower()}_ret")
    except Exception:
        return pd.Series(np.nan, index=index, name=f"{etf.lower()}_ret")


def _rsi(close: pd.Series, window: int = 14) -> pd.Series:
    """Wilder RSI — same formula used in TradingView / FactSet."""
    delta = close.diff()
    gain  = delta.clip(lower=0)
    loss  = -delta.clip(upper=0)
    avg_g = gain.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
    avg_l = loss.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
    rs    = avg_g / avg_l.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def _atr(high: pd.Series, low: pd.Series,
         close: pd.Series, window: int = 14) -> pd.Series:
    """Average True Range (Wilder smoothing)."""
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low  - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()


def _obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    """On-Balance Volume."""
    direction = np.sign(close.diff()).fillna(0)
    return (direction * volume).cumsum()


def _vwap(close: pd.Series, volume: pd.Series) -> pd.Series:
    """Cumulative VWAP (resets each day — here it's daily running VWAP)."""
    cum_vol = volume.cumsum()
    cum_pv  = (close * volume).cumsum()
    return cum_pv / cum_vol.replace(0, np.nan)


# ── Optional signal loaders (all return pd.Series indexed by date) ───────────

def _load_insider(ticker: str, dates: pd.Index) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Load insider net_shares, 7d rolling, 21d rolling from SQLite."""
    zeros = pd.Series(0.0, index=dates)
    try:
        conn = sqlite3.connect(INSIDER_DB)
        df = pd.read_sql(
            "SELECT date, net_shares FROM insider_flows WHERE ticker = ? ORDER BY date",
            conn, params=(ticker.upper(),), parse_dates=["date"]
        )
        conn.close()
        if df.empty:
            return zeros.copy(), zeros.copy(), zeros.copy()
        df = df.set_index(df["date"].dt.date)["net_shares"]
        net    = df.reindex(dates).fillna(0.0)
        roll7  = net.rolling(7,  min_periods=1).sum()
        roll21 = net.rolling(21, min_periods=1).sum()
        return net.rename("insider_net_shares"), roll7.rename("insider_7d"), roll21.rename("insider_21d")
    except Exception:
        return zeros.copy(), zeros.copy(), zeros.copy()


def _load_congress(ticker: str, dates: pd.Index) -> pd.Series:
    """Load congressional net shares from SQLite."""
    zeros = pd.Series(0.0, index=dates, name="congress_net_shares")
    try:
        conn = sqlite3.connect(CONGRESS_DB)
        df = pd.read_sql(
            "SELECT ds as date, congress_net_shares FROM congress_flows WHERE ticker = ? ORDER BY date",
            conn, params=(ticker.upper(),), parse_dates=["date"]
        )
        conn.close()
        if df.empty:
            return zeros
        df = df.set_index(df["date"].dt.date)["congress_net_shares"]
        return df.reindex(dates).fillna(0.0).rename("congress_net_shares")
    except Exception:
        return zeros


def _load_risk_flags(dates: pd.Index) -> pd.DataFrame:
    """Load pre-computed risk flags. Falls back to zeros if unavailable."""
    cols = ["risk_today", "risk_next_1d", "risk_next_3d", "risk_prev_1d"]
    zero_df = pd.DataFrame(0.0, index=dates, columns=cols)
    try:
        from signals.risk_gate import build_risk_features
        rf = build_risk_features(dates[0], dates[-1])
        rf = rf.set_index("date")[cols]
        return rf.reindex(dates).fillna(0.0)
    except Exception:
        return zero_df


# ══════════════════════════════════════════════════════════════════════════════
#  PUBLIC API  ←  THE ONLY FUNCTION YOU SHOULD IMPORT FROM THIS MODULE
# ══════════════════════════════════════════════════════════════════════════════

def build_feature_dataframe(
    ticker: str,
    start_date: str | date = "2018-01-01",
    end_date:   str | date | None = None,
    include_sentiment: bool = True,    # reads from SQLite cache, 0.0 if no data
) -> pd.DataFrame:
    """
    Build the canonical feature DataFrame for `ticker`.

    Parameters
    ----------
    ticker         : e.g. "AAPL"
    start_date     : ISO string or date object  (default: 2018-01-01)
    end_date       : ISO string or date object  (default: today)
    include_sentiment : if True, calls sentiment pipeline (slow, costs API calls)

    Returns
    -------
    pd.DataFrame with exactly OUTPUT_COLUMNS columns, clean RangeIndex,
    NaNs dropped from warm-up period only.
    """
    if end_date is None:
        end_date = datetime.today().strftime("%Y-%m-%d")

    start_str = str(start_date)
    end_str   = str(end_date)
    ticker    = ticker.upper().strip()

    # ── 1. Price data ─────────────────────────────────────────────────────────
    df = _download(ticker, start_str, end_str)

    required = {"date", "close", "volume"}
    missing  = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns after download: {missing}")

    # ── 2. Date index for reindexing optional signals ─────────────────────────
    date_index = pd.Index(df["date"])

    # ── 3. Price-based features ───────────────────────────────────────────────
    c = df["close"]

    df["return_1d"] = c.pct_change(1)
    df["return_3d"] = c.pct_change(3)
    df["return_5d"] = c.pct_change(5)

    df["ma_5"]  = c.rolling(5).mean()
    df["ma_10"] = c.rolling(10).mean()
    df["ma_20"] = c.rolling(20).mean()

    df["volatility_5d"]  = df["return_1d"].rolling(5).std()
    df["volatility_10d"] = df["return_1d"].rolling(10).std()

    # ── 4. Oscillators ────────────────────────────────────────────────────────
    df["rsi_14"] = _rsi(c)

    ema12 = c.ewm(span=12, adjust=False).mean()
    ema26 = c.ewm(span=26, adjust=False).mean()
    df["macd"]        = ema12 - ema26
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()

    ma20  = c.rolling(20).mean()
    std20 = c.rolling(20).std()
    df["bb_upper"] = ma20 + 2 * std20
    df["bb_lower"] = ma20 - 2 * std20
    df["bb_width"] = (4 * std20 / ma20.replace(0, np.nan))

    # ── 5. Volume features ────────────────────────────────────────────────────
    v = df["volume"].replace(0, np.nan)
    vol_mean = v.rolling(VOL_LOOKBACK).mean()
    vol_std  = v.rolling(VOL_LOOKBACK).std()

    df["volume_zscore"] = (v - vol_mean) / vol_std.replace(0, np.nan)
    df["volume_spike"]  = (df["volume_zscore"] > 2).astype(int)
    df["vwap"]          = _vwap(c, v)
    df["obv"]           = _obv(c, v)

    # ATR needs high/low — check they exist
    if {"high", "low"}.issubset(df.columns):
        df["atr"] = _atr(df["high"], df["low"], c)
    else:
        df["atr"] = np.nan

    # ── 6. Market / sector context ────────────────────────────────────────────
    spy = _market_return(SPY_TICKER, start_str, end_str, date_index)
    xlk = _market_return(SECTOR_ETF, start_str, end_str, date_index)
    df["spy_ret"] = spy.values
    df["xlk_ret"] = xlk.values

    # ── 7. Sentiment — reads from SQLite cache (run etl_sentiment.py daily) ────
    # Historical rows default to 0.0 (no past headlines available).
    # Live predictions use today's cached FinBERT score.
    # Set include_sentiment=False to skip entirely (faster, for batch training).
    if include_sentiment:
        try:
            from data.etl_sentiment import load_sentiment_scores
            sent_df = load_sentiment_scores(ticker, start_date=start_str, end_date=end_str)
            if not sent_df.empty:
                sent_df["date"] = pd.to_datetime(sent_df["date"]).dt.date
                sent_map = sent_df.set_index("date")["score"].to_dict()
                df["sentiment_score"] = df["date"].map(sent_map).fillna(0.0)
            else:
                df["sentiment_score"] = 0.0
        except Exception:
            df["sentiment_score"] = 0.0
    else:
        df["sentiment_score"] = 0.0

    # ── 8. Insider flows ──────────────────────────────────────────────────────
    ins_net, ins_7d, ins_21d = _load_insider(ticker, date_index)
    df["insider_net_shares"] = ins_net.values
    df["insider_7d"]         = ins_7d.values
    df["insider_21d"]        = ins_21d.values

    # ── 9. Congressional trading ──────────────────────────────────────────────
    congress = _load_congress(ticker, date_index)
    df["congress_net_shares"] = congress.values

    # ── 10. Risk flags ────────────────────────────────────────────────────────
    risk = _load_risk_flags(date_index)
    for col in ["risk_today", "risk_next_1d", "risk_next_3d", "risk_prev_1d"]:
        df[col] = risk[col].values if col in risk.columns else 0.0

    # ── 11. Pandemic regime ───────────────────────────────────────────────────
    dates_ts = pd.to_datetime(df["date"])
    df["is_pandemic"] = (
        (dates_ts >= PANDEMIC_START) & (dates_ts <= PANDEMIC_END)
    ).astype(int)

    # ── 12. Enforce output schema ─────────────────────────────────────────────
    # Add any missing columns as 0.0
    for col in OUTPUT_COLUMNS:
        if col not in df.columns:
            df[col] = 0.0

    df = df[OUTPUT_COLUMNS]

    # Drop warm-up NaNs (first ~26 rows from EMA/BB calculations)
    required_non_null = ["close", "ma_20", "rsi_14", "macd", "bb_upper"]
    df = df.dropna(subset=required_non_null).reset_index(drop=True)

    return df


def add_forecast_targets(
    df: pd.DataFrame,
    horizons: tuple[int, ...] = (1, 3, 5),
) -> pd.DataFrame:
    """
    Add binary classification targets to a feature DataFrame.

    For each horizon h, adds:
      target_{h}d = 1 if close[t+h] > close[t], else 0

    These are TRUE forward returns — no lookahead if you train only on rows
    where target is not NaN (i.e. exclude the last h rows).
    """
    df = df.copy()
    for h in horizons:
        df[f"target_{h}d"] = (df["close"].shift(-h) > df["close"]).astype(float)
        df.loc[df.index[-h:], f"target_{h}d"] = np.nan   # last h rows have no target
    return df
