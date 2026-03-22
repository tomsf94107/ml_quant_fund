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
import sqlite3
import yfinance as yf

# ── Constants ─────────────────────────────────────────────────────────────────
SPY_TICKER      = "SPY"
SECTOR_ETF      = "XLK"
VIX_TICKER      = "^VIX"
VOL_LOOKBACK    = 20        # sessions for volume z-score

# Sector ETF map — stock → best matching sector ETF
SECTOR_ETF_MAP = {
    "AAPL": "XLK", "MSFT": "XLK", "NVDA": "XLK", "AMD": "XLK",
    "GOOG": "XLK", "META": "XLK", "CRM": "XLK", "CRWD": "XLK",
    "DDOG": "XLK", "SNOW": "XLK", "DUOL": "XLK",
    "TSLA": "XLY", "AMZN": "XLY", "SHOP": "XLY",
    "AAPL": "XLK",
    "JNJ": "XLV", "PFE": "XLV", "UNH": "XLV", "MRNA": "XLV",
    "NVO": "XLV", "BSX": "XLV", "CNC": "XLV",
    "AXP": "XLF", "PYPL": "XLF",
    "MP": "XLB", "SLV": "XLB",
    "NFLX": "XLC", "ZM": "XLC",
    "PLTR": "XLK", "SMCI": "XLK", "TSM": "XLK",
    "RZLV": "XLK", "CRCL": "XLK",
}
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
    # Earnings surprise
    "eps_surprise", "rev_surprise",
    "days_to_earnings",
    "post_earnings_1d", "post_earnings_3d", "post_earnings_5d",
    # ── NEW v2 features ──────────────────────────────────────────────────────
    "return_20d", "return_60d",          # medium-term momentum
    "ma_50",                             # 50-day MA
    "ma5_above_ma20", "ma20_above_ma50", # MA crossover signals
    "high_52w_ratio", "low_52w_ratio",   # distance from 52w extremes
    "bb_pct",                            # BB position (0=at lower, 1=at upper)
    "rsi_above_70", "rsi_below_30",      # RSI extreme flags
    "vwap_dev_eod", "vol_surge_eod", "intraday_momentum",  # intraday-derived
    "obv_trend",                         # OBV vs 10d OBV mean
    "vix_close", "vix_ret",              # market fear gauge
    "oil_ret", "oil_spy_corr",           # crude oil price signal
    "dxy_ret", "yield_10y", "fear_greed", "beta_60d",
    "short_ratio", "short_pct_float", "vix_term_structure", "monday_sentiment",
    "sector_rel_ret",                    # stock return - sector ETF return
    "day_of_week", "is_month_end",       # calendar effects
    # ── NEW v3 features ──────────────────────────────────────────────────────
    "premarket_gap",                     # open vs prev close
    "es_overnight",                      # S&P500 futures overnight move
    "iv_skew_snap", "pc_ratio_snap",     # options IV skew + put/call ratio
    "analyst_upside", "analyst_buy_pct", "analyst_mult",  # analyst revisions
    "finbert_sentiment", "finbert_mult", # FinBERT NLP sentiment
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
                   index: pd.Index, return_close: bool = False):
    """Fetch ETF daily return, reindexed to match the main df's date index.
    If return_close=True, returns a DataFrame with both 'close' and 'ret' columns.
    """
    try:
        tmp = yf.download(etf, start=start, end=end,
                          auto_adjust=True, progress=False)
        if isinstance(tmp.columns, pd.MultiIndex):
            tmp.columns = tmp.columns.get_level_values(0)
        tmp = tmp.reset_index()
        tmp.columns = [c.strip().lower() for c in tmp.columns]
        tmp["date"] = pd.to_datetime(tmp["date"]).dt.date
        tmp = tmp.set_index("date")
        close_s = tmp["close"].reindex(index).ffill()
        ret_s   = close_s.pct_change().rename(f"{etf.lower()}_ret")
        if return_close:
            return pd.DataFrame({"close": close_s.values, "ret": ret_s.values},
                                 index=index)
        return ret_s
    except Exception:
        if return_close:
            return pd.DataFrame({"close": np.full(len(index), 20.0),
                                  "ret":   np.zeros(len(index))}, index=index)
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
    training_mode: bool = False,       # if True, skip slow live API calls
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
    o = df["open"] if "open" in df.columns else c  # open price

    # ── Pre-market gap ────────────────────────────────────────────────────────
    # How much did stock gap up/down from yesterday's close to today's open
    df["premarket_gap"] = (o - c.shift(1)) / c.shift(1)

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

    # ── 6b. Macro features — DXY, 10Y yield, Fear & Greed, Beta, Short interest ──
    # DXY (US Dollar index)
    try:
        _dxy = _market_return("DX-Y.NYB", start_str, end_str, date_index)
        df["dxy_ret"] = _dxy.fillna(0.0).values
    except Exception:
        df["dxy_ret"] = 0.0

    # VIX term structure — VIX/VIX3M ratio (Hull Chapter 20)
    # ratio > 1 = inverted = acute panic (short-term fear > long-term) = mean reverting, less dangerous
    # ratio < 1 = normal = sustained fear = more dangerous, suppresses BUY signals harder
    try:
        _vix3m = yf.download("^VIX3M", start=start_str, end=end_str,
                              auto_adjust=True, progress=False)
        _vix_raw = yf.download("^VIX", start=start_str, end=end_str,
                               auto_adjust=True, progress=False)
        if not _vix3m.empty and not _vix_raw.empty:
            if isinstance(_vix3m.columns, pd.MultiIndex):
                _vix3m.columns = _vix3m.columns.get_level_values(0)
            if isinstance(_vix_raw.columns, pd.MultiIndex):
                _vix_raw.columns = _vix_raw.columns.get_level_values(0)
            _vix3m.index = pd.to_datetime(_vix3m.index).normalize()
            _vix_raw.index = pd.to_datetime(_vix_raw.index).normalize()
            _vix3m_s = _vix3m["Close"].squeeze()
            _vix_s   = _vix_raw["Close"].squeeze()
            _ratio   = (_vix_s / _vix3m_s).dropna()
            _ratio_map = {d.date(): v for d, v in _ratio.items()}
            df["vix_term_structure"] = df["date"].map(_ratio_map).ffill().fillna(1.0)
        else:
            df["vix_term_structure"] = 1.0
    except Exception:
        df["vix_term_structure"] = 1.0



    # 10Y Treasury yield
    try:
        tnx_raw = yf.download("^TNX", start=start_str, end=end_str,
                               auto_adjust=True, progress=False)
        if not tnx_raw.empty:
            if isinstance(tnx_raw.columns, pd.MultiIndex):
                tnx_raw.columns = tnx_raw.columns.get_level_values(0)
            tnx_raw.index = pd.to_datetime(tnx_raw.index).normalize()
            tnx_series = tnx_raw["Close"].squeeze() / 100.0
            tnx_map = {d.date(): v for d, v in tnx_series.items()}
            df["yield_10y"] = df["date"].map(tnx_map).ffill().fillna(0.04)
        else:
            df["yield_10y"] = 0.04
    except Exception:
        df["yield_10y"] = 0.04

    # Fear & Greed Index (alternative.me — updated daily)
    try:
        import requests as _req
        _fg = _req.get("https://api.alternative.me/fng/?limit=1",
                       headers={"User-Agent": "MLQuantFund/1.0"}, timeout=5)
        if _fg.status_code == 200:
            _fg_val = float(_fg.json()["data"][0]["value"]) / 100.0
        else:
            _fg_val = 0.5
        df["fear_greed"] = _fg_val
    except Exception:
        df["fear_greed"] = 0.5

    # Monday sentiment score (Anthropic API — scored Sunday night)
    try:
        conn_sent = sqlite3.connect("data/sentiment.db")
        sent_row = conn_sent.execute("""
            SELECT sentiment_score, confidence FROM monday_sentiment
            WHERE ticker=? ORDER BY score_date DESC LIMIT 1
        """, (ticker,)).fetchone()
        conn_sent.close()
        if sent_row:
            # Decay sentiment signal over the week — full strength Monday, zero by Friday
            from utils.timezone import today_et
            dow = today_et().weekday()  # 0=Mon, 1=Tue, 2=Wed, 3=Thu, 4=Fri
            decay = max(0.0, 1.0 - (dow * 0.25))  # Mon=1.0, Tue=0.75, Wed=0.5, Thu=0.25, Fri=0.0
            df["monday_sentiment"] = sent_row[0] * sent_row[1] * decay
        else:
            df["monday_sentiment"] = 0.0
    except Exception:
        df["monday_sentiment"] = 0.0

    # 60-day rolling beta vs SPY
    try:
        _spy_ret = pd.Series(spy.values, index=df.index)
        _stk_ret = c.pct_change()
        _cov = _stk_ret.rolling(60).cov(_spy_ret)
        _var = _spy_ret.rolling(60).var()
        df["beta_60d"] = (_cov / _var.replace(0, np.nan)).fillna(1.0)
    except Exception:
        df["beta_60d"] = 1.0

    # Short interest ratio (from yfinance — updates bi-weekly)
    try:
        if not training_mode:
            _info = yf.Ticker(ticker).info
            df["short_ratio"]   = float(_info.get("shortRatio") or 0.0)
            df["short_pct_float"] = float(_info.get("shortPercentOfFloat") or 0.0)
        else:
            df["short_ratio"]   = 0.0
            df["short_pct_float"] = 0.0
    except Exception:
        df["short_ratio"]   = 0.0
        df["short_pct_float"] = 0.0

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

    # ── 8. Earnings surprise features ────────────────────────────────────────
    try:
        from data.etl_earnings import load_earnings_features
        earn = load_earnings_features(ticker, date_index)
        for col in ["eps_surprise", "rev_surprise", "days_to_earnings",
                    "post_earnings_1d", "post_earnings_3d", "post_earnings_5d"]:
            df[col] = earn[col].values if col in earn.columns else 0.0
    except Exception:
        for col in ["eps_surprise", "rev_surprise", "days_to_earnings",
                    "post_earnings_1d", "post_earnings_3d", "post_earnings_5d"]:
            df[col] = 0.0

    # ── 9. Insider flows ──────────────────────────────────────────────────────
    ins_net, ins_7d, ins_21d = _load_insider(ticker, date_index)
    df["insider_net_shares"] = ins_net.values
    df["insider_7d"]         = ins_7d.values
    df["insider_21d"]        = ins_21d.values

    # ── 10. Congressional trading ──────────────────────────────────────────────
    congress = _load_congress(ticker, date_index)
    df["congress_net_shares"] = congress.values

    # ── 11. Risk flags ────────────────────────────────────────────────────────
    risk = _load_risk_flags(date_index)
    for col in ["risk_today", "risk_next_1d", "risk_next_3d", "risk_prev_1d"]:
        df[col] = risk[col].values if col in risk.columns else 0.0

    # ── 11. Pandemic regime ───────────────────────────────────────────────────
    dates_ts = pd.to_datetime(df["date"])
    df["is_pandemic"] = (
        (dates_ts >= PANDEMIC_START) & (dates_ts <= PANDEMIC_END)
    ).astype(int)

    # ── 12. NEW v2 features ───────────────────────────────────────────────────

    # Medium-term momentum
    df["return_20d"] = c.pct_change(20)
    df["return_60d"] = c.pct_change(60)

    # MA50 and crossover signals
    df["ma_50"]           = c.rolling(50).mean()
    df["ma5_above_ma20"]  = (df["ma_5"] > df["ma_20"]).astype(int)
    df["ma20_above_ma50"] = (df["ma_20"] > df["ma_50"]).astype(int)

    # 52-week high/low ratios
    high_52w = c.rolling(252).max()
    low_52w  = c.rolling(252).min()
    df["high_52w_ratio"] = (c / high_52w.replace(0, np.nan)) - 1.0   # 0 = at 52w high
    df["low_52w_ratio"]  = (c / low_52w.replace(0, np.nan)) - 1.0    # 0 = at 52w low

    # BB position (0 = at lower band, 1 = at upper band)
    bb_range = (df["bb_upper"] - df["bb_lower"]).replace(0, np.nan)
    df["bb_pct"] = (c - df["bb_lower"]) / bb_range

    # RSI extreme flags
    df["rsi_above_70"] = (df["rsi_14"] > 70).astype(int)
    df["rsi_below_30"] = (df["rsi_14"] < 30).astype(int)

    # ── Intraday-derived daily features ──────────────────────────────────────
    # vwap_dev_eod: how far close was from VWAP at end of day
    df["vwap_dev_eod"] = (c - df["vwap"]) / df["vwap"].replace(0, np.nan)

    # vol_surge_eod: today's volume vs 20d average
    vol_avg = v.rolling(20).mean().replace(0, np.nan)
    df["vol_surge_eod"] = v / vol_avg

    # intraday_momentum: vwap_dev weighted by vol_surge
    df["intraday_momentum"] = df["vwap_dev_eod"] * df["vol_surge_eod"].fillna(1)

    # OBV trend (OBV minus its 10d mean, normalized by std)
    obv_ma  = df["obv"].rolling(10).mean()
    obv_std = df["obv"].rolling(10).std().replace(0, np.nan)
    df["obv_trend"] = (df["obv"] - obv_ma) / obv_std

    # VIX
    # ── Overnight futures (S&P500 futures ES=F) ─────────────────────────────
    try:
        es_ret = _market_return("ES=F", start_str, end_str, date_index)
        df["es_overnight"] = es_ret.values
    except Exception:
        df["es_overnight"] = 0.0

    try:
        vix_raw = _market_return(VIX_TICKER, start_str, end_str, date_index,
                                  return_close=True)
        df["vix_close"] = vix_raw["close"].values if "close" in vix_raw.columns else 20.0
        df["vix_ret"]   = vix_raw["ret"].values   if "ret"   in vix_raw.columns else 0.0
    except Exception:
        df["vix_close"] = 20.0
        df["vix_ret"]   = 0.0

    # ── Crude oil (USO as proxy for WTI) ────────────────────────────────────
    try:
        oil_raw = _market_return("USO", start_str, end_str, date_index)
        df["oil_ret"] = oil_raw.values
        df["oil_spy_corr"] = df["oil_ret"].rolling(20).corr(df["return_1d"]).fillna(0)
    except Exception:
        df["oil_ret"]      = 0.0
        df["oil_spy_corr"] = 0.0
    # ── FinBERT NLP sentiment ────────────────────────────────────────────────
    if training_mode:
        df["finbert_sentiment"] = 0.0
        df["finbert_mult"]      = 1.0
    else:
        try:
            from data.alpha_sources import get_earnings_call_sentiment
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as ex:
                fut = ex.submit(get_earnings_call_sentiment, ticker)
                try:
                    nlp = fut.result(timeout=10)
                    df["finbert_sentiment"] = nlp.get("sentiment_score")      or 0.0
                    df["finbert_mult"]      = nlp.get("earnings_multiplier") or 1.0
                except Exception:
                    df["finbert_sentiment"] = 0.0
                    df["finbert_mult"]      = 1.0
        except Exception:
            df["finbert_sentiment"] = 0.0
            df["finbert_mult"]      = 1.0

    # ── Analyst revisions ────────────────────────────────────────────────────
    if training_mode:
        df["analyst_upside"]  = 0.0
        df["analyst_buy_pct"] = 0.5
        df["analyst_mult"]    = 1.0
    else:
        try:
            from data.alpha_sources import get_analyst_revisions
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as ex:
                fut = ex.submit(get_analyst_revisions, ticker)
                try:
                    analyst = fut.result(timeout=5)
                    df["analyst_upside"]    = analyst.get("target_upside")      or 0.0
                    df["analyst_buy_pct"]   = analyst.get("buy_pct")             or 0.5
                    df["analyst_mult"]      = analyst.get("analyst_multiplier")  or 1.0
                except Exception:
                    df["analyst_upside"]  = 0.0
                    df["analyst_buy_pct"] = 0.5
                    df["analyst_mult"]    = 1.0
        except Exception:
            df["analyst_upside"]  = 0.0
            df["analyst_buy_pct"] = 0.5
            df["analyst_mult"]    = 1.0

    # ── Options IV skew (daily snapshot) ─────────────────────────────────────
    if training_mode:
        df["iv_skew_snap"]  = 0.0
        df["pc_ratio_snap"] = 1.0
    else:
        try:
            from features.options_flow import get_options_signal
            import functools, concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as ex:
                fut = ex.submit(get_options_signal, ticker)
                try:
                    opt = fut.result(timeout=5)
                    iv_skew_val = opt.get("iv_skew") or 0.0
                    pc_ratio_val = opt.get("put_call_ratio") or 1.0
                except Exception:
                    iv_skew_val, pc_ratio_val = 0.0, 1.0
            df["iv_skew_snap"]   = iv_skew_val
            df["pc_ratio_snap"]  = pc_ratio_val
        except Exception:
            df["iv_skew_snap"]  = 0.0
            df["pc_ratio_snap"] = 1.0

    # Sector-relative return (stock 1d return minus its sector ETF return)
    sector_sym = SECTOR_ETF_MAP.get(ticker, SECTOR_ETF)
    try:
        import signal as _signal
        def _timeout_handler(signum, frame): raise TimeoutError()
        _signal.signal(_signal.SIGALRM, _timeout_handler)
        _signal.alarm(8)  # 8 second timeout
        try:
            sec_ret = _market_return(sector_sym, start_str, end_str, date_index)
            df["sector_rel_ret"] = df["return_1d"].values - sec_ret.values
        finally:
            _signal.alarm(0)
    except Exception:
        df["sector_rel_ret"] = 0.0

    # Calendar effects
    df["day_of_week"]  = pd.to_datetime(df["date"]).dt.dayofweek   # 0=Mon 4=Fri
    df["is_month_end"] = pd.to_datetime(df["date"]).dt.is_month_end.astype(int)

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
