# v1.8 forecast_feature_engineering.py
# Adds Risk Event flags (risk_today / risk_next_1d / risk_next_3d / risk_prev_1d)
# and a nonzero fraction metric for your UI. Keeps your v1.7 features intact.
#
# If ml_quant_fund/utils/risk_events.py is missing, this file falls back to zeros.

from __future__ import annotations

import pandas as pd
import numpy as np
import yfinance as yf
import sqlite3
from datetime import datetime
from pathlib import Path


# -------------------- Risk Events (optional import) --------------------
# We attempt to import the helper you just created. If it isn't available,
# we define safe fallbacks that return zeros so the rest of the pipeline works.
DEFAULT_EVENTS_CAL_PATH = str(Path(__file__).resolve().parent / "data" / "events_calendar.csv")

try:
    # Preferred import path for your repo structure
    from ml_quant_fund.utils.risk_events import (
        load_events_calendar,
        make_risk_flags,
        nonzero_frac as risk_nonzero_frac,
    )
    _RISK_OK = True
except Exception:
    # Fallback: stubbed risk functions (all zeros)
    _RISK_OK = False

    def load_events_calendar(path: str) -> pd.DataFrame:
        # Empty calendar → no events
        return pd.DataFrame(columns=["date", "ticker", "category", "label", "severity"])

    def make_risk_flags(index: pd.DatetimeIndex, ticker: str,
                        events: pd.DataFrame | None = None,
                        weights: dict[str, float] | None = None) -> pd.DataFrame:
        idx = pd.DatetimeIndex(pd.to_datetime(index)).tz_localize(None).normalize()
        return pd.DataFrame(
            0.0,
            index=idx,
            columns=["risk_today", "risk_next_1d", "risk_next_3d", "risk_prev_1d"]
        )

    def risk_nonzero_frac(flags_df: pd.DataFrame) -> float:
        return 0.0

# -------------------- Load Insiders --------------------
def load_insider_flows(ticker: str, db_path: str = "insider_trades.db") -> pd.DataFrame:
    """
    Returns a DataFrame with columns:
      - date (datetime)
      - net_shares
      - insider_7d
      - insider_21d
    for the given ticker.
    """
    conn = sqlite3.connect(db_path)
    try:
        query = """
          SELECT date, net_shares, insider_7d, insider_21d
            FROM insider_flows
           WHERE ticker = ?
        ORDER BY date
        """
        df = pd.read_sql(query, conn, params=(ticker,), parse_dates=["date"])
    finally:
        conn.close()
    return df

# -------------------- Feature Builder --------------------
def build_feature_dataframe(
    ticker: str,
    start_date: str = "2018-01-01",
    end_date: str | None = None,
    events_calendar_path: str | None = DEFAULT_EVENTS_CAL_PATH,
    risk_weights: dict[str, float] | None = None,
) -> pd.DataFrame:
    """
    Builds a DataFrame of features for `ticker` between start_date and end_date.

    Adds:
      - Price features (returns, MAs, volatility)
      - RSI(14), MACD/Signal
      - Bollinger(20): upper/lower/width
      - Volume spike z-score + spike flag
      - Sentiment placeholder (0.0)
      - Insider flows (net_shares, insider_7d, insider_21d) merged from SQLite
      - Pandemic regime dummy (2020-03-01..2023-12-31)
      - Risk event flags (risk_today / risk_next_1d / risk_next_3d / risk_prev_1d)
        from events calendar (if available)

    Notes:
      - df.attrs["risk_nonzero_frac"] holds the fraction of rows with any risk flag > 0.
      - If risk helper/calendar is missing, risk columns are zeros and fraction is 0.0.
    """
    if end_date is None:
        end_date = datetime.today().strftime("%Y-%m-%d")

    # --- fetch price history ---
    df = yf.download(ticker, start=start_date, end=end_date, auto_adjust=True, progress=False)
    if df.empty:
        raise ValueError(f"No data found for {ticker}")

    # reset + rename
    df = df.reset_index()
    df.rename(columns={"Date": "date", "Close": "close", "Volume": "volume"}, inplace=True)

    # ensure core columns
    for col in ("date", "close", "volume"):
        if col not in df.columns:
            raise ValueError(f"❌ Missing required column: {col}")

    # --- Pandemic Regime Dummy ---
    df["is_pandemic"] = ((df["date"] >= pd.Timestamp("2020-03-01")) &
                         (df["date"] <= pd.Timestamp("2023-12-31"))).astype(int)

    # --- Price-Based Features ---
    df["return_1d"] = df["close"].pct_change()
    df["return_3d"] = df["close"].pct_change(3)
    df["return_5d"] = df["close"].pct_change(5)

    df["ma_5"]  = df["close"].rolling(window=5).mean()
    df["ma_10"] = df["close"].rolling(window=10).mean()
    df["ma_20"] = df["close"].rolling(window=20).mean()

    df["volatility_5d"]  = df["return_1d"].rolling(window=5).std()
    df["volatility_10d"] = df["return_1d"].rolling(window=10).std()

    # --- RSI (14) ---
    delta = df["close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    # Wilder's smoothing can be added later; simple MA is fine here
    avg_gain = gain.rolling(window=14, min_periods=14).mean()
    avg_loss = loss.rolling(window=14, min_periods=14).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    df["rsi_14"] = rsi  

    # --- MACD ---
    ema12 = df["close"].ewm(span=12, adjust=False).mean()
    ema26 = df["close"].ewm(span=26, adjust=False).mean()
    df["macd"] = ema12 - ema26
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()

    # --- Bollinger Bands (20-day) ---
    ma20 = df["close"].rolling(window=20).mean()
    std20 = df["close"].rolling(window=20).std()
    df["bollinger_upper"] = ma20 + 2 * std20
    df["bollinger_lower"] = ma20 - 2 * std20
    df["bollinger_width"] = (4 * std20 / ma20).replace([np.inf, -np.inf], np.nan)

    # --- Volume Spike Detection ---
    vol_mean = df["volume"].rolling(window=20).mean()
    vol_std = df["volume"].rolling(window=20).std()
    df["volume_zscore"] = (df["volume"] - vol_mean) / vol_std.replace(0, np.nan)
    df["volume_spike"] = (df["volume_zscore"] > 2).astype(int)

    # --- Sentiment Placeholder ---
    df["sentiment_score"] = 0.0  # TODO: plug in real sentiment pipeline

    # --- Insider-Trades Feature via SQLite aggregates ---
    try:
        ins_df = load_insider_flows(ticker)
        if not ins_df.empty:
            ins_df = ins_df.sort_values("date").set_index("date")
            df["insider_net_shares"] = df["date"].map(ins_df["net_shares"]).fillna(0.0)
            df["insider_7d"] = df["date"].map(ins_df["insider_7d"]).fillna(0.0)
            df["insider_21d"] = df["date"].map(ins_df["insider_21d"]).fillna(0.0)
        else:
            df["insider_net_shares"] = 0.0
            df["insider_7d"] = 0.0
            df["insider_21d"] = 0.0
    except Exception as e:
        print(f"⚠️ Insider-flows merge failed for {ticker}: {e}")
        df["insider_net_shares"] = 0.0
        df["insider_7d"] = 0.0
        df["insider_21d"] = 0.0

    # --- Risk Event Flags (calendar-based) ---
    try:
        events = None
        if events_calendar_path is not None:
            events = load_events_calendar(events_calendar_path)

        # Build risk flags against the date index
        risk_flags = make_risk_flags(
            pd.DatetimeIndex(df["date"]),
            ticker=ticker,
            events=events,
            weights=risk_weights,
        )
        # Ensure we have a merge-able column
        rf = risk_flags.copy()
        rf.index.name = "date"
        rf = rf.reset_index()

        df = df.merge(rf, on="date", how="left")

        # Fill any missing flags with zeros (e.g., for early lookback rows)
        for col in ["risk_today", "risk_next_1d", "risk_next_3d", "risk_prev_1d"]:
            if col not in df.columns:
                df[col] = 0.0
            df[col] = df[col].fillna(0.0).astype(float)

        # Attach nonzero fraction to attrs for UI readout
        df.attrs["risk_nonzero_frac"] = float(risk_nonzero_frac(risk_flags))
        # Also expose it as a convenience column if you want to print it quickly
        # (comment out if you prefer attrs-only)
        # df["risk_nonzero_frac"] = df.attrs["risk_nonzero_frac"]

    except Exception as e:
        # Any failure produces zeroed risk columns so downstream code is stable
        print(f"⚠️ Risk flags unavailable (using zeros): {e}")
        for col in ["risk_today", "risk_next_1d", "risk_next_3d", "risk_prev_1d"]:
            df[col] = 0.0
        df.attrs["risk_nonzero_frac"] = 0.0

    # --- Final cleanup ---
    # Drop NaNs from indicator warm-ups
    df = df.dropna().reset_index(drop=True)

    return df

# -------------------- Target Generator --------------------
def add_forecast_targets(df: pd.DataFrame, horizon_days: tuple[int, ...] = (1, 3)) -> pd.DataFrame:
    """
    Adds classification targets for up/down over given horizons.
    For each h in horizon_days:
      - ensures return_h d exists
      - creates target_h d = 1 if forward return over next h days > 0, else 0
    """
    df = df.copy()
    for h in horizon_days:
        return_col = f"return_{h}d"
        if return_col not in df.columns:
            df[return_col] = df["close"].pct_change(periods=h)
        df[f"target_{h}d"] = (df[return_col].shift(-h) > 0).astype(int)
    return df
