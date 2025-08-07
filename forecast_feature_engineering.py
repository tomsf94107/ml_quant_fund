# v1.5 forecast_feature_engineering.py
# Enhanced with Bollinger, Volume Spikes, Sentiment Placeholder, and Insider Trades

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime

# 1) bring in our insider-trades extractor
from data.etl_insider import fetch_insider_trades


# -------------------- Feature Builder --------------------
def build_feature_dataframe(ticker: str, start_date="2018-01-01", end_date=None):
    """
    Builds a DataFrame of features for `ticker` between start_date and end_date.
    """
    if end_date is None:
        end_date = datetime.today().strftime('%Y-%m-%d')

    # --- fetch price history ---
    df = yf.download(ticker, start=start_date, end=end_date, auto_adjust=True)
    if df.empty:
        raise ValueError(f"No data found for {ticker}")

    # reset + rename
    df = df.reset_index()
    df.rename(columns={"Date": "date", "Close": "close", "Volume": "volume"}, inplace=True)

    # sanity check
    for col in ("date", "close", "volume"):
        if col not in df.columns:
            raise ValueError(f"❌ Missing required column: {col}")

    # --- Price-Based Features ---
    df["return_1d"] = df["close"].pct_change()
    df["return_3d"] = df["close"].pct_change(3)
    df["return_5d"] = df["close"].pct_change(5)

    df["ma_5"]  = df["close"].rolling(window=5).mean()
    df["ma_10"] = df["close"].rolling(window=10).mean()
    df["ma_20"] = df["close"].rolling(window=20).mean()

    df["volatility_5d"]  = df["return_1d"].rolling(window=5).std()
    df["volatility_10d"] = df["return_1d"].rolling(window=10).std()

    # RSI (14)
    delta     = df["close"].diff()
    gain      = delta.where(delta > 0, 0)
    loss      = -delta.where(delta < 0, 0)
    avg_gain  = gain.rolling(window=14).mean()
    avg_loss  = loss.rolling(window=14).mean()
    rs        = avg_gain / avg_loss
    df["rsi_14"] = 100 - (100 / (1 + rs))

    # MACD
    ema12          = df["close"].ewm(span=12, adjust=False).mean()
    ema26          = df["close"].ewm(span=26, adjust=False).mean()
    df["macd"]        = ema12 - ema26
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()

    # --- Bollinger Bands (20-day) ---
    ma20 = df["ma_20"]
    std20 = df["close"].rolling(window=20).std()
    df["bollinger_upper"] = ma20 + 2 * std20
    df["bollinger_lower"] = ma20 - 2 * std20
    df["bollinger_width"] = (df["bollinger_upper"] - df["bollinger_lower"]) / ma20

    # --- Volume Spike Detection ---
    vol_mean20 = df["volume"].rolling(window=20).mean()
    vol_std20  = df["volume"].rolling(window=20).std()
    df["volume_zscore"] = (df["volume"] - vol_mean20) / vol_std20
    df["volume_spike"]  = (df["volume_zscore"] > 2).astype(int)

    # --- Optional: Sentiment Placeholder ---
    df["sentiment_score"] = 0.0  # TODO: plug in real sentiment pipeline

    # --- Insider-Trades Feature ---
    try:
        ins = fetch_insider_trades(ticker, mode="sheet-first")
        ins_ts = ins.set_index("ds")["net_shares"]
        df["insider_net_shares"] = df["date"].dt.date.map(lambda d: ins_ts.get(d, 0))
    except Exception as e:
        print(f"⚠️ Insider trades merge failed for {ticker}: {e}")
        df["insider_net_shares"] = 0

    # drop NaNs from rolling computations
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


# -------------------- Target Generator --------------------
def add_forecast_targets(df: pd.DataFrame, horizon_days=(1, 3)):
    df = df.copy()
    for h in horizon_days:
        return_col = f"return_{h}d"
        if return_col not in df.columns:
            df[return_col] = df["close"].pct_change(periods=h)
        df[f"target_{h}d"] = (df[return_col].shift(-h) > 0).astype(int)
    return df
