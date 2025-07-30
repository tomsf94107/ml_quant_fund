# v1.3 forecast_feature_engineering.py (Enhanced with Bollinger, Volume Spikes, Sentiment Placeholder)

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime

# -------------------- Feature Builder --------------------
def build_feature_dataframe(ticker: str, start_date="2018-01-01", end_date=None):
    if end_date is None:
        end_date = datetime.today().strftime('%Y-%m-%d')

    df = yf.download(ticker, start=start_date, end=end_date, auto_adjust=True)

    if df.empty:
        raise ValueError(f"No data found for {ticker}")

    df = df.reset_index()
    df.rename(columns={"Date": "date", "Close": "close", "Volume": "volume"}, inplace=True)

    required_cols = ["date", "close", "volume"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"❌ Missing required column: {col}")

    # --- Price-Based Features ---
    df["return_1d"] = df["close"].pct_change()
    df["return_3d"] = df["close"].pct_change(3)
    df["return_5d"] = df["close"].pct_change(5)

    df["ma_5"] = df["close"].rolling(window=5).mean()
    df["ma_10"] = df["close"].rolling(window=10).mean()
    df["ma_20"] = df["close"].rolling(window=20).mean()

    df["volatility_5d"] = df["return_1d"].rolling(window=5).std()
    df["volatility_10d"] = df["return_1d"].rolling(window=10).std()

    # RSI (14)
    delta = df["close"].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss
    df["rsi_14"] = 100 - (100 / (1 + rs))

    # MACD
    ema12 = df["close"].ewm(span=12, adjust=False).mean()
    ema26 = df["close"].ewm(span=26, adjust=False).mean()
    df["macd"] = ema12 - ema26
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()

    # --- New: Bollinger Bands (20-day) ---
    ma20 = df["close"].rolling(window=20).mean()
    std20 = df["close"].rolling(window=20).std()
    df["bollinger_upper"] = ma20 + 2 * std20
    df["bollinger_lower"] = ma20 - 2 * std20
    df["bollinger_width"] = (df["bollinger_upper"] - df["bollinger_lower"]) / ma20

    # --- New: Volume Spike Detection ---
    df["volume_zscore"] = (df["volume"] - df["volume"].rolling(window=20).mean()) / df["volume"].rolling(window=20).std()
    df["volume_spike"] = (df["volume_zscore"] > 2).astype(int)

    # --- Optional: Sentiment Placeholder (to be populated later) ---
    df["sentiment_score"] = 0.0  # TODO: Replace with real sentiment pipeline

    df.dropna(inplace=True)
    return df

# -------------------- Target Generator --------------------
def add_forecast_targets(df: pd.DataFrame, horizon_days=(1, 3)):
    df = df.copy()
    for h in horizon_days:
        col_name = f"target_{h}d"
        return_col = f"return_{h}d"

        if return_col not in df.columns:
            df[return_col] = df["close"].pct_change(periods=h)

        df[col_name] = (df[return_col].shift(-h) > 0).astype(int)
    return df

