"""
Intraday feature builder — fetches 5-min bars and computes
momentum, VWAP deviation, RSI, volume surge for 1hr/2hr/4hr horizons.
"""
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime
import pytz

ET = pytz.timezone("America/New_York")


def is_market_open() -> bool:
    now = datetime.now(ET)
    if now.weekday() >= 5:
        return False
    market_open  = now.replace(hour=9,  minute=30, second=0, microsecond=0)
    market_close = now.replace(hour=16, minute=0,  second=0, microsecond=0)
    return market_open <= now <= market_close


def minutes_since_open() -> int:
    now = datetime.now(ET)
    open_time = now.replace(hour=9, minute=30, second=0, microsecond=0)
    return max(0, int((now - open_time).total_seconds() / 60))


def build_intraday_features(ticker: str) -> pd.DataFrame:
    """
    Fetch today's 5-min bars and compute intraday features.
    Returns DataFrame with one row per 5-min bar.
    """
    data = yf.download(
        ticker, period="2d", interval="5m",
        progress=False, auto_adjust=True
    )
    if data.empty:
        return pd.DataFrame()

    # Fix MultiIndex — yfinance returns (field, ticker) MultiIndex
    if isinstance(data.columns, pd.MultiIndex):
        # level 0 = field (Close/High/Low), level 1 = ticker
        if "Close" in data.columns.get_level_values(0):
            data.columns = data.columns.get_level_values(0)
        elif "Close" in data.columns.get_level_values(1):
            data.columns = data.columns.get_level_values(1)
        else:
            # Flatten: take only the field part
            data.columns = [str(col[0]) for col in data.columns]

    df = data.copy()
    df.index = pd.to_datetime(df.index)

    # Keep only most recent trading day's bars
    if df.index.tzinfo:
        df.index = df.index.tz_convert(ET)
    # Get the last date that has data
    last_date = df.index[-1].date()
    df = df[df.index.date == last_date]

    if df.empty or len(df) < 5:
        return pd.DataFrame()

    # ── Features ─────────────────────────────────────────────────────────────
    df = df.copy()
    df["return_5m"]    = df["Close"].pct_change()
    df["return_30m"]   = df["Close"].pct_change(6)    # 6 × 5min = 30min
    df["return_1hr"]   = df["Close"].pct_change(12)   # 12 × 5min = 1hr
    df["return_2hr"]   = df["Close"].pct_change(24)
    df["vol_surge"]    = df["Volume"] / df["Volume"].rolling(12).mean()

    # VWAP
    df["vwap"] = (df["Close"] * df["Volume"]).cumsum() / df["Volume"].cumsum()
    df["vwap_dev"] = (df["Close"] - df["vwap"]) / df["vwap"]

    # RSI 14 on 5-min bars
    delta = df["Close"].diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    rs    = gain / loss.replace(0, np.nan)
    df["rsi_14"] = 100 - (100 / (1 + rs))

    # Momentum score: combines return, vwap_dev, rsi
    df["momentum_score"] = (
        df["return_1hr"].fillna(0) * 0.4 +
        df["vwap_dev"].fillna(0)   * 0.3 +
        ((df["rsi_14"].fillna(50) - 50) / 50) * 0.3
    )

    # Volume-weighted momentum
    df["vol_momentum"] = df["momentum_score"] * df["vol_surge"].fillna(1)

    return df.dropna(subset=["Close"])


def get_intraday_signal(ticker: str) -> dict:
    """
    Returns intraday signals for 1hr, 2hr, 4hr horizons.
    """
    result = {
        "ticker": ticker,
        "signal_1hr": "NEUTRAL",
        "signal_2hr": "NEUTRAL",
        "signal_4hr": "NEUTRAL",
        "prob_1hr": 0.5,
        "prob_2hr": 0.5,
        "prob_4hr": 0.5,
        "current_price": None,
        "vwap": None,
        "vwap_dev": None,
        "rsi_14": None,
        "vol_surge": None,
        "momentum_score": None,
        "minutes_since_open": minutes_since_open(),
        "market_open": is_market_open(),
        "error": None,
    }

    try:
        df = build_intraday_features(ticker)
        if df.empty:
            result["error"] = "No intraday data"
            return result

        last = df.iloc[-1]
        mom  = float(last.get("momentum_score", 0) or 0)
        rsi  = float(last.get("rsi_14", 50) or 50)
        vdev = float(last.get("vwap_dev", 0) or 0)
        vsur = float(last.get("vol_surge", 1) or 1)
        mso  = minutes_since_open()

        result["current_price"]    = round(float(last["Close"]), 2)
        result["vwap"]             = round(float(last["vwap"]), 2)   if "vwap"           in last else None
        result["vwap_dev"]         = round(vdev * 100, 2)
        result["rsi_14"]           = round(rsi, 1)
        result["vol_surge"]        = round(vsur, 2)
        result["momentum_score"]   = round(mom, 4)

        # Convert momentum → probability (sigmoid-like)
        def mom_to_prob(m, scale=8):
            return round(1 / (1 + np.exp(-m * scale)), 3)

        # 1hr signal — most sensitive to recent momentum
        p1 = mom_to_prob(mom * 1.0)
        # 2hr signal — smoother, uses 2hr return
        ret_2hr = float(last.get("return_2hr", 0) or 0)
        p2 = mom_to_prob((mom * 0.6 + ret_2hr * 0.4))
        # 4hr signal — rest of day, mean-revert toward VWAP
        # If late in day (>180min), mean reversion more likely
        late_day_factor = min(mso / 390, 1.0)  # 390 = full trading day mins
        p4 = mom_to_prob(mom * (1 - late_day_factor * 0.4) + vdev * late_day_factor * -2)

        def prob_to_signal(p):
            if p >= 0.60: return "UP"
            if p <= 0.40: return "DOWN"
            return "NEUTRAL"

        result["prob_1hr"]    = p1
        result["prob_2hr"]    = p2
        result["prob_4hr"]    = p4
        result["signal_1hr"]  = prob_to_signal(p1)
        result["signal_2hr"]  = prob_to_signal(p2)
        result["signal_4hr"]  = prob_to_signal(p4)

    except Exception as e:
        result["error"] = str(e)

    return result
