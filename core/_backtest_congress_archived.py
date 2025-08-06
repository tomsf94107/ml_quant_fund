# core/backtest_congress.py

import pandas as pd
from forecast_utils import build_feature_dataframe

def backtest_congress_signal(ticker: str, threshold: int = 0) -> pd.DataFrame:
    """
    Simple backtest: go long when yesterday’s net congressional buys
    exceed `threshold`, otherwise flat. Returns cumulative strategy
    vs. bench returns.
    """
    # Grab 1 year of data
    df = build_feature_dataframe(ticker, lookback=365)
    if df.empty or "congress_net_shares" not in df:
        return pd.DataFrame(columns=["cumstrat", "cumbench"])

    # Signal & returns
    df["signal"]     = df["congress_net_shares"].shift(1) > threshold
    df["future_ret"] = df["Close"].pct_change().shift(-1)
    df = df.dropna(subset=["signal", "future_ret"])

    # Strategy vs. benchmark PnL
    df["strategy_ret"] = df["signal"] * df["future_ret"]
    df["cumstrat"]     = (1 + df["strategy_ret"]).cumprod()
    df["cumbench"]     = (1 + df["future_ret"]).cumprod()

    return df[["cumstrat", "cumbench"]]
