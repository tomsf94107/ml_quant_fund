"""
signals/risk_gate.py
Builds daily risk flags for the training pipeline.
risk_today = 1 on high-impact event days (FOMC, CPI, jobs, earnings, VIX spikes)
           = 0 on normal days

Option B: real event flags — takes over from VIX proxy (Option A) going forward.
"""
import pandas as pd
import numpy as np

FOMC_DATES = [
    "2024-01-31", "2024-03-20", "2024-05-01", "2024-06-12",
    "2024-07-31", "2024-09-18", "2024-11-07", "2024-12-18",
    "2025-01-29", "2025-03-19", "2025-05-07", "2025-06-18",
    "2025-07-30", "2025-09-17", "2025-11-05", "2025-12-17",
    "2026-01-29", "2026-03-19", "2026-05-07", "2026-06-18",
]

CPI_DATES = [
    "2024-01-11", "2024-02-13", "2024-03-12", "2024-04-10",
    "2024-05-15", "2024-06-12", "2024-07-11", "2024-08-14",
    "2024-09-11", "2024-10-10", "2024-11-13", "2024-12-11",
    "2025-01-15", "2025-02-12", "2025-03-12", "2025-04-10",
    "2025-05-13", "2025-06-11", "2025-07-15", "2025-08-13",
    "2025-09-10", "2025-10-15", "2025-11-12", "2025-12-10",
    "2026-01-15", "2026-02-12", "2026-03-12", "2026-04-10",
]

VIX_SPIKE_PCT = 0.20


def build_risk_features(start_date, end_date) -> pd.DataFrame:
    """
    Build daily risk flags between start_date and end_date.
    Returns DataFrame with columns:
        risk_today, risk_next_1d, risk_next_3d, risk_prev_1d
    Indexed by business day dates.
    """
    import yfinance as yf

    dates = pd.bdate_range(str(start_date), str(end_date))
    df = pd.DataFrame(index=dates)
    df.index = pd.to_datetime(df.index)
    df["risk_today"] = 0.0

    for d in FOMC_DATES:
        ts = pd.Timestamp(d)
        if ts in df.index:
            df.loc[ts, "risk_today"] = 1.0

    for d in CPI_DATES:
        ts = pd.Timestamp(d)
        if ts in df.index:
            df.loc[ts, "risk_today"] = 1.0

    try:
        vix = yf.download("^VIX", start=str(start_date), end=str(end_date),
                          progress=False, auto_adjust=True)
        if not vix.empty:
            if hasattr(vix.columns, 'get_level_values'):
                vix.columns = vix.columns.get_level_values(0)
            vix_close = vix["Close"].squeeze()
            vix_ret = vix_close.pct_change().abs()
            for d in vix_ret[vix_ret > VIX_SPIKE_PCT].index:
                if d in df.index:
                    df.loc[d, "risk_today"] = 1.0
    except Exception:
        pass

    df["risk_next_1d"] = df["risk_today"].shift(-1).fillna(0)
    df["risk_next_3d"] = df["risk_today"].rolling(3).max().shift(-3).fillna(0)
    df["risk_prev_1d"] = df["risk_today"].shift(1).fillna(0)

    return df
