# signals/risk_gate.py
# Builds daily risk flags for the training pipeline.
# Primary: Unusual Whales economic calendar API
# Fallback: hardcoded FOMC/CPI dates + VIX spike detection

import os
import pandas as pd
import numpy as np

from features.uw_client import uw_get

# Fallback hardcoded dates
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


def _get_uw_economic_calendar(start_date: str, end_date: str) -> list[str]:
    """
    Read high-impact event dates from accuracy.db.economic_calendar.
    Returns list of date strings where risk = 1.
    
    May 13 2026: Switched from live UW fetch to DB read.
    Was making per-ticker, per-training-run UW API calls (~1250/week wasted).
    DB refreshed 3x/week M/W/F by scripts/refresh_economic_calendar.py.
    Worst-case staleness: 2 days. For 3-day-ahead risk_next_3d gate, fine.
    
    Falls back to empty list if DB unavailable → hardcoded FOMC_DATES + 
    CPI_DATES fallback in build_risk_features takes over.

    ML_QUANT_SKIP_UW_CALENDAR=1 env var still honored for explicit bypass.
    """
    import os
    if os.environ.get("ML_QUANT_SKIP_UW_CALENDAR") == "1":
        return []

    try:
        import sqlite3
        from pathlib import Path
        db_path = Path(__file__).parent.parent / "accuracy.db"
        if not db_path.exists():
            return []
        with sqlite3.connect(f"file:{db_path}?mode=ro", uri=True) as conn:
            rows = conn.execute("""
                SELECT DISTINCT event_date FROM economic_calendar
                WHERE event_date >= ? AND event_date <= ?
                  AND impact = 'High'
            """, (start_date, end_date)).fetchall()
        return [r[0] for r in rows]

    except Exception:
        return []


def build_risk_features(start_date, end_date) -> pd.DataFrame:
    """
    Build daily risk flags between start_date and end_date.
    Uses UW economic calendar as primary source.
    Falls back to hardcoded FOMC/CPI + VIX spike detection.
    """
    import yfinance as yf

    dates = pd.bdate_range(str(start_date), str(end_date))
    df = pd.DataFrame(index=dates)
    df.index = pd.to_datetime(df.index)
    df["risk_today"] = 0.0

    # Primary: UW economic calendar
    uw_dates = _get_uw_economic_calendar(str(start_date), str(end_date))
    if uw_dates:
        for d in uw_dates:
            ts = pd.Timestamp(d)
            if ts in df.index:
                df.loc[ts, "risk_today"] = 1.0
    else:
        # Fallback: hardcoded FOMC + CPI dates
        for d in FOMC_DATES + CPI_DATES:
            ts = pd.Timestamp(d)
            if ts in df.index:
                df.loc[ts, "risk_today"] = 1.0

    # ─── Calendar events (PIT-safe: scheduled in advance) ───────────────
    # These can populate risk_next_1d / risk_next_3d because future FOMC/CPI/
    # etc. dates ARE known at prediction time.
    df["risk_next_1d"] = df["risk_today"].shift(-1).fillna(0)
    df["risk_next_3d"] = df["risk_today"].rolling(3).max().shift(-3).fillna(0)
    df["risk_prev_1d"] = df["risk_today"].shift(1).fillna(0)

    # ─── VIX spike retrospection (PIT-safe: only past spikes used) ──────
    # FIXED May 24 2026: previously VIX spikes were added to risk_today AFTER
    # the shift was already computed — this leaked future VIX info into
    # risk_next_1d/3d (those features encoded "VIX will spike in N days").
    # Now: VIX spikes only feed risk_prev_1d (strictly past). risk_today and
    # risk_next_* contain ONLY calendar-known events.
    try:
        from features.yf_resilient import safe_yf_download
        vix = safe_yf_download(["^VIX"], start=str(start_date), end=str(end_date),
                               progress=False, auto_adjust=True)
        if vix is not None and not vix.empty:
            if hasattr(vix.columns, "get_level_values"):
                vix.columns = vix.columns.get_level_values(0)
            vix_close = vix["Close"].squeeze()
            vix_ret = vix_close.pct_change().abs()
            for d in vix_ret[vix_ret > VIX_SPIKE_PCT].index:
                # Shift forward by 1 day: today's spike becomes tomorrow's
                # "previous-day spike" — strictly retrospective.
                d_plus_1 = d + pd.tseries.offsets.BDay(1)
                if d_plus_1 in df.index:
                    df.loc[d_plus_1, "risk_prev_1d"] = 1.0
    except Exception:
        pass

    return df
