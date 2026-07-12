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
    # ── VIX spike source (REWIRED 2026-07-12) ───────────────────────────────
    # WAS: safe_yf_download(["^VIX"]). yfinance is XProtect-BLOCKED on this machine,
    # so that call returned EMPTY on every single invocation, the loop below never
    # executed, and risk_prev_1d has NEVER once recorded a VIX spike. Silent, total,
    # months-long failure -- the same shape as the 18 credential-less crons.
    #
    # NOW: accuracy.db.vix_history, populated by backfill_vix.py from VIXY (the VIX
    # short-term futures ETF) via Massive. Polygon returns 403 on I:VIX (index data
    # is a higher tier) and empty on ^VIX/VIX; VIXY comes through cleanly.
    #
    # CRITICAL CALIBRATION NOTE: VIXY tracks VIX FUTURES, not spot VIX. Futures move
    # far less -- a 20% spot move might be an 8-10% VIXY move. The old
    # VIX_SPIKE_PCT=0.20 would fire ZERO times on VIXY, leaving this gate dead in a
    # NEW way, which is worse than dead in an obvious way. So is_spike is computed
    # in the table as a SELF-CALIBRATING percentile: a spike is a day in the top 1%
    # of |daily move| over the trailing 252 days. No knowledge of the VIX/VIXY beta
    # is needed, and it adapts if the feed ever changes again.
    #
    # Reading a TABLE, not the wire: build_risk_features is called once per ticker
    # (~400x per pipeline run). Hitting an API 400 times for the same market-wide
    # series is how you get 429'd.
    try:
        import sqlite3 as _sq
        from pathlib import Path as _P
        _db = _P(__file__).resolve().parent.parent / "accuracy.db"
        _c = _sq.connect(f"file:{_db}?mode=ro", uri=True, timeout=30)
        _rows = _c.execute(
            "SELECT date FROM vix_history WHERE is_spike=1 AND date BETWEEN ? AND ?",
            (str(start_date), str(end_date))).fetchall()
        _c.close()
        if _rows:
            vix_ret = pd.Series(1.0, index=pd.to_datetime([r[0] for r in _rows]))
            for d in vix_ret.index:
                # Shift forward by 1 day: today's spike becomes tomorrow's
                # "previous-day spike" — strictly retrospective.
                d_plus_1 = d + pd.tseries.offsets.BDay(1)
                if d_plus_1 in df.index:
                    df.loc[d_plus_1, "risk_prev_1d"] = 1.0
    except Exception:
        pass

    return df
