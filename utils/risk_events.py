# risk_events.py
from __future__ import annotations
import pandas as pd
from pandas.tseries.offsets import BDay

CALENDAR_COLS = ["date", "ticker", "category", "label", "severity"]

def load_events_calendar(path: str) -> pd.DataFrame:
    """
    CSV schema:
      date (YYYY-MM-DD), ticker (e.g., CRWD or * for all),
      category (e.g., earnings, macro:FOMC, macro:CPI, macro:PPI),
      label (free text), severity (0..1)
    """
    try:
        cal = pd.read_csv(path)
    except FileNotFoundError:
        # Empty calendar if file not present
        return pd.DataFrame(columns=CALENDAR_COLS)

    # Basic cleaning
    cal = cal.rename(columns={c: c.lower() for c in cal.columns})
    missing = [c for c in CALENDAR_COLS if c not in cal.columns]
    if missing:
        raise ValueError(f"Calendar missing columns: {missing}")

    cal["date"] = pd.to_datetime(cal["date"]).dt.normalize()
    cal["ticker"] = cal["ticker"].fillna("*").str.upper()
    # clip + fill severity
    cal["severity"] = pd.to_numeric(cal["severity"], errors="coerce").fillna(1.0).clip(0, 1)
    return cal

def _filter_calendar(cal: pd.DataFrame, ticker: str) -> pd.DataFrame:
    # Keep events for this ticker OR global (*) macro events
    ticker = (ticker or "").upper()
    return cal[(cal["ticker"] == ticker) | (cal["ticker"] == "*")].copy()

def make_risk_flags(
    index: pd.DatetimeIndex,
    ticker: str,
    events: pd.DataFrame | None = None,
    weights: dict[str, float] | None = None,
) -> pd.DataFrame:
    """
    Returns a DataFrame with columns:
      risk_today, risk_next_1d, risk_next_3d, risk_prev_1d  (float 0..1)
    If no events provided, returns zeros.
    """
    idx = pd.DatetimeIndex(pd.to_datetime(index)).tz_localize(None).normalize()
    out = pd.DataFrame(0.0, index=idx, columns=["risk_today", "risk_next_1d", "risk_next_3d", "risk_prev_1d"])
    if events is None or events.empty:
        return out

    ev = _filter_calendar(events, ticker)
    if ev.empty:
        return out

    # Optional category weights (you can tweak)
    base = {
        "earnings": 1.0,
        "macro:FOMC": 0.9,
        "macro:CPI": 0.7,
        "macro:PPI": 0.6,
        "macro:PAYROLLS": 0.7,
    }
    if weights:
        base.update(weights)

    ev["w"] = ev["category"].map(lambda c: base.get(str(c), 0.5)) * ev["severity"]
    ev_dates = ev.groupby("date")["w"].sum()

    # Precompute business-day windows
    prev_1 = pd.Series(0.0, index=idx)
    today = pd.Series(0.0, index=idx)
    next_1 = pd.Series(0.0, index=idx)
    next_3 = pd.Series(0.0, index=idx)

    # Align weights onto index with date arithmetic
    today = today.add(ev_dates.reindex(idx, fill_value=0), fill_value=0)

    prev_map = ev_dates.copy()
    prev_map.index = prev_map.index + BDay(1)   # event yesterday -> flag today
    prev_1 = prev_1.add(prev_map.reindex(idx, fill_value=0), fill_value=0)

    next1_map = ev_dates.copy()
    next1_map.index = next1_map.index - BDay(1) # event tomorrow -> flag today
    next_1 = next_1.add(next1_map.reindex(idx, fill_value=0), fill_value=0)

    # “Within next 3 business days (excluding next_1)”
    for k in (2, 3):
        m = ev_dates.copy()
        m.index = m.index - BDay(k)
        next_3 = next_3.add(m.reindex(idx, fill_value=0), fill_value=0)

    # Clip to 1.0 so multiple events don’t explode the scale
    out["risk_today"] = today.clip(upper=1.0)
    out["risk_prev_1d"] = prev_1.clip(upper=1.0)
    out["risk_next_1d"] = next_1.clip(upper=1.0)
    out["risk_next_3d"] = next_3.clip(upper=1.0)
    return out

def nonzero_frac(flags_df: pd.DataFrame) -> float:
    """Fraction of rows where any risk flag > 0."""
    if flags_df.empty:
        return 0.0
    any_nonzero = (flags_df[["risk_today","risk_next_1d","risk_next_3d","risk_prev_1d"]].sum(axis=1) > 0)
    return float(any_nonzero.mean())
