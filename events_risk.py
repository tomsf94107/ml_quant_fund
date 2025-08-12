# events_risk.py
from __future__ import annotations
import os
from datetime import datetime, date, timedelta
from typing import List, Dict, Optional
import pandas as pd
import requests

ImpactW = {"Low": 1, "Medium": 2, "High": 3}

def _get(key_env: str, *secret_paths) -> Optional[str]:
    """Env first. If Streamlit is available, try st.secrets with nested paths."""
    v = os.getenv(key_env)
    if v: return v
    try:
        import streamlit as st  # only if running inside Streamlit
        cur = st.secrets
        for p in secret_paths:
            if isinstance(cur, dict) and p in cur: cur = cur[p]
            else: return None
        return cur if isinstance(cur, str) else None
    except Exception:
        return None

def _rows_to_df(rows: List[Dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    if df.empty: return df
    df["start"] = pd.to_datetime(df["start"], utc=True).dt.tz_convert("US/Eastern")
    df["end"]   = pd.to_datetime(df["end"],   utc=True).dt.tz_convert("US/Eastern")
    df["impact"]= df["impact"].fillna("Low")
    return df

def fetch_economic_fmp(start: date, end: date) -> pd.DataFrame:
    key = _get("FMP_API_KEY", "providers", "fmp_api_key")
    if not key: return pd.DataFrame()
    url = "https://financialmodelingprep.com/api/v3/economic_calendar"
    r = requests.get(url, params={"from": start.isoformat(),"to": end.isoformat(),"apikey": key}, timeout=15)
    r.raise_for_status()
    rows = []
    for it in r.json() or []:
        when = it.get("date"); 
        if not when: continue
        try:
            dt = datetime.fromisoformat(when.replace("Z","+00:00"))
        except Exception:
            continue
        imp_raw = (it.get("impact") or "").title()
        impact = "High" if "High" in imp_raw else ("Medium" if "Medium" in imp_raw else "Low")
        rows.append({
            "title": it.get("event") or "Economic Event",
            "category": "Economic",
            "start": dt,
            "end": dt + timedelta(minutes=30),
            "impact": impact,
            "tickers": "SPY,QQQ,IWM",
        })
    return _rows_to_df(rows)

def fetch_earnings_finnhub(start: date, end: date) -> pd.DataFrame:
    tok = _get("FINNHUB_TOKEN", "providers", "finnhub_token")
    if not tok: return pd.DataFrame()
    url = "https://finnhub.io/api/v1/calendar/earnings"
    r = requests.get(url, params={"from": start.isoformat(), "to": end.isoformat(), "token": tok}, timeout=15)
    r.raise_for_status()
    items = (r.json() or {}).get("earningsCalendar") or []
    rows = []
    for it in items:
        d = it.get("date")
        if not d: continue
        base = datetime.fromisoformat(d)
        time_hint = (it.get("time") or "").lower()
        if time_hint == "amc": when = base.replace(hour=16, minute=0)
        elif time_hint == "bmo": when = base.replace(hour=8,  minute=0)
        else: when = base.replace(hour=13, minute=0)
        sym = it.get("symbol","")
        rows.append({
            "title": f"Earnings: {sym}",
            "category": "Earnings",
            "start": when,
            "end": when + timedelta(hours=1),
            "impact": "High" if sym in {"AAPL","MSFT","GOOGL","AMZN","META","NVDA","TSLA"} else "Medium",
            "tickers": sym,
        })
    return _rows_to_df(rows)

def fetch_ipo_finnhub(start: date, end: date) -> pd.DataFrame:
    tok = _get("FINNHUB_TOKEN", "providers", "finnhub_token")
    if not tok: return pd.DataFrame()
    url = "https://finnhub.io/api/v1/calendar/ipo"
    r = requests.get(url, params={"from": start.isoformat(), "to": end.isoformat(), "token": tok}, timeout=15)
    r.raise_for_status()
    items = (r.json() or {}).get("ipoCalendar") or []
    rows = []
    for it in items:
        d = it.get("date") or it.get("ipoDate")
        if not d: continue
        when = datetime.fromisoformat(d)
        sym  = it.get("symbol") or it.get("ticker") or "IPO"
        rows.append({
            "title": f"IPO: {sym}",
            "category": "IPO/Lockup",
            "start": when.replace(hour=18, minute=0),
            "end":   when.replace(hour=20, minute=0),
            "impact": "Low",
            "tickers": sym,
        })
    return _rows_to_df(rows)

def load_events(start: date, end: date, use_fmp: bool=True, use_finnhub: bool=True) -> pd.DataFrame:
    dfs = []
    if use_fmp: dfs.append(fetch_economic_fmp(start, end))
    if use_finnhub:
        dfs += [fetch_earnings_finnhub(start, end), fetch_ipo_finnhub(start, end)]
    dfs = [d for d in dfs if d is not None and not d.empty]
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame(
        columns=["title","category","start","end","impact","tickers"]
    )

def build_risk_features(start: date, end: date,
                        use_fmp: bool=True, use_finnhub: bool=True) -> pd.DataFrame:
    """Return a DataFrame indexed by date with risk features:
       risk_today, risk_next_1d, risk_next_3d, risk_prev_1d
    """
    ev = load_events(start, end, use_fmp, use_finnhub)
    idx = pd.date_range(start, end, freq="D", tz="US/Eastern").date
    base = pd.DataFrame(index=pd.Index(idx, name="date"))
    if ev.empty:
        base[["risk_today","risk_next_1d","risk_next_3d","risk_prev_1d"]] = 0
        return base.reset_index()

    tmp = ev.copy()
    tmp["date"] = tmp["start"].dt.tz_convert("US/Eastern").dt.date
    tmp["w"]    = tmp["impact"].map(lambda x: ImpactW.get(str(x), 1))
    daily = tmp.groupby("date")["w"].sum().reindex(idx, fill_value=0)

    # future-known scheduled events: using next-day risk is allowed (known in advance)
    risk_today     = daily
    risk_next_1d   = daily.shift(-1, fill_value=0)
    risk_next_3d   = daily.shift(-1, fill_value=0) + daily.shift(-2, fill_value=0) + daily.shift(-3, fill_value=0)
    risk_prev_1d   = daily.shift(1,  fill_value=0)

    out = pd.DataFrame({
        "date": list(idx),
        "risk_today":   risk_today.values,
        "risk_next_1d": risk_next_1d.values,
        "risk_next_3d": risk_next_3d.values,
        "risk_prev_1d": risk_prev_1d.values,
    })
    return out
