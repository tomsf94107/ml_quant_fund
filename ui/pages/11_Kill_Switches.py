# ui/11_Kill_Switches.py
# Kill Switch Monitor Dashboard
# Monitors bucket-level kill switches from the AI Investment Playbook §10-12

import os, sys
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import sqlite3
import pandas as pd
import streamlit as st
from datetime import date, timedelta
from pathlib import Path

DB_PATH = Path(_ROOT) / "accuracy.db"

st.set_page_config(page_title="Kill Switches", page_icon="🚨", layout="wide")
st.title("🚨 Kill Switch Monitor")
st.caption("Bucket-level kill switches from the AI playbook §10-12 — checked daily.")


@st.cache_data(ttl=300)
def get_vix() -> float:
    try:
        import yfinance as yf
        v = yf.download("^VIX", period="5d", progress=False, auto_adjust=True)
        if not v.empty:
            if hasattr(v.columns, "get_level_values"):
                v.columns = v.columns.get_level_values(0)
            return float(v["Close"].iloc[-1])
    except Exception:
        pass
    return 20.0


@st.cache_data(ttl=300)
def get_vix_history(days: int = 10) -> list[float]:
    try:
        import yfinance as yf
        v = yf.download("^VIX", period="1mo", progress=False, auto_adjust=True)
        if not v.empty:
            if hasattr(v.columns, "get_level_values"):
                v.columns = v.columns.get_level_values(0)
            return v["Close"].tail(days).tolist()
    except Exception:
        pass
    return []


@st.cache_data(ttl=300)
def get_sleeve_drawdown() -> float:
    try:
        conn = sqlite3.connect(DB_PATH)
        rows = conn.execute("""
            SELECT prediction_date,
                   ROUND(100.0*SUM(CASE WHEN (prob_up>0.5 AND o.actual_return>0)
                                         OR (prob_up<=0.5 AND o.actual_return<0)
                                   THEN 1 ELSE 0 END)/COUNT(*),1) as acc
            FROM predictions p
            JOIN outcomes o ON p.ticker=o.ticker
                           AND p.prediction_date=o.prediction_date
                           AND p.horizon=o.horizon
            WHERE p.prediction_date >= ?
            GROUP BY p.prediction_date
            ORDER BY p.prediction_date DESC
            LIMIT 10
        """, (str(date.today() - timedelta(days=30)),)).fetchall()
        conn.close()
        if rows:
            accs = [r[1] for r in rows if r[1] is not None]
            if accs:
                return (sum(accs) / len(accs)) - 50.0
        return 0.0
    except Exception:
        return 0.0


@st.cache_data(ttl=300)
def get_recent_accuracy() -> float:
    try:
        conn = sqlite3.connect(DB_PATH)
        row = conn.execute("""
            SELECT ROUND(100.0*SUM(CASE WHEN (p.prob_up>0.5 AND o.actual_return>0)
                                         OR (p.prob_up<=0.5 AND o.actual_return<0)
                                   THEN 1 ELSE 0 END)/COUNT(*),1)
            FROM predictions p
            JOIN outcomes o ON p.ticker=o.ticker
                           AND p.prediction_date=o.prediction_date
                           AND p.horizon=o.horizon
            WHERE p.prediction_date >= ?
        """, (str(date.today() - timedelta(days=14)),)).fetchone()
        conn.close()
        return float(row[0]) if row and row[0] else 0.0
    except Exception:
        return 0.0


def _status_badge(status: str) -> str:
    if status == "ALERT": return "🔴"
    if status == "WATCH": return "🟡"
    return "🟢"


# ── Load live data ─────────────────────────────────────────────────────────────
vix          = get_vix()
vix_history  = get_vix_history(10)
accuracy_14d = get_recent_accuracy()

vix_above_25_days = sum(1 for v in vix_history if v > 25)
vix_above_30_days = sum(1 for v in vix_history if v > 30)


# ── KPIs ──────────────────────────────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)

vix_color = "normal" if vix < 20 else "inverse" if vix > 25 else "off"
c1.metric("VIX current", f"{vix:.1f}",
          "above 25 — reduce lotto" if vix > 25 else "normal range")

c2.metric("VIX >25 sessions (10d)", vix_above_25_days,
          "ALERT — trim lotto 50%" if vix_above_25_days >= 5 else "OK")

c3.metric("14-day accuracy", f"{accuracy_14d:.1f}%",
          "WATCH — below 50%" if accuracy_14d < 50 else "OK")

alerts = sum([
    vix_above_25_days >= 5,
    vix > 30,
    accuracy_14d < 45,
])
c4.metric("Active alerts", alerts, "require action" if alerts > 0 else "all clear")

st.markdown("---")


# ── VIX-based sleeve rules ─────────────────────────────────────────────────────
st.subheader("📊 VIX sleeve-level rules — §11.2")

vix_rules = [
    {
        "Rule": "VIX >25 for 5+ sessions → trim lotto 50%",
        "Threshold": "5 sessions",
        "Current": f"{vix_above_25_days} sessions",
        "Status": "ALERT" if vix_above_25_days >= 5 else "OK",
    },
    {
        "Rule": "VIX >30 for 10+ sessions → full risk-off",
        "Threshold": "10 sessions",
        "Current": f"{vix_above_30_days} sessions",
        "Status": "ALERT" if vix_above_30_days >= 10 else "WATCH" if vix_above_30_days >= 5 else "OK",
    },
    {
        "Rule": "VIX <15 for 20+ sessions → complacency warning",
        "Threshold": "20 sessions",
        "Current": f"{sum(1 for v in vix_history if v < 15)} sessions",
        "Status": "WATCH" if sum(1 for v in vix_history if v < 15) >= 15 else "OK",
    },
]

vix_df = pd.DataFrame(vix_rules)
vix_df["Status"] = vix_df["Status"].apply(lambda x: f"{_status_badge(x)} {x}")
st.dataframe(vix_df, use_container_width=True, hide_index=True)
st.markdown("---")


# ── Macro kill switches ────────────────────────────────────────────────────────
st.subheader("🌐 Macro kill switches — §12")
st.caption("Manual check required — update status based on latest earnings calls and news.")

macro_switches = [
    {
        "Trigger": "Any hyperscaler guides 2027 capex flat or lower",
        "Threshold": "Any 1 of 4",
        "Last checked": "Apr 17 2026",
        "Status": "OK",
        "Note": "All 4 guiding higher — Q2 calls due late Apr/May",
    },
    {
        "Trigger": "Memory contract prices decline QoQ for 2 consecutive quarters",
        "Threshold": "2 consecutive qtrs",
        "Last checked": "Apr 17 2026",
        "Status": "OK",
        "Note": "NAND +12% QoQ — healthy",
    },
    {
        "Trigger": "NVDA top-4 customer concentration >70%",
        "Threshold": ">70%",
        "Last checked": "Apr 17 2026",
        "Status": "WATCH",
        "Note": "~62% — rising. Monitor earnings.",
    },
    {
        "Trigger": "DC project cancellations >10 GW in 90 days",
        "Threshold": ">10 GW",
        "Last checked": "Apr 17 2026",
        "Status": "OK",
        "Note": "0 GW cancelled",
    },
    {
        "Trigger": "Oracle CDS spread >200bp for 15 days",
        "Threshold": ">200bp",
        "Last checked": "Apr 17 2026",
        "Status": "WATCH",
        "Note": "~180bp — approaching threshold",
    },
    {
        "Trigger": "2+ major AI productivity surveys report null/negative results",
        "Threshold": "2 surveys in 90d",
        "Last checked": "Apr 17 2026",
        "Status": "OK",
        "Note": "No new negative surveys",
    },
]

macro_df = pd.DataFrame(macro_switches)
macro_df["Status"] = macro_df["Status"].apply(lambda x: f"{_status_badge(x)} {x}")

def _color_status(val):
    if "ALERT" in val: return "color: #E24B4A; font-weight: 500"
    if "WATCH" in val: return "color: #BA7517; font-weight: 500"
    return "color: #1D9E75"

st.dataframe(
    macro_df.style.applymap(_color_status, subset=["Status"]),
    use_container_width=True, hide_index=True
)
st.markdown("---")


# ── Bucket status ──────────────────────────────────────────────────────────────
st.subheader("🪣 Bucket kill switch status — §10")
st.caption("Manual update required — based on order books, contract prices, and news.")

buckets = [
    {"Bucket":"Memory (MU, WDC, STX)",           "Kill Switch":"NAND/DRAM prices flat 2 qtrs",        "Status":"OK",    "Note":"NAND +12% QoQ"},
    {"Bucket":"Core Silicon (NVDA, AVGO, TSM)",   "Kill Switch":"NVDA top-4 >68% OR capex guides flat","Status":"WATCH", "Note":"GPU lead times falling — monitor"},
    {"Bucket":"Networking (ANET, ALAB, CLS)",     "Kill Switch":"Ethernet commitments scaled back",    "Status":"OK",    "Note":"ANET deferred revenue growing"},
    {"Bucket":"Power / Cooling (GEV, VRT, ETN)",  "Kill Switch":"DC cancellations >5 GW in 90d",       "Status":"OK",    "Note":"0 GW cancelled"},
    {"Bucket":"Nuclear (CCJ, OKLO)",               "Kill Switch":"SMR regulatory setback",              "Status":"OK",    "Note":"No new setbacks"},
    {"Bucket":"Neoclouds (CRWV, NBIS, APLD)",     "Kill Switch":"CDS spread widens >100bp in 30d",     "Status":"WATCH", "Note":"CRWV spreads elevated — watch"},
    {"Bucket":"Hyperscalers (MSFT, GOOGL, META)", "Kill Switch":"2 of 4 guide capex flat YoY",         "Status":"OK",    "Note":"Q2 calls pending late Apr"},
    {"Bucket":"Enterprise Software (PLTR, NOW)",  "Kill Switch":"NRR <110% for 2 consecutive qtrs",    "Status":"OK",    "Note":"Healthy NRR"},
    {"Bucket":"Cybersecurity (CRWD, PANW)",       "Kill Switch":"Major AI-driven sector re-rating",    "Status":"OK",    "Note":"No rerating events"},
    {"Bucket":"Physical AI (TSLA, ISRG, SYM)",    "Kill Switch":"Humanoid timeline slips >6 months",   "Status":"WATCH", "Note":"Optimus timeline uncertain"},
]

bdf = pd.DataFrame(buckets)
bdf["Status"] = bdf["Status"].apply(lambda x: f"{_status_badge(x)} {x}")

st.dataframe(
    bdf.style.applymap(_color_status, subset=["Status"]),
    use_container_width=True, hide_index=True
)

st.markdown("---")
st.caption("Update macro kill switches and bucket status manually after each major earnings call or news event. VIX rules update automatically.")
