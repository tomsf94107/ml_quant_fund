# ui/8_Dark_Pool.py
# Dark Pool & Options Skew Dashboard
# Shows institutional activity via Unusual Whales

import os, sys
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import sqlite3
import pandas as pd
import altair as alt
import streamlit as st
from datetime import date, timedelta
from pathlib import Path

DB_PATH = Path(_ROOT) / "accuracy.db"

st.set_page_config(page_title="Dark Pool & Options", page_icon="🌑", layout="wide")
st.title("🌑 Dark Pool & Options Skew")
st.caption("Institutional activity via Unusual Whales — dark pool ratio + 25-delta IV skew.")


def _load_meta() -> dict:
    try:
        mp = Path(_ROOT) / "tickers_metadata.csv"
        if mp.exists():
            df = pd.read_csv(mp)
            return df.set_index("ticker").to_dict("index")
    except Exception:
        pass
    return {}

_META = _load_meta()


@st.cache_data(ttl=300)
def load_dark_pool(days: int = 30) -> pd.DataFrame:
    if not DB_PATH.exists():
        return pd.DataFrame()
    try:
        cutoff = str(date.today() - timedelta(days=days))
        conn = sqlite3.connect(DB_PATH)
        df = pd.read_sql(
            "SELECT * FROM dark_pool_history WHERE date >= ? ORDER BY date DESC, dp_ratio DESC",
            conn, params=[cutoff]
        )
        conn.close()
        return df
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=300)
def load_skew(days: int = 30) -> pd.DataFrame:
    if not DB_PATH.exists():
        return pd.DataFrame()
    try:
        cutoff = str(date.today() - timedelta(days=days))
        conn = sqlite3.connect(DB_PATH)
        df = pd.read_sql(
            "SELECT * FROM options_skew_history WHERE date >= ? ORDER BY date DESC",
            conn, params=[cutoff]
        )
        conn.close()
        return df
    except Exception:
        return pd.DataFrame()


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Settings")
    days_back   = st.selectbox("History", [1, 7, 30], index=0,
                               format_func=lambda x: f"{x} day{'s' if x > 1 else ''}")
    dp_filter   = st.selectbox("Dark pool filter", ["All", "HIGH only", "NORMAL only", "LOW only"])
    skew_filter = st.selectbox("Skew filter", ["All", "BULLISH", "BEARISH", "NEUTRAL"])
    st.markdown("---")
    if st.button("🔄 Refresh cache"):
        st.cache_data.clear()
        st.rerun()


# ── Load data ─────────────────────────────────────────────────────────────────
dp_df   = load_dark_pool(days=days_back)
skew_df = load_skew(days=days_back)

if dp_df.empty:
    st.warning("No dark pool data yet. Run: uwsnapshot --date YYYY-MM-DD")
    st.stop()

latest_date = dp_df["date"].max()
dp_today    = dp_df[dp_df["date"] == latest_date].copy()
skew_today  = skew_df[skew_df["date"] == latest_date].copy() if not skew_df.empty else pd.DataFrame()

dp_today["bucket"] = dp_today["ticker"].map(lambda t: _META.get(t, {}).get("bucket", "—"))
dp_today["tier"]   = dp_today["ticker"].map(lambda t: _META.get(t, {}).get("tier", "—"))

if dp_filter != "All":
    sig_map = {"HIGH only": "HIGH", "NORMAL only": "NORMAL", "LOW only": "LOW"}
    dp_today = dp_today[dp_today["dp_signal"] == sig_map[dp_filter]]

if not skew_today.empty:
    merged = dp_today.merge(
        skew_today[["ticker", "skew_25d", "iv_rank", "skew_signal"]],
        on="ticker", how="left"
    )
    if skew_filter != "All":
        merged = merged[merged["skew_signal"].fillna("NEUTRAL") == skew_filter]
else:
    merged = dp_today.copy()
    merged["skew_25d"]    = None
    merged["iv_rank"]     = None
    merged["skew_signal"] = None


# ── KPIs ──────────────────────────────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
c1.metric("Tickers tracked",     len(dp_today))
c2.metric("Avg dark pool ratio", f"{dp_today['dp_ratio'].mean():.1%}")
c3.metric("High DP signals",     len(dp_today[dp_today["dp_signal"] == "HIGH"]))
if not skew_today.empty:
    c4.metric("Bearish skew count", len(skew_today[skew_today["skew_signal"] == "BEARISH"]))
else:
    c4.metric("Skew data", "Pending")

st.caption(f"Latest snapshot: {latest_date}")
st.markdown("---")


# ── Main table ────────────────────────────────────────────────────────────────
st.subheader("📋 Dark pool + options skew by ticker")

display = merged[["ticker","bucket","tier","dp_ratio","dp_signal","skew_25d","iv_rank","skew_signal"]].copy()
display["dp_ratio"] = display["dp_ratio"].apply(lambda x: f"{x:.1%}" if pd.notna(x) else "—")
display["skew_25d"] = display["skew_25d"].apply(lambda x: f"{x:+.3f}" if pd.notna(x) else "—")
display["iv_rank"]  = display["iv_rank"].apply(lambda x: f"{x:.0f}" if pd.notna(x) else "—")
display.columns     = ["Ticker","Bucket","Tier","DP Ratio","DP Signal","Skew 25d","IV Rank","Skew Signal"]

def _cdp(v):
    if v == "HIGH":    return "color: #E24B4A; font-weight: 500"
    if v == "LOW":     return "color: #3B6D11"
    return ""

def _csk(v):
    if v == "BEARISH": return "color: #E24B4A; font-weight: 500"
    if v == "BULLISH": return "color: #1D9E75; font-weight: 500"
    return ""

st.dataframe(
    display.style.applymap(_cdp, subset=["DP Signal"]).applymap(_csk, subset=["Skew Signal"]),
    use_container_width=True, hide_index=True,
)

st.markdown("---")


# ── Combined signal ───────────────────────────────────────────────────────────
st.subheader("🎯 Combined signal — high DP + bearish skew = potential distribution")

if "skew_signal" in merged.columns:
    combined = merged[
        (merged["dp_signal"] == "HIGH") & (merged["skew_signal"] == "BEARISH")
    ][["ticker","bucket","dp_ratio","skew_25d"]].copy()

    if not combined.empty:
        combined["dp_ratio"] = combined["dp_ratio"].apply(lambda x: f"{x:.1%}")
        combined["skew_25d"] = combined["skew_25d"].apply(lambda x: f"{x:+.3f}" if pd.notna(x) else "—")
        combined.columns = ["Ticker","Bucket","DP Ratio","Skew 25d"]
        st.dataframe(combined, use_container_width=True, hide_index=True)
    else:
        st.info("No tickers with both HIGH dark pool and BEARISH skew today.")

st.markdown("---")


# ── Trend chart ───────────────────────────────────────────────────────────────
st.subheader("📈 Dark pool ratio over time")
if days_back > 1 and not dp_df.empty:
    sel = st.selectbox("Ticker", sorted(dp_df["ticker"].unique()))
    trend = dp_df[dp_df["ticker"] == sel].sort_values("date")
    chart = (
        alt.Chart(trend).mark_line(point=True)
        .encode(
            x=alt.X("date:T", title="Date"),
            y=alt.Y("dp_ratio:Q", title="Dark Pool Ratio", axis=alt.Axis(format=".0%")),
            tooltip=["date:T", alt.Tooltip("dp_ratio:Q", format=".1%"), "dp_signal:N"],
        ).properties(height=250)
    )
    st.altair_chart(chart, use_container_width=True)
else:
    st.info("Select 7 or 30 days history to see trend.")
