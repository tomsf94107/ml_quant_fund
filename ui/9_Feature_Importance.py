# ui/9_Feature_Importance.py
# Feature Importance Dashboard
# Shows which features matter most across retrains

import os, sys
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import sqlite3
import pandas as pd
import altair as alt
import streamlit as st
from datetime import date, timedelta
from pathlib import Path

DB_PATH = Path(_ROOT) / "accuracy.db"

st.set_page_config(page_title="Feature Importance", page_icon="📊", layout="wide")
st.title("📊 Feature Importance")
st.caption("Which of the 79 features matter most — tracked across every retrain.")


@st.cache_data(ttl=300)
def load_importance(horizon: int = 1, days: int = 30) -> pd.DataFrame:
    if not DB_PATH.exists():
        return pd.DataFrame()
    try:
        cutoff = str(date.today() - timedelta(days=days))
        conn = sqlite3.connect(DB_PATH)
        df = pd.read_sql(
            """SELECT * FROM feature_importance_history
               WHERE horizon=? AND retrain_date>=?
               ORDER BY retrain_date DESC, importance DESC""",
            conn, params=[horizon, cutoff]
        )
        conn.close()
        return df
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=300)
def get_top_features(horizon: int = 1, days: int = 30, top_n: int = 20) -> pd.DataFrame:
    df = load_importance(horizon, days)
    if df.empty:
        return pd.DataFrame()
    return (
        df.groupby("feature")["importance"]
        .agg(avg_importance="mean", n_observations="count")
        .reset_index()
        .sort_values("avg_importance", ascending=False)
        .head(top_n)
    )


@st.cache_data(ttl=300)
def get_trending(horizon: int = 1, days: int = 60) -> tuple:
    df = load_importance(horizon, days)
    if df.empty:
        return pd.DataFrame(), pd.DataFrame()

    df["retrain_date"] = pd.to_datetime(df["retrain_date"])
    mid = df["retrain_date"].min() + (df["retrain_date"].max() - df["retrain_date"].min()) / 2

    early  = df[df["retrain_date"] <= mid].groupby("feature")["importance"].mean()
    recent = df[df["retrain_date"] > mid].groupby("feature")["importance"].mean()

    comp = pd.DataFrame({"early": early, "recent": recent}).dropna()
    if comp.empty:
        return pd.DataFrame(), pd.DataFrame()

    comp["change_pct"] = (comp["recent"] - comp["early"]) / comp["early"].replace(0, 1e-10) * 100

    rising   = comp[comp["change_pct"] > 20].sort_values("change_pct", ascending=False).reset_index()
    declining= comp[comp["change_pct"] < -20].sort_values("change_pct").reset_index()
    return rising, declining


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Settings")
    horizon  = st.selectbox("Horizon", [1, 3, 5], format_func=lambda x: f"{x}d")
    days_back= st.selectbox("Period", [30, 60, 90], format_func=lambda x: f"{x} days")
    top_n    = st.slider("Top N features", 10, 30, 20)
    st.markdown("---")
    if st.button("🔄 Refresh"):
        st.cache_data.clear()
        st.rerun()


# ── Load ──────────────────────────────────────────────────────────────────────
top_df             = get_top_features(horizon, days_back, top_n)
rising, declining  = get_trending(horizon, days_back)
raw_df             = load_importance(horizon, days_back)

if top_df.empty:
    st.warning("No feature importance data yet. Run retrain first — data populates automatically after each retrain.")
    st.stop()

# ── KPIs ──────────────────────────────────────────────────────────────────────
retrains = raw_df["retrain_date"].nunique() if not raw_df.empty else 0
c1, c2, c3, c4 = st.columns(4)
c1.metric("Total features", len(top_df))
c2.metric("Retrains tracked", retrains)
c3.metric("Rising features",  len(rising)   if not rising.empty   else 0)
c4.metric("Declining features", len(declining) if not declining.empty else 0)
st.markdown("---")


# ── Top features bar chart ────────────────────────────────────────────────────
st.subheader(f"Top {top_n} features by avg importance — {horizon}d horizon")

chart = (
    alt.Chart(top_df)
    .mark_bar()
    .encode(
        x=alt.X("avg_importance:Q", title="Avg importance (gain)"),
        y=alt.Y("feature:N", sort="-x", title="Feature"),
        color=alt.Color("avg_importance:Q",
                        scale=alt.Scale(scheme="blues"),
                        legend=None),
        tooltip=["feature:N",
                 alt.Tooltip("avg_importance:Q", format=".4f"),
                 "n_observations:Q"],
    )
    .properties(height=max(300, top_n * 20))
)
st.altair_chart(chart, use_container_width=True)
st.markdown("---")


# ── Rising / Declining ────────────────────────────────────────────────────────
col_r, col_d = st.columns(2)

with col_r:
    st.subheader("📈 Rising features")
    st.caption(f"Gaining >20% importance in last {days_back} days")
    if not rising.empty:
        rising_display = rising[["feature","change_pct","recent","early"]].copy()
        rising_display["change_pct"] = rising_display["change_pct"].apply(lambda x: f"+{x:.0f}%")
        rising_display["recent"]     = rising_display["recent"].apply(lambda x: f"{x:.4f}")
        rising_display["early"]      = rising_display["early"].apply(lambda x: f"{x:.4f}")
        rising_display.columns = ["Feature","Change","Recent Imp","Early Imp"]
        st.dataframe(rising_display, use_container_width=True, hide_index=True)
    else:
        st.info("Not enough retrain history yet.")

with col_d:
    st.subheader("📉 Declining features")
    st.caption(f"Losing >20% importance in last {days_back} days")
    if not declining.empty:
        declining_display = declining[["feature","change_pct","recent","early"]].copy()
        declining_display["change_pct"] = declining_display["change_pct"].apply(lambda x: f"{x:.0f}%")
        declining_display["recent"]     = declining_display["recent"].apply(lambda x: f"{x:.4f}")
        declining_display["early"]      = declining_display["early"].apply(lambda x: f"{x:.4f}")
        declining_display.columns = ["Feature","Change","Recent Imp","Early Imp"]
        st.dataframe(declining_display, use_container_width=True, hide_index=True)
    else:
        st.info("No significantly declining features.")

st.markdown("---")


# ── Feature over time ─────────────────────────────────────────────────────────
st.subheader("📈 Feature importance over time")
if not raw_df.empty:
    all_features = sorted(raw_df["feature"].unique())
    sel_features = st.multiselect("Select features", all_features,
                                  default=all_features[:5] if len(all_features) >= 5 else all_features)

    if sel_features:
        trend_df = raw_df[raw_df["feature"].isin(sel_features)].copy()
        trend_df["retrain_date"] = pd.to_datetime(trend_df["retrain_date"])

        chart2 = (
            alt.Chart(trend_df)
            .mark_line(point=True)
            .encode(
                x=alt.X("retrain_date:T", title="Retrain date"),
                y=alt.Y("importance:Q",   title="Importance (gain)"),
                color=alt.Color("feature:N", title="Feature"),
                tooltip=["retrain_date:T","feature:N",
                         alt.Tooltip("importance:Q", format=".4f")],
            )
            .properties(height=300)
        )
        st.altair_chart(chart2, use_container_width=True)
