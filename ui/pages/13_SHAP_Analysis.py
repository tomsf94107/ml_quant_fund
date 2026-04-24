# ui/pages/13_SHAP_Analysis.py
# SHAP feature importance analysis — shows which features actually drive predictions
# vs gain-based importance which can be misleading

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

st.set_page_config(page_title="SHAP Analysis", page_icon="🔬", layout="wide")
st.title("🔬 SHAP Feature Analysis")
st.caption("True feature importance using SHAP values — more accurate than XGBoost gain-based ranking.")


@st.cache_data(ttl=300)
def load_shap_data(horizon: int = 1, days: int = 30) -> pd.DataFrame:
    if not DB_PATH.exists():
        return pd.DataFrame()
    try:
        cutoff = str(date.today() - timedelta(days=days))
        conn = sqlite3.connect(DB_PATH, timeout=30)
        df = pd.read_sql("""
            SELECT ticker, horizon, feature, importance, shap_importance,
                   rank, retrain_date
            FROM feature_importance_history
            WHERE horizon=? AND retrain_date>=?
            ORDER BY retrain_date DESC, ticker, rank
        """, conn, params=[horizon, cutoff])
        conn.close()
        return df
    except Exception as e:
        st.error(f"DB error: {e}")
        return pd.DataFrame()


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Settings")
    horizon   = st.selectbox("Horizon", [1, 3, 5], format_func=lambda x: f"{x}-day")
    days_back = st.selectbox("History", [7, 30, 90], index=1,
                              format_func=lambda x: f"{x} days")
    view_mode = st.radio("View", ["Top features (all tickers)",
                                   "Compare SHAP vs Gain",
                                   "Per-ticker breakdown",
                                   "Feature trend"])
    st.markdown("---")
    if st.button("🔄 Refresh cache"):
        st.cache_data.clear()
        st.rerun()


df = load_shap_data(horizon=horizon, days=days_back)

if df.empty:
    st.warning(f"No SHAP data for {horizon}-day horizon in last {days_back} days. Run retrain to populate.")
    st.stop()

latest_date = df["retrain_date"].max()
latest      = df[df["retrain_date"] == latest_date].copy()

# ── KPIs ──────────────────────────────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
c1.metric("Tickers analyzed", latest["ticker"].nunique())
c2.metric("Features tracked", latest["feature"].nunique())
c3.metric("Latest retrain",   latest_date)
c4.metric("SHAP populated",
          f"{(latest['shap_importance'] > 0).sum()}/{len(latest)}")

st.markdown("---")


# ════════════════════════════════════════
# VIEW 1 — Top features across all tickers
# ════════════════════════════════════════
if view_mode == "Top features (all tickers)":
    st.subheader("Top features — averaged across all tickers")

    agg = latest.groupby("feature").agg(
        avg_shap=("shap_importance", "mean"),
        avg_gain=("importance", "mean"),
        n_tickers=("ticker", "nunique"),
    ).reset_index()
    agg = agg.sort_values("avg_shap", ascending=False).head(25)

    col_a, col_b = st.columns([1, 1])

    with col_a:
        st.markdown("**Top 25 by SHAP (true impact)**")
        chart = (
            alt.Chart(agg)
            .mark_bar()
            .encode(
                x=alt.X("avg_shap:Q", title="Mean |SHAP|"),
                y=alt.Y("feature:N", sort="-x", title=""),
                tooltip=["feature", "avg_shap", "avg_gain", "n_tickers"],
                color=alt.value("#378ADD"),
            )
            .properties(height=500)
        )
        st.altair_chart(chart, use_container_width=True)

    with col_b:
        st.markdown("**Same features — by gain (comparison)**")
        chart2 = (
            alt.Chart(agg)
            .mark_bar()
            .encode(
                x=alt.X("avg_gain:Q", title="Mean gain"),
                y=alt.Y("feature:N", sort="-color", title=""),
                tooltip=["feature", "avg_shap", "avg_gain", "n_tickers"],
                color=alt.value("#BA7517"),
            )
            .properties(height=500)
        )
        st.altair_chart(chart2, use_container_width=True)

    st.markdown("---")
    st.subheader("Detailed table")
    display = agg.copy()
    display["avg_shap"] = display["avg_shap"].round(4)
    display["avg_gain"] = display["avg_gain"].round(2)
    display.columns = ["Feature", "Avg SHAP", "Avg Gain", "Tickers"]
    st.dataframe(display, use_container_width=True, hide_index=True)


# ════════════════════════════════════════
# VIEW 2 — Compare SHAP vs Gain disagreement
# ════════════════════════════════════════
elif view_mode == "Compare SHAP vs Gain":
    st.subheader("SHAP vs Gain — which features are misranked?")
    st.caption("Features where SHAP and Gain disagree significantly. Gain often overweights frequently-split features that aren't actually predictive.")

    agg = latest.groupby("feature").agg(
        avg_shap=("shap_importance", "mean"),
        avg_gain=("importance", "mean"),
    ).reset_index()

    # Normalize both to 0-1 for comparison
    if agg["avg_shap"].max() > 0:
        agg["shap_norm"] = agg["avg_shap"] / agg["avg_shap"].max()
    else:
        agg["shap_norm"] = 0
    if agg["avg_gain"].max() > 0:
        agg["gain_norm"] = agg["avg_gain"] / agg["avg_gain"].max()
    else:
        agg["gain_norm"] = 0

    # Disagreement = absolute difference
    agg["disagreement"] = (agg["gain_norm"] - agg["shap_norm"]).round(3)
    agg["verdict"] = agg["disagreement"].apply(
        lambda d: "⚠️ Gain overrates" if d > 0.2 else
                  "📈 SHAP undervalued" if d < -0.2 else
                  "✅ Agree"
    )

    # Show biggest disagreements
    top_disagree = agg.reindex(agg["disagreement"].abs().sort_values(ascending=False).index).head(20)

    display = top_disagree[["feature", "avg_shap", "avg_gain", "shap_norm", "gain_norm", "verdict"]].copy()
    display.columns = ["Feature", "SHAP", "Gain", "SHAP (norm)", "Gain (norm)", "Verdict"]
    display["SHAP"] = display["SHAP"].round(4)
    display["Gain"] = display["Gain"].round(2)
    display["SHAP (norm)"] = display["SHAP (norm)"].round(3)
    display["Gain (norm)"] = display["Gain (norm)"].round(3)

    def _color_verdict(v):
        if "overrates" in v: return "color: #E24B4A; font-weight: 500"
        if "undervalued" in v: return "color: #1D9E75; font-weight: 500"
        return ""

    st.dataframe(
        display.style.applymap(_color_verdict, subset=["Verdict"]),
        use_container_width=True, hide_index=True,
    )

    st.markdown("---")
    st.info("""
    **How to read:**
    - **Gain overrates** — XGBoost uses this feature in many splits but SHAP says it doesn't actually drive predictions. Candidate to drop.
    - **SHAP undervalued** — Feature rarely used in splits but when used, it's highly predictive. Keep.
    - **Agree** — Both methods rank it similarly.
    """)


# ════════════════════════════════════════
# VIEW 3 — Per-ticker breakdown
# ════════════════════════════════════════
elif view_mode == "Per-ticker breakdown":
    st.subheader("Per-ticker SHAP analysis")
    tickers = sorted(latest["ticker"].unique())
    ticker  = st.selectbox("Ticker", tickers)

    ticker_df = latest[latest["ticker"] == ticker].sort_values("shap_importance", ascending=False).head(20)

    if ticker_df.empty:
        st.warning(f"No data for {ticker}")
        st.stop()

    st.markdown(f"**Top 20 features for {ticker} ({horizon}-day prediction)**")

    chart = (
        alt.Chart(ticker_df)
        .mark_bar()
        .encode(
            x=alt.X("shap_importance:Q", title="SHAP importance"),
            y=alt.Y("feature:N", sort="-x", title=""),
            tooltip=["feature", "shap_importance", "importance", "rank"],
            color=alt.value("#378ADD"),
        )
        .properties(height=500)
    )
    st.altair_chart(chart, use_container_width=True)

    display = ticker_df[["feature", "shap_importance", "importance", "rank"]].copy()
    display.columns = ["Feature", "SHAP", "Gain", "Gain Rank"]
    display["SHAP"] = display["SHAP"].round(4)
    display["Gain"] = display["Gain"].round(2)
    st.dataframe(display, use_container_width=True, hide_index=True)


# ════════════════════════════════════════
# VIEW 4 — Feature trend over time
# ════════════════════════════════════════
elif view_mode == "Feature trend":
    st.subheader("Feature importance trend over time")

    features = sorted(df["feature"].unique())
    default_features = ["vwap", "obv", "atr", "yield_10y"][:4]
    selected_features = st.multiselect(
        "Features to track",
        features,
        default=[f for f in default_features if f in features],
    )

    if not selected_features:
        st.info("Select at least one feature to plot")
        st.stop()

    trend_df = df[df["feature"].isin(selected_features)].copy()
    trend_agg = trend_df.groupby(["retrain_date", "feature"]).agg(
        avg_shap=("shap_importance", "mean")
    ).reset_index()
    trend_agg["retrain_date"] = pd.to_datetime(trend_agg["retrain_date"])

    chart = (
        alt.Chart(trend_agg)
        .mark_line(point=True)
        .encode(
            x=alt.X("retrain_date:T", title="Retrain date"),
            y=alt.Y("avg_shap:Q", title="Avg SHAP importance"),
            color=alt.Color("feature:N", title="Feature"),
            tooltip=["retrain_date:T", "feature", "avg_shap"],
        )
        .properties(height=400)
    )
    st.altair_chart(chart, use_container_width=True)

    st.info("Rising SHAP = feature gaining importance. Declining = feature becoming less useful — candidate to drop.")
