# ui/2_Accuracy.py
# Forecast Accuracy Dashboard — reads from accuracy/sink.py
# Shows real prediction vs outcome accuracy, not training metrics.

import os, sys
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import pandas as pd
import altair as alt
import streamlit as st
from datetime import date, timedelta

import os as _os
def _load_accuracy_meta() -> dict:
    try:
        import pandas as _pm
        _mp = _os.path.join(_ROOT, "tickers_metadata.csv")
        if _os.path.exists(_mp):
            _df = _pm.read_csv(_mp)
            return _df.set_index("ticker").to_dict("index")
    except Exception:
        pass
    return {}

from accuracy.sink import (
    load_accuracy, load_prediction_history,
    reconcile_outcomes, update_accuracy_cache,
    get_spy_relative_accuracy, get_eod_accuracy_summary,
)

st.set_page_config(page_title="Forecast Accuracy", page_icon="🎯", layout="wide")
_TICKER_META = _load_accuracy_meta()
st.title("🎯 Forecast Accuracy Dashboard")
st.caption("Live prediction accuracy — how often the model was actually right after the fact.")

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Settings")
    horizon     = st.selectbox("Horizon", [1, 3, 5], format_func=lambda x: f"{x}d")
    window_days = st.selectbox("Rolling window", [30, 60, 90, 180], index=2,
                               format_func=lambda x: f"{x}d")

    st.markdown("---")
    if st.button("🔁 Reconcile outcomes"):
        with st.spinner("Fetching actual outcomes from yfinance..."):
            try:
                n = reconcile_outcomes()
                st.success(f"✓ {n} new outcomes recorded")
            except Exception as e:
                st.error(f"Failed: {e}")

    if st.button("♻️ Recompute accuracy cache"):
        with st.spinner("Computing metrics..."):
            try:
                df = update_accuracy_cache(window_days=window_days)
                st.success(f"✓ Cache updated — {len(df)} ticker×horizon rows")
            except Exception as e:
                st.error(f"Failed: {e}")

    if st.button("🔄 Clear cache"):
        st.cache_data.clear()
        st.rerun()

# ── Load data ─────────────────────────────────────────────────────────────────
@st.cache_data(ttl=300)
def _load(horizon, window_days):
    return load_accuracy(horizon=horizon, window_days=window_days)

# Auto-reconcile on page load
with st.spinner("Reconciling outcomes..."):
    try:
        n = reconcile_outcomes()
        if n > 0:
            st.success(f"✓ {n} new outcomes reconciled")
    except Exception:
        pass

acc_df = _load(horizon, window_days)

# ── SPY-relative accuracy (always show) ──────────────────────────────────────
st.subheader("📊 Daily Performance vs SPY")
st.caption("Are our tickers outperforming the market? More meaningful than BUY accuracy with small sample size.")
try:
    spy_results = get_spy_relative_accuracy()
    if spy_results:
        sdf = pd.DataFrame(spy_results)
        sdf["spy_ret"]       = sdf["spy_ret"].apply(lambda x: f"{x:+.2%}" if x == x else "N/A")
        sdf["avg_ret"]       = sdf["avg_ret"].apply(lambda x: f"{x:+.2%}")
        sdf["avg_vs_spy"]    = sdf["avg_vs_spy"].apply(lambda x: f"{x:+.2%}" if x == x else "N/A")
        sdf["pct_beat_spy"]  = sdf["pct_beat_spy"].apply(lambda x: f"{x:.0%}" if x == x else "N/A")
        sdf["buy_acc"]       = sdf["buy_acc"].apply(lambda x: f"{x:.0%}" if x is not None else "—")
        sdf.columns = ["Date","SPY Return","Avg Ticker Return","Avg vs SPY","% Beat SPY","# BUYs","BUY Acc"]
        st.dataframe(sdf, use_container_width=True, hide_index=True)

        valid = [r for r in spy_results if r["avg_vs_spy"] == r["avg_vs_spy"]]
        if valid:
            avg_vs_spy = sum(r["avg_vs_spy"] for r in valid) / len(valid)
            avg_beat   = sum(r["pct_beat_spy"] for r in valid) / len(valid)
            col1, col2 = st.columns(2)
            col1.metric("Avg daily alpha vs SPY", f"{avg_vs_spy:+.2%}")
            col2.metric("% of tickers beating SPY", f"{avg_beat:.0%}")
except Exception as e:
    st.warning(f"SPY comparison unavailable: {e}")

st.markdown("---")

# ── BUY/SELL signal accuracy ──────────────────────────────────────────────────
st.subheader("🎯 BUY/SELL Signal Accuracy")
st.caption("Only meaningful after 60+ BUY/SELL signals across different market conditions.")
try:
    eod_acc = get_eod_accuracy_summary()
    if eod_acc:
        edf = pd.DataFrame(eod_acc)
        edf["accuracy"]   = edf["accuracy"].apply(lambda x: f"{x:.1%}" if x is not None else "N/A")
        edf["avg_return"] = edf["avg_return"].apply(lambda x: f"{x:+.2%}" if x is not None else "N/A")
        edf.columns = ["Ticker", "# Outcomes", "Accuracy", "Avg Return"]
        st.dataframe(edf, use_container_width=True, hide_index=True)
        valid = [r for r in eod_acc if r["accuracy"] is not None]
        total_buys = sum(r["n"] for r in valid)
        if valid:
            avg = sum(r["accuracy"] for r in valid) / len(valid)
            st.caption(f"Overall: {avg:.1%} across {total_buys} BUY/SELL signals · Need 60+ for statistical significance")
    else:
        st.info("No BUY/SELL signals recorded yet.")
except Exception as e:
    st.warning(f"BUY/SELL accuracy unavailable: {e}")

st.markdown("---")
st.subheader("📈 Model Accuracy Cache")
if acc_df.empty:
    st.info("Run **Reconcile outcomes** then **Recompute accuracy cache** in the sidebar to populate this section.")
    st.stop()

# ── KPI row ───────────────────────────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
c1.metric("Tickers tracked",  len(acc_df))
c2.metric("Avg Accuracy",     f"{acc_df['accuracy'].mean():.1%}")
c3.metric("Avg ROC-AUC",      f"{acc_df['roc_auc'].mean():.3f}")
c4.metric("Avg Brier Score",  f"{acc_df['brier_score'].mean():.3f}")

st.caption(
    "**Brier score**: lower is better (0 = perfect, 0.25 = random). "
    "**ROC-AUC**: higher is better (0.5 = random, 1.0 = perfect)."
)

# ── Accuracy by Bucket ───────────────────────────────────────────────────────
st.subheader("🪣 Accuracy by Bucket")
if not acc_df.empty and _TICKER_META:
    acc_df["bucket"] = acc_df["ticker"].map(lambda t: _TICKER_META.get(t, {}).get("bucket", "Unknown"))
    acc_df["tier"]   = acc_df["ticker"].map(lambda t: _TICKER_META.get(t, {}).get("tier", "Unknown"))

    bucket_acc = (
        acc_df.groupby("bucket")
        .agg(avg_accuracy=("accuracy","mean"), n_tickers=("ticker","count"))
        .reset_index()
        .sort_values("avg_accuracy", ascending=False)
    )
    tier_acc = (
        acc_df.groupby("tier")
        .agg(avg_accuracy=("accuracy","mean"), n_tickers=("ticker","count"))
        .reset_index()
        .sort_values("avg_accuracy", ascending=False)
    )

    col_b, col_t = st.columns(2)
    with col_b:
        st.caption("By Bucket")
        st.dataframe(
            bucket_acc.style.format({"avg_accuracy": "{:.1%}"}),
            use_container_width=True, hide_index=True
        )
    with col_t:
        st.caption("By Tier")
        st.dataframe(
            tier_acc.style.format({"avg_accuracy": "{:.1%}"}),
            use_container_width=True, hide_index=True
        )
else:
    st.info("Add tickers_metadata.csv to see bucket-level accuracy.")

st.markdown("---")

# ── Leaderboard table ─────────────────────────────────────────────────────────
st.subheader("📋 Accuracy by Ticker")

def _color_accuracy(val):
    if pd.isna(val): return ""
    if val >= 0.60:  return "color: #00c853"
    if val >= 0.55:  return "color: #ffab00"
    return "color: #ff1744"

styled = (
    acc_df[["ticker", "bucket", "tier", "accuracy", "roc_auc", "brier_score", "n_predictions"]]
    .style
    .format({
        "accuracy":    "{:.1%}",
        "roc_auc":     "{:.3f}",
        "brier_score": "{:.3f}",
    })
    .applymap(_color_accuracy, subset=["accuracy"])
)
st.dataframe(styled, use_container_width=True, hide_index=True)

# ── Bar chart ─────────────────────────────────────────────────────────────────
st.subheader("📊 Accuracy Comparison")
metric_choice = st.selectbox("Metric", ["accuracy", "roc_auc", "brier_score"])

chart = (
    alt.Chart(acc_df.sort_values(metric_choice, ascending=(metric_choice == "brier_score")))
    .mark_bar()
    .encode(
        x=alt.X("ticker:N", sort=None, title="Ticker"),
        y=alt.Y(f"{metric_choice}:Q", title=metric_choice.replace("_", " ").title()),
        color=alt.Color(
            f"{metric_choice}:Q",
            scale=alt.Scale(
                scheme="redyellowgreen" if metric_choice != "brier_score" else "redyellowgreen",
                reverse=(metric_choice == "brier_score"),
            ),
            legend=None,
        ),
        tooltip=["ticker", "accuracy", "roc_auc", "brier_score", "n_predictions"],
    )
    .properties(height=350)
)
st.altair_chart(chart, use_container_width=True)

# ── Trend chart for selected ticker ───────────────────────────────────────────
st.subheader("📈 Accuracy Over Time")
sel_ticker = st.selectbox("Select ticker", acc_df["ticker"].tolist())
days_back  = st.slider("Days back", 30, 180, 90, step=10)

hist = load_prediction_history(sel_ticker, horizon=horizon, days=days_back)

if hist.empty:
    st.info(f"No prediction history for {sel_ticker} yet.")
else:
    hist["rolling_acc"] = hist["correct"].rolling(20, min_periods=5).mean()

    base = alt.Chart(hist).encode(
        x=alt.X("prediction_date:T", title="Date")
    )

    acc_line = base.mark_line(color="#00c853").encode(
        y=alt.Y("rolling_acc:Q", title="Rolling 20-day Accuracy",
                scale=alt.Scale(domain=[0, 1])),
        tooltip=["prediction_date:T",
                 alt.Tooltip("rolling_acc:Q", format=".1%"),
                 "signal:N", "actual_up:Q"],
    )

    signal_pts = base.mark_circle(size=30).encode(
        y=alt.Y("prob_up:Q", title="Predicted Prob"),
        color=alt.condition(
            "datum.actual_up == 1",
            alt.value("#00c853"),
            alt.value("#ff1744"),
        ),
        tooltip=["prediction_date:T",
                 alt.Tooltip("prob_up:Q", format=".1%"),
                 "signal:N", "actual_up:Q",
                 alt.Tooltip("actual_return:Q", format=".2%")],
    )

    st.altair_chart(
        alt.layer(acc_line, signal_pts)
        .resolve_scale(y="independent")
        .properties(
            title=f"{sel_ticker} — Predicted Probability vs Actual Outcome",
            height=350,
        ),
        use_container_width=True,
    )

    # Raw table
    with st.expander("🧾 Raw prediction history"):
        display = hist[["prediction_date", "prob_up", "signal",
                        "confidence", "actual_up", "actual_return", "correct"]].copy()
        display["predicted"] = display["prob_up"].apply(lambda x: "⬆️ UP" if x > 0.5 else "⬇️ DOWN")
        display["actual"]    = display["actual_up"].apply(lambda x: "⬆️ UP" if x == 1 else "⬇️ DOWN")
        display["correct"]   = display["correct"].apply(lambda x: "✅" if x == 1 else "❌")
        display = display[["prediction_date", "prob_up", "predicted", "actual",
                           "actual_return", "correct", "signal", "confidence"]]
        display.columns = ["Date", "Prob Up", "Predicted", "Actual",
                          "Return", "Correct", "Signal", "Confidence"]
        st.dataframe(
            display.style.format({
                "Prob Up": "{:.1%}",
                "Return":  "{:.2%}",
            }),
            use_container_width=True,
            hide_index=True,
        )

    csv = hist.to_csv(index=False).encode()
    st.download_button(
        f"⬇️ Download {sel_ticker} history",
        csv,
        file_name=f"{sel_ticker}_accuracy_{horizon}d.csv",
        mime="text/csv",
        key="dl_acc",
    )
