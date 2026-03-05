# ui/3_Leaderboard.py
# Signal Leaderboard — ranks all tickers by backtest performance metrics.
# Replaces v2.5 pg3 Signal Leaderboard (which used MAE/MSE/R² and Google Sheets).

import os, sys
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import altair as alt
import streamlit as st

from features.builder import build_feature_dataframe
from signals.generator import generate_signals, DEFAULT_CONFIDENCE_THRESHOLD, DEFAULT_BLOCK_TAU

st.set_page_config(page_title="Signal Leaderboard", page_icon="🏆", layout="wide")
st.title("🏆 Signal Leaderboard")
st.caption("Ranks all tickers by backtest Sharpe ratio. Run Strategy to refresh.")

# ── Helpers ───────────────────────────────────────────────────────────────────
def _load_tickers() -> list[str]:
    path = Path(_ROOT) / "tickers.txt"
    if path.exists():
        return [t.strip().upper() for t in path.read_text().splitlines() if t.strip()]
    return ["AAPL", "NVDA", "TSLA", "AMD"]

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Settings")
    horizon              = st.selectbox("Horizon", [1, 3, 5], format_func=lambda x: f"{x}d")
    confidence_threshold = st.slider("Confidence threshold",
                                     0.50, 0.95, DEFAULT_CONFIDENCE_THRESHOLD, 0.01)
    block_tau            = st.slider("Block when risk_next_3d ≥", 0, 6, DEFAULT_BLOCK_TAU, 1)
    start_date           = st.date_input("Start", value=date(2022, 1, 1))
    end_date             = st.date_input("End",   value=date.today())

    tickers = st.multiselect(
        "Tickers",
        options=_load_tickers(),
        default=_load_tickers(),
    )

    run = st.button("🚀 Run Leaderboard", type="primary")

# ── Run ───────────────────────────────────────────────────────────────────────
if run:
    rows = []
    progress = st.progress(0, text="Building leaderboard...")

    for i, tkr in enumerate(tickers):
        progress.progress(i / len(tickers), text=f"Processing {tkr}...")
        try:
            df = build_feature_dataframe(
                tkr,
                start_date=start_date.isoformat(),
                end_date=end_date.isoformat(),
            )
            if df.empty:
                continue

            result = generate_signals(
                ticker=tkr,
                df=df,
                horizon=horizon,
                confidence_threshold=confidence_threshold,
                block_tau=block_tau,
            )

            if result.error:
                continue

            m = result.metrics
            rows.append({
                "ticker":         tkr,
                "signal":         result.today_signal,
                "prob":           result.today_prob_eff,
                "sharpe":         m.sharpe,
                "cagr":           m.cagr,
                "max_drawdown":   m.max_drawdown,
                "accuracy":       m.accuracy,
                "n_trades":       m.n_trades,
                "exposure":       m.exposure,
                "profit_factor":  m.profit_factor,
            })
        except Exception as e:
            st.warning(f"⚠️ {tkr}: {e}")

    progress.progress(1.0, text="Done.")

    if not rows:
        st.error("No results. Check tickers and date range.")
        st.stop()

    lb = pd.DataFrame(rows).sort_values("sharpe", ascending=False).reset_index(drop=True)
    lb.index += 1   # rank starts at 1
    st.session_state["leaderboard"] = lb

# ── Display ───────────────────────────────────────────────────────────────────
lb = st.session_state.get("leaderboard")

if lb is None:
    st.info("Click **Run Leaderboard** in the sidebar to generate rankings.")
    st.stop()

# ── Top 3 podium ──────────────────────────────────────────────────────────────
st.subheader("🥇 Top Performers")
top3 = lb.head(3)
cols = st.columns(3)
medals = ["🥇", "🥈", "🥉"]
for col, (_, row), medal in zip(cols, top3.iterrows(), medals):
    signal_color = "🟢" if row["signal"] == "BUY" else "🔴"
    col.metric(
        label=f"{medal} {row['ticker']}",
        value=f"Sharpe {row['sharpe']:.2f}",
        delta=f"{signal_color} {row['signal']}  p={row['prob']:.1%}",
    )
    col.caption(
        f"CAGR {row['cagr']:.1%} · "
        f"MaxDD {row['max_drawdown']:.1%} · "
        f"Acc {row['accuracy']:.1%}"
    )

st.divider()

# ── Full leaderboard table ────────────────────────────────────────────────────
st.subheader("📋 Full Rankings")

def _color_signal(val):
    return "color: #00c853" if val == "BUY" else "color: #ff1744"

def _color_sharpe(val):
    if pd.isna(val): return ""
    if val >= 1.5:  return "color: #00c853"
    if val >= 0.5:  return "color: #ffab00"
    return "color: #ff1744"

styled = (
    lb.reset_index()[["index", "ticker", "signal", "prob", "sharpe",
                       "cagr", "max_drawdown", "accuracy",
                       "n_trades", "profit_factor"]]
    .rename(columns={"index": "rank"})
    .style
    .format({
        "prob":          "{:.1%}",
        "sharpe":        "{:.2f}",
        "cagr":          "{:.1%}",
        "max_drawdown":  "{:.1%}",
        "accuracy":      "{:.1%}",
        "profit_factor": "{:.2f}",
    })
    .applymap(_color_signal,  subset=["signal"])
    .applymap(_color_sharpe,  subset=["sharpe"])
)
st.dataframe(styled, use_container_width=True, hide_index=True)

# ── Scatter: Sharpe vs CAGR ───────────────────────────────────────────────────
st.subheader("📊 Sharpe vs CAGR")

scatter = (
    alt.Chart(lb.reset_index())
    .mark_circle(size=120)
    .encode(
        x=alt.X("sharpe:Q", title="Sharpe Ratio"),
        y=alt.Y("cagr:Q",   title="CAGR", axis=alt.Axis(format=".0%")),
        color=alt.Color(
            "signal:N",
            scale=alt.Scale(
                domain=["BUY", "HOLD"],
                range=["#00c853", "#ff1744"],
            ),
        ),
        size=alt.Size("n_trades:Q", legend=None),
        tooltip=[
            "ticker",
            alt.Tooltip("sharpe:Q",       format=".2f"),
            alt.Tooltip("cagr:Q",         format=".1%"),
            alt.Tooltip("max_drawdown:Q", format=".1%"),
            alt.Tooltip("accuracy:Q",     format=".1%"),
            "signal",
            alt.Tooltip("prob:Q",         format=".1%"),
        ],
        text="ticker:N",
    )
    .properties(height=400, title="Green = BUY signal today · Size = number of trades")
)

labels = scatter.mark_text(align="left", dx=8, fontSize=11).encode(text="ticker:N")
st.altair_chart((scatter + labels).interactive(), use_container_width=True)

# ── Bar: Max Drawdown ─────────────────────────────────────────────────────────
st.subheader("🛡️ Max Drawdown by Ticker")
dd_chart = (
    alt.Chart(lb.reset_index().sort_values("max_drawdown"))
    .mark_bar()
    .encode(
        x=alt.X("ticker:N", sort=None),
        y=alt.Y("max_drawdown:Q", title="Max Drawdown",
                axis=alt.Axis(format=".0%")),
        color=alt.Color(
            "max_drawdown:Q",
            scale=alt.Scale(scheme="redyellowgreen", reverse=True),
            legend=None,
        ),
        tooltip=["ticker", alt.Tooltip("max_drawdown:Q", format=".1%")],
    )
    .properties(height=300)
)
st.altair_chart(dd_chart, use_container_width=True)

# ── Download ──────────────────────────────────────────────────────────────────
csv = lb.reset_index().to_csv(index=False).encode()
st.download_button(
    "⬇️ Download leaderboard CSV",
    csv,
    file_name="leaderboard.csv",
    mime="text/csv",
    key="dl_lb",
)
