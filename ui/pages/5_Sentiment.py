# ui/5_Sentiment.py
# Sentiment control panel — run FinBERT on selected tickers, view scores.
# Manual override for breaking news. Shows last update time per ticker.

import os, sys
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import sqlite3
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd
import altair as alt
import streamlit as st

from data.etl_sentiment import (
    run_sentiment_etl, load_sentiment_scores,
    DB_PATH, DEFAULT_SOURCES, SOURCES_WITH_KEYS, _current_time_slot,
)

st.set_page_config(page_title="Sentiment", page_icon="📰", layout="wide")
st.title("📰 Sentiment Control Panel")
st.caption("FinBERT scoring across 10 news sources. Select tickers and run on demand or on schedule.")

# ── Helpers ───────────────────────────────────────────────────────────────────
def _load_tickers() -> list[str]:
    path = Path(_ROOT) / "tickers.txt"
    if path.exists():
        return [t.strip().upper() for t in path.read_text().splitlines() if t.strip()]
    return ["AAPL", "NVDA", "TSLA", "AMD"]


def _load_latest_scores() -> pd.DataFrame:
    """Load the most recent sentiment score per ticker from the DB."""
    if not DB_PATH.exists():
        return pd.DataFrame()
    try:
        conn = sqlite3.connect(DB_PATH)
        df = pd.read_sql("""
            SELECT ticker, date, time_slot, score,
                   positive_pct, negative_pct, neutral_pct,
                   n_headlines, sources_used, created_at
            FROM sentiment_scores
            WHERE id IN (
                SELECT MAX(id) FROM sentiment_scores GROUP BY ticker
            )
            ORDER BY score DESC
        """, conn)
        conn.close()
        return df
    except Exception:
        return pd.DataFrame()


def _load_history(ticker: str, days: int = 14) -> pd.DataFrame:
    cutoff = (date.today() - timedelta(days=days)).isoformat()
    if not DB_PATH.exists():
        return pd.DataFrame()
    try:
        conn = sqlite3.connect(DB_PATH)
        df = pd.read_sql("""
            SELECT date, time_slot, score, positive_pct, negative_pct, n_headlines
            FROM sentiment_scores
            WHERE ticker = ? AND date >= ?
            ORDER BY date, time_slot
        """, conn, params=(ticker.upper(), cutoff))
        conn.close()
        df["date"] = pd.to_datetime(df["date"])
        return df
    except Exception:
        return pd.DataFrame()


def _available_sources() -> list[str]:
    """Return sources that are usable (free or have keys configured)."""
    free = ["Google", "Yahoo", "StockTwits", "EDGAR", "Reddit"]
    keyed = [s for s, key in SOURCES_WITH_KEYS.items() if os.getenv(key) or
             (hasattr(st, "secrets") and st.secrets.get(key))]
    return free + keyed


# ── Sidebar: source status ────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🔌 Source Status")
    free_srcs  = ["Google", "Yahoo", "StockTwits", "EDGAR", "Reddit"]
    keyed_srcs = list(SOURCES_WITH_KEYS.keys())

    for s in free_srcs:
        st.markdown(f"🟢 **{s}** — free")

    for s, key in SOURCES_WITH_KEYS.items():
        has_key = bool(os.getenv(key) or st.secrets.get(key, None))
        icon    = "🟢" if has_key else "🔴"
        label   = "active" if has_key else f"add `{key}` to secrets.toml"
        st.markdown(f"{icon} **{s}** — {label}")

    st.markdown("---")
    st.markdown("## ⏰ Schedule")
    st.caption("""Run `crontab -e` and add:
```
0  6  * * 1-5  cd ~/Desktop/ML_Quant_Fund && python -m data.etl_sentiment
0 12  * * 1-5  cd ~/Desktop/ML_Quant_Fund && python -m data.etl_sentiment
45 15 * * 1-5  cd ~/Desktop/ML_Quant_Fund && python -m data.etl_sentiment
```
This covers pre-market, midday, and close.""")


# ══════════════════════════════════════════════════════════════════════════════
#  CURRENT SCORES TABLE
# ══════════════════════════════════════════════════════════════════════════════

st.subheader("📊 Latest Scores")

scores_df = _load_latest_scores()

if scores_df.empty:
    st.info("No sentiment data yet. Run sentiment below to get started.")
else:
    scores_df["created_at"] = pd.to_datetime(scores_df["created_at"])
    minutes_ago = ((datetime.utcnow() - scores_df["created_at"]).dt.total_seconds() / 60).round(0)
    scores_df["updated"] = minutes_ago.apply(
        lambda m: f"{int(m)}m ago" if m < 60
        else f"{int(m//60)}h {int(m%60)}m ago"
    )

    def _sentiment_bar(score: float) -> str:
        filled = int(abs(score) * 10)
        bar    = "█" * filled + "░" * (10 - filled)
        return f"{'📈' if score >= 0 else '📉'} {bar} {score:+.3f}"

    scores_df["sentiment"] = scores_df["score"].apply(_sentiment_bar)

    st.dataframe(
        scores_df[["ticker", "sentiment", "positive_pct", "negative_pct",
                   "n_headlines", "time_slot", "updated"]]
        .style.format({
            "positive_pct": "{:.0f}%",
            "negative_pct": "{:.0f}%",
        }),
        use_container_width=True,
        hide_index=True,
    )

    # Bull/Bear bar chart
    chart = (
        alt.Chart(scores_df)
        .mark_bar()
        .encode(
            x=alt.X("ticker:N", sort="-y", title="Ticker"),
            y=alt.Y("score:Q",  title="Sentiment Score (-1 to +1)"),
            color=alt.condition(
                "datum.score >= 0",
                alt.value("#00c853"),
                alt.value("#ff1744"),
            ),
            tooltip=[
                "ticker",
                alt.Tooltip("score:Q",        format="+.3f"),
                alt.Tooltip("positive_pct:Q", format=".0f", title="Positive %"),
                alt.Tooltip("negative_pct:Q", format=".0f", title="Negative %"),
                "n_headlines",
                "time_slot",
                "updated",
            ],
        )
        .properties(height=300, title="Current Sentiment Score by Ticker")
    )
    st.altair_chart(chart, use_container_width=True)


st.divider()


# ══════════════════════════════════════════════════════════════════════════════
#  MANUAL RUN PANEL
# ══════════════════════════════════════════════════════════════════════════════

st.subheader("🚀 Run Sentiment Now")
st.caption("Use this for breaking news — Trump tariff tweet, Fed surprise, Ukraine escalation, earnings miss, etc.")

all_tickers    = _load_tickers()
active_sources = _available_sources()

col1, col2 = st.columns([2, 1])

with col1:
    selected_tickers = st.multiselect(
        "Select tickers to score",
        options=all_tickers,
        default=all_tickers[:5],   # default to first 5 to avoid freezing
        help="Select only the tickers you care about. Each takes ~4 seconds.",
    )

    selected_sources = st.multiselect(
        "Sources to use",
        options=active_sources,
        default=active_sources,
        help="Deselect slow sources if you need results fast.",
    )

with col2:
    slot_override = st.selectbox(
        "Label this run as",
        ["auto-detect", "pre_market", "midday", "close", "intraday", "breaking_news"],
        help="Intraday/breaking_news won't overwrite scheduled slots.",
    )
    force_rerun = st.checkbox(
        "Force re-run",
        value=False,
        help="Re-run even if this slot is already cached today.",
    )

# Estimate time
est_seconds = len(selected_tickers) * 4
est_label   = f"~{est_seconds}s" if est_seconds < 60 else f"~{est_seconds//60}m {est_seconds%60}s"

st.caption(f"⏱ Estimated time: **{est_label}** for {len(selected_tickers)} ticker(s)")

if st.button("▶️ Run Sentiment Now", type="primary", disabled=not selected_tickers):
    slot = None if slot_override == "auto-detect" else slot_override
    results = {}

    progress = st.progress(0, text="Starting FinBERT...")
    status   = st.empty()

    for i, tkr in enumerate(selected_tickers):
        progress.progress(i / len(selected_tickers), text=f"Scoring {tkr}...")
        status.caption(f"Running {tkr} ({i+1}/{len(selected_tickers)})")

        try:
            r = run_sentiment_etl(
                tickers=[tkr],
                sources=selected_sources,
                time_slot=slot,
                force=force_rerun,
                verbose=False,
            )
            results.update(r)
        except Exception as e:
            st.warning(f"⚠️ {tkr}: {e}")

    progress.progress(1.0, text="Done.")
    status.empty()

    if results:
        st.success(f"✓ Scored {len(results)} tickers")
        result_rows = [
            {"ticker": t, "score": s,
             "signal": "📈 Bullish" if s > 0.1 else ("📉 Bearish" if s < -0.1 else "➡️ Neutral")}
            for t, s in sorted(results.items(), key=lambda x: x[1], reverse=True)
        ]
        st.dataframe(
            pd.DataFrame(result_rows)
            .style.format({"score": "{:+.3f}"}),
            use_container_width=True,
            hide_index=True,
        )
        st.rerun()


st.divider()


# ══════════════════════════════════════════════════════════════════════════════
#  SENTIMENT HISTORY
# ══════════════════════════════════════════════════════════════════════════════

st.subheader("📈 Sentiment History")

if not scores_df.empty:
    hist_ticker = st.selectbox(
        "Ticker",
        options=scores_df["ticker"].tolist(),
        key="hist_ticker",
    )
    hist_days = st.slider("Days back", 3, 30, 14)
    hist_df   = _load_history(hist_ticker, days=hist_days)

    if hist_df.empty:
        st.info(f"No history for {hist_ticker} yet.")
    else:
        hist_df["label"] = hist_df["date"].dt.strftime("%m-%d") + " " + hist_df["time_slot"]

        line = (
            alt.Chart(hist_df)
            .mark_line(point=True)
            .encode(
                x=alt.X("date:T", title="Date"),
                y=alt.Y("score:Q", title="Sentiment Score",
                        scale=alt.Scale(domain=[-1, 1])),
                color=alt.Color("time_slot:N", title="Slot"),
                tooltip=[
                    "date:T", "time_slot",
                    alt.Tooltip("score:Q",        format="+.3f"),
                    alt.Tooltip("positive_pct:Q", format=".0f", title="Pos%"),
                    alt.Tooltip("negative_pct:Q", format=".0f", title="Neg%"),
                    "n_headlines",
                ],
            )
            .properties(height=300, title=f"{hist_ticker} — Sentiment Over Time")
        )

        zero = alt.Chart(pd.DataFrame({"y": [0]})).mark_rule(
            strokeDash=[4, 4], color="gray", opacity=0.5
        ).encode(y="y:Q")

        st.altair_chart((line + zero).interactive(), use_container_width=True)
else:
    st.info("Run sentiment first to see history.")


# ══════════════════════════════════════════════════════════════════════════════
#  ADD NEWS_API KEY REMINDER
# ══════════════════════════════════════════════════════════════════════════════

if not os.getenv("NEWS_API_KEY") and not st.secrets.get("NEWS_API_KEY", None):
    with st.expander("💡 Enable NewsAPI for geopolitical coverage"):
        st.markdown("""
**NewsAPI** adds Reuters, AP, Bloomberg summaries, WSJ — critical for current macro environment
(Trump tariffs, Ukraine/Russia, China tensions, Fed speak).

1. Get a free key at **newsapi.org** (100 req/day free, $50/month for more)
2. Add to your `.streamlit/secrets.toml`:
```toml
NEWS_API_KEY = "your_key_here"
```
3. Restart the app — NewsAPI will be automatically enabled as a source.
""")
