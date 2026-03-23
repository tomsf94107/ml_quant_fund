# ui/pages/5_Sentiment.py
# Sentiment dashboard — Anthropic API daily scores with accuracy tracking

import os, sys
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import sqlite3
import json
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd
import altair as alt
import streamlit as st

st.set_page_config(page_title="Sentiment", page_icon="📰", layout="wide")
st.title("📰 Sentiment Dashboard")
st.caption("Daily news sentiment scored by Anthropic API. Decays Mon=100% → Fri=0%. ~$0.12/day for 92 tickers.")

# Always resolve to project root (2 levels up from ui/pages/)
_PROJECT_ROOT = Path(__file__).parent.parent.parent
SENT_DB = _PROJECT_ROOT / "data" / "sentiment.db"
ACC_DB  = _PROJECT_ROOT / "accuracy.db"

def _load_latest() -> pd.DataFrame:
    if not SENT_DB.exists(): return pd.DataFrame()
    try:
        conn = sqlite3.connect(SENT_DB)
        df = pd.read_sql("""
            SELECT ticker, score_date, sentiment_score, sentiment_label,
                   confidence, headlines
            FROM monday_sentiment
            WHERE id IN (SELECT MAX(id) FROM monday_sentiment GROUP BY ticker)
            ORDER BY sentiment_score DESC
        """, conn)
        conn.close()
        return df
    except Exception:
        return pd.DataFrame()

def _load_history(days: int = 30) -> pd.DataFrame:
    if not SENT_DB.exists(): return pd.DataFrame()
    try:
        cutoff = (date.today() - timedelta(days=days)).isoformat()
        conn = sqlite3.connect(SENT_DB)
        df = pd.read_sql("""
            SELECT ticker, score_date, sentiment_score, sentiment_label, confidence
            FROM monday_sentiment WHERE score_date >= ?
            ORDER BY score_date DESC
        """, conn, params=(cutoff,))
        conn.close()
        return df
    except Exception:
        return pd.DataFrame()

def _load_accuracy_by_sentiment() -> pd.DataFrame:
    """Compare accuracy on days with bullish vs bearish sentiment."""
    if not SENT_DB.exists() or not ACC_DB.exists(): return pd.DataFrame()
    try:
        conn_s = sqlite3.connect(SENT_DB)
        conn_a = sqlite3.connect(ACC_DB)
        sent = pd.read_sql("SELECT ticker, score_date, sentiment_label, sentiment_score FROM monday_sentiment", conn_s)
        preds = pd.read_sql("""
            SELECT p.ticker, p.prediction_date, p.prob_up,
                   o.actual_return,
                   CASE WHEN o.actual_return > 0 THEN 1 ELSE 0 END as actual_up,
                   CASE WHEN (p.prob_up > 0.5 AND o.actual_return > 0) OR
                             (p.prob_up <= 0.5 AND o.actual_return <= 0)
                   THEN 1 ELSE 0 END as correct
            FROM predictions p
            JOIN outcomes o ON p.ticker=o.ticker
                AND p.prediction_date=o.prediction_date
                AND p.horizon=o.horizon
            WHERE p.horizon=1
        """, conn_a)
        conn_s.close()
        conn_a.close()

        merged = preds.merge(sent, left_on=["ticker","prediction_date"],
                             right_on=["ticker","score_date"], how="inner")
        if merged.empty: return pd.DataFrame()

        summary = merged.groupby("sentiment_label").agg(
            count=("correct","count"),
            accuracy=("correct","mean"),
            avg_return=("actual_return","mean")
        ).reset_index()
        summary["accuracy"] = summary["accuracy"].round(3)
        summary["avg_return"] = summary["avg_return"].round(4)
        return summary
    except Exception:
        return pd.DataFrame()

# ── Summary metrics ───────────────────────────────────────────────────────────
df = _load_latest()

if df.empty:
    st.info("No sentiment data yet. Run scripts/daily_sentiment.py to populate.")
    st.stop()

bullish = len(df[df["sentiment_label"] == "BULLISH"])
bearish = len(df[df["sentiment_label"] == "BEARISH"])
neutral = len(df[df["sentiment_label"] == "NEUTRAL"])
latest_date = df["score_date"].max()

m1, m2, m3, m4 = st.columns(4)
m1.metric("Last scored", latest_date)
m2.metric("Bullish", bullish, f"{bullish/len(df):.0%}")
m3.metric("Bearish", bearish, f"{bearish/len(df):.0%}")
m4.metric("Neutral", neutral, f"{neutral/len(df):.0%}")

st.markdown("---")

# ── Accuracy by sentiment label ───────────────────────────────────────────────
st.subheader("📐 Accuracy by sentiment")
st.caption("Does sentiment predict next-day direction? Green = sentiment helps, Red = hurts.")

acc_df = _load_accuracy_by_sentiment()
if not acc_df.empty and acc_df["count"].sum() >= 10:
    for _, row in acc_df.iterrows():
        col1, col2, col3, col4 = st.columns([2,2,2,3])
        icon = "🟢" if row["sentiment_label"] == "BULLISH" else "🔴" if row["sentiment_label"] == "BEARISH" else "⚪"
        col1.markdown(f"{icon} **{row['sentiment_label']}**")
        col2.metric("Accuracy", f"{row['accuracy']:.1%}")
        col3.metric("Avg return", f"{row['avg_return']:+.2%}")
        col4.metric("Samples", int(row["count"]))
else:
    st.info(f"Need 10+ matched predictions to show accuracy. Currently {acc_df['count'].sum() if not acc_df.empty else 0} matched.")

st.markdown("---")

# ── Top signals today ─────────────────────────────────────────────────────────
col_bull, col_bear = st.columns(2)

with col_bull:
    st.subheader("🟢 Most bullish today")
    top_bull = df[df["sentiment_label"]=="BULLISH"].head(10)
    if not top_bull.empty:
        for _, row in top_bull.iterrows():
            headlines = json.loads(row["headlines"]) if row["headlines"] else []
            with st.expander(f"{row['ticker']} — score={row['sentiment_score']:+.2f} conf={row['confidence']:.1f}"):
                for h in headlines[:3]:
                    st.caption(f"• {h}")
    else:
        st.info("No bullish signals today")

with col_bear:
    st.subheader("🔴 Most bearish today")
    top_bear = df[df["sentiment_label"]=="BEARISH"].sort_values("sentiment_score").head(10)
    if not top_bear.empty:
        for _, row in top_bear.iterrows():
            headlines = json.loads(row["headlines"]) if row["headlines"] else []
            with st.expander(f"{row['ticker']} — score={row['sentiment_score']:+.2f} conf={row['confidence']:.1f}"):
                for h in headlines[:3]:
                    st.caption(f"• {h}")
    else:
        st.info("No bearish signals today")

st.markdown("---")

# ── Full table with filters ───────────────────────────────────────────────────
st.subheader("📋 All tickers")
fc1, fc2, fc3 = st.columns(3)
label_filter = fc1.selectbox("Label", ["ALL","BULLISH","BEARISH","NEUTRAL"])
min_conf     = fc2.slider("Min confidence", 0.0, 1.0, 0.0, 0.1)
show_all     = fc3.checkbox("Show all", value=False)

filtered = df.copy()
if label_filter != "ALL":
    filtered = filtered[filtered["sentiment_label"] == label_filter]
filtered = filtered[filtered["confidence"] >= min_conf]
if not show_all:
    filtered = filtered.head(10)

# Build card HTML
CARD_CSS = """<style>
.sent-card{background:var(--color-background-primary);border:0.5px solid var(--color-border-tertiary);
border-radius:10px;padding:10px 16px;margin-bottom:6px;display:grid;
grid-template-columns:70px 90px 80px 80px 1fr;align-items:center;gap:12px}
.sent-hdr{background:var(--color-background-secondary);border-radius:8px;padding:6px 16px;
display:grid;grid-template-columns:70px 90px 80px 80px 1fr;gap:12px;margin-bottom:6px}
</style>"""

html = CARD_CSS
html += '<div class="sent-hdr"><span style="font-size:11px;color:var(--color-text-secondary)">Ticker</span><span style="font-size:11px;color:var(--color-text-secondary)">Label</span><span style="font-size:11px;color:var(--color-text-secondary)">Score</span><span style="font-size:11px;color:var(--color-text-secondary)">Confidence</span><span style="font-size:11px;color:var(--color-text-secondary)">Headlines</span></div>'

for _, row in filtered.iterrows():
    label = row["sentiment_label"]
    border = "#3B6D11" if label=="BULLISH" else "#A32D2D" if label=="BEARISH" else "#888780"
    label_color = "color:var(--color-text-success)" if label=="BULLISH" else "color:var(--color-text-danger)" if label=="BEARISH" else "color:var(--color-text-secondary)"
    score_color = "color:var(--color-text-success)" if row["sentiment_score"]>0 else "color:var(--color-text-danger)" if row["sentiment_score"]<0 else ""
    headlines = json.loads(row["headlines"]) if row["headlines"] else []
    first_headline = headlines[0][:60]+"..." if headlines else "—"

    html += f"""<div class="sent-card" style="border-left:3px solid {border}">
  <div style="font-size:16px;font-weight:500">{row['ticker']}</div>
  <div style="font-size:13px;{label_color}">{label}</div>
  <div style="font-size:14px;font-weight:500;{score_color}">{row['sentiment_score']:+.2f}</div>
  <div style="font-size:13px;color:var(--color-text-secondary)">{row['confidence']:.1f}</div>
  <div style="font-size:12px;color:var(--color-text-secondary)">{first_headline}</div>
</div>"""

st.html(html)

st.markdown("---")
tickers_count = len([t.strip() for t in open(_PROJECT_ROOT / "tickers.txt").readlines() if t.strip()])
st.caption(f"Sentiment scored daily at 3 PM Vietnam time (4 AM ET) via Anthropic API · {tickers_count} tickers · ~$0.13/day")
