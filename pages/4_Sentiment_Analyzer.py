# sentiment_app.py v2.3 — full feature set + source filter
import os
import datetime
import certifi
import torch
import yfinance as yf
import streamlit as st
import pandas as pd
import requests
from collections import Counter, defaultdict
from dotenv import load_dotenv
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    pipeline
)

# Shared multi-source sentiment util
from sentiment_utils import get_sentiment_scores

# ─── 1) Password Protection ────────────────────────────────────────────────
def check_login():
    pwd = st.text_input("Enter password:", type="password")
    if pwd != st.secrets.get("app_password", "MlQ@nt@072025"):
        st.stop()
check_login()

# ─── 2) Page Config ───────────────────────────────────────────────────────
st.set_page_config(page_title="Sentiment Analyzer", layout="wide")
st.title("🧠 Live Stock Sentiment Analyzer")
st.caption(f"🕒 Updated: {datetime.datetime.now():%Y-%m-%d %H:%M:%S}")

# ─── 3) Env & Certifi ─────────────────────────────────────────────────────
load_dotenv()
os.environ["SSL_CERT_FILE"] = certifi.where()

# ─── 4) Fallback Model ────────────────────────────────────────────────────
# (for cases FinBERT fails)
@st.cache_resource(show_spinner="🔁 Loading fallback model…")
def load_fallback():
    return pipeline("sentiment-analysis",
                    model="distilbert-base-uncased-finetuned-sst-2-english")
FALLBACK_MODEL = load_fallback()

# ─── 5) Sector Definitions ─────────────────────────────────────────────────
SECTORS = {
    "Technology": ["AAPL", "MSFT", "GOOGL", "NVDA"],
    "Financials": ["JPM", "GS", "BAC", "WFC"],
    "Healthcare": ["PFE", "JNJ", "MRK", "UNH"]
}

# ─── 6) UI: Ticker + Source Filter ────────────────────────────────────────
ticker = st.text_input("Enter ticker (e.g., AAPL, MSFT):", "AAPL").upper()

ALL_SOURCES = [
    "NewsAPI","Google","Yahoo","Reddit",
    "Pushshift","StockTwits","EDGAR",
    "Marketaux","AlphaV"
]
selected_sources = st.multiselect(
    "Filter by headline sources:",
    options=ALL_SOURCES,
    default=ALL_SOURCES
)

# ─── 7) Main Analysis Block ────────────────────────────────────────────────
if st.button("Analyze Sentiment"):

    with st.status(f"Analyzing sentiment for {ticker}…", expanded=True):
        # Pass the user's selection into our shared util
        sentiment = get_sentiment_scores(
            ticker,
            sources=selected_sources,
            log_to_csv=False
        )
        st.success("✅ Analysis done.")

    # Breakdown metrics
    st.markdown("### 📊 Sentiment Breakdown")
    c1, c2, c3 = st.columns(3)
    c1.metric("👍 Positive", f"{sentiment['positive']}%")
    c2.metric("😐 Neutral", f"{sentiment['neutral']}%")
    c3.metric("👎 Negative", f"{sentiment['negative']}%")

    # Bar chart
    df_plot = pd.DataFrame.from_dict(
        {"positive": sentiment["positive"],
         "neutral":  sentiment["neutral"],
         "negative": sentiment["negative"]},
        orient="index",
        columns=["percent"]
    )
    st.bar_chart(df_plot, use_container_width=True)

    # Text summary
    mood = (
        "bullish 📈" if sentiment["positive"] > max(sentiment["neutral"], sentiment["negative"])
        else "bearish 📉" if sentiment["negative"] > max(sentiment["positive"], sentiment["neutral"])
        else "mixed ⚖️"
    )
    st.markdown(
        f"🗣️ For **{ticker}**, overall sentiment is *{mood}* — "
        f"{sentiment['positive']}% 👍, {sentiment['neutral']}% 😐, {sentiment['negative']}% 👎."
    )

    # Price trend
    st.markdown("### 💹 Recent Price Trend")
    data = yf.download(ticker, period="5d", interval="1h", progress=False)
    if data.empty:
        st.warning("⚠️ No price data available.")
    else:
        st.line_chart(data["Close"], use_container_width=True)

# ─── 8) Divider & Sector View ─────────────────────────────────────────────
st.divider()
st.subheader("🏢 Sector-wide Sentiment View")

sector_results: dict[str, dict[str, dict[str,float]]] = defaultdict(dict)
for sector, tickers in SECTORS.items():
    with st.spinner(f"Analyzing {sector} sector…"):
        for tkr in tickers:
            # reuse same filter for sector
            sec_sent = get_sentiment_scores(
                tkr,
                sources=selected_sources,
                log_to_csv=False
            )
            sector_results[sector][tkr] = sec_sent

    st.markdown(f"#### 🏭 {sector}")
    df_sec = pd.DataFrame(sector_results[sector]).T
    st.dataframe(df_sec)
    st.bar_chart(df_sec, use_container_width=True)
