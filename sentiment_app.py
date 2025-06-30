import streamlit as st
from test_live_sentiment import get_sentiment_scores
from datetime import datetime
import certifi
import os

# ---- SSL fix ----
os.environ["SSL_CERT_FILE"] = certifi.where()
st.write(f"✅ SSL_CERT_FILE set to: {os.environ['SSL_CERT_FILE']}")

# ---- Streamlit UI ----
st.set_page_config(page_title="Live Ticker Sentiment", layout="centered")
st.title("🧠 Live Stock Sentiment Analyzer")
st.caption(f"🕒 Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# ---- Input ----
ticker = st.text_input("Enter a stock ticker (e.g., AAPL, MSFT):", "AAPL").strip().upper()

if ticker:
    st.info(f"Fetching sentiment for: `{ticker}`...")
    try:
        sentiment = get_sentiment_scores(ticker)
        st.success("✅ Sentiment Analysis Complete")
        st.bar_chart(sentiment["news"])
        st.markdown("**📊 Legend:**\n- 🟢 Positive = Bullish\n- ⚪ Neutral = Mixed\n- 🔴 Negative = Bearish")
        st.json(sentiment)
    except Exception as e:
        st.error(f"⚠️ Error: {e}")
