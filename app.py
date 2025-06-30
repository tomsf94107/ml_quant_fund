import streamlit as st
import matplotlib.pyplot as plt
from sentiment_utils import get_sentiment_scores
import os
import certifi

# Ensure SSL cert file is set
os.environ["SSL_CERT_FILE"] = certifi.where()

st.set_page_config(page_title="Stock Sentiment Dashboard", layout="centered")
st.title("📊 Stock News Sentiment Analyzer")

ticker = st.text_input("Enter stock ticker (e.g., AAPL, MSFT)", "MSFT")

if st.button("Analyze"):
    with st.spinner("Analyzing..."):
        scores = get_sentiment_scores(ticker.upper())

    st.success(f"Sentiment for ${ticker.upper()}")

    st.subheader("News Sentiment Breakdown")
    st.bar_chart(scores["news"])

    st.markdown("🔍 **Legend:**")
    st.markdown("- **Positive** = Bullish news")
    st.markdown("- **Neutral** = Mixed sentiment")
    st.markdown("- **Negative** = Bearish news")

