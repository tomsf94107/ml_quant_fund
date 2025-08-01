import streamlit as st
from core.backtest_congress import backtest_congress_signal
import pandas as pd

st.title("📊 Congress Trading Backtest")
ticker = st.text_input("Ticker", "AAPL").upper()
threshold = st.number_input("Buy threshold (net shares)", value=0)

if st.button("Run Backtest"):
    df = backtest_congress_signal(ticker, threshold)
    if df.empty:
        st.warning("No data or congress feature missing.")
    else:
        st.line_chart(df, use_container_width=True)
        final = df.iloc[-1]
        st.metric("Strategy vs Bench", f"{final['cumstrat']:.2f}", f"vs {final['cumbench']:.2f}")
