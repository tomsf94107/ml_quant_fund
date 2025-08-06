# pages/5_Congress_Trading.py

import os                                    # ← don’t forget this
import streamlit as st
import pandas as pd
from data.etl_congress import fetch_congress_trades
from core.backtest_congress import backtest_congress_signal

# Page config
st.set_page_config(page_title="Congress Trading Dashboard", layout="wide")

# ─── Debug your key ─────────────────────────────────────────────────────────
st.subheader("🔑 Debug: Quiver Key Check")
st.text(f'os.getenv("QUIVER_API_KEY")     = {os.getenv("QUIVER_API_KEY")!r}')
st.text(f'st.secrets.get("QUIVER_API_KEY") = {st.secrets.get("QUIVER_API_KEY")!r}')
st.markdown("---")

# ─── Title and description ─────────────────────────────────────────────────
st.title("🏛️ Congress Trading Dashboard")
st.markdown(
    "The Stock Trading on Congressional Knowledge Act requires U.S. Senators and Representatives to "
    "publicly file any financial transaction within 45 days. Here we pull those disclosures, show recent "
    "aggregate trading activity for a symbol, and backtest a simple long-short strategy based on their trades."
)

# ─── Sidebar inputs ─────────────────────────────────────────────────────────
st.sidebar.header("🔍 Search & Settings")
ticker    = st.sidebar.text_input("Stock ticker", "AAPL").upper()
threshold = st.sidebar.number_input(
    "Net shares threshold for long signal",
    min_value=-100_000, max_value=100_000,
    value=0, step=100
)
run = st.sidebar.button("Fetch & Backtest")

if run:
    with st.spinner(f"Loading Congress trades for {ticker}..."):
        try:
            trades = fetch_congress_trades(ticker)
        except Exception as e:
            st.error(f"Error fetching trades: {e}")
            st.stop()

    if trades.empty:
        st.warning("No congressional trades found for this ticker.")
    else:
        # Recent trades table
        st.subheader("📋 Recent Aggregated Trades")
        st.dataframe(trades.tail(10).reset_index(), use_container_width=True)

        # Backtest chart
        st.subheader("📈 Backtest: Congress Long-Short Strategy")
        bt = backtest_congress_signal(ticker, threshold)
        if bt.empty:
            st.warning("Backtest returned no data.")
        else:
            st.line_chart(bt[['cumstrat', 'cumbench']], use_container_width=True)

            # Performance metrics
            final = bt.iloc[-1]
            strat_return = (final['cumstrat'] - 1) * 100
            bench_return = (final['cumbench'] - 1) * 100
            c1, c2 = st.columns(2)
            c1.metric("Strategy Cumulative Return", f"{strat_return:.2f}%")
            c2.metric("Benchmark Cumulative Return", f"{bench_return:.2f}%")

else:
    st.info("Use the sidebar to enter a ticker and click Fetch & Backtest.")
