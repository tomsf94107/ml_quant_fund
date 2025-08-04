# pages/5_Insider_Trading.py

import os, sys
# ensure project root is on sys.path (if needed)
root = os.path.abspath(os.path.join(__file__, os.pardir, os.pardir))
if root not in sys.path:
    sys.path.append(root)

import streamlit as st
import pandas as pd
from data.etl_insider import fetch_insider_trades

st.set_page_config(page_title="Insider Trading Viewer", layout="wide")
st.title("🗃️ Insider Trading Data")

# Sidebar inputs
ticker = st.sidebar.text_input("Ticker symbol", "AAPL").upper()
count  = st.sidebar.number_input("Number of filings to fetch", min_value=5, max_value=100, value=20)

if st.sidebar.button("Fetch Insider Trades"):
    with st.spinner(f"Loading last {count} Form 4 filings for {ticker}…"):
        # explicitly pull from RSS, limiting to `count`
        df = fetch_insider_trades(ticker, mode="rss")
        if df.empty:
            st.warning("No insider filings found.")
            st.stop()

        # If you really want to limit the number of rows shown:
        df = df.sort_values("ds", ascending=False).head(count)

        st.success(f"Fetched {len(df)} filings.")

        # Display raw table
        st.subheader("📄 Raw Insider Filings")
        st.dataframe(df, use_container_width=True)

        # Plot net shares over time
        st.subheader("📈 Net Insider Shares Over Time")
        st.line_chart(df.set_index("ds")[["net_shares"]], use_container_width=True)

        # Metrics
        total_net   = int(df["net_shares"].sum())
        total_buys  = int(df["num_buy_tx"].sum())
        total_sells = int(df["num_sell_tx"].sum())

        c1, c2, c3 = st.columns(3)
        c1.metric("Total Net Shares", f"{total_net}")
        c2.metric("Total Buy Transactions", f"{total_buys}")
        c3.metric("Total Sell Transactions", f"{total_sells}")

        # Bar chart of buy vs sell counts
        st.subheader("🔢 Transactions Count Breakdown")
        st.bar_chart(pd.DataFrame({
            "Buys":  [total_buys],
            "Sells": [total_sells]
        }), use_container_width=True)

else:
    st.info("Use the sidebar to enter a ticker and click ‘Fetch Insider Trades’.")
