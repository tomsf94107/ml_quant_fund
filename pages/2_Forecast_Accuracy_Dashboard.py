# v2.2 — 2_Forecast_Accuracy_Dashboard.py with st.secrets integration

import pandas as pd
import streamlit as st
from forecast_utils import get_gsheet_logger

st.set_page_config(layout="wide")
st.title("📊 Forecast Accuracy Dashboard")

# ---- Load Data from Google Sheet ----
def load_accuracy_log_from_gsheet():
    try:
        sheet = get_gsheet_logger()
        data = sheet.get_all_records()
        return pd.DataFrame(data)
    except Exception as e:
        st.error(f"❌ Failed to load forecast logs: {e}")
        return pd.DataFrame()

acc_df = load_accuracy_log_from_gsheet()
if acc_df.empty:
    st.stop()

# ---- Clean + Process ----
acc_df['timestamp'] = pd.to_datetime(acc_df['timestamp'])
acc_df = acc_df.sort_values("timestamp", ascending=False)

# ---- Per-Ticker Filter ----
unique_tickers = acc_df['ticker'].unique().tolist()
def_tickers = unique_tickers[:5]
selected = st.multiselect("Select tickers to view:", unique_tickers, default=def_tickers)
acc_df = acc_df[acc_df['ticker'].isin(selected)]

# ---- Show Only Latest Per Ticker ----
latest_only = st.checkbox("Show only latest log per ticker", value=True)
if latest_only:
    acc_df = acc_df.sort_values("timestamp", ascending=False).drop_duplicates(subset="ticker", keep="first")

# ---- View Table + Charts ----
st.dataframe(acc_df.reset_index(drop=True))

if not acc_df.empty:
    st.subheader("📈 Trend Over Time")
    chart_df = acc_df.sort_values("timestamp")
    st.line_chart(chart_df.set_index("timestamp")[["mae", "mse", "r2"]])
