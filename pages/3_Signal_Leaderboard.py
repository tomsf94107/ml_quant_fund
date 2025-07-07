# v2.1 — 3_Signal_Leaderboard.py (reads from Google Sheets via Streamlit Secrets)

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import json
import gspread
from oauth2client.service_account import ServiceAccountCredentials

st.set_page_config(layout="wide")
st.title("📊 Signal Leaderboard")

# -------- Load Data from Google Sheets --------
def load_eval_data_from_gsheet():
    try:
        scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
        creds_dict = st.secrets["gcp_service_account"]
        creds = ServiceAccountCredentials.from_json_keyfile_dict(creds_dict, scope)
        client = gspread.authorize(creds)
        sheet = client.open("forecast_evaluation_log").sheet1
        data = sheet.get_all_records()
        return pd.DataFrame(data)
    except Exception as e:
        st.error(f"❌ Failed to load evaluation logs from Google Sheets: {e}")
        return pd.DataFrame()

df = load_eval_data_from_gsheet()
if df.empty:
    st.warning("No evaluation data found.")
    st.stop()

# -------- Sidebar Filters --------
df["timestamp"] = pd.to_datetime(df["timestamp"])
with st.sidebar:
    st.header("📂 Filters")
    tickers = sorted(df["ticker"].unique())
    selected = st.multiselect("Select tickers to view", tickers, default=tickers)
    date_range = st.date_input(
        "Select date range",
        [df["timestamp"].min().date(), df["timestamp"].max().date()]
    )

# -------- Apply Filters --------
filtered = df[
    (df["ticker"].isin(selected)) &
    (df["timestamp"].dt.date >= date_range[0]) &
    (df["timestamp"].dt.date <= date_range[1])
]

# -------- Latest Per Ticker --------
latest_df = (
    filtered.sort_values("timestamp")
    .groupby("ticker")
    .tail(1)
    .sort_values("r2", ascending=False)
)

st.subheader("🏅 Latest Forecast Performance Per Ticker")
st.dataframe(
    latest_df.set_index("ticker")[["timestamp", "mae", "mse", "r2"]],
    use_container_width=True
)

# -------- Visual Leaderboard --------
st.subheader("📈 Visual Comparison")
metric = st.selectbox("Select metric to rank", ["mae", "mse", "r2"])

fig, ax = plt.subplots(figsize=(10, 5))
latest_df_sorted = latest_df.sort_values(metric, ascending=(metric != "r2"))
ax.barh(latest_df_sorted["ticker"], latest_df_sorted[metric], color="skyblue")
ax.set_xlabel(metric.upper())
ax.set_ylabel("Ticker")
ax.set_title(f"Latest {metric.upper()} by Ticker")
st.pyplot(fig)

# -------- Download --------
csv = latest_df.to_csv(index=False).encode()
st.download_button("📥 Download Latest Metrics", csv, file_name="latest_forecast_scores.csv")