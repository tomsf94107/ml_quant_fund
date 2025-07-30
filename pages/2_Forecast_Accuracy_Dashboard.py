# v3.2 — Forecast Accuracy Dashboard (Merged Final Version)

import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import os
import gspread
from oauth2client.service_account import ServiceAccountCredentials
from forecast_utils import get_gsheet_logger, run_auto_retrain_all

st.set_page_config(layout="wide")
st.title("📊 Forecast Accuracy Dashboard")

# ---- Load from Google Sheets ----
def load_accuracy_log_from_gsheet():
    sheet = get_gsheet_logger()
    data = sheet.get_all_records()
    return pd.DataFrame(data)

# ---- Fallback: Local Forecast Logs ----
def load_eval_data_from_csv():
    dfs = []
    folder = "forecast_logs"
    if not os.path.exists(folder):
        return pd.DataFrame()
    for file in os.listdir(folder):
        if file.endswith("_xgb_log.csv"):
            df = pd.read_csv(os.path.join(folder, file))
            dfs.append(df)
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()

# ---- Try Sheets First ----
try:
    acc_df = load_accuracy_log_from_gsheet()
    source = "Google Sheets ✅"
except Exception as e:
    st.warning(f"⚠️ Google Sheets failed: {e}")
    acc_df = load_eval_data_from_csv()
    source = "Local CSV fallback 🗂"

if acc_df.empty:
    st.error("❌ No accuracy data found.")
    st.stop()

st.caption(f"📂 Data Source: {source}")

# ---- Sidebar Controls ----
acc_df['timestamp'] = pd.to_datetime(acc_df['timestamp'])
with st.sidebar:
    st.header("📂 Filters")
    tickers = sorted(acc_df["ticker"].unique())
    selected = st.multiselect("Select tickers", tickers, default=tickers)
    date_range = st.date_input(
        "Select date range",
        [acc_df["timestamp"].min().date(), acc_df["timestamp"].max().date()]
    )
    if st.button("🔁 Run Batch Retrain"):
        run_auto_retrain_all()
        st.success("✅ Batch retrain complete!")

# ---- Apply Filters ----
filtered = acc_df[
    (acc_df["ticker"].isin(selected)) &
    (acc_df["timestamp"].dt.date >= date_range[0]) &
    (acc_df["timestamp"].dt.date <= date_range[1])
]

if filtered.empty:
    st.warning("⚠️ No logs match current filters.")
    st.stop()

# ---- Latest Metrics Per Ticker ----
latest_df = (
    filtered.sort_values("timestamp")
    .groupby("ticker")
    .tail(1)
    .sort_values("r2", ascending=False)
)

# ---- Calculate delta_r2 (improvement vs previous log) ----
latest_df["delta_r2"] = latest_df["r2"] - (
    filtered.groupby("ticker")["r2"]
    .nth(-2)
    .reindex(latest_df["ticker"])
    .values
)

st.subheader("🏅 Latest Forecast Performance Per Ticker")
st.dataframe(
    latest_df.set_index("ticker")[["timestamp", "mae", "mse", "r2", "delta_r2"]],
    use_container_width=True
)

# ---- Metric Comparison Chart ----
st.subheader("📈 Visual Comparison")
metric = st.selectbox("Select metric to compare", ["mae", "mse", "r2", "delta_r2"])

fig, ax = plt.subplots(figsize=(10, 5))
sorted_df = latest_df.sort_values(metric, ascending=(metric != "r2" and metric != "delta_r2"))
ax.barh(sorted_df["ticker"], sorted_df[metric], color="skyblue")
ax.set_xlabel(metric.upper())
ax.set_title(f"{metric.upper()} by Ticker")
st.pyplot(fig)

# ---- Trend Chart ----
st.subheader("📉 Trend Over Time")
chart_df = filtered.sort_values("timestamp")
st.line_chart(chart_df.set_index("timestamp")[["mae", "mse", "r2"]])

# ---- Download ----
csv = latest_df.to_csv(index=False).encode()
st.download_button("📥 Download Latest Accuracy Log", csv, file_name="latest_forecast_scores.csv")
