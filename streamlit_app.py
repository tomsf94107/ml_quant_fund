# v10 - Combines v8 and v9 into one master version with SHAP patch

import os
from dotenv import load_dotenv
load_dotenv()

import streamlit as st
from streamlit_autorefresh import st_autorefresh
import yfinance as yf
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from datetime import datetime
import smtplib
from email.mime.text import MIMEText
import shap
import matplotlib.pyplot as plt
import base64
import tempfile
import zipfile
import gspread
from oauth2client.service_account import ServiceAccountCredentials

from forecast_utils import (
    forecast_price_trend,
    forecast_today_movement,
    auto_retrain_forecast_model,
    compute_rolling_accuracy,
    get_latest_forecast_log
)

# -------------------- Helper Functions --------------------
def train_model(df, features, target_col):
    X = df[features]
    y = df[target_col]
    X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False, test_size=0.2)
    model = XGBClassifier(max_depth=3, learning_rate=0.1, use_label_encoder=False, eval_metric='logloss')
    model.fit(X_train, y_train)
    return model, X_test, y_test

def evaluate_strategy(df, y_test, y_pred, y_prob, threshold):
    df_test = df.iloc[-len(y_test):].copy()
    df_test['Pred'] = y_pred
    df_test['Prob'] = y_prob
    df_test['Signal'] = (df_test['Prob'] > threshold).astype(int)
    df_test['Strategy'] = df_test['Signal'].shift(1) * df_test['Return_1D']
    df_test['Market'] = df_test['Return_1D']
    df_test.dropna(subset=['Strategy', 'Market'], inplace=True)
    df_test[['Strategy', 'Market']] = (1 + df_test[['Strategy', 'Market']]).cumprod()
    acc = accuracy_score(y_test, y_pred)
    sharpe = np.sqrt(252) * df_test['Strategy'].pct_change().mean() / df_test['Strategy'].pct_change().std()
    max_dd = ((df_test['Strategy'] / df_test['Strategy'].cummax()) - 1).min()
    cagr = (df_test['Strategy'].iloc[-1] / df_test['Strategy'].iloc[0]) ** (252 / len(df_test)) - 1
    return df_test, acc, sharpe, max_dd, cagr

def send_alert_email(ticker, prob):
    try:
        msg = MIMEText(f"High-confidence BUY signal for {ticker} with prob={prob:.2f}")
        msg['Subject'] = f"Trading Alert: {ticker}"
        msg['From'] = os.getenv("EMAIL_SENDER")
        msg['To'] = os.getenv("EMAIL_RECEIVER")
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(os.getenv("EMAIL_SENDER"), os.getenv("EMAIL_PASSWORD"))
            server.send_message(msg)
        st.success(f"Email alert sent for {ticker}.")
    except Exception as e:
        st.error(f"Email failed: {e}")

def plot_shap(model, X_test):
    explainer = shap.Explainer(model)
    shap_values = explainer(X_test)
    st.write("Feature Importance (SHAP)")
    fig = plt.figure(figsize=(8, 4))
    shap.plots.beeswarm(shap_values, max_display=5, show=False)
    st.pyplot(fig)

def load_forecast_tickers():
    path = "tickers.csv"
    if not os.path.exists(path):
        return ["AAPL", "MSFT"]
    with open(path, "r") as f:
        return [line.strip().upper() for line in f if line.strip()]

def save_forecast_tickers(ticker_list):
    with open("tickers.csv", "w") as f:
        for tkr in ticker_list:
            f.write(tkr.strip().upper() + "\n")

def load_accuracy_log_from_gsheet():
    try:
        scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
        creds = ServiceAccountCredentials.from_json_keyfile_name("keys/mlquan-0515c30186b6.json", scope)
        client = gspread.authorize(creds)
        sheet = client.open("forecast_evaluation_log").sheet1
        data = sheet.get_all_records()
        return pd.DataFrame(data)
    except Exception as e:
        st.error(f"⚠️ Failed to load Google Sheet: {e}")
        return pd.DataFrame()

# 🔐 Password Protection
def check_login():
    password = st.text_input("Enter password:", type="password")
    if password != st.secrets.get("app_password", "MlQ@nt@072025"):
        st.stop()
check_login()

# ---- Streamlit UI ----
st.set_page_config(layout="wide")
st_autorefresh(interval=5 * 60 * 1000, key="refresh")
st.title("📈 ML-Based Stock Strategy Dashboard")
st.caption(f"🕒 Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# ---- Forecast Section ----
with st.expander("📅 Forecast Price Trends (Prophet Model)"):
    ticker_input = st.text_input("Enter a ticker for 3-month forecast", "AAPL")
    if ticker_input:
        auto_retrain_forecast_model(ticker_input.upper())
    if st.button("Run Forecast"):
        forecast_df, err = forecast_price_trend(ticker_input.upper())
        if err:
            st.warning(err)
        else:
            st.subheader(f"🗓️ 3-Month Price Forecast for {ticker_input.upper()}")
            future_df = forecast_df[forecast_df['ds'] > pd.Timestamp.today()]
            st.line_chart(future_df.set_index("ds")[["yhat", "yhat_lower", "yhat_upper"]])

            st.subheader("🗓️ Today's Movement Prediction")
            movement, err2 = forecast_today_movement(ticker_input.upper())
            if err2:
                st.warning(err2)
            else:
                st.success(movement)

            log_path = get_latest_forecast_log(ticker_input.upper())
            if log_path:
                df_acc = compute_rolling_accuracy(log_path)
                st.subheader("📈 Rolling Forecast Accuracy")
                fig, ax = plt.subplots(figsize=(8, 4))
                ax.plot(df_acc["ds"], df_acc["7d_accuracy"], label="7-Day Accuracy")
                ax.plot(df_acc["ds"], df_acc["30d_accuracy"], label="30-Day Accuracy")
                ax.set_ylabel("Accuracy")
                ax.set_ylim(0, 1.05)
                ax.legend()
                st.pyplot(fig)

                latest = df_acc.iloc[-1]
                if latest['correct']:
                    st.success("✅ Latest forecast direction was correct!")
                else:
                    st.error("❌ Latest forecast direction was wrong.")

# ---- Sidebar Config ----
with st.sidebar:
    tickers = st.text_input("Enter comma-separated tickers:", "AAPL,MSFT").upper().split(',')
    start_date = st.date_input("Start date", pd.to_datetime("2018-01-01"))
    end_date = st.date_input("End date", datetime.today())
    confidence_threshold = st.slider("Confidence threshold", 0.5, 0.99, 0.7)
    enable_email = st.toggle("📧 Send Email Alerts", value=True)
    enable_zip_download = st.toggle("📦 Download ZIP of all results", value=True)
    enable_shap = st.toggle("📊 Show SHAP Explainability", value=True)

    with st.expander("📋 Manage Forecast Tickers"):
        curr = "\n".join(load_forecast_tickers())
        tick_edit = st.text_area("Edit tickers (one per line):", curr, height=150)
        if st.button("💾 Save Tickers"):
            save_forecast_tickers(tick_edit.split("\n"))
            st.success("tickers.csv updated!")
        st.caption("tickers.csv should be in your project root, one ticker per line, no header.")

log_files = []
if "live_signals" not in st.session_state:
    st.session_state["live_signals"] = {}

# ---- Strategy Execution Section ----
if st.button("🚀 Run Strategy"):
    st.subheader("📱 Live Signals Dashboard")
    for tkr, val in st.session_state["live_signals"].items():
        signal = "🟢 BUY" if val["signal"] == 1 else "🔴 HOLD"
        st.markdown(f"**{tkr}** → {signal} ({val['confidence']*100:.1f}%)")

    for ticker in tickers:
        st.subheader(f"📊 {ticker} Strategy")
        df = yf.download(ticker, start=start_date, end=end_date, auto_adjust=True)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = ['_'.join(col).strip() for col in df.columns.values]
        close_col = f'Close_{ticker}' if f'Close_{ticker}' in df.columns else 'Close'
        if close_col not in df.columns:
            st.warning(f"Missing expected column {close_col} in {ticker}.")
            continue

        df['Return_1D'] = df[close_col].pct_change()
        df['Target'] = (df['Return_1D'].shift(-1) > 0).astype(int)
        df['RSI'] = df[close_col].diff().apply(lambda x: max(x, 0)).rolling(14).mean() / \
                    df[close_col].diff().abs().rolling(14).mean() * 100
        df['MACD'] = df[close_col].ewm(span=12).mean() - df[close_col].ewm(span=26).mean()
        df['Signal'] = df['MACD'].ewm(span=9).mean()
        df.dropna(inplace=True)

        features = ['RSI', 'MACD', 'Signal']
        if not all(col in df.columns for col in features):
            st.warning(f"Missing features for {ticker}. Skipping.")
            continue

        model, X_test, y_test = train_model(df, features, 'Target')
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        df_test, acc, sharpe, max_dd, cagr = evaluate_strategy(df, y_test, y_pred, y_prob, confidence_threshold)

        st.metric("Accuracy", f"{acc:.2f}")
        st.metric("Sharpe Ratio", f"{sharpe:.2f}")
        st.metric("Max Drawdown", f"{max_dd:.2%}")
        st.metric("CAGR", f"{cagr:.2%}")
        st.line_chart(df_test[['Strategy', 'Market']])

        csv = df_test.to_csv(index=False).encode()
        st.download_button(f"🗕️ Download CSV - {ticker}", csv, file_name=f"{ticker}_strategy.csv")
        log_files.append((f"{ticker}_strategy.csv", csv))

        if enable_email and not df_test.empty:
            latest = df_test.iloc[-1]
            if latest['Signal'] == 1 and latest['Prob'] > confidence_threshold:
                send_alert_email(ticker, latest['Prob'])

        if enable_shap:
            plot_shap(model, X_test)

        st.session_state["live_signals"][ticker] = {
            "signal": int(df_test.iloc[-1]["Signal"]),
            "confidence": float(df_test.iloc[-1]["Prob"])
        }

    if enable_zip_download and log_files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".zip") as tmp_zip:
            with zipfile.ZipFile(tmp_zip.name, "w") as zipf:
                for filename, content in log_files:
                    zipf.writestr(filename, content)
            with open(tmp_zip.name, "rb") as f:
                b64 = base64.b64encode(f.read()).decode()
                st.markdown(f'<a href="data:application/zip;base64,{b64}" download="strategy_logs.zip">📦 Download All Logs as ZIP</a>', unsafe_allow_html=True)

# ---- Accuracy Dashboard ----
st.subheader("📊 Forecast Accuracy Dashboard (from Google Sheet)")
acc_df = load_accuracy_log_from_gsheet()
if not acc_df.empty:
    acc_df['timestamp'] = pd.to_datetime(acc_df['timestamp'])
    acc_df = acc_df.sort_values("timestamp", ascending=False)

    st.dataframe(acc_df)

    st.line_chart(acc_df.set_index("timestamp")[["mae", "mse", "r2"]])
else:
    st.warning("No forecast accuracy data found.")
