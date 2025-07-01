# v5.2 - Forecast + Auto-Retrain Integration (Fixed datetime merge)

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
from forecast_utils import (
    forecast_price_trend,
    forecast_today_movement,
    auto_retrain_forecast_model,
)

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
        # Auto retrain every time forecast section is touched
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

# ---- Sidebar Config ----
with st.sidebar:
    tickers = st.text_input("Enter comma-separated tickers:", "AAPL,MSFT").upper().split(',')
    start_date = st.date_input("Start date", pd.to_datetime("2018-01-01"))
    end_date = st.date_input("End date", datetime.today())
    confidence_threshold = st.slider("Confidence threshold", 0.5, 0.99, 0.7)
    enable_email = st.toggle("📧 Send Email Alerts", value=True)
    enable_zip_download = st.toggle("📦 Download ZIP of all results", value=True)
    enable_shap = st.toggle("📊 Show SHAP Explainability", value=True)

log_files = []
if "live_signals" not in st.session_state:
    st.session_state["live_signals"] = {}

# ---- Run Strategy Section ----
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

        X = df[features]
        y = df['Target']

        if len(X) < 50:
            st.warning(f"Not enough data to train model for {ticker}. Skipping.")
            continue

        X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False, test_size=0.2)
        model = XGBClassifier(max_depth=3, learning_rate=0.1, use_label_encoder=False, eval_metric='logloss')
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        df_test = df.iloc[-len(y_test):].copy()
        df_test['Pred'] = y_pred
        df_test['Prob'] = y_prob
        df_test['Signal'] = (df_test['Prob'] > confidence_threshold).astype(int)
        df_test['Strategy'] = df_test['Signal'].shift(1) * df_test['Return_1D']
        df_test['Market'] = df_test['Return_1D']

        df_test.dropna(subset=['Strategy', 'Market'], inplace=True)
        df_test[['Strategy', 'Market']] = (1 + df_test[['Strategy', 'Market']]).cumprod()

        acc = accuracy_score(y_test, y_pred)
        sharpe = np.sqrt(252) * df_test['Strategy'].pct_change().mean() / df_test['Strategy'].pct_change().std()
        max_dd = ((df_test['Strategy'] / df_test['Strategy'].cummax()) - 1).min()
        cagr = (df_test['Strategy'].iloc[-1] / df_test['Strategy'].iloc[0]) ** (252 / len(df_test)) - 1

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
                try:
                    msg = MIMEText(f"High-confidence BUY signal for {ticker} with prob={latest['Prob']:.2f}")
                    msg['Subject'] = f"Trading Alert: {ticker}"
                    msg['From'] = os.getenv("EMAIL_SENDER")
                    msg['To'] = os.getenv("EMAIL_RECEIVER")

                    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
                        server.login(os.getenv("EMAIL_SENDER"), os.getenv("EMAIL_PASSWORD"))
                        server.send_message(msg)
                    st.success(f"Email alert sent for {ticker}.")
                except Exception as e:
                    st.error(f"Email failed: {e}")

        if enable_shap:
            explainer = shap.Explainer(model)
            shap_values = explainer(X_test)
            st.write("Feature Importance (SHAP)")
            fig, ax = plt.subplots()
            shap.plots.beeswarm(shap_values, max_display=5, show=False)
            st.pyplot(fig)

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
