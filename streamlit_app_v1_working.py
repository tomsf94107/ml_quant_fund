# streamlit_app.py
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

# ---- Helper: RSI ----
def compute_rsi(series, window=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window).mean()
    avg_loss = loss.rolling(window).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

# ---- Helper: Metrics ----
def compute_backtest_metrics(df):
    returns = df['Strategy'].pct_change().dropna()
    if returns.empty or returns.std() == 0:
        return 0.0, 0.0, 0.0
    sharpe = np.sqrt(252) * returns.mean() / returns.std()
    max_dd = ((df['Strategy'] / df['Strategy'].cummax()) - 1).min()
    cagr = (df['Strategy'].iloc[-1] / df['Strategy'].iloc[0]) ** (252 / len(df)) - 1
    return sharpe, max_dd, cagr

# ---- Streamlit UI ----
st.set_page_config(layout="wide")
st_autorefresh(interval=5 * 60 * 1000, key="refresh")
st.title("📈 ML-Based Stock Strategy Dashboard")
st.caption(f"🕒 Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

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

if st.button("🚀 Run Strategy"):
    st.subheader("📡 Live Signals Dashboard")
    for tkr, val in st.session_state["live_signals"].items():
        signal = "🟢 BUY" if val["signal"] == 1 else "🔴 HOLD"
        st.markdown(f"**{tkr}** → {signal} ({val['confidence']*100:.1f}%)")

    for ticker in tickers:
        st.subheader(f"📊 {ticker} Strategy")
        df = yf.download(ticker, start=start_date, end=end_date, auto_adjust=True)

        # ---- FIX: Flatten MultiIndex ----
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = ['_'.join(col).strip() for col in df.columns.values]

        # ---- Identify correct Close column ----
        close_col = f'Close_{ticker}' if f'Close_{ticker}' in df.columns else 'Close'
        if close_col not in df.columns:
            st.warning(f"Missing expected column {close_col} in {ticker}.")
            continue

        df['Return_1D'] = df[close_col].pct_change()
        df['Target'] = (df['Return_1D'].shift(-1) > 0).astype(int)
        df['RSI'] = compute_rsi(df[close_col])
        df['MACD'] = df[close_col].ewm(span=12, adjust=False).mean() - df[close_col].ewm(span=26, adjust=False).mean()
        df['Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
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

        if 'Strategy' in df_test.columns and 'Market' in df_test.columns:
            df_test.dropna(subset=['Strategy', 'Market'], inplace=True)
            df_test[['Strategy', 'Market']] = (1 + df_test[['Strategy', 'Market']]).cumprod()
        else:
            st.warning(f"Missing strategy columns for {ticker}.")
            continue

        acc = accuracy_score(y_test, y_pred)
        sharpe, max_dd, cagr = compute_backtest_metrics(df_test)

        st.metric("Accuracy", f"{acc:.2f}")
        st.metric("Sharpe Ratio", f"{sharpe:.2f}")
        st.metric("Max Drawdown", f"{max_dd:.2%}")
        st.metric("CAGR", f"{cagr:.2%}")

        if not df_test[['Strategy', 'Market']].isnull().values.any():
            st.line_chart(df_test[['Strategy', 'Market']])

        csv = df_test.to_csv(index=False).encode()
        st.download_button(f"📥 Download CSV - {ticker}", csv, file_name=f"{ticker}_strategy.csv")
        log_files.append((f"{ticker}_strategy.csv", csv))

        if enable_email and not df_test.empty:
            latest = df_test.iloc[-1]
            if latest['Signal'] == 1 and latest['Prob'] > confidence_threshold:
                try:
                    from_email = os.getenv("EMAIL_SENDER")
                    to_email = os.getenv("EMAIL_RECEIVER")
                    password = os.getenv("EMAIL_PASSWORD")

                    msg = MIMEText(f"High-confidence BUY signal for {ticker} with prob={latest['Prob']:.2f}")
                    msg['Subject'] = f"Trading Alert: {ticker}"
                    msg['From'] = from_email
                    msg['To'] = to_email

                    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
                        server.login(from_email, password)
                        server.send_message(msg)
                    st.success(f"Email alert sent for {ticker}.")
                except Exception as e:
                    st.error(f"Email failed for {ticker}: {e}")

        if enable_shap:
            explainer = shap.Explainer(model)
            shap_values = explainer(X_test)
            st.write("Feature Importance (SHAP)")
            fig, ax = plt.subplots()
            shap.plots.beeswarm(shap_values, max_display=5, show=False)
            st.pyplot(fig)

        if not df_test.empty:
            latest = df_test.iloc[-1]
            st.session_state["live_signals"][ticker] = {
                "signal": int(latest["Signal"]),
                "confidence": round(float(latest["Prob"]), 2)
            }

    if enable_zip_download and log_files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".zip") as tmp_zip:
            with zipfile.ZipFile(tmp_zip.name, "w") as zipf:
                for filename, content in log_files:
                    zipf.writestr(filename, content)

            with open(tmp_zip.name, "rb") as f:
                zip_bytes = f.read()
                b64 = base64.b64encode(zip_bytes).decode()
                href = f'<a href="data:application/zip;base64,{b64}" download="strategy_logs.zip">📦 Download All Logs as ZIP</a>'
                st.markdown(href, unsafe_allow_html=True)
