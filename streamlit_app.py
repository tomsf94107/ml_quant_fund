import os
from dotenv import load_dotenv
load_dotenv()

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from datetime import datetime
import smtplib
from email.mime.text import MIMEText
import alpaca_trade_api as tradeapi

# ---- Helper: RSI ----
def compute_rsi(series, window=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window).mean()
    avg_loss = loss.rolling(window).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

# ---- Helper: MACD ----
def compute_macd(data):
    ema12 = data['Close'].ewm(span=12, adjust=False).mean()
    ema26 = data['Close'].ewm(span=26, adjust=False).mean()
    data['MACD'] = ema12 - ema26
    data['Signal'] = data['MACD'].ewm(span=9, adjust=False).mean()
    return data

# ---- Streamlit UI ----
st.title("📈 ML-Based Stock Strategy Backtester")

ticker = st.text_input("Enter stock ticker:", "AAPL").upper()
start_date = st.date_input("Start date", pd.to_datetime("2018-01-01"))
end_date = st.date_input("End date", datetime.today())

enable_trading = st.toggle("⚠️ Enable Auto-Trading (Alpaca)", value=False)

if st.button("Run Strategy"):
    df = yf.download(ticker, start=start_date, end=end_date)
    
    if df.empty:
        st.error("⚠️ No data found for this ticker and date range.")
    else:
        df = df.dropna()
        df['Return_1D'] = df['Adj Close'].pct_change()
        df['Target'] = (df['Return_1D'].shift(-1) > 0).astype(int)
        df['RSI'] = compute_rsi(df['Adj Close'])
        df = compute_macd(df)
        df = df.dropna()

        features = ['RSI', 'MACD', 'Signal']
        X = df[features]
        y = df['Target']

        X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False, test_size=0.2)

        model = XGBClassifier(max_depth=3, learning_rate=0.1, use_label_encoder=False, eval_metric='logloss')
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        acc = accuracy_score(y_test, y_pred)

        df_test = df.iloc[-len(y_test):].copy()
        df_test['Pred'] = y_pred
        df_test['Prob'] = y_prob
        df_test['Signal'] = (df_test['Prob'] > 0.7).astype(int)
        df_test['Strategy'] = df_test['Signal'].shift(1) * df_test['Return_1D']
        df_test['Market'] = df_test['Return_1D']
        df_test[['Strategy', 'Market']] = (1 + df_test[['Strategy', 'Market']]).cumprod()

        st.metric("Model Accuracy", f"{acc:.2f}")
        st.line_chart(df_test[['Strategy', 'Market']])

        with st.expander("See raw classification report"):
            st.text(classification_report(y_test, y_pred))

        # Email alert
        latest = df_test.iloc[-1]
        if latest['Signal'] == 1 and latest['Prob'] > 0.7:
            try:
                from_email = os.getenv("EMAIL_SENDER")
                to_email = os.getenv("EMAIL_RECEIVER")
                password = os.getenv("EMAIL_PASSWORD")

                msg = MIMEText(f"High-confidence BUY signal for {ticker} with prob={latest['Prob']:.2f}")
                msg["Subject"] = f"Trading Alert: {ticker}"
                msg["From"] = from_email
                msg["To"] = to_email

                with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
                    server.login(from_email, password)
                    server.send_message(msg)

                st.success("✅ Email alert sent.")
            except Exception as e:
                st.error(f"Email alert failed: {e}")

            # Alpaca auto-trading
            if enable_trading:
                try:
                    alpaca = tradeapi.REST(os.getenv("ALPACA_KEY"), os.getenv("ALPACA_SECRET"), "https://paper-api.alpaca.markets")
                    alpaca.submit_order(symbol=ticker, qty=1, side='buy', type='market', time_in_force='gtc')
                    st.warning("🚨 Auto-trade executed via Alpaca")
                except Exception as e:
                    st.error(f"Alpaca trade failed: {e}")
