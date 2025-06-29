# streamlit_app.py

import os
import pandas as pd
import numpy as np
import yfinance as yf
import xgboost as xgb
import talib
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from datetime import datetime
import smtplib
from email.mime.text import MIMEText

# ---- Helper: RSI ----
def calculate_rsi(df, period=14):
    df['RSI'] = talib.RSI(df['Close'], timeperiod=period)
    return df

# ---- Helper: MACD ----
def calculate_macd(df):
    macd, macdsignal, macdhist = talib.MACD(df['Close'])
    df['MACD'] = macd
    df['MACD_Signal'] = macdsignal
    return df

# ---- Helper: Features ----
def create_features(df):
    df = calculate_rsi(df)
    df = calculate_macd(df)
    df['Return'] = df['Close'].pct_change()
    df['Direction'] = np.where(df['Return'] > 0, 1, 0)
    df = df.dropna()
    return df

# ---- Model ----
def train_model(df):
    features = ['RSI', 'MACD', 'MACD_Signal']
    X = df[features]
    y = df['Direction']
    model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    model.fit(X, y)
    return model

# ---- Strategy ----
def apply_strategy(df, model):
    features = ['RSI', 'MACD', 'MACD_Signal']
    X = df[features]
    df['Prediction'] = model.predict(X)
    df['Strategy'] = df['Prediction'].shift(1) * df['Return']
    df.dropna(subset=['Strategy', 'Return'], inplace=True)
    return df

# ---- Performance ----
def compute_performance(df):
    cumulative_strategy = (df['Strategy'] + 1).cumprod()
    cumulative_market = (df['Return'] + 1).cumprod()
    return cumulative_strategy, cumulative_market

# ---- Streamlit UI ----
st.title("📈 ML-Based Stock Strategy Backtester")
ticker = st.text_input("Enter stock ticker", value="AAPL")
if st.button("🚀 Run Strategy"):
    try:
        df = yf.download(ticker, start="2020-01-01")
        df = create_features(df)
        model = train_model(df)
        df = apply_strategy(df, model)

        strategy, market = compute_performance(df)
        st.subheader(f"📊 {ticker} Strategy")
        st.line_chart(pd.DataFrame({"Strategy": strategy, "Market": market}))
        st.success("Strategy executed successfully.")
    except Exception as e:
        st.error(f"❌ Error running strategy: {str(e)}")
