import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import xgboost as xgb
import matplotlib.pyplot as plt
from datetime import datetime
import smtplib
from email.mime.text import MIMEText

# ---- Helper: RSI ----
def compute_rsi(prices, period=14):
    delta = prices.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

# ---- Helper: MACD ----
def compute_macd(prices):
    exp1 = prices.ewm(span=12, adjust=False).mean()
    exp2 = prices.ewm(span=26, adjust=False).mean()
    macd = exp1 - exp2
    signal = macd.ewm(span=9, adjust=False).mean()
    return macd, signal

# ---- Strategy Logic ----
def generate_features(df):
    df['Return'] = df['Close'].pct_change()
    df['RSI'] = compute_rsi(df['Close'])
    macd, signal = compute_macd(df['Close'])
    df['MACD'] = macd
    df['Signal'] = signal
    df['MA10'] = df['Close'].rolling(window=10).mean()
    df['MA50'] = df['Close'].rolling(window=50).mean()
    df['Volume_Change'] = df['Volume'].pct_change()
    df.dropna(inplace=True)
    return df

def train_model(df):
    features = ['RSI', 'MACD', 'Signal', 'MA10', 'MA50', 'Volume_Change']
    df['Target'] = (df['Return'].shift(-1) > 0).astype(int)
    X = df[features]
    y = df['Target']
    model = xgb.XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.1)
    model.fit(X, y)
    return model, features

def run_strategy(ticker):
    df = yf.download(ticker, period='1y')
    df = generate_features(df)
    model, features = train_model(df)
    df['Prediction'] = model.predict(df[features])
    df['Strategy'] = df['Prediction'].shift(1) * df['Return']
    df.dropna(inplace=True)
    return df

def plot_results(df):
    df[['Return', 'Strategy']].cumsum().apply(np.exp).plot(figsize=(10, 5))
    plt.title("Strategy vs Buy & Hold")
    plt.grid(True)
    st.pyplot(plt)

# ---- Streamlit App ----
st.set_page_config(layout="wide")
st.title("🚀 ML-Based Stock Strategy Backtester")

ticker = st.text_input("Enter stock ticker (e.g. AAPL)", value="AAPL")

if st.button("Run Strategy"):
    with st.spinner("Running backtest..."):
        try:
            result = run_strategy(ticker)
            plot_results(result)
            st.success("Backtest complete.")
        except Exception as e:
            st.error(f"Error: {e}")
