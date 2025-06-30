import os
import traceback
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
import matplotlib.pyplot as plt
import seaborn as sns
import smtplib
from email.message import EmailMessage
import certifi

from test_live_sentiment import get_sentiment_scores
from sentiment_utils import analyze_sentiment, summarize_sentiments, fetch_news_titles

# ✅ Email env check
email_sender = os.getenv("EMAIL_SENDER")
email_receiver = os.getenv("EMAIL_RECEIVER")
email_password = os.getenv("EMAIL_PASSWORD")
if not all([email_sender, email_receiver, email_password]):
    st.warning("⚠️ EMAIL_SENDER / EMAIL_RECEIVER / EMAIL_PASSWORD missing from environment.")
else:
    st.success(f"✅ EMAIL_SENDER loaded: {email_sender}")

# SSL cert
os.environ["SSL_CERT_FILE"] = certifi.where()

SECTOR_MAP = {
    'AAPL': 'Technology', 'MSFT': 'Technology', 'GOOG': 'Technology', 'META': 'Technology',
    'AMZN': 'Consumer', 'TSLA': 'Automotive', 'JNJ': 'Healthcare', 'JPM': 'Financial',
    'XOM': 'Energy', 'NVDA': 'Semiconductors', 'NVO': 'Healthcare', 'UNH': 'Healthcare',
    'PFE': 'Healthcare', 'MRNA': 'Healthcare'
}

st.set_page_config(layout="wide")
st_autorefresh(interval=5 * 60 * 1000, key="refresh")

st.title("📈 ML-Based Stock Strategy Dashboard + Sentiment")
st.caption(f"🕒 Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# Sidebar
with st.sidebar:
    tickers = st.text_input("Enter comma-separated tickers:", "AAPL,MSFT").upper().split(',')
    start_date = st.date_input("Start date", pd.to_datetime("2018-01-01"))
    end_date = st.date_input("End date", datetime.today())
    confidence_threshold = st.slider("Confidence threshold", 0.5, 0.99, 0.7)
    sector_filter = st.selectbox("Filter by sector (optional):", ["All"] + sorted(set(SECTOR_MAP.values())))
    show_heatmap = st.checkbox("Show Sentiment Heatmap", value=True)
    show_leaderboard = st.checkbox("Show Performance Leaderboard", value=True)
    enable_email = st.checkbox("Send Email Alert on Strategy Trigger")

if sector_filter != "All":
    tickers = [t for t in tickers if SECTOR_MAP.get(t) == sector_filter]

results = []
sentiment_matrix = {}

# ---- Live Sentiment ----
if tickers:
    st.subheader(f"🧠 Live Sentiment for {tickers[0].upper()}")
    sentiment = get_sentiment_scores(tickers[0].upper())
    col1, col2 = st.columns(2)
    with col1:
        st.bar_chart(sentiment["news"])
    with col2:
        st.markdown("**Legend:**\n- **Positive** = Bullish\n- **Neutral** = Mixed\n- **Negative** = Bearish")

# ---- Main Loop ----
for ticker in tickers:
    df = yf.download(ticker, start=start_date, end=end_date, auto_adjust=True)
    if df.empty:
        st.warning(f"No data for {ticker}")
        continue

    df['Return_1D'] = df['Close'].pct_change()
    df['Target'] = (df['Return_1D'].shift(-1) > 0).astype(int)
    df['RSI'] = df['Close'].diff().apply(lambda x: max(x, 0)).rolling(14).mean() / df['Close'].diff().abs().rolling(14).mean()
    df['RSI'] *= 100
    ema12 = df['Close'].ewm(span=12).mean()
    ema26 = df['Close'].ewm(span=26).mean()
    df['MACD'] = ema12 - ema26
    df['Signal'] = df['MACD'].ewm(span=9).mean()
    df.dropna(inplace=True)

    news = fetch_news_titles(ticker)
    sentiments = [analyze_sentiment(title) for title in news]
    summary = summarize_sentiments(sentiments)
    df['Sentiment_Positive'] = summary['positive']
    df['Sentiment_Negative'] = summary['negative']
    sentiment_matrix[ticker] = summary

    X = df[['RSI', 'MACD', 'Signal', 'Sentiment_Positive', 'Sentiment_Negative']]
    y = df['Target']

    if len(X) < 50:
        st.warning(f"Not enough data for {ticker}")
        continue

    X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False, test_size=0.2)
    model = XGBClassifier(max_depth=3, learning_rate=0.1, eval_metric='logloss')
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    df_test = df.iloc[-len(y_test):].copy()
    df_test['Prob'] = y_prob
    df_test['Signal'] = (df_test['Prob'] > confidence_threshold).astype(int)
    df_test['Strategy'] = df_test['Signal'].shift(1) * df_test['Return_1D']
    df_test['Market'] = df_test['Return_1D']
    df_test.dropna(subset=['Strategy', 'Market'], inplace=True)
    df_test[['Strategy', 'Market']] = (1 + df_test[['Strategy', 'Market']]).cumprod()

    sharpe = np.sqrt(252) * df_test['Strategy'].pct_change().mean() / df_test['Strategy'].pct_change().std()
    max_dd = ((df_test['Strategy'] / df_test['Strategy'].cummax()) - 1).min()
    cagr = (df_test['Strategy'].iloc[-1] / df_test['Strategy'].iloc[0]) ** (252 / len(df_test)) - 1

    results.append({
        'Ticker': ticker,
        'Sharpe': round(sharpe, 2),
        'MaxDD': f"{max_dd:.2%}",
        'CAGR': f"{cagr:.2%}",
        'Positive': summary['positive'],
        'Negative': summary['negative']
    })

    # ---- Email Alert ----
    if enable_email and df_test['Signal'].iloc[-1] == 1:
        try:
            msg = EmailMessage()
            msg["Subject"] = f"[ALERT] Buy Signal for {ticker}"
            msg["From"] = email_sender
            msg["To"] = email_receiver
            msg.set_content(
                f"ML Strategy triggered a BUY signal for {ticker}.\n"
                f"Confidence: {df_test['Prob'].iloc[-1]:.2%}\n"
                f"Sharpe Ratio: {sharpe:.2f}\n"
                f"Strategy CAGR: {cagr:.2%}"
            )
            with smtplib.SMTP_SSL("smtp.gmail.com", 465) as smtp:
                smtp.login(email_sender, email_password)
                smtp.send_message(msg)
        except Exception as e:
            st.warning(f"Email alert failed: {e}")
            st.text(traceback.format_exc())

    # ---- Trend Chart ----
    st.subheader(f"📉 Strategy vs Market: {ticker}")
    st.line_chart(df_test[['Strategy', 'Market']])

# ---- Leaderboard ----
if show_leaderboard and results:
    st.subheader("🏆 Strategy Leaderboard")
    st.dataframe(pd.DataFrame(results).sort_values("Sharpe", ascending=False).set_index("Ticker"))

# ---- Heatmap ----
if show_heatmap and sentiment_matrix:
    st.subheader("🔥 Sentiment Heatmap")
    df_heat = pd.DataFrame(sentiment_matrix).T[['positive', 'negative']]
    fig, ax = plt.subplots(figsize=(8, len(df_heat) * 0.4 + 1))
    sns.heatmap(df_heat, annot=True, fmt=".2f", cmap="RdYlGn", ax=ax)
    st.pyplot(fig)
