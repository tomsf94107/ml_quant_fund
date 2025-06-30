import os
import datetime
import requests
import certifi
import yfinance as yf
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import torch
from collections import Counter, defaultdict
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# ✅ Environment
os.environ["SSL_CERT_FILE"] = certifi.where()
load_dotenv()
st.success(f"✅ SSL_CERT_FILE set to: {os.environ['SSL_CERT_FILE']}")

FINBERT_PATH = "yiyanghkust/finbert-tone"
SECTORS = {
    "Technology": ["AAPL", "MSFT", "GOOGL", "NVDA"],
    "Financials": ["JPM", "GS", "BAC", "WFC"],
    "Healthcare": ["PFE", "JNJ", "MRK", "UNH"]
}

@st.cache_resource(show_spinner="🔁 Loading FinBERT...")
def load_finbert():
    tokenizer = AutoTokenizer.from_pretrained(FINBERT_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(FINBERT_PATH)
    return tokenizer, model

@st.cache_resource(show_spinner="🔁 Loading fallback model...")
def load_fallback():
    return pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

def summarize_sentiments(sentiments: list[str]) -> dict:
    counter = Counter(sentiments)
    total = sum(counter.values())
    return {
        "positive": round(counter["positive"] / total * 100, 2) if total else 0.0,
        "neutral": round(counter["neutral"] / total * 100, 2) if total else 0.0,
        "negative": round(counter["negative"] / total * 100, 2) if total else 0.0,
    }

def fetch_news_titles(ticker: str) -> list[str]:
    titles = []
    try:
        api_key = os.getenv("NEWS_API_KEY")
        if not api_key:
            st.error("❌ Missing NEWS_API_KEY.")
            return []
        url = f"https://newsapi.org/v2/everything?q={ticker}&apiKey={api_key}&pageSize=10&sortBy=publishedAt"
        res = requests.get(url).json()
        titles = [a["title"] + " " + a.get("description", "") for a in res.get("articles", [])]
    except Exception as e:
        st.error(f"🛑 News API error: {e}")
    return titles

def analyze_with_finbert(texts):
    tokenizer, model = load_finbert()
    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
    return probs[:, 2] - probs[:, 0]

@st.cache_data(show_spinner="📈 Analyzing sentiment...")
def get_sentiment_scores(ticker: str) -> dict:
    news = fetch_news_titles(ticker)
    sentiments = []
    if not news:
        return {"news": {"positive": 0, "neutral": 0, "negative": 0}}
    try:
        scores = analyze_with_finbert(news)
        for s in scores.tolist():
            if s >= 0.05:
                sentiments.append("positive")
            elif s <= -0.05:
                sentiments.append("negative")
            else:
                sentiments.append("neutral")
    except Exception as e:
        st.warning(f"⚠️ FinBERT failed, using fallback: {e}")
        try:
            fallback = load_fallback()
            sentiments = [s["label"].lower() for s in fallback(news)]
        except Exception as fallback_e:
            st.error(f"❌ Fallback model failed: {fallback_e}")
            return {"news": {"positive": 0, "neutral": 0, "negative": 0}}
    return {"news": summarize_sentiments(sentiments)}

def generate_summary(name: str, sentiment: dict):
    pos = sentiment['positive']
    neg = sentiment['negative']
    neu = sentiment['neutral']
    mood = "bullish 📈" if pos > max(neg, neu) else "bearish 📉" if neg > max(pos, neu) else "mixed ⚖️"
    return f"🗣️ For **{name}**, sentiment is currently *{mood}* — {pos}% 👍, {neu}% 😐, {neg}% 👎."

def plot_price_trend(ticker: str):
    data = yf.download(ticker, period="5d", interval="1h")
    if data.empty:
        st.warning("⚠️ No price data available.")
        return
    st.line_chart(data["Close"], use_container_width=True)

def render_sector_sentiment():
    st.subheader("🏢 Sector-wide Sentiment View")
    sector_results = defaultdict(dict)
    for sector, tickers in SECTORS.items():
        with st.spinner(f"Analyzing {sector} sector..."):
            for ticker in tickers:
                result = get_sentiment_scores(ticker)
                sector_results[sector][ticker] = result["news"]
    for sector, data in sector_results.items():
        st.markdown(f"#### 🏭 {sector}")
        df = pd.DataFrame(data).T
        st.dataframe(df)
        st.bar_chart(df)

# ✅ Streamlit UI
st.title("🧠 Live Stock Sentiment Analyzer")
st.caption(f"🕒 Updated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

ticker = st.text_input("Enter ticker (e.g., AAPL, MSFT):", value="AAPL")
if st.button("Analyze"):
    with st.status(f"Analyzing sentiment for: {ticker.upper()}...", expanded=True):
        sentiment = get_sentiment_scores(ticker)
        st.success("✅ Analysis done.")
        st.bar_chart(sentiment["news"])
        st.markdown("#### 💬 Summary")
        st.markdown(generate_summary(ticker.upper(), sentiment["news"]))
        st.markdown("#### 💹 Price Trend")
        plot_price_trend(ticker)

st.divider()
render_sector_sentiment()
