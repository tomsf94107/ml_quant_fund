import os
import datetime
import requests
import certifi
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import torch
from collections import Counter
import streamlit as st

# ✅ SSL Cert for requests
os.environ["SSL_CERT_FILE"] = certifi.where()
load_dotenv()
st.success(f"✅ SSL_CERT_FILE set to: {os.environ['SSL_CERT_FILE']}")

FINBERT_PATH = "yiyanghkust/finbert-tone"

@st.cache_resource(show_spinner="🔁 Loading FinBERT model...")
def load_finbert():
    tokenizer = AutoTokenizer.from_pretrained(FINBERT_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(FINBERT_PATH)
    return tokenizer, model

@st.cache_resource(show_spinner="🔁 Loading DistilBERT fallback model...")
def load_fallback_pipeline():
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
            st.error("❌ Missing NEWS_API_KEY in environment.")
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
    return probs[:, 2] - probs[:, 0]  # positive - negative

@st.cache_data(show_spinner="📈 Analyzing news sentiment...")
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
        st.warning(f"⚠️ FinBERT failed, using DistilBERT fallback: {e}")
        try:
            sentiment_pipeline = load_fallback_pipeline()
            sentiments = [s["label"].lower() for s in sentiment_pipeline(news)]
        except Exception as fallback_e:
            st.error(f"❌ Fallback model also failed: {fallback_e}")
            return {"news": {"positive": 0, "neutral": 0, "negative": 0}}

    return {"news": summarize_sentiments(sentiments)}

# ✅ Streamlit App UI
st.title("🧠 Live Stock Sentiment Analyzer")
st.caption(f"🕒 Last updated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

ticker = st.text_input("Enter a stock ticker (e.g., AAPL, MSFT):", value="AAPL")

if ticker:
    with st.status(f"Fetching sentiment for: **{ticker.upper()}**...", expanded=True):
        sentiment_data = get_sentiment_scores(ticker)
        st.success("✅ Sentiment Analysis Complete")

    st.bar_chart(sentiment_data["news"])

    st.markdown("### 📊 Legend:")
    st.markdown("- 🟢 **Positive** = Bullish")
    st.markdown("- ⚪️ **Neutral** = Mixed")
    st.markdown("- 🔴 **Negative** = Bearish")

    st.json(sentiment_data)
