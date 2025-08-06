import os
import requests
import certifi
from dotenv import load_dotenv
from transformers import pipeline
from collections import Counter
import streamlit as st

# ✅ Environment Setup
os.environ["SSL_CERT_FILE"] = certifi.where()
load_dotenv()
print("✅ SSL_CERT_FILE set to:", os.environ["SSL_CERT_FILE"])

# ✅ Load DistilBERT sentiment pipeline once (cached)
@st.cache_resource(show_spinner=False)
def load_sentiment_pipeline():
    return pipeline(
        "sentiment-analysis",
        model="distilbert-base-uncased-finetuned-sst-2-english",
        device=-1  # Force CPU for Mac compatibility
    )

# ✅ Fetch news headlines using NewsAPI
def fetch_news_titles(ticker: str) -> list[str]:
    titles = []
    try:
        news_key = os.getenv("NEWS_API_KEY")
        if news_key:
            url = f"https://newsapi.org/v2/everything?q={ticker}&apiKey={news_key}&pageSize=10&sortBy=publishedAt"
            res = requests.get(url).json()
            articles = [a["title"] + " " + a.get("description", "") for a in res.get("articles", [])]
            titles.extend(articles)
        else:
            print("❌ NEWS_API_KEY not found in environment")
    except Exception as e:
        print("🛑 News fetch error:", e)
    return titles

# ✅ Summarize sentiment results
def summarize_sentiments(sentiments: list[str]) -> dict:
    counter = Counter(sentiments)
    total = sum(counter.values())
    return {
        "positive": round(counter["positive"] / total * 100, 2) if total else 0.0,
        "neutral": round(counter["neutral"] / total * 100, 2) if total else 0.0,
        "negative": round(counter["negative"] / total * 100, 2) if total else 0.0,
    }

# ✅ Compute sentiment for a ticker (cached)
@st.cache_data(show_spinner=False)
def get_sentiment_scores(ticker: str) -> dict:
    news = fetch_news_titles(ticker)
    sentiments = []

    if news:
        try:
            pipe = load_sentiment_pipeline()
            for s in pipe(news):
                label = s["label"].lower()
                if label not in ["positive", "negative"]:
                    sentiments.append("neutral")
                else:
                    sentiments.append(label)
        except Exception as e:
            print("Sentiment analysis error:", e)

    return {"news": summarize_sentiments(sentiments)}

# ✅ Run standalone
if __name__ == "__main__":
    ticker = "MSFT"
    result = get_sentiment_scores(ticker)
    print(f"\n📊 News sentiment for ${ticker}:")
    print(result)
