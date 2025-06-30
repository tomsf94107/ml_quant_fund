import os
import requests
import certifi
from dotenv import load_dotenv

import torch
from transformers import pipeline
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from collections import Counter
from functools import lru_cache

# ---- Environment Setup ----
os.environ["SSL_CERT_FILE"] = certifi.where()
load_dotenv()

# ---- DistilBERT Setup (Fallback & Default on Streamlit Cloud) ----
sentiment_pipeline = pipeline(
    "sentiment-analysis",
    model="distilbert-base-uncased-finetuned-sst-2-english",
    truncation=True,
    max_length=512
)

# ---- VADER Setup (optional secondary method) ----
vader = SentimentIntensityAnalyzer()

# ---- Sentiment Analyzer Wrapper ----
def analyze_sentiment(text: str) -> str:
    try:
        result = sentiment_pipeline(text[:512])[0]  # Use only first 512 tokens
        label = result["label"].lower()
        return label if label in ["positive", "neutral", "negative"] else "neutral"
    except Exception as e:
        print("DistilBERT error:", e)
        # fallback to VADER
        score = vader.polarity_scores(text)["compound"]
        if score >= 0.05:
            return "positive"
        elif score <= -0.05:
            return "negative"
        else:
            return "neutral"

# ---- Helper: Summarize ----
def summarize_sentiments(sentiments: list[str]) -> dict:
    counter = Counter(sentiments)
    total = sum(counter.values())
    return {
        "positive": round(counter["positive"] / total * 100, 2) if total else 0.0,
        "neutral": round(counter["neutral"] / total * 100, 2) if total else 0.0,
        "negative": round(counter["negative"] / total * 100, 2) if total else 0.0,
    }

# ---- News Fetcher ----
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
        print("News error:", e)
    return titles

# ---- Aggregator ----
@lru_cache(maxsize=64)
def get_sentiment_scores(ticker: str, hours_back: int = 48) -> dict:
    news = fetch_news_titles(ticker)
    sentiments = [analyze_sentiment(article) for article in news]
    return {
        "news": summarize_sentiments(sentiments)
    }
