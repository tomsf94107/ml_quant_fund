import os
import datetime
import requests
import certifi
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from collections import Counter

# ✅ Environment Setup
os.environ["SSL_CERT_FILE"] = certifi.where()  # Permanent fix for SSL
load_dotenv()  # Load API keys from .env
print("✅ SSL_CERT_FILE set to:", os.environ["SSL_CERT_FILE"])

FINBERT_PATH = "yiyanghkust/finbert-tone"

# ✅ Lazy-load FinBERT inside a function to avoid Streamlit crash
def load_finbert():
    tokenizer = AutoTokenizer.from_pretrained(FINBERT_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(FINBERT_PATH)
    return tokenizer, model

# ✅ Analyze sentiment using FinBERT
def analyze_finbert(texts):
    tokenizer, model = load_finbert()
    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
    return probs[:, 2] - probs[:, 0]  # positive - negative

# ✅ Summarize sentiment results
def summarize_sentiments(sentiments: list[str]) -> dict:
    counter = Counter(sentiments)
    total = sum(counter.values())
    return {
        "positive": round(counter["positive"] / total * 100, 2) if total else 0.0,
        "neutral": round(counter["neutral"] / total * 100, 2) if total else 0.0,
        "negative": round(counter["negative"] / total * 100, 2) if total else 0.0,
    }

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
        print("News error:", e)
    return titles

# ✅ Get sentiment breakdown for news
def get_news_sentiment(ticker: str) -> dict:
    news = fetch_news_titles(ticker)
    sentiments = []
    if news:
        try:
            scores = analyze_finbert(news)
            for s in scores.tolist():
                if s >= 0.05:
                    sentiments.append("positive")
                elif s <= -0.05:
                    sentiments.append("negative")
                else:
                    sentiments.append("neutral")
        except Exception as e:
            print("FinBERT error:", e)
    return summarize_sentiments(sentiments)

# ✅ Final callable from Streamlit app
def get_sentiment_scores(ticker: str) -> dict:
    summary = get_news_sentiment(ticker)
    return {
        "news": {
            "positive": summary["positive"],
            "neutral": summary["neutral"],
            "negative": summary["negative"]
        }
    }

# ✅ Run standalone for debugging
if __name__ == "__main__":
    ticker = "MSFT"
    result = get_sentiment_scores(ticker)
    print(f"\n📊 News sentiment for ${ticker}:")
    print(result)
