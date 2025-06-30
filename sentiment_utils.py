import os
import datetime
import requests
import certifi
os.environ["SSL_CERT_FILE"] = certifi.where()

from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from collections import Counter
from functools import lru_cache

# ---- Environment Setup ----
load_dotenv()

# ---- FinBERT Setup ----
FINBERT_PATH = "yiyanghkust/finbert-tone"
tokenizer = AutoTokenizer.from_pretrained(FINBERT_PATH)
model = AutoModelForSequenceClassification.from_pretrained(FINBERT_PATH)

# ---- VADER Setup ----
vader = SentimentIntensityAnalyzer()

# ---- Helper: FinBERT ----
def analyze_finbert(texts):
    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
    return probs[:, 2] - probs[:, 0]

# ---- Helper: VADER ----
def analyze_sentiment(text: str) -> str:
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

    news_sentiments = []
    if news:
        try:
            finbert_result = analyze_finbert(news)
            for score in finbert_result.tolist():
                if score >= 0.05:
                    news_sentiments.append("positive")
                elif score <= -0.05:
                    news_sentiments.append("negative")
                else:
                    news_sentiments.append("neutral")
        except Exception as e:
            print("FinBERT error:", e)

    return {
        "news": summarize_sentiments(news_sentiments)
    }
