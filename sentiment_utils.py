# sentiment_utils.py v3.4-final — adds ticker None/empty guard in fetch_news_titles()

import os
import datetime
import requests
import certifi
import pandas as pd
import csv
import torch
from collections import Counter
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ✅ Optional: load from .env if not running in Streamlit
try:
    from dotenv import load_dotenv
    load_dotenv()
except:
    pass

# ✅ Streamlit secrets fallback
try:
    import streamlit as st
    ST_SECRETS = st.secrets
except ImportError:
    ST_SECRETS = {}

# ✅ SSL Fix (macOS crash fix)
os.environ["SSL_CERT_FILE"] = certifi.where()

# ✅ Load FinBERT once (cached)
FINBERT_PATH = "yiyanghkust/finbert-tone"
_tokenizer = None
_model = None

def load_finbert():
    global _tokenizer, _model
    if _tokenizer is None or _model is None:
        _tokenizer = AutoTokenizer.from_pretrained(FINBERT_PATH)
        _model = AutoModelForSequenceClassification.from_pretrained(FINBERT_PATH)
    return _tokenizer, _model

# ✅ Helper: summarize predictions
def summarize_sentiments(sentiments: list[str]) -> dict:
    counter = Counter(sentiments)
    total = sum(counter.values())
    return {
        "positive": round(counter["positive"] / total * 100, 2) if total else 0.0,
        "neutral": round(counter["neutral"] / total * 100, 2) if total else 0.0,
        "negative": round(counter["negative"] / total * 100, 2) if total else 0.0,
    }

# ✅ News fetcher with fallback
def fetch_news_titles(ticker: str) -> list[str]:
    if not ticker or not isinstance(ticker, str):
        print("⚠️ fetch_news_titles() received invalid or empty ticker.")
        return []

    api_key = os.getenv("NEWS_API_KEY") or ST_SECRETS.get("NEWS_API_KEY")
    if not api_key:
        raise RuntimeError("❌ Missing NEWS_API_KEY in environment or Streamlit secrets.")

    url = f"https://newsapi.org/v2/everything?q={ticker}&apiKey={api_key}&pageSize=10&sortBy=publishedAt"
    try:
        res = requests.get(url, timeout=5)
        res.raise_for_status()
        articles = res.json().get("articles", [])
        return [
            (a.get("title") or "") + " " + (a.get("description") or "")
            for a in articles if a.get("title") or a.get("description")
        ]
    except Exception as e:
        print(f"❌ NewsAPI error: {e}")
        return []

# ✅ FinBERT scorer (returns score polarity)
def analyze_with_finbert(texts: list[str]):
    tokenizer, model = load_finbert()
    inputs = tokenizer(texts, padding=True, truncation=True, max_length=512, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
    return probs[:, 2] - probs[:, 0]

# ✅ Main function: get sentiment for ticker
def get_sentiment_scores(ticker: str, log_to_csv=False) -> dict:
    try:
        news = fetch_news_titles(ticker)
        if not news:
            return {"positive": 0.0, "neutral": 0.0, "negative": 0.0}

        scores = analyze_with_finbert(news)
        sentiments = []
        for s in scores.tolist():
            if s >= 0.05:
                sentiments.append("positive")
            elif s <= -0.05:
                sentiments.append("negative")
            else:
                sentiments.append("neutral")

        summary = summarize_sentiments(sentiments)

        if log_to_csv:
            today = datetime.datetime.now().strftime("%Y-%m-%d")
            filename = "sentiment_scores.csv"
            file_exists = os.path.isfile(filename)
            with open(filename, mode="a", newline="") as f:
                writer = csv.writer(f)
                if not file_exists:
                    writer.writerow(["date", "ticker", "positive", "neutral", "negative"])
                writer.writerow([today, ticker, summary["positive"], summary["neutral"], summary["negative"]])

        print(f"✅ {ticker} Sentiment: {summary}")
        return summary

    except Exception as e:
        print(f"Sentiment error for {ticker}: {e}")
        return {"positive": 0.0, "neutral": 0.0, "negative": 0.0}
