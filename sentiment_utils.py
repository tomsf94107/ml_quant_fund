# sentiment_utils.py  v3.5  — multi‑source news → FinBERT sentiment
# ---------------------------------------------------------------------
# Adds Google News RSS, Yahoo Finance, and Reddit headlines on top of
# NewsAPI.  Uses only free / scrapable endpoints (no paid API needed).
# ---------------------------------------------------------------------

from __future__ import annotations

import os, time, random, datetime, csv, requests, certifi
from collections import Counter
from typing import List, Dict

import feedparser           # RSS parser (pure‑python, tiny)
import yfinance as yf        # already a project dependency
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Optional .env support for local runs -----------------------------------------------------
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

# Streamlit secrets fallback --------------------------------------------------------------
try:
    import streamlit as st
    ST_SECRETS = st.secrets  # type: ignore
except Exception:
    ST_SECRETS = {}

# SSL fix for macOS / certifi --------------------------------------------------------------
os.environ["SSL_CERT_FILE"] = certifi.where()

# -----------------------------------------------------------------------------------------
# FinBERT (cached singleton)
# -----------------------------------------------------------------------------------------

_FINBERT_MODEL = "yiyanghkust/finbert-tone"
_tokenizer = None
_model = None

def _load_finbert():
    global _tokenizer, _model
    if _tokenizer is None or _model is None:
        _tokenizer = AutoTokenizer.from_pretrained(_FINBERT_MODEL)
        _model     = AutoModelForSequenceClassification.from_pretrained(_FINBERT_MODEL)
    return _tokenizer, _model

# -----------------------------------------------------------------------------------------
# Helper: summarise list["positive"|"neutral"|"negative"] → % dict
# -----------------------------------------------------------------------------------------

def _summarise(sentiments: List[str]) -> Dict[str, float]:
    c = Counter(sentiments)
    total = float(sum(c.values())) or 1.0  # avoid div/0
    return {
        "positive": round(c["positive"] / total * 100, 2),
        "neutral" : round(c["neutral"]  / total * 100, 2),
        "negative": round(c["negative"] / total * 100, 2),
    }

# -----------------------------------------------------------------------------------------
# 1) NewsAPI helper (kept from previous version) -------------------------------------------
# -----------------------------------------------------------------------------------------

_NEWS_HEADERS = {"User-Agent": "ml-quant-sentiment/0.1"}

def _newsapi_titles(ticker: str, api_key: str, page_size: int = 10) -> List[str]:
    url = (
        "https://newsapi.org/v2/everything"
        f"?q={ticker}&apiKey={api_key}&pageSize={page_size}&sortBy=publishedAt"
    )
    try:
        resp = requests.get(url, headers=_NEWS_HEADERS, timeout=5)
        resp.raise_for_status()
        items = resp.json().get("articles", [])
        return [
            f"{a.get('title','')} {a.get('description','')}".strip()
            for a in items if a.get("title") or a.get("description")
        ]
    except Exception as ex:
        print(f"❌ NewsAPI error for {ticker}: {ex}")
        return []

# -----------------------------------------------------------------------------------------
# 2) Google News RSS -----------------------------------------------------------------------
# -----------------------------------------------------------------------------------------

def _google_news_titles(ticker: str) -> List[str]:
    rss_url = (
        "https://news.google.com/rss/search?q="
        f"{ticker}+stock&hl=en-US&gl=US&ceid=US:en"
    )
    try:
        feed = feedparser.parse(rss_url)
        return [e.title for e in feed.entries if e.title]
    except Exception as ex:
        print(f"❌ Google News RSS error for {ticker}: {ex}")
        return []

# -----------------------------------------------------------------------------------------
# 3) Yahoo Finance headlines ---------------------------------------------------------------
# -----------------------------------------------------------------------------------------

def _yahoo_finance_titles(ticker: str) -> List[str]:
    try:
        news_items = yf.Ticker(ticker).news or []
        return [n["title"] for n in news_items if n.get("title")]
    except Exception as ex:
        print(f"❌ Yahoo Finance fetch error for {ticker}: {ex}")
        return []

# -----------------------------------------------------------------------------------------
# 4) Reddit r/stocks search (public JSON, no key) -----------------------------------------
# -----------------------------------------------------------------------------------------

_REDDIT_HEADERS = {"User-Agent": "Mozilla/5.0 sentiment-fetcher"}

def _reddit_titles(ticker: str, limit: int = 20) -> List[str]:
    url = (
        "https://www.reddit.com/search.json"
        f"?q={ticker}&limit={limit}&type=link&restrict_sr=on&sr_detail=true"
    )
    try:
        resp = requests.get(url, headers=_REDDIT_HEADERS, timeout=5)
        resp.raise_for_status()
        data = resp.json()
        return [child["data"]["title"] for child in data["data"]["children"]]
    except Exception as ex:
        print(f"❌ Reddit fetch error for {ticker}: {ex}")
        return []

# -----------------------------------------------------------------------------------------
# Master fetcher — aggregates all free sources --------------------------------------------
# -----------------------------------------------------------------------------------------

def fetch_news_titles(ticker: str) -> List[str]:
    if not ticker or not isinstance(ticker, str):
        print("⚠️ fetch_news_titles(): empty or invalid ticker")
        return []

    titles: List[str] = []

    # 1) NewsAPI if key available
    api_key = os.getenv("NEWS_API_KEY") or ST_SECRETS.get("NEWS_API_KEY")
    if api_key:
        titles += _newsapi_titles(ticker, api_key)

    # 2) Google News
    titles += _google_news_titles(ticker)

    # 3) Yahoo Finance
    titles += _yahoo_finance_titles(ticker)

    # 4) Reddit (be polite → small sleep)
    time.sleep(random.uniform(0.3, 0.7))
    titles += _reddit_titles(ticker)

    # Clean & deduplicate
    cleaned = {t.strip() for t in titles if t and len(t) > 15}
    return list(cleaned)[:30]  # cap to 30 headlines for speed

# -----------------------------------------------------------------------------------------
# FinBERT inference -----------------------------------------------------------------------
# -----------------------------------------------------------------------------------------

def _finbert_polarity(texts: List[str]):
    tok, mdl = _load_finbert()
    batch = tok(texts, truncation=True, padding=True, max_length=512, return_tensors="pt")
    with torch.no_grad():
        logits = mdl(**batch).logits
    probs = torch.nn.functional.softmax(logits, dim=-1)
    # polarity score = positive prob - negative prob
    return (probs[:, 2] - probs[:, 0]).tolist()

# -----------------------------------------------------------------------------------------
# Public API ------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------

def get_sentiment_scores(ticker: str, *, log_to_csv: bool = False) -> Dict[str, float]:
    """Fetches headlines from multiple free sources, passes them through FinBERT,
    and returns a percentage breakdown of positive / neutral / negative.
    Always returns a dict with three keys; values sum to 100 (or 0 if no data)."""

    try:
        headlines = fetch_news_titles(ticker)
        if not headlines:
            return {"positive": 0.0, "neutral": 0.0, "negative": 0.0}

        polarities = _finbert_polarity(headlines)
        sentiments = [
            "positive" if p >= 0.05 else "negative" if p <= -0.05 else "neutral"
            for p in polarities
        ]
        summary = _summarise(sentiments)

        if log_to_csv:
            _log_to_csv(ticker, summary)

        print(f"✅ {ticker} sentiment -> {summary}")
        return summary

    except Exception as ex:
        print(f"❌ Sentiment pipeline error for {ticker}: {ex}")
        return {"positive": 0.0, "neutral": 0.0, "negative": 0.0}

# -----------------------------------------------------------------------------------------
# CSV logger (optional) -------------------------------------------------------------------
# -----------------------------------------------------------------------------------------

def _log_to_csv(ticker: str, summary: Dict[str, float]):
    today = datetime.datetime.now().strftime("%Y-%m-%d")
    fname = "sentiment_scores.csv"
    new_file = not os.path.exists(fname)
    with open(fname, "a", newline="") as f:
        writer = csv.writer(f)
        if new_file:
            writer.writerow(["date", "ticker", "positive", "neutral", "negative"])
        writer.writerow([today, ticker, summary["positive"], summary["neutral"], summary["negative"]])
