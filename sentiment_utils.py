# sentiment_utils.py  v3.7  — add StockTwits, SEC EDGAR, Marketaux, AlphaVantage
# ---------------------------------------------------------------------
# Free-only sentiment pipeline.  Headline sources:
#   1. NewsAPI (if key present)
#   2. Google News RSS  (50 items)
#   3. Yahoo Finance news
#   4. Reddit live JSON  (search endpoint, polite UA + retry)
#   5. Pushshift (historical Reddit submissions)
#   6. StockTwits API  (30 most recent posts)
#   7. SEC EDGAR RSS (Atom feed of filings)
#   8. Marketaux news endpoint
#   9. AlphaVantage News & Sentiment (beta)
# Headlines deduped and capped at 80, then scored by FinBERT.
# ---------------------------------------------------------------------

from __future__ import annotations
import os, time, random, datetime, csv, requests, certifi
from collections import Counter
from typing import List, Dict

import feedparser            # RSS/Atom parser (pure-python)
import yfinance as yf         # Yahoo Finance client
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
)

# ------------------------------------------------------------------
# Optional .env + Streamlit secrets support
# ------------------------------------------------------------------
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

try:
    import streamlit as st
    ST_SECRETS = st.secrets  # type: ignore
except Exception:
    ST_SECRETS = {}

# certifi fix for some OSes
os.environ["SSL_CERT_FILE"] = certifi.where()

# ------------------------------------------------------------------
# FinBERT (lazy-loaded singleton)
# ------------------------------------------------------------------
_FINBERT_MODEL = "yiyanghkust/finbert-tone"
_tokenizer = None
_model = None

def _load_finbert():
    global _tokenizer, _model
    if _tokenizer is None or _model is None:
        _tokenizer = AutoTokenizer.from_pretrained(_FINBERT_MODEL)
        _model     = AutoModelForSequenceClassification.from_pretrained(_FINBERT_MODEL)
    return _tokenizer, _model

# ------------------------------------------------------------------
# Utility: convert list[str] → % dict
# ------------------------------------------------------------------
def _summarise(sentiments: List[str]) -> Dict[str, float]:
    c = Counter(sentiments)
    total = float(sum(c.values())) or 1.0
    return {
        "positive": round(c["positive"] / total * 100, 2),
        "neutral" : round(c["neutral" ] / total * 100, 2),
        "negative": round(c["negative"] / total * 100, 2),
    }

# ------------------------------------------------------------------
# NewsAPI helper
# ------------------------------------------------------------------
_NEWS_HEADERS = {"User-Agent": "ml-quant-sentiment/0.4"}

def _newsapi_titles(ticker: str, api_key: str, page_size: int = 30) -> List[str]:
    url = ("https://newsapi.org/v2/everything?"
           f"q={ticker}&apiKey={api_key}&pageSize={page_size}&sortBy=publishedAt")
    try:
        r = requests.get(url, headers=_NEWS_HEADERS, timeout=8)
        r.raise_for_status()
        arts = r.json().get("articles", [])
        return [f"{a.get('title','')} {a.get('description','')}".strip()
                for a in arts if a.get("title") or a.get("description")]
    except Exception as ex:
        print(f"❌ NewsAPI error for {ticker}: {ex}")
        return []

# ------------------------------------------------------------------
# Google News RSS (50)
# ------------------------------------------------------------------
def _google_news_titles(ticker: str, max_items: int = 50) -> List[str]:
    rss = ("https://news.google.com/rss/search?q="
           f"{ticker}+stock&hl=en-US&gl=US&ceid=US:en")
    try:
        feed = feedparser.parse(rss)
        return [e.title for e in feed.entries[:max_items] if e.title]
    except Exception as ex:
        print(f"❌ Google RSS error for {ticker}: {ex}")
        return []

# ------------------------------------------------------------------
# Yahoo Finance
# ------------------------------------------------------------------
def _yahoo_finance_titles(ticker: str) -> List[str]:
    try:
        items = yf.Ticker(ticker).news or []
        return [n.get("title","") for n in items if n.get("title")]
    except Exception as ex:
        print(f"❌ Yahoo news error for {ticker}: {ex}")
        return []

# ------------------------------------------------------------------
# Reddit live search (retry + UA)
# ------------------------------------------------------------------
_REDDIT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Ubuntu; Linux x86_64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/125.0.0.0 Safari/537.36 sentiment-fetcher/0.4"
    )
}
def _reddit_titles(ticker: str, limit: int = 20) -> List[str]:
    url = ("https://www.reddit.com/search.json"
           f"?q={ticker}&limit={limit}&type=link&restrict_sr=on&sr_detail=true")
    for attempt in (1, 2):
        try:
            r = requests.get(url, headers=_REDDIT_HEADERS, timeout=6)
            r.raise_for_status()
            data = r.json()
            return [c["data"]["title"] for c in data["data"]["children"]]
        except requests.exceptions.HTTPError as ex:
            if r.status_code == 403 and attempt == 1:
                time.sleep(1.5)
                continue
            print(f"❌ Reddit error for {ticker}: {ex}")
            break
        except Exception as ex:
            print(f"❌ Reddit error for {ticker}: {ex}")
            break
    return []

# ------------------------------------------------------------------
# Pushshift historical Reddit
# ------------------------------------------------------------------
def _pushshift_titles(ticker: str, size: int = 30) -> List[str]:
    url = ("https://api.pushshift.io/reddit/search/submission/"
           f"?q={ticker}&size={size}&fields=title")
    try:
        r = requests.get(url, headers=_REDDIT_HEADERS, timeout=8)
        r.raise_for_status()
        entries = r.json().get("data", [])
        return [d.get("title","") for d in entries if d.get("title")]
    except Exception as ex:
        print(f"❌ Pushshift error for {ticker}: {ex}")
        return []

# ------------------------------------------------------------------
# StockTwits API
# ------------------------------------------------------------------
def _stocktwits_titles(ticker: str) -> List[str]:
    url = f"https://api.stocktwits.com/api/2/streams/symbol/{ticker}.json"
    try:
        data = requests.get(url, timeout=5).json().get("messages", [])
        return [m.get("body","") for m in data if m.get("body")]
    except Exception as ex:
        print(f"❌ StockTwits error for {ticker}: {ex}")
        return []

# ------------------------------------------------------------------
# SEC EDGAR RSS (Atom)
# ------------------------------------------------------------------
SEC_EDGAR_HEADERS = {
    "User-Agent": "MyMLApp/1.0 (mailto:tomsf94107@gmail.com)"
}

def _sec_edgar_titles(ticker: str, count: int = 40) -> List[str]:
    url = (
        "https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&"
        f"CIK={ticker}&type=&owner=exclude&count={count}&output=atom"
    )
    try:
        # ↓ use SEC_EDGAR_HEADERS here
        r = requests.get(url, headers=SEC_EDGAR_HEADERS, timeout=8)
        r.raise_for_status()
        feed = feedparser.parse(r.text)
        return [e.title for e in feed.entries if e.title]
    except Exception as ex:
        print(f"❌ SEC EDGAR RSS error for {ticker}: {ex}")
        return []

# ------------------------------------------------------------------
# Marketaux news endpoint
# ------------------------------------------------------------------
def _marketaux_titles(ticker: str, page_size: int = 20) -> List[str]:
    token = os.getenv("MARKETAUX_API_KEY") or ST_SECRETS.get("MARKETAUX_API_KEY")
    if not token:
        return []
    url = ("https://api.marketaux.com/v1/news/all?"
           f"api_token={token}&symbols={ticker}&limit={page_size}")
    try:
        arts = requests.get(url, timeout=6).json().get("data", [])
        return [a.get("title","") for a in arts if a.get("title")]
    except Exception as ex:
        print(f"❌ Marketaux error for {ticker}: {ex}")
        return []

# ------------------------------------------------------------------
# Alpha Vantage News & Sentiment (beta)
# ------------------------------------------------------------------
def _alpha_vantage_titles(ticker: str) -> List[str]:
    key = os.getenv("ALPHA_VANTAGE_KEY") or ST_SECRETS.get("ALPHA_VANTAGE_KEY")
    if not key:
        return []
    url = ("https://www.alphavantage.co/query?"
           f"function=NEWS_SENTIMENT&tickers={ticker}&apikey={key}")
    try:
        feed = requests.get(url, timeout=8).json().get("feed", [])
        return [f.get("title","") for f in feed if f.get("title")]
    except Exception as ex:
        print(f"❌ AlphaVantage error for {ticker}: {ex}")
        return []


# ------------------------------------------------------------------
# Aggregator
# ------------------------------------------------------------------
MAX_HEADLINES = 80

def fetch_news_titles(
    ticker: str,
    sources: List[str] | None = None,    # ← optional filter
    **kwargs,
) -> List[str]:
    """Aggregates headlines from our 9 free sources, filtered & capped."""
    ALL_SRCS = {
        "NewsAPI":    lambda: _newsapi_titles(
                            ticker,
                            os.getenv("NEWS_API_KEY") or ST_SECRETS.get("NEWS_API_KEY"),
                            kwargs.get("page_size", 30)
                        ),
        "Google":     lambda: _google_news_titles(
                            ticker,
                            kwargs.get("max_items", 50)
                        ),
        "Yahoo":      lambda: _yahoo_finance_titles(ticker),
        "Reddit":     lambda: _reddit_titles(
                            ticker,
                            kwargs.get("limit", 20)
                        ),
        "Pushshift":  lambda: _pushshift_titles(
                            ticker,
                            kwargs.get("size", 30)
                        ),
        "StockTwits": lambda: _stocktwits_titles(ticker),
        "EDGAR":      lambda: _sec_edgar_titles(
                            ticker,
                            kwargs.get("count", 40)
                        ),
        "Marketaux":  lambda: _marketaux_titles(
                            ticker,
                            kwargs.get("page_size", 20)
                        ),
        "AlphaV":     lambda: _alpha_vantage_titles(ticker),
    }

    if not ticker:
        return []

    # Default to all sources if none selected
    pick = sources or list(ALL_SRCS.keys())

    titles: List[str] = []
    for src in pick:
        fetch_fn = ALL_SRCS.get(src)
        if fetch_fn:
            titles += fetch_fn()

    # Dedupe, filter too-short strings, and cap
    cleaned = {t.strip() for t in titles if t and len(t) > 15}
    return list(cleaned)[: kwargs.get("max_headlines", MAX_HEADLINES)]

# ------------------------------------------------------------------
# FinBERT inference
# ------------------------------------------------------------------
def _finbert_polarity(texts: List[str]) -> List[float]:
    tok, mdl = _load_finbert()
    batch = tok(texts, truncation=True, padding=True, max_length=512, return_tensors="pt")
    with torch.no_grad():
        logits = mdl(**batch).logits
    probs = torch.nn.functional.softmax(logits, dim=-1)
    return (probs[:, 2] - probs[:, 0]).tolist()

# ------------------------------------------------------------------
# Public API
# ------------------------------------------------------------------

def get_sentiment_scores(
    ticker: str,
    sources: List[str] | None = None,
    *,
    log_to_csv: bool = False
) -> Dict[str, float]:
    """
    Return percent breakdown of positive/neutral/negative (sum≈100).
    Optional `sources` list filters which feeds to include.
    """
    try:
        # ← pass the user’s source filter here
        headlines = fetch_news_titles(ticker, sources=sources)

        if not headlines:
            return {"positive": 0.0, "neutral": 0.0, "negative": 0.0}

        pol = _finbert_polarity(headlines)
        labels = [
            "positive" if p >= 0.05
            else "negative" if p <= -0.05
            else "neutral"
            for p in pol
        ]
        summary = _summarise(labels)

        if log_to_csv:
            _log_to_csv(ticker, summary)

        return summary

    except Exception as ex:
        print(f"❌ Sentiment pipeline error for {ticker}: {ex}")
        return {"positive": 0.0, "neutral": 0.0, "negative": 0.0}

# ------------------------------------------------------------------
# CSV logger (optional)
# ------------------------------------------------------------------

def _log_to_csv(ticker: str, summary: Dict[str, float]) -> None:
    today = datetime.datetime.now().strftime("%Y-%m-%d")
    fname = "sentiment_scores.csv"
    new = not os.path.exists(fname)
    with open(fname, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["date","ticker","positive","neutral","negative"])
        w.writerow([today, ticker, summary["positive"], summary["neutral"], summary["negative"]])
