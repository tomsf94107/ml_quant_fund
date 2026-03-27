#!/usr/bin/env python3
"""
Monday Sentiment Scorer — ML Quant Fund
Fetches news for all tickers via Yahoo Finance RSS + Google News RSS + yfinance,
scores sentiment using Anthropic API, stores in sentiment.db for Monday predictions.

Run: Sunday night ~9 PM ET (after US markets close Friday, before Asia opens)
Cron: 0 8 * * 1 (8 AM Vietnam Monday = 9 PM ET Sunday)
"""
from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import sqlite3
import json
import time
import logging
import urllib.request
from datetime import date, datetime, timedelta
from pathlib import Path

import yfinance as yf
import anthropic

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
log = logging.getLogger(__name__)

DB_PATH = Path("data/sentiment.db")
ROOT    = Path(__file__).parent.parent

BROWSER_HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
}

def _init_db(conn):
    conn.execute("""
        CREATE TABLE IF NOT EXISTS monday_sentiment (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker          TEXT NOT NULL,
            score_date      TEXT NOT NULL,
            sentiment_score REAL NOT NULL,
            sentiment_label TEXT NOT NULL,
            confidence      REAL NOT NULL,
            headlines       TEXT NOT NULL,
            created_at      TEXT NOT NULL,
            UNIQUE(ticker, score_date)
        )
    """)
    conn.commit()


def _fetch_rss(url: str) -> list:
    """Fetch and parse an RSS feed with browser user agent. Returns list of entries."""
    try:
        import feedparser
        req      = urllib.request.Request(url, headers=BROWSER_HEADERS)
        response = urllib.request.urlopen(req, timeout=10)
        feed     = feedparser.parse(response.read())
        return feed.entries
    except Exception:
        return []


def _get_headlines(ticker: str, days_back: int = 3) -> list[str]:
    """
    Fetch recent news headlines from 3 sources (always all run):
      1. Yahoo Finance RSS  — up to 20 entries, same source as yfinance but fresher
      2. Google News RSS    — up to 100 entries, real-time broad coverage
      3. yfinance           — structured news API

    Deduplicates by first 60 chars of lowercased title.
    Returns up to 10 unique headlines.
    """
    cutoff    = datetime.utcnow() - timedelta(days=days_back)
    headlines = []
    seen      = set()

    def _clean(title: str) -> str:
        return title.strip().lower()

    def _is_duplicate(title: str) -> bool:
        c = _clean(title)
        for s in seen:
            if c[:60] == s[:60]:
                return True
        return False

    def _add(title: str) -> bool:
        title = title.strip()
        if not title or _is_duplicate(title):
            return False
        headlines.append(title)
        seen.add(_clean(title))
        return True

    def _parse_rss_entry(entry) -> str | None:
        """Extract title from RSS entry, filter by recency."""
        if hasattr(entry, 'published_parsed') and entry.published_parsed:
            try:
                pub = datetime(*entry.published_parsed[:6])
                if pub < cutoff:
                    return None
            except Exception:
                pass
        return entry.get('title', '').strip() or None

    # --- Source 1: Yahoo Finance RSS ---
    try:
        url     = f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={ticker}&region=US&lang=en-US"
        entries = _fetch_rss(url)
        for entry in entries[:20]:
            title = _parse_rss_entry(entry)
            if title:
                _add(title)
    except Exception as e:
        log.debug(f"Yahoo RSS failed for {ticker}: {e}")

    # --- Source 2: Google News RSS ---
    try:
        import urllib.parse
        query   = urllib.parse.quote(f"{ticker} stock")
        url     = f"https://news.google.com/rss/search?q={query}&hl=en-US&gl=US&ceid=US:en"
        entries = _fetch_rss(url)
        for entry in entries[:20]:
            title = _parse_rss_entry(entry)
            if title:
                # Google News appends " - Source Name" — strip for cleaner scoring
                if ' - ' in title:
                    title = title.rsplit(' - ', 1)[0].strip()
                _add(title)
    except Exception as e:
        log.debug(f"Google News RSS failed for {ticker}: {e}")

    # --- Source 3: yfinance ---
    try:
        t    = yf.Ticker(ticker)
        news = t.news or []
        for item in news[:15]:
            inner   = item.get("content", item)
            pub_str = inner.get("pubDate", "") or inner.get("displayTime", "")
            title   = inner.get("title", "") or item.get("title", "")
            if not title:
                continue
            if pub_str:
                try:
                    pub_dt = datetime.fromisoformat(
                        pub_str.replace("Z", "+00:00")
                    ).replace(tzinfo=None)
                    if pub_dt < cutoff:
                        continue
                except Exception:
                    pass
            _add(title)
    except Exception as e:
        log.debug(f"yfinance news failed for {ticker}: {e}")

    return headlines[:10]  # top 10 unique headlines across all sources


def _score_sentiment(ticker: str, headlines: list[str], client: anthropic.Anthropic) -> dict:
    """Score sentiment of headlines using Anthropic API."""
    if not headlines:
        return {"score": 0.0, "label": "NEUTRAL", "confidence": 0.0}

    headlines_text = "\n".join(f"- {h}" for h in headlines)
    prompt = f"""You are a financial sentiment analyst. Analyze these recent news headlines for {ticker} and return ONLY a JSON object with no other text.

Headlines (up to 10, from multiple sources including Yahoo Finance, Google News, and yfinance):
{headlines_text}

Return exactly this JSON format:
{{"score": <float between -1.0 and 1.0>, "label": "<BULLISH|BEARISH|NEUTRAL>", "confidence": <float between 0.0 and 1.0>}}

Where:
- score: -1.0 = very bearish, 0.0 = neutral, 1.0 = very bullish
- label: overall direction based on the full set of headlines
- confidence: how clear and consistent the signal is across all headlines (0=mixed/unclear, 1=very clear)"""

    try:
        msg = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=150,
            messages=[{"role": "user", "content": prompt}]
        )
        raw = msg.content[0].text.strip()
        raw = raw.replace("```json", "").replace("```", "").strip()
        if not raw:
            return {"score": 0.0, "label": "NEUTRAL", "confidence": 0.0}
        result = json.loads(raw)
        return {
            "score":      max(-1.0, min(1.0, float(result.get("score", 0.0)))),
            "label":      result.get("label", "NEUTRAL").upper(),
            "confidence": max(0.0, min(1.0, float(result.get("confidence", 0.0)))),
        }
    except Exception as e:
        log.warning(f"Sentiment API failed for {ticker}: {e}")
        return {"score": 0.0, "label": "NEUTRAL", "confidence": 0.0}


def run_monday_sentiment():
    """Main function — score all tickers and store results."""
    today = date.today().isoformat()
    log.info(f"Monday sentiment scorer — {today}")

    # Load tickers
    tickers = [t.strip() for t in open(ROOT / "tickers.txt").readlines() if t.strip()]
    log.info(f"Scoring {len(tickers)} tickers")

    client = anthropic.Anthropic()
    conn   = sqlite3.connect(DB_PATH)
    _init_db(conn)

    scored  = 0
    bullish = 0
    bearish = 0

    for i, ticker in enumerate(tickers, 1):
        try:
            headlines = _get_headlines(ticker, days_back=3)
            result    = _score_sentiment(ticker, headlines, client)

            conn.execute("""
                INSERT OR REPLACE INTO monday_sentiment
                (ticker, score_date, sentiment_score, sentiment_label, confidence, headlines, created_at)
                VALUES (?,?,?,?,?,?,?)
            """, (
                ticker, today,
                result["score"], result["label"], result["confidence"],
                json.dumps(headlines),
                datetime.now().isoformat()
            ))
            conn.commit()

            scored += 1
            if result["label"] == "BULLISH": bullish += 1
            if result["label"] == "BEARISH": bearish += 1

            if i % 10 == 0:
                log.info(f"  [{i}/{len(tickers)}] scored {scored}, bullish={bullish}, bearish={bearish}")

            time.sleep(0.5)  # rate limit

        except Exception as e:
            log.error(f"{ticker}: {e}")

    conn.close()
    log.info(f"Done — {scored} scored, {bullish} bullish, {bearish} bearish")
    return scored

if __name__ == "__main__":
    run_monday_sentiment()
