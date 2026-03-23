#!/usr/bin/env python3
"""
Monday Sentiment Scorer — ML Quant Fund
Fetches weekend news for all tickers via yfinance, scores sentiment
using Anthropic API, stores in sentiment.db for Monday predictions.

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
from datetime import date, datetime, timedelta
from pathlib import Path

import yfinance as yf
import anthropic

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
log = logging.getLogger(__name__)

DB_PATH = Path("data/sentiment.db")
ROOT    = Path(__file__).parent.parent

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

def _get_headlines(ticker: str, days_back: int = 3) -> list[str]:
    """Fetch recent news headlines for a ticker via yfinance."""
    try:
        t = yf.Ticker(ticker)
        news = t.news or []
        cutoff = datetime.now() - timedelta(days=days_back)
        headlines = []
        for item in news[:10]:
            # Handle both old and new yfinance news structure
            inner = item.get("content", item)
            pub_str = inner.get("pubDate", "") or inner.get("displayTime", "")
            title = inner.get("title", "") or item.get("title", "")
            if not title:
                continue
            if pub_str:
                try:
                    pub_dt = datetime.fromisoformat(pub_str.replace("Z", "+00:00")).replace(tzinfo=None)
                    if pub_dt < cutoff:
                        continue
                except Exception:
                    pass
            headlines.append(title)
        return headlines[:5]  # max 5 headlines per ticker
    except Exception:
        return []

def _score_sentiment(ticker: str, headlines: list[str], client: anthropic.Anthropic) -> dict:
    """Score sentiment of headlines using Anthropic API."""
    if not headlines:
        return {"score": 0.0, "label": "NEUTRAL", "confidence": 0.0}

    headlines_text = "\n".join(f"- {h}" for h in headlines)
    prompt = f"""You are a financial sentiment analyst. Analyze these recent news headlines for {ticker} and return ONLY a JSON object with no other text.

Headlines:
{headlines_text}

Return exactly this JSON format:
{{"score": <float between -1.0 and 1.0>, "label": "<BULLISH|BEARISH|NEUTRAL>", "confidence": <float between 0.0 and 1.0>}}

Where:
- score: -1.0 = very bearish, 0.0 = neutral, 1.0 = very bullish
- label: overall direction
- confidence: how clear the signal is (0=unclear, 1=very clear)"""

    try:
        msg = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=150,
            messages=[{"role": "user", "content": prompt}]
        )
        raw = msg.content[0].text.strip()
        # Strip markdown code blocks if present
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

    scored = 0
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
