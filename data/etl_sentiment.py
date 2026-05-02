# data/etl_sentiment.py
# ─────────────────────────────────────────────────────────────────────────────
# Sentiment ETL. Wraps sentiment_utils.py pipeline, caches results to SQLite.
#
# Why we need this:
#   - FinBERT inference on 80 headlines takes 3-5s per ticker
#   - Running it 31× per dashboard load = 2+ minutes of freezing
#   - Solution: run once daily (cron/scheduler), cache to SQLite
#   - builder.py reads from cache in <1ms
#
# Historical data problem:
#   - We can't go back and get 2022 news headlines
#   - sentiment_score = 0.0 for all historical rows (training data)
#   - Model treats 0.0 as "no sentiment signal" — valid, not a bug
#   - Going forward, real scores accumulate daily and improve predictions
#   - This means sentiment adds value immediately for LIVE signals
#
# Zero Streamlit imports. Backend only.
# Schedule: 3x daily on weekdays (pre-market, midday, close)
#   crontab -e
#   0  6  * * 1-5  cd ~/Desktop/ML_Quant_Fund && python -m data.etl_sentiment
#   0 12  * * 1-5  cd ~/Desktop/ML_Quant_Fund && python -m data.etl_sentiment
#   45 15 * * 1-5  cd ~/Desktop/ML_Quant_Fund && python -m data.etl_sentiment
# Manual override: python -m data.etl_sentiment --tickers AAPL NVDA --force --slot intraday
# ─────────────────────────────────────────────────────────────────────────────

from __future__ import annotations

import os
import sqlite3
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd

# ── Config ────────────────────────────────────────────────────────────────────
DB_PATH = Path(os.getenv("SENTIMENT_DB_PATH", "sentiment.db"))

# Sources to use by default — ordered by reliability.
# Google + Yahoo are free and require no API keys.
# Add NewsAPI/AlphaVantage/NYTimes keys to secrets.toml to enable more sources.
DEFAULT_SOURCES = ["Google", "Yahoo", "EDGAR", "Reddit"]  # StockTwits removed — API unreliable


# ══════════════════════════════════════════════════════════════════════════════
#  DATABASE SETUP
# ══════════════════════════════════════════════════════════════════════════════

def _init_db(db_path: Path = DB_PATH) -> sqlite3.Connection:
    """Create sentiment_scores table if it doesn't exist."""
    conn = sqlite3.connect(db_path)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS sentiment_scores (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker       TEXT    NOT NULL,
            date         TEXT    NOT NULL,
            time_slot    TEXT    NOT NULL DEFAULT 'daily',
            score        REAL    NOT NULL DEFAULT 0.0,
            positive_pct REAL    NOT NULL DEFAULT 0.0,
            negative_pct REAL    NOT NULL DEFAULT 0.0,
            neutral_pct  REAL    NOT NULL DEFAULT 0.0,
            n_headlines  INTEGER NOT NULL DEFAULT 0,
            sources_used TEXT    NOT NULL DEFAULT '',
            created_at   TEXT    NOT NULL,
            UNIQUE(ticker, date, time_slot)
        )
    """)
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_sent_ticker_date "
        "ON sentiment_scores(ticker, date)"
    )
    conn.commit()
    return conn


# ══════════════════════════════════════════════════════════════════════════════
#  FETCH + SCORE
# ══════════════════════════════════════════════════════════════════════════════

def _compute_sentiment(
    ticker: str,
    sources: list[str] = DEFAULT_SOURCES,
) -> dict:
    """
    Fetch headlines and run FinBERT. Returns dict with score and breakdown.
    score = positive_pct - negative_pct  (range: -100 to +100)
    Normalized to [-1, +1] for use as a feature.
    """
    try:
        from data.sentiment_utils import fetch_news_titles, _finbert_polarity
    except ImportError:
        from sentiment_utils import fetch_news_titles, _finbert_polarity

    headlines = fetch_news_titles(ticker, sources=sources or DEFAULT_SOURCES)

    if not headlines:
        return {
            "score": 0.0,
            "positive_pct": 0.0,
            "negative_pct": 0.0,
            "neutral_pct":  0.0,
            "n_headlines":  0,
        }

    pol = _finbert_polarity(headlines)

    positive = sum(1 for p in pol if p >=  0.05)
    negative = sum(1 for p in pol if p <= -0.05)
    neutral  = len(pol) - positive - negative
    n        = len(pol)

    pos_pct  = positive / n * 100
    neg_pct  = negative / n * 100
    neu_pct  = neutral  / n * 100

    # Net sentiment score normalized to [-1, +1]
    score = (pos_pct - neg_pct) / 100.0

    return {
        "score":        round(score,   4),
        "positive_pct": round(pos_pct, 2),
        "negative_pct": round(neg_pct, 2),
        "neutral_pct":  round(neu_pct, 2),
        "n_headlines":  n,
    }


# ══════════════════════════════════════════════════════════════════════════════
#  PUBLIC API — WRITE
# ══════════════════════════════════════════════════════════════════════════════

# ── Sources that need API keys ────────────────────────────────────────────────
SOURCES_WITH_KEYS = {
    "NewsAPI":   "NEWS_API_KEY",
    "AlphaV":    "ALPHA_VANTAGE_KEY",
    "Marketaux": "MARKETAUX_API_KEY",
    "NYTimes":   "NYT_API_KEY",
}

def _get_active_sources(requested: list[str]) -> list[str]:
    """Filter sources to those that are available (have keys or are free)."""
    free = {"Google", "Yahoo", "StockTwits", "EDGAR", "Reddit", "Pushshift"}
    active = []
    for s in requested:
        if s in free:
            active.append(s)
        elif s in SOURCES_WITH_KEYS:
            key_name = SOURCES_WITH_KEYS[s]
            if os.getenv(key_name):
                active.append(s)
            # else silently skip — no key available
    return active


def _current_time_slot() -> str:
    """Return the time slot label based on current hour."""
    h = datetime.now().hour
    if h < 9:   return "pre_market"
    if h < 13:  return "midday"
    if h < 16:  return "close"
    return "after_hours"


def run_sentiment_etl(
    tickers:   list[str],
    sources:   list[str] = DEFAULT_SOURCES,
    as_of:     date | None = None,
    time_slot: str | None = None,   # pre_market / midday / close / intraday
    db_path:   Path = DB_PATH,
    verbose:   bool = True,
    force:     bool = False,
) -> dict[str, float]:
    """
    Run sentiment pipeline for selected tickers and cache results.

    Parameters
    ----------
    tickers   : list of ticker strings — select only what you need
    sources   : which news sources to pull from
    as_of     : date to tag scores with (default: today)
    time_slot : label for this run (auto-detected from time if None)
    db_path   : SQLite file path
    verbose   : print progress
    force     : overwrite existing scores for this slot

    Returns dict: {ticker: sentiment_score}
    """
    as_of     = as_of or date.today()
    time_slot = time_slot or _current_time_slot()
    active    = _get_active_sources(sources)
    conn      = _init_db(db_path)
    scores    = {}
    now       = datetime.utcnow().isoformat()

    if verbose:
        print(f"  Slot: {time_slot}  Sources: {active}")

    for ticker in tickers:
        ticker = ticker.upper().strip()

        # Skip if already cached for this slot today (unless force=True)
        if not force:
            existing = conn.execute(
                "SELECT score FROM sentiment_scores WHERE ticker=? AND date=? AND time_slot=?",
                (ticker, str(as_of), time_slot)
            ).fetchone()
            if existing:
                scores[ticker] = existing[0]
                if verbose:
                    print(f"  {ticker} — cached {time_slot} ({existing[0]:+.3f})")
                continue

        if verbose:
            print(f"  {ticker} — running FinBERT ({len(sources)} sources)...")

        try:
            result = _compute_sentiment(ticker, sources=sources)
        except Exception as e:
            print(f"  ⚠ {ticker} failed: {e}")
            result = {
                "score": 0.0, "positive_pct": 0.0,
                "negative_pct": 0.0, "neutral_pct": 0.0, "n_headlines": 0,
            }

        conn.execute("""
            INSERT OR REPLACE INTO sentiment_scores
                (ticker, date, time_slot, score, positive_pct, negative_pct,
                 neutral_pct, n_headlines, sources_used, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            ticker, str(as_of), time_slot,
            result["score"], result["positive_pct"], result["negative_pct"],
            result["neutral_pct"], result["n_headlines"],
            ",".join(active), now,
        ))
        conn.commit()

        scores[ticker] = result["score"]

        if verbose:
            print(
                f"    ✓ score={result['score']:+.3f}  "
                f"+{result['positive_pct']:.0f}% "
                f"-{result['negative_pct']:.0f}% "
                f"n={result['n_headlines']}"
            )

    conn.close()
    return scores


# ══════════════════════════════════════════════════════════════════════════════
#  PUBLIC API — READ
# ══════════════════════════════════════════════════════════════════════════════

def load_sentiment_scores(
    ticker:     str,
    start_date: str | date | None = None,
    end_date:   str | date | None = None,
    db_path:    Path = DB_PATH,
) -> pd.DataFrame:
    """
    Read cached sentiment scores for a ticker over a date range.
    Called by features/builder.py.

    Returns DataFrame with columns: date, ticker, score, n_headlines
    Returns empty DataFrame if DB doesn't exist or no rows found.
    """
    if not db_path.exists():
        return pd.DataFrame()

    conn  = sqlite3.connect(db_path)
    # Get latest score per day (most recent time_slot wins)
    # Filter is_corrupted=1 rows (257 FinBERT-bug rows Feb 4 -> Mar 8 2026)
    # IS NULL covers rows from before the column was added.
    query = """
        SELECT date, ticker, score, positive_pct, negative_pct, n_headlines
        FROM sentiment_scores
        WHERE ticker = ?
        AND (is_corrupted = 0 OR is_corrupted IS NULL)
        AND id IN (
            SELECT MAX(id) FROM sentiment_scores
            WHERE ticker = ?
            AND (is_corrupted = 0 OR is_corrupted IS NULL)
            GROUP BY date
        )
    """
    params: list = [ticker.upper(), ticker.upper()]

    if start_date:
        query += " AND date >= ?"
        params.append(str(start_date))
    if end_date:
        query += " AND date <= ?"
        params.append(str(end_date))

    query += " ORDER BY date"

    try:
        df = pd.read_sql(query, conn, params=params, parse_dates=["date"])
        conn.close()
        return df
    except Exception as e:
        conn.close()
        print(f"  ⚠ Sentiment DB read failed for {ticker}: {e}")
        return pd.DataFrame()


def get_sentiment_score(
    ticker:  str,
    as_of:   date | None = None,
    db_path: Path = DB_PATH,
) -> float:
    """
    Get a single sentiment score for ticker on a specific date.
    Falls back to most recent score if exact date not found.
    Returns 0.0 if no data available.

    This is the function called by features/builder.py for live predictions.
    """
    as_of = as_of or date.today()

    if not db_path.exists():
        return 0.0

    conn = sqlite3.connect(db_path)
    try:
        # Try exact date first
        row = conn.execute(
            "SELECT score FROM sentiment_scores WHERE ticker=? AND date=?",
            (ticker.upper(), str(as_of))
        ).fetchone()

        if row:
            conn.close()
            return float(row[0])

        # Fall back to most recent score within last 3 days
        cutoff = str(as_of - timedelta(days=3))
        row = conn.execute(
            "SELECT score FROM sentiment_scores "
            "WHERE ticker=? AND date >= ? ORDER BY date DESC LIMIT 1",
            (ticker.upper(), cutoff)
        ).fetchone()

        conn.close()
        return float(row[0]) if row else 0.0

    except Exception:
        conn.close()
        return 0.0


# ══════════════════════════════════════════════════════════════════════════════
#  CLI
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse
    from models.train_all import DEFAULT_TICKERS

    parser = argparse.ArgumentParser(description="Run daily sentiment ETL")
    parser.add_argument("--tickers", nargs="+", default=None)
    parser.add_argument("--sources", nargs="+", default=DEFAULT_SOURCES)
    parser.add_argument("--force",   action="store_true",
                        help="Re-run even if today's score is already cached")
    parser.add_argument("--slot",    type=str, default=None,
                        help="Time slot label: pre_market / midday / close / intraday")
    args = parser.parse_args()

    tickers = args.tickers or DEFAULT_TICKERS
    print(f"\nRunning sentiment ETL for {len(tickers)} tickers...")
    print(f"Sources: {args.sources}\n")

    scores = run_sentiment_etl(tickers, sources=args.sources, force=args.force, time_slot=args.slot)

    print(f"\nDone. Summary:")
    for tkr, score in sorted(scores.items(), key=lambda x: x[1], reverse=True):
        bar = "█" * int(abs(score) * 20)
        sign = "+" if score >= 0 else "-"
        print(f"  {tkr:<6} {sign}{abs(score):.3f}  {bar}")
