# data/etl_gdelt.py
# ─────────────────────────────────────────────────────────────────────────────
# GDELT historical sentiment backfill.
# Fetches past news headlines from GDELT (free, global, back to 2015)
# and runs FinBERT to populate sentiment.db with historical scores.
#
# Why GDELT:
#   - Free, no API key needed
#   - 500M+ articles in 65 languages
#   - Updated every 15 minutes
#   - Covers geopolitics, macro, company news
#
# Strategy:
#   - Run nightly for a few tickers at a time (avoids overloading GDELT)
#   - After 2-3 weeks you have meaningful historical data
#   - Retrain monthly as coverage improves
#
# Run manually:   python -m data.etl_gdelt --tickers AAPL NVDA --days 30
# Run nightly:    45 23 * * 1-5  cd ~/Desktop/ML_Quant_Fund && python -m data.etl_gdelt
# ─────────────────────────────────────────────────────────────────────────────

from __future__ import annotations

import os
import time
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd
import requests

from data.etl_sentiment import (
    _init_db, _compute_sentiment, DB_PATH, DEFAULT_SOURCES
)

# ── GDELT config ──────────────────────────────────────────────────────────────
GDELT_API   = "https://api.gdeltproject.org/api/v2/doc/doc"
GDELT_DELAY = 5.0   # seconds between requests — 5s reduces 429 errors significantly
MAX_ARTICLES = 40   # per ticker per day — enough for FinBERT signal


# ══════════════════════════════════════════════════════════════════════════════
#  GDELT FETCHER
# ══════════════════════════════════════════════════════════════════════════════

def _fetch_gdelt_headlines(
    ticker:    str,
    as_of:     date,
    max_items: int = MAX_ARTICLES,
) -> list[str]:
    """
    Fetch headlines from GDELT for a ticker on a specific past date.
    Uses GDELT's ArtList mode to get article titles.

    Returns list of headline strings.
    """
    # GDELT timerange format: YYYYMMDDHHMMSS
    start_dt = datetime.combine(as_of, datetime.min.time())
    end_dt   = start_dt + timedelta(days=1)
    fmt      = "%Y%m%d%H%M%S"

    # Search for ticker name — use common name for better coverage
    # e.g. "AAPL" → "Apple", "NVDA" → "Nvidia"
    query = _ticker_to_query(ticker)

    params = {
        "query":     f'{query} sourcelang:eng',
        "mode":      "ArtList",
        "maxrecords": min(max_items, 250),
        "startdatetime": start_dt.strftime(fmt),
        "enddatetime":   end_dt.strftime(fmt),
        "format":    "json",
        "sort":      "DateDesc",
    }

    try:
        r = requests.get(GDELT_API, params=params, timeout=15)
        r.raise_for_status()
        data   = r.json()
        arts   = data.get("articles") or []
        titles = [a.get("title", "") for a in arts if a.get("title")]
        return [t.strip() for t in titles if len(t.strip()) > 15][:max_items]
    except Exception as e:
        print(f"    ⚠ GDELT error for {ticker} on {as_of}: {e}")
        return []


def _ticker_to_query(ticker: str) -> str:
    """Map ticker to a better search query for GDELT."""
    mapping = {
        "AAPL":  "Apple Inc",
        "NVDA":  "Nvidia",
        "TSLA":  "Tesla",
        "AMD":   "AMD semiconductor",
        "MSFT":  "Microsoft",
        "GOOG":  "Google Alphabet",
        "META":  "Meta Facebook",
        "AMZN":  "Amazon",
        "PLTR":  "Palantir",
        "CRWD":  "CrowdStrike",
        "SHOP":  "Shopify",
        "SNOW":  "Snowflake",
        "DDOG":  "Datadog",
        "MRNA":  "Moderna",
        "UNH":   "UnitedHealth",
        "JNJ":   "Johnson Johnson",
        "PFE":   "Pfizer",
        "NVO":   "Novo Nordisk",
        "NFLX":  "Netflix",
        "PYPL":  "PayPal",
        "SMCI":  "Super Micro Computer",
        "CNC":   "Centene",
        "TSM":   "TSMC Taiwan Semiconductor",
        "AXP":   "American Express",
        "BSX":   "Boston Scientific",
        "DUOL":  "Duolingo",
        "MP":    "MP Materials rare earth",
        "OPEN":  "Opendoor Technologies",
        "ZM":    "Zoom Video",
        "CRCL":  "Circle financial",
        "RZLV":  "Razor Labs",
        "SLV":   "silver ETF iShares",
    }
    return mapping.get(ticker.upper(), ticker)


# ══════════════════════════════════════════════════════════════════════════════
#  BACKFILL RUNNER
# ══════════════════════════════════════════════════════════════════════════════

def run_gdelt_backfill(
    tickers:     list[str],
    days_back:   int  = 30,
    db_path:     Path = DB_PATH,
    verbose:     bool = True,
    skip_existing: bool = True,   # don't re-score dates already in DB
    max_per_run: int  = 5,        # max tickers per nightly run (avoid rate limits)
) -> dict[str, int]:
    """
    Backfill historical sentiment using GDELT.

    For each ticker × date, fetches headlines and runs FinBERT.
    Results tagged with time_slot='gdelt_backfill'.

    Parameters
    ----------
    tickers      : list of tickers to backfill
    days_back    : how many calendar days to go back
    db_path      : SQLite path
    verbose      : print progress
    skip_existing: skip dates already scored (safe to re-run)
    max_per_run  : cap tickers per run to avoid overloading GDELT

    Returns dict: {ticker: dates_scored}
    """
    conn    = _init_db(db_path)
    results = {}
    now_str = datetime.utcnow().isoformat()

    # Cap tickers per run
    tickers = tickers[:max_per_run]

    for ticker in tickers:
        ticker = ticker.upper().strip()
        dated  = 0

        if verbose:
            print(f"\n  {ticker} — backfilling {days_back} days via GDELT...")

        for d in range(days_back, 0, -1):
            as_of = date.today() - timedelta(days=d)

            # Skip weekends
            if as_of.weekday() >= 5:
                continue

            # Skip if already scored
            if skip_existing:
                row = conn.execute(
                    "SELECT id FROM sentiment_scores "
                    "WHERE ticker=? AND date=? AND time_slot='gdelt_backfill'",
                    (ticker, str(as_of))
                ).fetchone()
                if row:
                    continue

            headlines = _fetch_gdelt_headlines(ticker, as_of)

            if not headlines:
                if verbose:
                    print(f"    {as_of} — no headlines found")
                time.sleep(GDELT_DELAY)
                continue

            # Run FinBERT on fetched headlines
            try:
                try:
                    from data.sentiment_utils import _finbert_polarity
                except ImportError:
                    from sentiment_utils import _finbert_polarity

                pol      = _finbert_polarity(headlines)
                n        = len(pol)
                positive = sum(1 for p in pol if p >=  0.05)
                negative = sum(1 for p in pol if p <= -0.05)
                neutral  = n - positive - negative

                pos_pct = positive / n * 100
                neg_pct = negative / n * 100
                neu_pct = neutral  / n * 100
                score   = round((pos_pct - neg_pct) / 100.0, 4)

                conn.execute("""
                    INSERT OR REPLACE INTO sentiment_scores
                        (ticker, date, time_slot, score,
                         positive_pct, negative_pct, neutral_pct,
                         n_headlines, sources_used, created_at)
                    VALUES (?, ?, 'gdelt_backfill', ?, ?, ?, ?, ?, 'GDELT', ?)
                """, (
                    ticker, str(as_of), score,
                    round(pos_pct, 2), round(neg_pct, 2), round(neu_pct, 2),
                    n, now_str,
                ))
                conn.commit()
                dated += 1

                if verbose:
                    bar  = "█" * int(abs(score) * 10)
                    sign = "+" if score >= 0 else "-"
                    print(f"    {as_of}  {sign}{abs(score):.3f}  {bar}  n={n}")

            except Exception as e:
                if verbose:
                    print(f"    {as_of} — FinBERT failed: {e}")

            time.sleep(GDELT_DELAY)   # polite delay between GDELT calls

        results[ticker] = dated
        if verbose:
            print(f"  ✓ {ticker} — {dated} dates scored")

    conn.close()
    return results


# ══════════════════════════════════════════════════════════════════════════════
#  CLI
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse
    from models.train_all import DEFAULT_TICKERS

    parser = argparse.ArgumentParser(description="GDELT historical sentiment backfill")
    parser.add_argument("--tickers",  nargs="+", default=None,
                        help="Tickers to backfill (default: first 5 from DEFAULT_TICKERS)")
    parser.add_argument("--days",     type=int, default=30,
                        help="Days to go back (default 30)")
    parser.add_argument("--max",      type=int, default=5,
                        help="Max tickers per run (default 5, avoids rate limits)")
    parser.add_argument("--no-skip",  action="store_true",
                        help="Re-score even if date already in DB")
    args = parser.parse_args()

    tickers = args.tickers or DEFAULT_TICKERS

    print(f"\nGDELT backfill: {len(tickers[:args.max])} tickers × {args.days} days")
    print(f"Estimated time: ~{len(tickers[:args.max]) * args.days * GDELT_DELAY / 60:.0f} minutes\n")

    results = run_gdelt_backfill(
        tickers=tickers,
        days_back=args.days,
        skip_existing=not args.no_skip,
        max_per_run=args.max,
    )

    total = sum(results.values())
    print(f"\nDone. {total} total dates scored.")
    for tkr, n in results.items():
        print(f"  {tkr:<6} {n} dates")
