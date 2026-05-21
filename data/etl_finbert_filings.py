"""
data/etl_finbert_filings.py
─────────────────────────────────────────────────────────────────────────────
FinBERT historical backfill + forward-incremental ETL for SEC filings.

Session A (May 21 2026): 8-K + NT-10Q/NT-10K only.
Sessions B-D extend to 10-Q, 10-K, S-*, DEF 14A, 425, SC 14D9, 6-K.

Usage:
  python -m data.etl_finbert_filings --ticker AAPL --days-back 365
  python -m data.etl_finbert_filings --all --days-back 365
  python -m data.etl_finbert_filings --incremental --days-back 7
"""

from __future__ import annotations
import argparse
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from data.sec_section_parser import (
    get_cik, list_filings, fetch_filing_html,
    extract_8k_items, extract_8k_press_release, extract_nt_notice,
    is_earnings_8k,
)

DB_PATH = Path("data/sentiment.db")
TICKERS_FILE = Path("tickers.txt")
SESSION_A_FORMS = ["8-K", "NT 10-Q", "NT 10-K"]

GUIDANCE_RAISED = [
    "raised guidance", "increased outlook", "raised our outlook",
    "exceeded expectations", "record revenue", "beat expectations",
    "strong demand", "raised full-year",
]
GUIDANCE_LOWERED = [
    "withdrew guidance", "lowered guidance", "reduced outlook",
    "challenging environment", "below expectations", "missed expectations",
    "softer than expected", "headwinds",
]

_FINBERT_PIPELINE = None


def _get_finbert():
    global _FINBERT_PIPELINE
    if _FINBERT_PIPELINE is None:
        from transformers import pipeline
        _FINBERT_PIPELINE = pipeline(
            "sentiment-analysis",
            model="ProsusAI/finbert",
            tokenizer="ProsusAI/finbert",
            device=-1,
        )
    return _FINBERT_PIPELINE


def score_text(text: str, max_chars: int = 4000) -> dict:
    if not text or len(text) < 50:
        return {"sentiment_score": 0.0, "sentiment_label": "neutral",
                "confidence": 0.0, "section_text_len": 0}

    text = text[:max_chars]
    pipe = _get_finbert()
    CHUNK = 1800
    chunks = [text[i:i + CHUNK] for i in range(0, len(text), CHUNK)]

    pos_score = neg_score = neu_score = 0.0
    n = 0
    for chunk in chunks:
        try:
            result = pipe(chunk, truncation=True, max_length=512)
            if not result:
                continue
            r = result[0]
            label = r["label"].lower()
            score = float(r["score"])
            if label == "positive":
                pos_score += score
            elif label == "negative":
                neg_score += score
            else:
                neu_score += score
            n += 1
        except Exception:
            continue

    if n == 0:
        return {"sentiment_score": 0.0, "sentiment_label": "neutral",
                "confidence": 0.0, "section_text_len": len(text)}

    pos_avg = pos_score / n
    neg_avg = neg_score / n
    neu_avg = neu_score / n
    compound = pos_avg - neg_avg

    if pos_avg > neg_avg and pos_avg > neu_avg:
        label, conf = "positive", pos_avg
    elif neg_avg > pos_avg and neg_avg > neu_avg:
        label, conf = "negative", neg_avg
    else:
        label, conf = "neutral", neu_avg

    return {"sentiment_score": round(compound, 4), "sentiment_label": label,
            "confidence": round(conf, 4), "section_text_len": len(text)}


def _detect_guidance(text: str) -> Optional[str]:
    tl = text.lower()
    if any(w in tl for w in GUIDANCE_RAISED):
        return "RAISED"
    if any(w in tl for w in GUIDANCE_LOWERED):
        return "LOWERED"
    return None


def _earnings_multiplier(sentiment: float, guidance: Optional[str]) -> float:
    if guidance == "RAISED":
        return 1.10
    if guidance == "LOWERED":
        return 0.90
    if sentiment > 0.5:
        return 1.05
    if sentiment < -0.5:
        return 0.95
    return 1.0


def process_8k(conn, ticker, cik, filing) -> int:
    accn = filing["accession"]
    filing_date = filing["filing_date"]

    cur = conn.execute(
        "SELECT COUNT(*) FROM finbert_filings WHERE ticker=? AND accession=?",
        (ticker, accn))
    if cur.fetchone()[0] > 0:
        return 0

    html = fetch_filing_html(cik, accn, filing["primary_document"])
    if not html:
        _insert_error_row(conn, ticker, accn, filing_date, "8-K", "primary_doc_fetch_failed")
        return 1

    items = extract_8k_items(html)
    if not items:
        _insert_error_row(conn, ticker, accn, filing_date, "8-K", "no_items_extracted")
        return 1

    is_earn = is_earnings_8k(items)
    rows_inserted = 0

    if is_earn:
        pr_text = extract_8k_press_release(cik, accn)
        if pr_text:
            scores = score_text(pr_text)
            guidance = _detect_guidance(pr_text)
            mult = _earnings_multiplier(scores["sentiment_score"], guidance)
            _insert_row(conn, ticker, accn, filing_date, "8-K", "press_release",
                        scores, mult, guidance, 1, 0)
            rows_inserted += 1

    # Filings with parseable items
    for item_no, item_text in items.items():
        if item_no == "_whole":
            continue
        scores = score_text(item_text)
        section_name = f"item_{item_no.replace('.', '_')}"
        _insert_row(conn, ticker, accn, filing_date, "8-K", section_name,
                    scores, None, None, int(is_earn), 0)
        rows_inserted += 1

    # Fallback: filings with no Item X.XX headers (governance/board changes,
    # debt issuance, etc) get scored as whole_doc + press_release if available
    if "_whole" in items and rows_inserted == 0:
        scores = score_text(items["_whole"])
        pr_text = extract_8k_press_release(cik, accn)
        if pr_text:
            pr_scores = score_text(pr_text)
            guidance = _detect_guidance(pr_text)
            mult = _earnings_multiplier(pr_scores["sentiment_score"], guidance)
            _insert_row(conn, ticker, accn, filing_date, "8-K", "press_release",
                        pr_scores, mult, guidance, 0, 0)
            rows_inserted += 1
        _insert_row(conn, ticker, accn, filing_date, "8-K", "whole_doc",
                    scores, None, None, 0, 0)
        rows_inserted += 1

    conn.commit()
    return rows_inserted


def process_nt_notice(conn, ticker, cik, filing) -> int:
    accn = filing["accession"]
    filing_date = filing["filing_date"]
    form = filing["form"]

    cur = conn.execute(
        "SELECT COUNT(*) FROM finbert_filings WHERE ticker=? AND accession=?",
        (ticker, accn))
    if cur.fetchone()[0] > 0:
        return 0

    html = fetch_filing_html(cik, accn, filing["primary_document"])
    if not html:
        _insert_error_row(conn, ticker, accn, filing_date, form, "fetch_failed")
        return 1

    text = extract_nt_notice(html)
    if not text:
        _insert_error_row(conn, ticker, accn, filing_date, form, "no_text_extracted")
        return 1

    scores = score_text(text)
    _insert_row(conn, ticker, accn, filing_date, form, "late_notice",
                scores, None, None, 0, 1)
    conn.commit()
    return 1


def _insert_row(conn, ticker, accn, filing_date, filing_type, section,
                scores, earnings_mult, guidance, is_earnings, is_late_notice):
    conn.execute("""
        INSERT INTO finbert_filings
        (ticker, accession, filing_date, filing_type, section,
         sentiment_score, sentiment_label, confidence, section_text_len,
         earnings_multiplier, guidance, is_earnings, is_late_notice,
         error, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, NULL, ?)
    """, (
        ticker, accn, filing_date, filing_type, section,
        scores["sentiment_score"], scores["sentiment_label"],
        scores["confidence"], scores["section_text_len"],
        earnings_mult, guidance, is_earnings, is_late_notice,
        datetime.now(timezone.utc).isoformat(),
    ))


def _insert_error_row(conn, ticker, accn, filing_date, filing_type, error_msg):
    conn.execute("""
        INSERT OR IGNORE INTO finbert_filings
        (ticker, accession, filing_date, filing_type, section,
         sentiment_score, sentiment_label, confidence, section_text_len,
         earnings_multiplier, guidance, is_earnings, is_late_notice,
         error, created_at)
        VALUES (?, ?, ?, ?, '_error', NULL, NULL, NULL, NULL,
                NULL, NULL, 0, 0, ?, ?)
    """, (
        ticker, accn, filing_date, filing_type, error_msg,
        datetime.now(timezone.utc).isoformat(),
    ))
    conn.commit()


def ingest_ticker(conn, ticker, days_back=365, verbose=True) -> dict:
    cik = get_cik(ticker)
    if not cik:
        return {"ticker": ticker, "error": "cik_lookup_failed", "rows": 0}

    filings = list_filings(cik, days_back=days_back, form_types=SESSION_A_FORMS)
    if verbose:
        print(f"  {ticker}: CIK={cik}, {len(filings)} filings to process")

    total_rows = 0
    for i, f in enumerate(filings):
        form = f["form"]
        if verbose:
            print(f"    [{i+1}/{len(filings)}] {f['filing_date']} {form:10s} {f['accession']}")
        try:
            if form == "8-K":
                n = process_8k(conn, ticker, cik, f)
            elif form in ("NT 10-Q", "NT 10-K"):
                n = process_nt_notice(conn, ticker, cik, f)
            else:
                continue
            total_rows += n
        except Exception as e:
            if verbose:
                print(f"      ERROR: {e}")
            _insert_error_row(conn, ticker, f["accession"], f["filing_date"], form, str(e)[:200])
            total_rows += 1

    return {"ticker": ticker, "filings": len(filings), "rows": total_rows}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ticker", type=str)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--days-back", type=int, default=365)
    ap.add_argument("--incremental", action="store_true")
    ap.add_argument("--verbose", action="store_true", default=True)
    args = ap.parse_args()

    if args.ticker:
        tickers = [args.ticker.upper()]
    elif args.all or args.incremental:
        if not TICKERS_FILE.exists():
            print(f"ERROR: {TICKERS_FILE} not found")
            sys.exit(1)
        tickers = [t.strip().upper() for t in TICKERS_FILE.read_text().splitlines()
                   if t.strip() and not t.strip().startswith("#")]
    else:
        print("ERROR: provide --ticker X or --all")
        sys.exit(1)

    print(f"FinBERT filings ingest: {len(tickers)} tickers, last {args.days_back}d")
    print(f"Forms: {SESSION_A_FORMS}")
    print()

    conn = sqlite3.connect(DB_PATH)
    total_rows = total_filings = 0
    for i, ticker in enumerate(tickers):
        print(f"[{i+1}/{len(tickers)}] {ticker}")
        result = ingest_ticker(conn, ticker, days_back=args.days_back, verbose=args.verbose)
        if "error" in result:
            print(f"  ERROR: {result['error']}")
        else:
            total_filings += result["filings"]
            total_rows += result["rows"]
    conn.close()
    print()
    print(f"Done: {total_rows} rows from {total_filings} filings across {len(tickers)} tickers")


if __name__ == "__main__":
    main()
