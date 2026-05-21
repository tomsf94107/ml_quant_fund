# FinBERT Comprehensive SEC Filings Design — May 21 2026

Scope: full historical FinBERT coverage across 8-K, 10-Q, 10-K, NT-, S-, DEF 14A, 425, SC 14D9, 6-K filings for all 125 tickers.

EXCLUDED from FinBERT scope (captured elsewhere as structured data):
- 13D, 13G (institutional ownership) — captured via institutional_trades ingest
- Form 4 (insider transactions) — captured via insider_flows table

## Filing types in scope
| Filing | Sections | Notes |
|---|---|---|
| 8-K | Items + press-release exhibit | High signal |
| 10-Q | MD&A + Risk Factors | Medium signal |
| 10-K | MD&A + Risk Factors + Business | Medium signal |
| NT-10-Q / NT-10-K | Whole | Rare, strong negative |
| S-1, S-3 | Risk Factors + Use of Proceeds | Dilution signal |
| DEF 14A | Comp + Risk Oversight | Low priority |
| 425, SC 14D9 | Whole | M&A catalyst |
| 6-K | Press release | Foreign issuer |

## Database schema (in data/sentiment.db)

CREATE TABLE finbert_filings (
    ticker             TEXT NOT NULL,
    accession          TEXT NOT NULL,
    filing_date        TEXT NOT NULL,
    filing_type        TEXT NOT NULL,
    section            TEXT NOT NULL,
    sentiment_score    REAL,
    sentiment_label    TEXT,
    confidence         REAL,
    section_text_len   INTEGER,
    earnings_multiplier REAL,
    guidance           TEXT,
    is_earnings        INTEGER,
    is_late_notice     INTEGER,
    error              TEXT,
    created_at         TEXT NOT NULL,
    PRIMARY KEY (ticker, accession, section)
);

## Build sequence (4 sessions)
- Session A: 8-K + NT-* (in progress)
- Session B: 10-Q
- Session C: 10-K
- Session D: S-*, DEF 14A, 425, SC 14D9, 6-K + forward-incremental + feature wiring

## Output features (5)
- finbert_sentiment (latest score)
- finbert_mult (earnings multiplier)
- finbert_days_since (recency 0-90)
- finbert_is_earnings_5d (binary)
- finbert_is_late_notice_30d (binary)

## Sessions A-D total: ~11 hours work + 6 hours compute
