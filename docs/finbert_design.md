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

---

## Session E — LLM Structured Filing Extraction (planned, future)

**Motivation:** FinBERT only outputs a single sentiment number per text chunk. It cannot extract structured fields like guidance changes, EPS/revenue beats, new product announcements, or specific P&L drivers. To capture these signals — which research shows can add 3-7% AUC lift vs bag-of-words sentiment — we need a real LLM-based extractor.

### Architecture

Use Claude API (Haiku for cost efficiency) to extract structured fields from each filing. New columns persist alongside FinBERT scores.

### Schema additions

```sql
ALTER TABLE finbert_filings ADD COLUMN guidance_change TEXT;
  -- 'raised' | 'maintained' | 'lowered' | 'withdrawn' | NULL
ALTER TABLE finbert_filings ADD COLUMN eps_vs_consensus TEXT;
  -- 'beat' | 'inline' | 'miss' | NULL
ALTER TABLE finbert_filings ADD COLUMN rev_vs_consensus TEXT;
  -- 'beat' | 'inline' | 'miss' | NULL
ALTER TABLE finbert_filings ADD COLUMN buyback_announced INTEGER;
ALTER TABLE finbert_filings ADD COLUMN buyback_amount_b REAL;
ALTER TABLE finbert_filings ADD COLUMN dividend_change TEXT;
  -- 'increased' | 'maintained' | 'cut' | 'initiated' | 'suspended' | NULL
ALTER TABLE finbert_filings ADD COLUMN new_product_announced INTEGER;
ALTER TABLE finbert_filings ADD COLUMN ceo_change INTEGER;
ALTER TABLE finbert_filings ADD COLUMN cfo_change INTEGER;
ALTER TABLE finbert_filings ADD COLUMN restructuring_announced INTEGER;
ALTER TABLE finbert_filings ADD COLUMN segment_strength TEXT;
  -- JSON: {"data_center": "strong", "gaming": "weak"} or NULL
ALTER TABLE finbert_filings ADD COLUMN key_positives TEXT;
  -- JSON array of bullet strings
ALTER TABLE finbert_filings ADD COLUMN key_negatives TEXT;
  -- JSON array of bullet strings
ALTER TABLE finbert_filings ADD COLUMN llm_extracted_at TEXT;
ALTER TABLE finbert_filings ADD COLUMN llm_model TEXT;
```

### Extraction prompt template
You are a financial analyst extracting structured data from a SEC filing.
Read the filing text and return ONLY a JSON object with these fields:
{
"guidance_change": "raised|maintained|lowered|withdrawn|null",
"eps_vs_consensus": "beat|inline|miss|null",
"rev_vs_consensus": "beat|inline|miss|null",
"buyback_announced": true|false,
"buyback_amount_b": <number in billions>|null,
"dividend_change": "increased|maintained|cut|initiated|suspended|null",
"new_product_announced": true|false,
"ceo_change": true|false,
"cfo_change": true|false,
"restructuring_announced": true|false,
"segment_strength": {"segment_name": "strong|weak|stable", ...}|null,
"key_positives": ["<bullet 1>", "<bullet 2>", ...],
"key_negatives": ["<bullet 1>", "<bullet 2>", ...]
}
Be strict: only set true/positive if explicitly stated. Use null if unclear.
FILING TEXT:
{filing_text}

### Feature engineering

From the JSON columns, derive these model features (added to FEATURE_COLUMNS):

| Feature | Type | Source |
|---|---|---|
| llm_guidance_score | int (-2 to +2) | -2=withdrawn, -1=lowered, 0=maintained, +1=raised |
| llm_eps_beat | int (-1/0/+1) | miss/inline/beat |
| llm_rev_beat | int (-1/0/+1) | miss/inline/beat |
| llm_buyback_pct_mcap | float | buyback_amount_b / market_cap |
| llm_dividend_score | int (-2 to +2) | suspended/cut/maintained/increased/initiated |
| llm_new_product | int (0/1) | binary |
| llm_exec_change | int (0/1/2) | 0=none, 1=CFO change, 2=CEO change |
| llm_positives_count | int | len(key_positives) |
| llm_negatives_count | int | len(key_negatives) |

Total: 9 new features. FEATURE_COLUMNS would grow from 82 to 91.

### Compute budget

- 1,248 existing filings × ~5,000 tokens each = ~6.2M input tokens
- ~500 tokens output per filing = ~625k output tokens
- Claude Haiku: ~$5 total backfill
- Claude Sonnet: ~$30 total backfill (better extraction quality)
- Forward incremental: ~10 filings/day × $0.01-0.05 = trivial

### Build sequence

1. Schema migration on finbert_filings
2. Build data/etl_llm_filing_extract.py with extraction prompt
3. Test on 5 known filings (NVDA Q4 26, AAPL Q2 26, RZLV recent, etc) — manually verify accuracy
4. Backfill across 1,248 filings (~30-60 min wall clock at 5 req/sec)
5. Add feature loader load_llm_features_pit() in data/alpha_sources.py
6. Wire into builder.py + classifier.py FEATURE_COLUMNS
7. Walk-forward backtest to measure lift

### Risk

- LLM hallucination risk: must include "null if unclear" in prompt, validate against schema
- Cost runaway: cap at $50/month with rate limiting
- Model drift: re-validate quarterly against held-out filings

### Expected impact

Research benchmark (Frankel et al 2022, Jegadeesh & Wu 2013, others): structured earnings extraction adds 3-7% AUC lift vs sentiment-only. We currently have 0% lift from FinBERT (insufficient history + sparse signal). Session E should be the dominant contributor when measured.

### Session A vs Session E

| Aspect | Session A (today, FinBERT) | Session E (future, LLM) |
|---|---|---|
| Output | 1 sentiment number | 9 structured fields |
| Compute cost | Free (local CPU) | ~$5 backfill + $1/month |
| History captured | Yes (1y of 8-Ks) | Same filings, more detail |
| Predictive lift | ~0-1% | ~3-7% |
| Engineering complexity | Moderate (parser + scorer) | Higher (prompt engineering + JSON validation) |

Session E should ideally run AFTER FinBERT proves the baseline data plumbing works.
