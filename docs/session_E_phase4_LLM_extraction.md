# Session E Phase 4: LLM Filing Text Extraction

**Status:** Designed, NOT shipped. Implementation deferred.
**Date:** May 22, 2026
**Estimated effort:** 4-5h code + ~$9 LLM cost (one-time backfill)
**Predecessor phases:** Phase 1 (existing eps_surprise), Phase 2 (rev_growth from Polygon), Phase 3 (8-K Item codes from edgartools)

---

## Why LLM (not structured)

Phases 1-3 covered everything available from structured sources:

| Feature category | Source | Phase |
|---|---|---|
| EPS surprise | earnings_surprises (yfinance ETL) | Phase 1 (existing) |
| Revenue growth | Polygon Financials | Phase 2 (shipped today) |
| 8-K event codes | edgartools (Item 5.02, 1.01, etc) | Phase 3 (shipped today) |
| Sentiment | FinBERT scores | Session A (prior) |

What remains genuinely needs natural language understanding:

- llm_guidance_score (-2 to +2): forward guidance tone in earnings releases / 8-Ks
- llm_buyback_pct_mcap: announced buyback as % of market cap
- llm_dividend_score (-2 to +2): dividend initiation/increase/cut
- llm_new_product: binary, new product launch mentioned
- llm_positives_count / llm_negatives_count: qualitative tone signals

These are NOT in 8-K Item codes (Item 5.02 just says exec change; LLM tells us CFO promoted to CEO after activist investor pressure). They are in filing text but require understanding context.

---

## Implementation paths

### Path A: refetch + store + extract (RECOMMENDED long-term)
1. Schema migration: add text column to finbert_filings (or new table filing_text)
2. Refetch 2,846 historical filings from SEC EDGAR (10 req/sec = ~5 min)
3. Store cleaned text
4. LLM-extract 5 features per filing
5. Modify Session A ETL to store text going forward

Cost breakdown:
- SEC refetch: ~5 min (rate-limited, free)
- Storage: ~30 MB text (negligible)
- LLM:
  - Haiku 4.5: ~$9 (1248 unique filings × ~5000 input × $1/M + 500 output × $5/M)
  - Sonnet 4.5: ~$30 (3x cost, higher quality)
- Code: 4-5 hours

Pros:
- One-time cost; all historical data available for training
- Future filings automatic; re-extractable if prompts improve

Cons:
- Larger blast radius (touches Session A ETL)
- LLM cost recurring if prompts revised

### Path B: forward-only extraction (CHEAPER, slower signal)
1. Modify Session A ETL: when fetching new filing, also call LLM
2. Store extracted features in new columns of finbert_filings
3. Historical 1,248 filings remain LLM-blank

Cost: ~$0.01-0.05/day; 1-2 hours code; no refetch.

Pros: low effort/cost; lower risk.
Cons: ~12 months until model has training signal; lost historical; walk-forward unusable for past.

Honest recommendation: Path A if budget allows; Path B if delaying.

---

## Schema migration

```sql
ALTER TABLE finbert_filings ADD COLUMN text TEXT;
ALTER TABLE finbert_filings ADD COLUMN llm_guidance_score REAL;
ALTER TABLE finbert_filings ADD COLUMN llm_buyback_pct_mcap REAL;
ALTER TABLE finbert_filings ADD COLUMN llm_dividend_score REAL;
ALTER TABLE finbert_filings ADD COLUMN llm_new_product INTEGER;
ALTER TABLE finbert_filings ADD COLUMN llm_positives_count INTEGER;
ALTER TABLE finbert_filings ADD COLUMN llm_negatives_count INTEGER;
ALTER TABLE finbert_filings ADD COLUMN llm_extracted_at TEXT;
ALTER TABLE finbert_filings ADD COLUMN llm_model_version TEXT;
```

Idempotent: re-run extraction skips rows with non-NULL llm_extracted_at.

---

## Prompt design

System prompt:
You are a structured financial analyst extracting features from SEC filings.
Return ONLY valid JSON. No prose, no markdown fences.
Score guidance and dividends from -2 (very negative) to +2 (very positive).
Score 0 if not mentioned.
For buyback, report dollar amount announced / market cap as decimal (0.05 = 5% of mcap).
For new_product, return 1 if new product/service launch announced, 0 otherwise.
For positives/negatives, count distinct items, each capped at 5.

User prompt per filing:
Filing type: {filing_type}
Filing date: {filing_date}
Ticker: {ticker}
Market cap (estimate): ${market_cap}M
Text:
{first 5000 chars}
Return JSON:
{
"guidance_score": -2 to +2,
"buyback_pct_mcap": 0.0 to 1.0,
"dividend_score": -2 to +2,
"new_product": 0 or 1,
"positives_count": 0 to 5,
"negatives_count": 0 to 5
}

Use Haiku 4.5 for cost. Cross-validate on 100 random filings against manual labels.

---

## Rule #1 audit

| Letter | Concern | Mitigation |
|---|---|---|
| (a) Audit | LLM behavior on empty text, foreign-filer text, non-English | Verify on 5-ticker pilot before full backfill |
| (b) Silent errors | LLM returns invalid JSON, rate limits, API key fails | Explicit logging; raise after 3 retries |
| (c) Flag-flip audit | ML_QUANT_LLM_EXTRACT=1 to enable | Single env flag, env-gated like inst features |
| (d) Verify script | Hand-label 50 random filings, check LLM agreement | Target >70% agreement before trusting |
| (e) Built-not-known cold | Token costs vary by filing length; some filings are 50k tokens | Truncate to first 5000 chars; document cost variance |
| (f) Test real data | Pilot 10 filings, inspect output | Required before backfill |
| (g) Gap-check | finbert_filings is read by Session A ETL and FinBERT loader | Need to test both still work after schema migration |
| (h) Verify chain | LLM → DB → loader → feature → model → output | End-to-end test with 1 ticker after backfill |
| (i) Compiled OK ≠ verified | AST pass not enough | Manual review of 100 sample extractions |

Key risk: LLM extraction is a black box. Output quality varies. Recommend keeping FinBERT scores AS WELL as LLM features (not replacing) so model can use both.

---

## Validation strategy

Before backfilling all 1,248 filings:
1. Pick 10 known-good filings (recent NVDA earnings 8-K, AMD product launch 8-K, etc)
2. Hand-label expected feature values
3. Run LLM on those 10
4. Compare. Aim for >80% agreement on discrete features, MAE <0.5 on continuous

After backfill:
1. Compute feature distributions, check for outliers
2. Sanity-check tickers with known events (e.g., NVDA buyback announcements should have llm_buyback_pct_mcap > 0 on those filing_dates)
3. Run walk-forward backtest WITH and WITHOUT LLM features
4. Decide if marginal AUC lift justifies $9 recurring cost (if any)

---

## Outstanding decisions before implementing

1. Path A vs Path B (refetch + backfill vs forward-only)?
2. Haiku ($9) vs Sonnet ($30)?
3. Truncation strategy for long filings (5000 chars or smarter section selection)?
4. Should LLM features replace or supplement FinBERT scores?
5. Refresh policy: re-extract if prompt changes? Or freeze v1?

These should be settled before writing any code.
