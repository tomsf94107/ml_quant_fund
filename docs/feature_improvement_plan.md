# Feature Improvement Plan — May 21 2026

Generated after systematic feature audit revealed 11 dead features
(7 with train/serve mismatch, 3 genuinely broken pipelines, 1 correct-static).

This document plans the 4 follow-up items in priority order.

---

## ROI Summary

| # | Item | Effort | Expected AUC | Risk |
|---|------|--------|--------------|------|
| 1 | Drop 4 dead-and-unfixable features | 30 min | +0.001 (cleaner) | Very low |
| 2 | Sector ETF features | 2-3 hours | +0.005 to +0.015 | Low |
| 3 | FinBERT historical backfill | 6-8 hours setup + 2 hours compute | +0.005 to +0.015 | Medium |
| 4 | pc_ratio_change_5d + iv_skew_change features (future) | 8-12 hours | +0.003 to +0.010 | Medium |

**Total commit budget:** ~20 hours of focused work spread over 3-4 sessions.

---

## Item 1 — Drop dead features (PRIORITY: SHIP TODAY OR NEXT SESSION)

### Scope

Remove the following from FEATURE_COLUMNS in models/classifier.py:

| Feature | Reason |
|---------|--------|
| analyst_upside | yfinance live-only, lagging, low ML value |
| analyst_buy_pct | yfinance live-only, lagging, low ML value |
| analyst_mult | duplicates model prob confidence |
| fear_greed | alternative.me snapshot broadcast (no historical) |

**Keep but flag for later:**
- iv_skew_snap, pc_ratio_snap — kept for now; replaced by item #4 features
- finbert_sentiment, finbert_mult — kept; fixed by item #3
- short_ratio, short_pct_float — kept; backfillable via Massive (separate session)
- rev_surprise — kept; fix etl_earnings to populate (separate session)
- is_pandemic — correct as-is (historical static flag)

### Implementation

1. Edit models/classifier.py FEATURE_COLUMNS
2. Backup file as .bak.before_drop_dead4_<date>
3. Verify Pipeline B retrain succeeds with N-4 features
4. Verify no other code references the dropped columns
5. Commit

### Risk mitigation

- Models retrain on next Pipeline B with 4 fewer columns. Universe AUC may shift +/- 0.005.
- If AUC drops materially (>0.01), restore from backup.

---

## Item 2 — Sector ETF features (PRIORITY: HIGH)

### Hypothesis

May 2026 regression (-7% to -21% on OKLO/CEG/ORIC/MRNA) was thematic. Model
saw individual stock features but had no signal that the sector was rolling
over (XLV for biotech, etc).

### Scope

Add 8 sector return features to features/builder.py:

| Feature | ETF | Sector |
|---------|-----|--------|
| xle_ret_5d | XLE | Energy |
| xlv_ret_5d | XLV | Healthcare |
| xlf_ret_5d | XLF | Financials |
| xlk_ret_5d | XLK | Technology |
| xlu_ret_5d | XLU | Utilities |
| xli_ret_5d | XLI | Industrials |
| xlp_ret_5d | XLP | Consumer Staples |
| xly_ret_5d | XLY | Consumer Discretionary |

Already exists: xlk_ret (single-day). Extend to 5-day return on all 8 sectors.

### Implementation

1. In features/builder.py, find the existing xlk_ret block
2. Replicate the pattern for all 8 ETFs at the 5-day horizon
3. Add the 8 column names to FEATURE_COLUMNS
4. Verify build with one ticker before Pipeline B retrain
5. Walk-forward backtest to measure AUC delta

### Data source

Massive /v2/aggs/ticker/{ETF}/range/... — already in use for stock prices.
No new API integration needed.

### Risk mitigation

- ETF returns are highly correlated with broad market features (spy_ret).
- Mitigation: keep both. XGBoost will handle correlation. Worst case: feature importances are distributed across correlated features.

---

## Item 3 — FinBERT historical backfill (PRIORITY: MEDIUM-HIGH)

### Architecture goal

Eliminate train/serve mismatch by:
- Building a historical database of FinBERT sentiment per (ticker, filing_date)
- Querying that database identically in training AND inference modes
- Same code path -> no mismatch

### Database schema

```sql
CREATE TABLE finbert_history (
    ticker         TEXT NOT NULL,
    filing_date    TEXT NOT NULL,  -- YYYY-MM-DD
    accession      TEXT NOT NULL,  -- SEC filing accession ID
    sentiment_score REAL,           -- FinBERT compound score [-1, 1]
    earnings_mult  REAL,            -- derived multiplier ~ [0.8, 1.2]
    filing_type    TEXT,            -- 8-K, 10-Q, 10-K
    is_earnings    INTEGER,         -- 1 if this is an earnings 8-K
    raw_text_len   INTEGER,         -- length of text scored (for QA)
    created_at     TEXT,
    PRIMARY KEY (ticker, accession)
);

CREATE INDEX idx_finbert_ticker_date ON finbert_history(ticker, filing_date);
```

### One-time backfill

1. Pull 8-K filings for each of 125 tickers going back 365 days from SEC EDGAR
2. Filter to earnings 8-Ks using existing heuristic in alpha_sources.py
3. Run FinBERT on each filing's text
4. Store in finbert_history table

Compute estimate: 125 tickers x ~4 8-Ks/year x 5 years = ~2,500 inferences
x ~3 sec each (CPU) = ~2 hours batch compute on a single core
x ~30 min wall clock with 4-core parallelism

### Forward incremental (daily)

Add to Pipeline A (Stage 1.5, after insider ETL):
- Check SEC EDGAR for new 8-Ks per ticker since last run
- Run FinBERT on new filings
- Append to finbert_history

~5 minutes/day, ~10-20 new filings to score across universe.

### Feature load (replaces current training_mode skip block)

```python
# OLD (training_mode skip — caused train/serve mismatch):
if training_mode:
    df["finbert_sentiment"] = 0.0
    df["finbert_mult"]      = 1.0
else:
    # ... live API fetch

# NEW (same code path both modes):
finbert = load_finbert_for_ticker_pit(ticker, date_index)
df["finbert_sentiment"] = finbert["sentiment_score"]
df["finbert_mult"]      = finbert["earnings_mult"]
```

load_finbert_for_ticker_pit() queries finbert_history for the most-recent
filing with filing_date < as_of_date per row. PIT-safe.

### Risk mitigation

- FinBERT model is ~440MB. Bundle into repo? Or use HuggingFace cached download.
- 8-K extraction can fail (PDF vs HTML). Retry pattern + skip-and-log for failures.
- Filing date vs trade date timing: 8-Ks file after market close; use filing_date < as_of_date strictly.

---

## Item 4 — Options change features (PRIORITY: LOW — DEFER)

### Why deferred

pc_ratio_snap and iv_skew_snap as snapshots are weak signal. Changes are
the right shape:

| Better feature | Captures |
|----------------|----------|
| pc_ratio_change_5d | Sentiment shifting (vs static level) |
| pc_ratio_zscore_30d | Unusual positioning vs ticker's norm |
| iv_skew_change_5d | Hedging-demand shifting |
| iv_skew_zscore_30d | Tail-risk pricing vs ticker's norm |

### Implementation outline (when prioritized)

1. pc_ratio_change_5d — Massive aggregates provide put_volume + call_volume historically. Simple computation. ~3 hours.
2. iv_skew_change_5d — Requires Black-Scholes IV inversion (Massive doesn't expose historical IV directly). Compute IV from option close prices, find 25-delta contracts, take skew. ~6-8 hours + API call volume.

### Open questions before commit

- Will Massive Options Starter rate limit allow full backfill? (To verify)
- Will computed IV match Massive's snapshot IV closely enough to trust? (Validate on overlapping 30-day window)

---

## Commit cadence

Recommended order of attack:

1. Today/next session: Item 1 (drop dead) — 30 min, low risk.
2. Session 2: Item 2 (sector ETF) — 2-3 hours, measurable AUC delta.
3. Session 3-4: Item 3 (FinBERT) — 6-8 hours; do design review before code.
4. Defer indefinitely: Item 4 (options change) — revisit only if item 2+3 deliver expected AUC gains and we still need more edge.

Each shipped item gets:
- Walk-forward backtest before/after
- Honest AUC delta documented in commit message
- Memory note if it lands or fails

---

Generated 2026-05-21 from systematic dead-feature audit.
