# Audit Findings — Sunday May 24, 2026 (Morning)

Conducted as part of Priority #1 (AUC reconciliation) before building diagnostic_audit.py.

## Suspect features identified

`build_feature_dataframe(training_mode=True)` produces 107 columns. Of these, 8 had "suspicious" names (forward-looking):
- risk_next_1d, risk_next_3d
- post_earnings_1d, post_earnings_3d, post_earnings_5d
- expected_move_perc, pre_earnings_drift, post_earnings_drift

## Audit results per suspect

### CLEAN

**post_earnings_1d/3d/5d** — `data/etl_earnings.py:load_earnings_features`
- Computes "is current date d between 0 and window days AFTER report date rd"
- Uses (d - rd).days — strictly backward-looking
- Filters earnings_surprises.created_at <= as_of when as_of is passed
- **Verdict: not leakage**

**finbert_sentiment_earnings** — `data/alpha_sources.py:load_finbert_pit`
- Uses filing_date < asof filter strictly
- Same code path for training and live inference
- Reads from data/sentiment.db.finbert_filings
- **Verdict: PIT-correct**

**expected_move_perc / pre_earnings_drift / post_earnings_drift** — `data/etl_earnings.py:load_uw_earnings_features`
- Skipped in training_mode=True (set to 0.0)
- Only computed in live inference mode
- **Verdict: CLEAN for PIT training**

### LEAK FOUND

**risk_next_1d / risk_next_3d** — `signals/risk_gate.py:build_risk_features`

Implementation pattern:
- VIX spikes detected from vix_ret > VIX_SPIKE_PCT (uses pct_change, looks at the spike day itself)
- Sets risk_today[T] = 1 for spike days
- Then: df["risk_next_1d"] = df["risk_today"].shift(-1)
- Then: df["risk_next_3d"] = df["risk_today"].rolling(3).max().shift(-3)

**Problem:**
- VIX spikes are not known in advance
- The shift(-1) and shift(-3) encode FUTURE VIX spikes in TODAY rows
- Training data has spurious "I know tomorrow VIX spikes" signal
- Production (where shift can not access future) shows 0 for unknown future spikes
- Train/serve mismatch

**Honest distinction:**
- FOMC/CPI dates: scheduled months ahead — NOT leakage
- Earnings dates: scheduled weeks ahead — NOT leakage
- **VIX spikes: NOT known in advance — LEAKAGE**

**Severity (estimate):**
- VIX spikes >5% occur ~5-15 times per year
- Train data has these labeled "correctly"
- Test/production data does not
- Could account for 5-10pp inflation in train AUC if model finds the signal

### NOT TRUE PIT

**Path A `models/train_cross_sectional.py:validate_oos`**

- Calls build_feature_dataframe(ticker, start_date=start_date) per ticker
- No end_date parameter per row
- Features built using FULL available history per ticker
- The chronological split happens AFTER feature build

**Compare to analysis/walk_forward.py:load_panel_pit:**
- Calls build_feature_dataframe(ticker, end_date=prediction_date, training_mode=True) per row
- Features built using ONLY data up to prediction_date
- **This is the honest PIT path**

**Implication:**
- Path A's reported AUC of 0.58-0.59 is between proper PIT and full-leak
- True PIT AUC for Path A is UNKNOWN
- Currently running test (PID 12511) uses honest PIT path with cross-sectional pooled data
- Result will be the real reconciliation point

### CLEAN with caveat

**Path A train median used for test fill**
- test_df[FEATURE_COLUMNS] = test_df[FEATURE_COLUMNS].fillna(train_df[FEATURE_COLUMNS].median())
- Correct usage of train-only statistics
- **OK**

## Summary

| AUC source | Reliability |
|---|---|
| PIT production config (0.44) | TRUSTABLE — true PIT, properly purged, may be over-regularized |
| Per-ticker walk-forward (0.51-0.53) | UNCERTAIN — has VIX leak, does not use end_date PIT |
| Path A 5-fold (0.58-0.59) | INFLATED — has VIX leak AND not true PIT |

The honest baseline is somewhere around 0.44-0.55 for both architectures.

## Next steps

1. **Wait for PID 12511 (PIT --config default)** — tells us if loose XGB params alone fix per-ticker
2. **Fix VIX leak** in signals/risk_gate.py:build_risk_features
3. **Rebuild Path A panel with PIT-strict features** (per-row end_date)
4. **Re-validate Path A** after fixes
5. **Compare honest PIT AUC across architectures**

## Open questions

- How much of Path A 0.58-0.59 is real architecture lift vs leaks?
- Does fixing the VIX leak drop Path A close to per-ticker baseline?
- Should risk_next_* only include calendar events, not VIX spikes?
