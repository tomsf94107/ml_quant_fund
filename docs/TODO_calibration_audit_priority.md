# Calibration Audit — TOMORROW (Sun May 24) IMMEDIATE PRIORITY

**Why:** Production data shows model is badly calibrated. >=0.80 confidence BUYs at h=5 hit only 40%. This is BEFORE we can build position sizing, exits, dashboard, or any execution improvements.

**Stop:** No position-sizing, no exit logic, no dashboard until this is understood.

## The smoking gun

| h | prob bucket | n | hit% |
|---|---|---|---|
| 3 | 0.55-0.65 | 75 | **~39%** (loses money) |
| 3 | 0.70-0.80 | 188 | 63.8% (avg prob said 75%) |
| 5 | 0.60-0.65 | 54 | **42.6%** (loses money) |
| 5 | >=0.80 | 35 | **40.0%** (worst bucket = highest "confidence") |

**A well-calibrated model has hit rate ≈ predicted prob, monotone in confidence.**

## Step 1: Reproduce the calibration table

```bash
sqlite3 accuracy.db "
SELECT 
    horizon,
    CASE 
      WHEN prob_up < 0.55 THEN 'A. <0.55'
      WHEN prob_up < 0.60 THEN 'B. 0.55-0.60'
      WHEN prob_up < 0.65 THEN 'C. 0.60-0.65'
      WHEN prob_up < 0.70 THEN 'D. 0.65-0.70'
      WHEN prob_up < 0.80 THEN 'E. 0.70-0.80'
      ELSE 'F. >=0.80'
    END AS bucket,
    COUNT(*) AS n,
    ROUND(AVG(prob_up)*100, 1) AS avg_prob,
    ROUND(AVG(actual_up)*100, 1) AS hit_pct,
    ROUND(AVG(actual_up)*100 - AVG(prob_up)*100, 1) AS calibration_error
FROM predictions p JOIN outcomes o USING(ticker, prediction_date, horizon)
WHERE p.signal = 'BUY' AND o.actual_up IS NOT NULL
GROUP BY horizon, bucket
ORDER BY horizon, bucket;
"
```

## Step 2: Test multiplier inflation hypothesis

If prob_up is multiplied (risk_mult × sent_mult × regime_mult × ...) the high-conf bucket may be inflated. Compare:

```bash
sqlite3 accuracy.db "
SELECT 
    horizon,
    CASE 
      WHEN prob_raw < 0.55 THEN 'A. <0.55'
      WHEN prob_raw < 0.60 THEN 'B. 0.55-0.60'
      WHEN prob_raw < 0.65 THEN 'C. 0.60-0.65'
      WHEN prob_raw < 0.70 THEN 'D. 0.65-0.70'
      WHEN prob_raw < 0.80 THEN 'E. 0.70-0.80'
      ELSE 'F. >=0.80'
    END AS prob_raw_bucket,
    COUNT(*) AS n,
    ROUND(AVG(prob_raw)*100, 1) AS avg_prob_raw,
    ROUND(AVG(actual_up)*100, 1) AS hit_pct,
    ROUND(AVG(actual_up)*100 - AVG(prob_raw)*100, 1) AS cal_err_raw
FROM predictions p JOIN outcomes o USING(ticker, prediction_date, horizon)
WHERE p.signal = 'BUY' AND o.actual_up IS NOT NULL AND prob_raw IS NOT NULL
GROUP BY horizon, prob_raw_bucket
ORDER BY horizon, prob_raw_bucket;
"
```

**If prob_raw is well-calibrated but prob_up isn't, multipliers are the culprit.**

## Step 3: Test each multiplier's impact

For each multiplier (risk, sent, regime, options, squeeze, intraday, fg):
- Group BUYs by multiplier value
- Check if hit rate varies systematically

```bash
sqlite3 accuracy.db "
SELECT 
    CASE WHEN risk_mult > 1.05 THEN 'risk_boost' 
         WHEN risk_mult < 0.95 THEN 'risk_penalty' 
         ELSE 'risk_neutral' END AS risk_cat,
    horizon,
    COUNT(*) AS n,
    ROUND(AVG(actual_up)*100, 1) AS hit_pct
FROM predictions p JOIN outcomes o USING(ticker, prediction_date, horizon)
WHERE signal='BUY' AND actual_up IS NOT NULL
GROUP BY horizon, risk_cat
ORDER BY horizon, risk_cat;
"
```

Repeat for sent_mult, regime_mult, options_mult, squeeze_mult, intraday_mult, fg_mult.

## Step 4: Check if Path A (GLOBAL) is better-calibrated

After Monday Pipeline C runs with new code, prob_up_global will populate. Then:

```bash
sqlite3 accuracy.db "
SELECT 
    horizon,
    CASE 
      WHEN prob_up_global < 0.55 THEN 'A. <0.55'
      WHEN prob_up_global < 0.65 THEN 'B. 0.55-0.65'
      WHEN prob_up_global < 0.75 THEN 'C. 0.65-0.75'
      ELSE 'D. >=0.75'
    END AS bucket,
    COUNT(*) AS n,
    ROUND(AVG(prob_up_global)*100, 1) AS avg_gl,
    ROUND(AVG(actual_up)*100, 1) AS hit_pct
FROM predictions p JOIN outcomes o USING(ticker, prediction_date, horizon)
WHERE prob_up_global IS NOT NULL AND actual_up IS NOT NULL
GROUP BY horizon, bucket
ORDER BY horizon, bucket;
"
```

## Possible findings and what to do

### Finding A: prob_raw IS well-calibrated, multipliers break it
**Fix:** Cap multiplier compound effect. Or replace multiplier system with proper rule-based bucket overrides.

### Finding B: prob_raw is also broken
**Fix:** Re-fit isotonic calibration on recent data. Verify training data hasn't leaked test signal.

### Finding C: One specific multiplier is destructive
**Fix:** Remove or invert it for the affected horizons.

### Finding D: It's regime-specific (the recent period is unusual)
**Fix:** Wait for more data, OR train regime-conditional models.

### Finding E: Tickers in the high-conf bucket are pathological (penny stocks, etc.)
**Fix:** Universe filter, OR adjust model on these subgroups.

## Decision rule

Don't build position sizing / exits / dashboard until at least ONE of these is true:
1. prob_raw is shown to be well-calibrated
2. Calibration can be fixed via isotonic re-fit
3. Path A (GLOBAL) is shown to be better-calibrated and gets promoted
4. We accept production model is broken and build bucket-OVERRIDE position sizing based on actual hit rates instead of trusting prob_up

## Cross-reference to roadmap

The 4-week roadmap doc (docs/roadmap_4week_quant_completeness.md) assumed accuracy was good. This audit may invalidate Week 1 items 1.1 and 1.2 until calibration is fixed.

Week 1 NEW priority:
1. **Calibration audit** (THIS doc, ~half day)
2. **Calibration fix** (depends on finding, ~half day to 1 day)
3. Then Week 1.4 (regression tests) — still relevant
4. Then 1.3 (dashboard) — still relevant
5. Position sizing (1.1) and stops (1.2) ONLY after calibration fix or bucket-override approach
