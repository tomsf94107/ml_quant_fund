# Native NaN + Missing Indicators A/B — May 27 2026

## Goal
Test if ML_QUANT_NATIVE_NAN=1 (skip fillna) and/or ML_QUANT_MISSING_INDICATORS=1
(add {feature}_has_value columns) improve per-ticker model AUC by reviving
dead inst_* (dark pool) features that were median-filled into uselessness.

## Method
Trained per-ticker ensemble model (XGB+LGB) on h=5 target for 5 tickers
under 4 configurations: default, native_nan, indicators, both.
3-way temporal split (60/20/20), reported OOS AUC on test holdout.

## Results

| Config | Mean AUC | vs default |
|---|---|---|
| default | 0.4758 | baseline |
| native_nan | 0.4674 | -0.84pp |
| indicators | 0.4828 | +0.70pp |
| both | 0.4829 | +0.71pp |

Per-ticker AUC range: 0.40-0.52 across all configs. All within noise.

**Inst feature importance: ZERO in every config.**

## Honest interpretation

- Per-ticker model performance ~ random (~0.50 AUC) on post-2024 test
- Imputation strategy doesn't meaningfully change this
- Inst features remain dead because 10% coverage isn't enough for
  trees to extract signal, regardless of NaN handling
- Indicator columns slightly better (+0.7pp) but within noise (n=96 test rows)

## Decision

**Do NOT enable either flag in production for per-ticker models.**

Reasons:
1. No measurable AUC improvement
2. Adds complexity without benefit
3. Underlying issue is data sparsity, not preprocessing

## Reversal-friendly state

Flags remain in code (commits 11b4367 ancestor) but DEFAULT OFF.
If inst data coverage improves later (e.g. 6 months of new data),
re-run this test. Flags ready to enable instantly.

## Future test

Try same flags on GLOBAL ranker (180K rows across all tickers).
Cross-sectional model has more inst data per training example.
Might show measurable benefit where per-ticker doesn't.

## Code path
- features/builder.py: ML_QUANT_MISSING_INDICATORS extends OUTPUT_COLUMNS
- models/classifier.py: ML_QUANT_MISSING_INDICATORS extends FEATURE_COLUMNS,
                         ML_QUANT_NATIVE_NAN gates 2 fillna sites
- models/ensemble.py: ML_QUANT_NATIVE_NAN gates 1 fillna site
- scripts/test_native_nan_ab.py: A/B test runner

## Backups
- features/builder.py.bak.before_missing_indicators_20260527
- models/classifier.py.bak.before_nativenan_20260527
- models/classifier.py.bak.before_missing_indicators_20260527
- models/ensemble.py.bak.before_nativenan_20260527
