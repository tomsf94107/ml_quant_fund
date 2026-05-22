# TODO: Refactor build_feature_dataframe to separate features from diagnostics

**Created:** May 22, 2026
**Priority:** Medium
**Effort:** 1-2 hours
**Tags:** cleanup, architecture, log-noise

---

## Problem

`build_feature_dataframe` returns a single DataFrame with 107 columns:
- ~90 are model FEATURES (trained on, used in prediction)
- ~17 are NON-FEATURES (diagnostic, output-only, used by dashboard / backtest)

Non-feature columns include: close, volume, macd_signal, expected_move_perc,
pre_earnings_drift, post_earnings_drift, is_earnings_week, fear_greed,
es_overnight.

When `EnsembleResult.predict_proba` runs, it filters df to `self.feature_cols`
(the 90 model inputs) and warns about any extras. The warning currently fires
375x per Pipeline C run for the 17 non-feature columns. Noisy.

The filter logic is correct — it catches the real bug class (silent feature
mismatch). But the noise dilutes its value.

## Proper fix

Restructure `build_feature_dataframe` to return EITHER:

**Option D-1:** Two DataFrames
```python
def build_feature_dataframe(ticker, ...) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (df_features, df_context).
    
    df_features: ~90 cols, ONLY model training/inference inputs.
    df_context: diagnostic + output columns (close, volume, dates, etc).
    """
```

**Option D-2:** Single DataFrame with metadata
```python
def build_feature_dataframe(ticker, ...) -> pd.DataFrame:
    """Returns df. df.attrs['feature_cols'] lists the model inputs."""
```

D-2 is less invasive (no signature change for existing callers).

## Implementation steps for D-2

1. Define FEATURE_COLUMNS as the authoritative model input list
2. In `build_feature_dataframe`, set `df.attrs['feature_cols'] = FEATURE_COLUMNS`
3. In `EnsembleResult.predict_proba`, use `X.attrs.get('feature_cols', list(X.columns))` to filter
4. Verify all callers (signals/generator.py, models/train_all_batched.py, walk_forward.py, etc.) preserve the attrs dict through their .copy() / .loc[] operations
5. Add a unit test verifying df.attrs survives reshape ops

## Rule #1 audit reminders

- (a) Audit ALL callers of build_feature_dataframe before changing signature
- (b) df.attrs is lost in many pandas operations (groupby, merge) — fragile
- (g) Gap-check: predict_proba currently uses self.feature_cols, NOT df.attrs.
      The filter is safe regardless. Cleanup is purely for log noise.

## Why not tonight

- Larger blast radius than warranted at end of long session
- The current warning is functionally correct, just noisy
- 1-2hr proper work vs 5min hacky whitelist

## Related

- The defensive filter in models/ensemble.py (commit da72dae May 22) is the
  CORRECT fix for the bug class (silent feature mismatch). It stays in place
  permanently. This TODO is purely log noise cleanup.
