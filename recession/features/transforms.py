"""
recession/features/transforms.py

Feature transformations applied in Step 4 of the pipeline. Pure functions —
input is a pandas Series indexed by month, output is a pandas Series indexed
by month. No DB access, no global state, no `as_of` plumbing (that lives in
the loader, not here — these functions just compute on whatever you give them).

Transforms implemented:
  1. yoy_pct(s):                       (s_t / s_{t-12}) - 1
  2. log_return_12m(s):                log(s_t / s_{t-12})
  3. hamilton_detrend(s, h=24):        Hamilton (2018) regression filter
  4. at_risk_threshold_theoretical:    fixed-threshold binary dummy
  5. at_risk_threshold_empirical:      data-driven threshold (refit per fold)

Design rules (from v1.1.1 + Phase 1 lessons):
- Every transform is point-in-time honest: only uses data <= the index of
  the output point. No look-ahead.
- Hamilton's choice of h=24 (months) is the published default for monthly
  data. Hamilton's 2018 paper specifically argues against HP-filter due to
  endpoint look-ahead. We don't implement HP-filter for that reason.
- Empirical at-risk thresholds are fit on a *training fold only*; the fitted
  threshold is then applied forward. Never fit on full history.
"""
from __future__ import annotations

import math
from typing import Literal, Optional

import numpy as np
import pandas as pd


# =============================================================================
# Helpers
# =============================================================================

def _ensure_monotonic_monthly(s: pd.Series) -> pd.Series:
    """Verify the series is monthly-indexed, sorted, and has no duplicates.

    This catches a class of bugs where the caller passes in raw DB rows
    without a proper DatetimeIndex.
    """
    if not isinstance(s.index, pd.DatetimeIndex):
        raise TypeError(
            f"transforms expect a DatetimeIndex, got {type(s.index).__name__}"
        )
    if not s.index.is_monotonic_increasing:
        raise ValueError("transforms expect a monotonically-increasing index")
    if s.index.has_duplicates:
        raise ValueError("transforms expect a unique index (no duplicate months)")
    return s


# =============================================================================
# 1. YoY percent change
# =============================================================================

def yoy_pct(s: pd.Series) -> pd.Series:
    """12-month percent change. (s_t / s_{t-12}) - 1.

    Used for inflation features (CPILFESL, PCEPILFE, CES0500000003) where the
    raw level is non-stationary but YoY change is stationary.

    First 12 observations become NaN (no prior-year value to compare to).

    Args:
        s: monthly time series with DatetimeIndex.

    Returns:
        Series of same shape; first 12 values are NaN.
    """
    _ensure_monotonic_monthly(s)
    return (s / s.shift(12)) - 1.0


def log_return_12m(s: pd.Series) -> pd.Series:
    """12-month log return. log(s_t / s_{t-12}).

    For positive-valued series only (e.g., SP500). Will produce NaN if any
    value is non-positive.

    Already shipped in Step 3.7 as SP500_RET_12M; this is the in-memory
    equivalent for any other series we want to transform similarly.
    """
    _ensure_monotonic_monthly(s)
    if (s <= 0).any():
        # Replace non-positive with NaN to avoid log domain errors;
        # caller is expected to handle a series with non-positive values.
        s = s.where(s > 0)
    return np.log(s / s.shift(12))


# =============================================================================
# 2. Hamilton (2018) detrending
# =============================================================================

def hamilton_detrend(
    s: pd.Series,
    h: int = 24,
    p: int = 4,
) -> pd.Series:
    """Hamilton (2018) regression filter for trend/cycle decomposition.

    Implements the OLS regression in Hamilton's "Why You Should Never Use
    the Hodrick-Prescott Filter" (Review of Economics and Statistics, 2018).

    The model: y_{t+h} = β_0 + β_1·y_t + β_2·y_{t-1} + β_3·y_{t-2} + β_4·y_{t-3} + v_{t+h}

    The "cycle" component at time t+h is the residual: v̂_{t+h} = y_{t+h} - ŷ_{t+h}

    For monthly data, h=24 (2 years ahead) is Hamilton's published default.
    p=4 is also the published default (4 lags).

    Properties:
    - Uses only past data: y_t, y_{t-1}, y_{t-2}, y_{t-3} predict y_{t+h}.
      No look-ahead at any point.
    - First (h + p - 1) = 27 observations are NaN: not enough lags + horizon.
    - The cycle is by construction stationary if y is integrated of order 1.

    Args:
        s: monthly time series.
        h: forecast horizon used for filtering (default 24 = 2 years).
        p: number of lags in the right-hand side (default 4).

    Returns:
        Cycle component, same index as s. First (h + p - 1) values are NaN.
    """
    _ensure_monotonic_monthly(s)
    if h < 1 or p < 1:
        raise ValueError(f"h and p must be >= 1; got h={h}, p={p}")

    n = len(s)
    if n < h + p:
        # Not enough data — return all-NaN
        return pd.Series(np.nan, index=s.index, name=s.name)

    # Build the design matrix:
    # row t represents the regression target y_{t+h} regressed on
    # [y_t, y_{t-1}, ..., y_{t-(p-1)}].
    # We can only form rows where t in [p-1, n-h-1].
    # Output cycle is filled at index t+h.

    y = s.values.astype(float)
    rows_X = []
    rows_y = []
    rows_target_idx = []

    for t in range(p - 1, n - h):
        # Right-hand side: y_t, y_{t-1}, ..., y_{t-(p-1)}
        x = [y[t - k] for k in range(p)]
        if any(np.isnan(v) for v in x) or np.isnan(y[t + h]):
            continue
        rows_X.append([1.0] + x)             # intercept + lags
        rows_y.append(y[t + h])
        rows_target_idx.append(t + h)

    if not rows_X:
        return pd.Series(np.nan, index=s.index, name=s.name)

    X = np.array(rows_X)
    Y = np.array(rows_y)

    # OLS: β = (X'X)^-1 X'Y
    try:
        beta, *_ = np.linalg.lstsq(X, Y, rcond=None)
    except np.linalg.LinAlgError:
        return pd.Series(np.nan, index=s.index, name=s.name)

    # Compute residuals at the target indices
    cycle = pd.Series(np.nan, index=s.index, name=s.name)
    Y_hat = X @ beta
    residuals = Y - Y_hat
    for idx, resid in zip(rows_target_idx, residuals):
        cycle.iloc[idx] = resid

    return cycle


# =============================================================================
# 3. At-risk threshold dummies
# =============================================================================

def at_risk_theoretical(
    s: pd.Series,
    threshold: float,
    direction: Literal["above", "below", "below_or_equal", "above_or_equal"],
) -> pd.Series:
    """Binary at-risk dummy with a fixed (literature-defined) threshold.

    Used for features where there's a published threshold:
        T10Y3M             < 0       (yield curve inverted)
        NFCI               > 0.25    (financial conditions tight)
        BAA10Y             > 3.5     (credit blowout)
        SP500_DRAWDOWN_12M ≤ -0.10   (10%+ drawdown)

    Args:
        s: monthly series.
        threshold: numeric cutoff.
        direction: which side of threshold = at-risk.

    Returns:
        Series of 0/1 (NaN preserved).
    """
    _ensure_monotonic_monthly(s)
    if direction == "above":
        out = (s > threshold).astype(float)
    elif direction == "below":
        out = (s < threshold).astype(float)
    elif direction == "above_or_equal":
        out = (s >= threshold).astype(float)
    elif direction == "below_or_equal":
        out = (s <= threshold).astype(float)
    else:
        raise ValueError(f"Unknown direction: {direction}")

    # Preserve NaN where source was NaN
    out[s.isna()] = np.nan
    return out


def at_risk_empirical_fit(
    s_train: pd.Series,
    target_train: pd.Series,
    direction: Literal["higher_means_recession", "lower_means_recession"],
    recession_quantile: float = 0.25,
) -> float:
    """Fit an empirical at-risk threshold from a training fold.

    Method:
    - Subset s_train to months where target_train==1 (pre-recession months).
    - Compute the value at the `recession_quantile` of those months.
    - That value is the threshold.

    For "higher_means_recession" features (e.g., NFCI), threshold = 75th
    percentile of recession months — feature must be ≥ that to fire.

    For "lower_means_recession" features (e.g., T10Y3M), threshold = 25th
    percentile of recession months — feature must be ≤ that to fire.

    Why these percentiles: a 25th percentile of recession months means "if
    the feature is at this level, ~25% of typical pre-recession months had
    a worse value — i.e., we're in the recession-favoring tail." More
    conservative than median (50th) which would fire too often, less
    extreme than 10th which would rarely fire.

    Refit per fold to avoid look-ahead leakage.

    Args:
        s_train: feature values from training fold.
        target_train: target labels (0/1) aligned to s_train (same index).
        direction: which way the feature points.
        recession_quantile: percentile to use (default 0.25).

    Returns:
        Fitted threshold (float).
    """
    if not s_train.index.equals(target_train.index):
        raise ValueError("s_train and target_train must have identical indices")

    recession_mask = (target_train == 1)
    s_recession = s_train[recession_mask].dropna()

    if len(s_recession) < 5:
        # Not enough recession observations to fit a meaningful percentile.
        # Return NaN; caller should treat dummy as all-zero or skip the feature.
        return float("nan")

    if direction == "higher_means_recession":
        # Threshold = lower quantile of recession values
        # Feature ≥ threshold = at-risk
        # Use 1 - recession_quantile so threshold is "low end of recession zone"
        threshold = float(s_recession.quantile(1.0 - recession_quantile))
    elif direction == "lower_means_recession":
        # Threshold = upper quantile of recession values
        # Feature ≤ threshold = at-risk
        threshold = float(s_recession.quantile(recession_quantile))
    else:
        raise ValueError(f"Unknown direction: {direction}")

    return threshold


def at_risk_empirical_apply(
    s: pd.Series,
    threshold: float,
    direction: Literal["higher_means_recession", "lower_means_recession"],
) -> pd.Series:
    """Apply a previously-fitted empirical threshold to a series.

    Args:
        s: feature values to transform (any time period).
        threshold: from at_risk_empirical_fit().
        direction: same direction used in fit.

    Returns:
        Series of 0/1 (NaN preserved).
    """
    _ensure_monotonic_monthly(s)
    if math.isnan(threshold):
        # Fit returned NaN — produce all-zero dummy with NaN preserved
        out = pd.Series(0.0, index=s.index, name=s.name)
        out[s.isna()] = np.nan
        return out

    if direction == "higher_means_recession":
        out = (s >= threshold).astype(float)
    elif direction == "lower_means_recession":
        out = (s <= threshold).astype(float)
    else:
        raise ValueError(f"Unknown direction: {direction}")

    out[s.isna()] = np.nan
    return out


# =============================================================================
# Feature → transform routing
# =============================================================================

# Configuration: which features get which transforms.
# Read from features_registry.detrend_method when possible; this dict overrides
# / extends for cases not covered by the registry.
#
# Format: feature_name -> ('transform_name', kwargs_dict)
TRANSFORM_REGISTRY: dict[str, tuple[str, dict]] = {
    # Inflation: YoY (also tagged in features_registry.detrend_method)
    "CPILFESL":           ("yoy_pct", {}),
    "PCEPILFE":           ("yoy_pct", {}),
    "CES0500000003":      ("yoy_pct", {}),

    # Trending features: Hamilton detrending
    "INDPRO":             ("hamilton", {"h": 24, "p": 4}),
    "DTWEXBGS":           ("hamilton", {"h": 24, "p": 4}),
    # SP500 levels are detrended via SP500_RET_12M and SP500_DRAWDOWN_12M
    # (Step 3.7), so we don't apply Hamilton to raw SP500 here.

    # Already-stationary features: no transform (the model uses raw values)
    # T10Y3M, T10Y2Y, NFCI, BAA10Y, EBP, REAL_FFR_GAP, NAPMPI, CFNAI,
    # SAHMREALTIME, JTSQUR, JTSLDR, ISRATIO, SP500_RET_12M, SP500_DRAWDOWN_12M
    # — all stationary by construction, no entry needed.
}


# At-risk dummy configuration.
# Theoretical thresholds match T5 conditions for interpretability.
AT_RISK_THEORETICAL: dict[str, dict] = {
    "T10Y3M":              {"threshold": 0.0,   "direction": "below"},
    "NFCI":                {"threshold": 0.25,  "direction": "above"},
    "BAA10Y":              {"threshold": 3.5,   "direction": "above"},
    "SP500_DRAWDOWN_12M":  {"threshold": -0.10, "direction": "below_or_equal"},
}

# Features that get empirical at-risk thresholds (refit per fold).
# Direction is from the exploration report's univariate AUC analysis,
# computed on RAW values. For features that get a transform first
# (INDPRO via Hamilton), the direction may shift — the "higher_means_recession"
# tag here applies to the cycle component output by Hamilton, which is
# expected to behave similarly but should be re-validated in Step 5.
AT_RISK_EMPIRICAL: dict[str, dict] = {
    "DTWEXBGS":      {"direction": "lower_means_recession"},
    "DRTSCILM":      {"direction": "higher_means_recession"},
    "EBP":           {"direction": "higher_means_recession"},
    "INDPRO":        {"direction": "higher_means_recession"},  # post-Hamilton: cycle component (re-validate in Step 5)
    "REAL_FFR_GAP":  {"direction": "higher_means_recession"},
    "T10Y2Y":        {"direction": "lower_means_recession"},
}


def get_transform_for_feature(feature_name: str) -> Optional[tuple[str, dict]]:
    """Look up the transform (if any) for a given feature."""
    return TRANSFORM_REGISTRY.get(feature_name)
