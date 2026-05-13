"""
features/alpha_transformations.py
─────────────────────────────────────────────────────────────────────────────
Alpha transformation operators (Finding Alphas Ch. 10 / WorldQuant BRAIN).

17 pure functions that take a panel DataFrame and return a transformed
panel. Designed to compose into alpha expressions like:
    decay_linear(ts_rank(zscore(returns_5d), 20), 5)

Each operator:
  - Takes panel-shaped data: rows = dates, cols = tickers (or MultiIndex)
  - Returns same shape, preserving index
  - Handles NaN gracefully (skipna where it makes sense)
  - Does NOT use future information (causal/rolling only for ts_* ops)

Convention:
  cs_* = cross-sectional (across tickers, per date)
  ts_* = time-series (across dates, per ticker)
  Other = pointwise (scalar transform)

Reference:
  Finding Alphas (Tulchinsky), Ch. 10 — operator catalog
  WorldQuant BRAIN platform operator set
  Memory #11 — build from scratch, no prior drafts exist
"""
from __future__ import annotations
import pandas as pd
import numpy as np


# ══════════════════════════════════════════════════════════════════════════════
#  CROSS-SECTIONAL OPERATORS (across tickers, per date)
# ══════════════════════════════════════════════════════════════════════════════

def cs_rank(df: pd.DataFrame, pct: bool = True) -> pd.DataFrame:
    """
    Cross-sectional rank within each date row.
    
    pct=True: returns rank as percentile in [0, 1] (most common)
    pct=False: returns integer rank
    """
    return df.rank(axis=1, pct=pct, method="average")


def cs_zscore(df: pd.DataFrame) -> pd.DataFrame:
    """Cross-sectional z-score within each date row: (x - mean) / std."""
    mean = df.mean(axis=1)
    std = df.std(axis=1).replace(0, np.nan)
    return df.sub(mean, axis=0).div(std, axis=0)


def cs_mean(df: pd.DataFrame) -> pd.Series:
    """Cross-sectional mean per date (returns Series, not panel)."""
    return df.mean(axis=1)


def cs_std(df: pd.DataFrame) -> pd.Series:
    """Cross-sectional std per date (returns Series)."""
    return df.std(axis=1)


def cs_demean(df: pd.DataFrame) -> pd.DataFrame:
    """Subtract cross-sectional mean from each row (dollar-neutralize prep)."""
    return df.sub(df.mean(axis=1), axis=0)


# ══════════════════════════════════════════════════════════════════════════════
#  TIME-SERIES OPERATORS (across dates, per ticker)
# ══════════════════════════════════════════════════════════════════════════════

def ts_mean(df: pd.DataFrame, window: int) -> pd.DataFrame:
    """Rolling mean over `window` days, per ticker."""
    return df.rolling(window=window, min_periods=max(1, window // 2)).mean()


def ts_std(df: pd.DataFrame, window: int) -> pd.DataFrame:
    """Rolling std over `window` days, per ticker."""
    return df.rolling(window=window, min_periods=max(2, window // 2)).std()


def ts_rank(df: pd.DataFrame, window: int) -> pd.DataFrame:
    """
    Rolling rank of latest value within trailing `window` days, per ticker.
    Returns percentile in [0, 1].
    """
    def _rank_last(x):
        x = x.dropna()
        if len(x) < 2:
            return np.nan
        return (x.rank(pct=True).iloc[-1])
    return df.rolling(window=window, min_periods=max(2, window // 2)).apply(_rank_last, raw=False)


def ts_delta(df: pd.DataFrame, window: int) -> pd.DataFrame:
    """Difference from value `window` days ago: x[t] - x[t-window]."""
    return df.diff(periods=window)


def ts_corr(df_a: pd.DataFrame, df_b: pd.DataFrame, window: int) -> pd.DataFrame:
    """
    Rolling correlation between two panels over `window` days, per ticker.
    Both panels must have same shape + columns.
    """
    if df_a.shape != df_b.shape:
        raise ValueError(f"ts_corr shape mismatch: {df_a.shape} vs {df_b.shape}")
    return df_a.rolling(window=window, min_periods=max(3, window // 2)).corr(df_b)


def ts_max(df: pd.DataFrame, window: int) -> pd.DataFrame:
    """Rolling max over `window` days, per ticker."""
    return df.rolling(window=window, min_periods=1).max()


def ts_min(df: pd.DataFrame, window: int) -> pd.DataFrame:
    """Rolling min over `window` days, per ticker."""
    return df.rolling(window=window, min_periods=1).min()


def ts_argmax(df: pd.DataFrame, window: int) -> pd.DataFrame:
    """
    Days-ago of max value within trailing `window` days, per ticker.
    Range: 0 (max is today) to window-1 (max is oldest).
    """
    def _argmax(x):
        if x.isna().all():
            return np.nan
        return float(len(x) - 1 - x.values.argmax())
    return df.rolling(window=window, min_periods=1).apply(_argmax, raw=False)


def ts_decay_linear(df: pd.DataFrame, window: int) -> pd.DataFrame:
    """
    Linearly decay-weighted mean over `window` days.
    Most recent gets weight `window`, oldest gets weight 1.
    Heavily used in WorldQuant BRAIN alphas — emphasizes recent signal.
    """
    weights = np.arange(1, window + 1, dtype=float)
    weights /= weights.sum()

    def _decay(x):
        if x.isna().all() or len(x.dropna()) < max(2, window // 2):
            return np.nan
        # Pad if too few values (shouldn't happen with min_periods, defensive)
        if len(x) < window:
            return np.nan
        return float(np.nansum(x.values * weights))

    return df.rolling(window=window, min_periods=window).apply(_decay, raw=False)


def ts_sma(df: pd.DataFrame, window: int) -> pd.DataFrame:
    """Simple moving average — alias for ts_mean, kept for naming parity."""
    return ts_mean(df, window)


# ══════════════════════════════════════════════════════════════════════════════
#  POINTWISE / SCALAR OPERATORS
# ══════════════════════════════════════════════════════════════════════════════

def signed_power(df: pd.DataFrame, p: float) -> pd.DataFrame:
    """
    Signed power: sign(x) * |x|^p. Useful for emphasizing extremes (p<1)
    or dampening them (p>1). Preserves direction.
    """
    return np.sign(df) * np.power(np.abs(df), p)


def scale(df: pd.DataFrame, target_sum: float = 1.0) -> pd.DataFrame:
    """
    Scale each row so absolute values sum to `target_sum`.
    Standard prep for portfolio weights (Σ|w| = 1 = fully invested).
    """
    abs_sum = df.abs().sum(axis=1).replace(0, np.nan)
    return df.div(abs_sum / target_sum, axis=0)


# ══════════════════════════════════════════════════════════════════════════════
#  GROUP / NEUTRALIZATION OPERATORS
# ══════════════════════════════════════════════════════════════════════════════

def group_neutralize(df: pd.DataFrame, groups: dict) -> pd.DataFrame:
    """
    Subtract group mean from each member, per date.
    
    Args:
      df: panel (rows=dates, cols=tickers)
      groups: dict mapping ticker → group_label (e.g. ticker→sector)
    
    Returns:
      Panel where each value is x - mean_of_its_group_that_date.
      Tickers not in groups dict pass through unchanged.
    """
    out = df.copy()
    # Build group → list of tickers
    group_map = {}
    for ticker, group_label in groups.items():
        if ticker in df.columns:
            group_map.setdefault(group_label, []).append(ticker)

    for group_label, tickers in group_map.items():
        if len(tickers) < 2:
            continue  # group of 1 can't be neutralized meaningfully
        group_mean = df[tickers].mean(axis=1)
        for t in tickers:
            out[t] = df[t] - group_mean
    return out


def indneutralize(df: pd.DataFrame, industries: dict) -> pd.DataFrame:
    """Alias for group_neutralize with industry/sector groups."""
    return group_neutralize(df, industries)


# ══════════════════════════════════════════════════════════════════════════════
#  REGISTRY — for use by future explode_panel() in features/builder.py
# ══════════════════════════════════════════════════════════════════════════════

# Maps operator name → (function, default window or None)
# Window-bearing ops will be applied with multiple windows in explode_panel.
ALPHA_OPS = {
    "cs_rank":       (cs_rank,       None),
    "cs_zscore":     (cs_zscore,     None),
    "cs_demean":     (cs_demean,     None),
    "ts_mean":       (ts_mean,       [5, 10, 20]),
    "ts_std":        (ts_std,        [5, 10, 20]),
    "ts_rank":       (ts_rank,       [5, 10, 20]),
    "ts_delta":      (ts_delta,      [1, 5, 10]),
    "ts_max":        (ts_max,        [5, 10, 20]),
    "ts_min":        (ts_min,        [5, 10, 20]),
    "ts_argmax":     (ts_argmax,     [5, 10, 20]),
    "ts_decay_linear": (ts_decay_linear, [5, 10]),
    "signed_power":  (signed_power,  None),  # called with p=0.5 typically
    "scale":         (scale,         None),
}

# Operators NOT in registry but exported for direct call:
#   cs_mean, cs_std (return Series, not panel — used in compositions)
#   ts_sma (alias of ts_mean)
#   ts_corr (binary op, needs two panels)
#   group_neutralize, indneutralize (need group dict)


__all__ = [
    # cross-sectional
    "cs_rank", "cs_zscore", "cs_mean", "cs_std", "cs_demean",
    # time-series
    "ts_mean", "ts_std", "ts_rank", "ts_delta", "ts_corr",
    "ts_max", "ts_min", "ts_argmax", "ts_decay_linear", "ts_sma",
    # pointwise
    "signed_power", "scale",
    # group
    "group_neutralize", "indneutralize",
    # registry
    "ALPHA_OPS",
]
