"""
recession/features/breadth.py

DIFFUSION_BREADTH engineered feature.

Concept: a diffusion index. For each month, count what fraction of features
are currently in their "recession-favoring zone". When many indicators
simultaneously point toward recession, that BREADTH of deterioration is
itself a signal — broader than any single indicator.

This is the classic diffusion-index idea (used by the Conference Board LEI,
ISM, etc.): the share of components moving in the bad direction.

Outputs (per the user's Decision: "both"):
  - BREADTH_GLOBAL          fraction of ALL scored features recession-favoring
  - BREADTH_TIER_<label>    fraction within each tier (yield_credit, labor, ...)

Design:
- A feature is "recession-favoring" if it's on the recession side of its
  threshold. We reuse the at-risk dummy logic — a feature with an at-risk
  dummy == 1 is recession-favoring.
- Features without a defined at-risk threshold are EXCLUDED from the breadth
  computation (they don't have a "zone"). Breadth = fired / scored, where
  scored = features that have a threshold AND non-NaN value that month.
- NaN handling: a feature with NaN value that month is NOT counted in the
  denominator (we can't tell which zone it's in). This keeps breadth honest
  at the start of history when many features have no data.

This module does NOT compute thresholds itself — it receives a panel of
already-computed at-risk dummies (0/1/NaN) and aggregates them. The caller
(builder.py) is responsible for producing the dummies via transforms.py.
That keeps breadth.py a pure aggregation function with no leakage surface.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd


# =============================================================================
# Core: compute breadth from a panel of at-risk dummies
# =============================================================================

def compute_breadth(
    at_risk_dummies: pd.DataFrame,
    feature_tiers: dict[str, int],
    tier_labels: Optional[dict[int, str]] = None,
) -> pd.DataFrame:
    """Compute global and per-tier diffusion breadth.

    Args:
        at_risk_dummies: DataFrame indexed by month, columns are feature
            names, values are 0 / 1 / NaN. A 1 means "this feature is in
            its recession-favoring zone this month". NaN means "unknown
            this month" (feature had no data) — excluded from both
            numerator and denominator.
        feature_tiers: {feature_name -> tier_int}. Only features present
            in BOTH this dict and the dummies DataFrame are scored.
        tier_labels: optional {tier_int -> short_label}. Used for naming
            the per-tier columns. If absent, columns are 'BREADTH_TIER_<n>'.

    Returns:
        DataFrame indexed by the same months, with columns:
            BREADTH_GLOBAL
            BREADTH_TIER_<label>   (one per tier that has >=1 scored feature)

        Values are in [0, 1] or NaN (NaN when no features were scorable
        that month — e.g., very early history).
    """
    if at_risk_dummies.empty:
        return pd.DataFrame(index=at_risk_dummies.index)

    tier_labels = tier_labels or {}

    # Restrict to features that have a tier assignment AND are in the panel
    scored_features = [
        f for f in at_risk_dummies.columns if f in feature_tiers
    ]
    if not scored_features:
        # Nothing scorable — return all-NaN global column
        out = pd.DataFrame(index=at_risk_dummies.index)
        out["BREADTH_GLOBAL"] = np.nan
        return out

    dummies = at_risk_dummies[scored_features]

    out = pd.DataFrame(index=at_risk_dummies.index)

    # --- Global breadth ---
    # numerator = count of 1s per row; denominator = count of non-NaN per row
    fired_global = dummies.sum(axis=1, skipna=True)
    scored_global = dummies.notna().sum(axis=1)
    # Avoid divide-by-zero: where scored == 0, breadth is NaN
    out["BREADTH_GLOBAL"] = np.where(
        scored_global > 0,
        fired_global / scored_global,
        np.nan,
    )

    # --- Per-tier breadth ---
    # Group features by tier
    tiers_present: dict[int, list[str]] = {}
    for f in scored_features:
        t = feature_tiers[f]
        tiers_present.setdefault(t, []).append(f)

    for tier, feats in sorted(tiers_present.items()):
        label = tier_labels.get(tier, str(tier))
        col_name = f"BREADTH_TIER_{label}"
        tier_dummies = dummies[feats]
        fired = tier_dummies.sum(axis=1, skipna=True)
        scored = tier_dummies.notna().sum(axis=1)
        out[col_name] = np.where(scored > 0, fired / scored, np.nan)

    return out


# =============================================================================
# Helper: build at-risk dummies from a raw feature panel
# =============================================================================

def build_at_risk_dummies(
    panel: pd.DataFrame,
    theoretical: dict[str, dict],
    empirical_thresholds: Optional[dict[str, dict]] = None,
) -> pd.DataFrame:
    """Build a panel of 0/1/NaN at-risk dummies from a raw feature panel.

    This is a convenience that wraps the at-risk logic for the breadth
    use-case. It applies:
      - theoretical thresholds (fixed, from transforms.AT_RISK_THEORETICAL)
      - empirical thresholds (pre-fitted per fold; passed in as a dict
        mapping feature -> {'threshold': float, 'direction': str})

    Args:
        panel: raw feature values (months x features).
        theoretical: {feature -> {'threshold': float, 'direction': str}}.
            direction in {'above','below','above_or_equal','below_or_equal'}.
        empirical_thresholds: {feature -> {'threshold': float,
            'direction': 'higher_means_recession'|'lower_means_recession'}}.
            These thresholds must already be fitted (on a training fold)
            by the caller — this function only APPLIES them. Pass None if
            no empirical dummies are wanted.

    Returns:
        DataFrame of dummies (0/1/NaN), one column per feature that had a
        threshold defined. Features without thresholds are omitted.
    """
    dummies = pd.DataFrame(index=panel.index)

    # Theoretical (fixed-threshold) dummies
    for feat, cfg in theoretical.items():
        if feat not in panel.columns:
            continue
        s = panel[feat]
        thr = cfg["threshold"]
        direction = cfg["direction"]
        if direction == "above":
            d = (s > thr).astype(float)
        elif direction == "below":
            d = (s < thr).astype(float)
        elif direction == "above_or_equal":
            d = (s >= thr).astype(float)
        elif direction == "below_or_equal":
            d = (s <= thr).astype(float)
        else:
            raise ValueError(f"Unknown direction {direction!r} for {feat}")
        d[s.isna()] = np.nan
        dummies[feat] = d

    # Empirical (fitted-threshold) dummies
    if empirical_thresholds:
        for feat, cfg in empirical_thresholds.items():
            if feat not in panel.columns:
                continue
            thr = cfg.get("threshold")
            if thr is None or (isinstance(thr, float) and np.isnan(thr)):
                # Threshold wasn't fitted (sparse training data) — skip
                continue
            s = panel[feat]
            direction = cfg["direction"]
            if direction == "higher_means_recession":
                d = (s >= thr).astype(float)
            elif direction == "lower_means_recession":
                d = (s <= thr).astype(float)
            else:
                raise ValueError(
                    f"Unknown empirical direction {direction!r} for {feat}"
                )
            d[s.isna()] = np.nan
            # If a feature appears in BOTH theoretical and empirical,
            # theoretical wins (it's literature-grounded). Skip if present.
            if feat not in dummies.columns:
                dummies[feat] = d

    return dummies


# =============================================================================
# Top-level convenience
# =============================================================================

def add_breadth_columns(
    panel: pd.DataFrame,
    at_risk_dummies: pd.DataFrame,
    feature_tiers: dict[str, int],
    tier_labels: Optional[dict[int, str]] = None,
) -> pd.DataFrame:
    """Compute breadth and append the columns to a feature panel.

    Args:
        panel: the feature panel to append breadth columns to.
        at_risk_dummies: 0/1/NaN dummies (from build_at_risk_dummies).
        feature_tiers: {feature -> tier_int}.
        tier_labels: optional {tier_int -> label}.

    Returns:
        panel + BREADTH_GLOBAL + BREADTH_TIER_* columns.
    """
    breadth = compute_breadth(at_risk_dummies, feature_tiers, tier_labels)
    out = panel.copy()
    for col in breadth.columns:
        out[col] = breadth[col]
    return out
