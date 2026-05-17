"""
recession/features/freshness.py

DATA_FRESHNESS engineered feature.

Concept: in live use, a recession model works with data of varying staleness.
Jobless claims update weekly with a ~1 week lag. JOLTS lags ~5-6 weeks.
GDP-adjacent series can lag a quarter. When the model makes a prediction
"as of today", some inputs reflect last week and some reflect two months ago.

DATA_FRESHNESS quantifies this. For a given as_of date, it measures how
many days have elapsed between when each feature's most recent observation
was published (its vintage_date) and the as_of date.

Why it matters: a model can learn "when my freshest credit data is already
40 days old, my prediction is less reliable" — or simply that stale-data
regimes behave differently. It also serves as a data-quality monitor.

Two outputs:
  - FRESHNESS_MEAN_DAYS   average staleness across all features (in days)
  - FRESHNESS_MAX_DAYS    worst-case staleness (the most-lagged feature)

Behaviour for historical training rows:
  For a deep-historical observation_month (say 1995-03), by the time of any
  reasonable as_of (say 2008-09-01) every 1995 value was long since
  published. So freshness for old rows is near-constant and large. The
  signal only VARIES near the as_of edge, where some features have recent
  data and others lag. That's intentional — freshness is a "data edge"
  feature, informative for live prediction and near-flat for old training
  data. Models that don't find it useful will simply down-weight it.

This module reads vintage_date directly from features_monthly. It needs
DB access (unlike transforms.py / breadth.py which are pure functions).
"""
from __future__ import annotations

import sqlite3
from datetime import date as _date
from pathlib import Path
from typing import Optional

import pandas as pd


DEFAULT_DB_PATH = Path.cwd() / "recession.db"


# =============================================================================
# Core
# =============================================================================

def compute_freshness(
    as_of: str,
    feature_names: list[str],
    obs_months: pd.DatetimeIndex,
    db_path: Path = DEFAULT_DB_PATH,
) -> pd.DataFrame:
    """Compute DATA_FRESHNESS columns for a set of observation months.

    For each observation_month M and each feature F, find the latest
    vintage_date V such that V <= as_of (the value the model would have
    known at as_of). Freshness for (F, M) = (as_of - V) in days.

    Then per month, aggregate across features:
        FRESHNESS_MEAN_DAYS  = mean staleness across features
        FRESHNESS_MAX_DAYS   = max staleness (worst-lagged feature)

    Args:
        as_of: ISO date string. The reference point for staleness.
            Every vintage with vintage_date <= as_of is eligible.
        feature_names: list of features to include in the freshness calc.
        obs_months: the observation months to produce freshness for
            (typically the index of the feature panel).
        db_path: path to recession.db.

    Returns:
        DataFrame indexed by obs_months, with columns
        FRESHNESS_MEAN_DAYS and FRESHNESS_MAX_DAYS.
        NaN where no feature had any vintage <= as_of for that month.
    """
    as_of_date = _date.fromisoformat(as_of)

    if not feature_names:
        out = pd.DataFrame(index=obs_months)
        out["FRESHNESS_MEAN_DAYS"] = float("nan")
        out["FRESHNESS_MAX_DAYS"] = float("nan")
        return out

    # Pull, for each (feature, observation_month), the latest vintage_date
    # that is <= as_of.
    placeholders = ",".join("?" * len(feature_names))
    sql = f"""
        WITH ranked AS (
            SELECT
                feature_name,
                observation_month,
                vintage_date,
                ROW_NUMBER() OVER (
                    PARTITION BY feature_name, observation_month
                    ORDER BY vintage_date DESC
                ) AS rn
            FROM features_monthly
            WHERE feature_name IN ({placeholders})
              AND vintage_date <= ?
        )
        SELECT feature_name, observation_month, vintage_date
        FROM ranked
        WHERE rn = 1
    """
    params = list(feature_names) + [as_of]

    conn = sqlite3.connect(db_path)
    try:
        rows = conn.execute(sql, params).fetchall()
    finally:
        conn.close()

    # Build {observation_month -> list of staleness-in-days}
    from collections import defaultdict
    staleness: dict[str, list[int]] = defaultdict(list)
    for feature_name, obs_month, vintage_date in rows:
        try:
            v = _date.fromisoformat(vintage_date)
        except (TypeError, ValueError):
            continue
        days = (as_of_date - v).days
        # Defensive: vintage shouldn't be after as_of (SQL filters it),
        # but guard anyway.
        if days < 0:
            continue
        staleness[obs_month].append(days)

    # Aggregate per observation month
    out = pd.DataFrame(index=obs_months)
    mean_vals = []
    max_vals = []
    for m in obs_months:
        key = m.strftime("%Y-%m-%d")
        vals = staleness.get(key, [])
        if vals:
            mean_vals.append(sum(vals) / len(vals))
            max_vals.append(float(max(vals)))
        else:
            mean_vals.append(float("nan"))
            max_vals.append(float("nan"))

    out["FRESHNESS_MEAN_DAYS"] = mean_vals
    out["FRESHNESS_MAX_DAYS"] = max_vals
    return out


# =============================================================================
# Convenience
# =============================================================================

def add_freshness_columns(
    panel: pd.DataFrame,
    as_of: str,
    db_path: Path = DEFAULT_DB_PATH,
    feature_names: Optional[list[str]] = None,
) -> pd.DataFrame:
    """Compute freshness and append the columns to a feature panel.

    Args:
        panel: feature panel (months x features).
        as_of: ISO date string — the staleness reference point.
        db_path: path to recession.db.
        feature_names: which features to base freshness on. Defaults to
            all columns of `panel` (excluding any engineered columns that
            don't correspond to real features — those are skipped because
            they won't be in features_monthly anyway).

    Returns:
        panel + FRESHNESS_MEAN_DAYS + FRESHNESS_MAX_DAYS.
    """
    if feature_names is None:
        # Use panel columns, but exclude engineered columns that aren't
        # real DB features (PCs, breadth, prior freshness). They simply
        # won't be found in features_monthly, but filtering keeps the
        # SQL query lean.
        feature_names = [
            c for c in panel.columns
            if not c.endswith("_PC1")
            and not c.startswith("BREADTH_")
            and not c.startswith("FRESHNESS_")
        ]

    freshness = compute_freshness(
        as_of=as_of,
        feature_names=feature_names,
        obs_months=panel.index,
        db_path=db_path,
    )
    out = panel.copy()
    for col in freshness.columns:
        out[col] = freshness[col]
    return out
