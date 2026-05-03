"""
recession/features/pit_loader.py

Point-in-time feature loader. The single most important module in Step 4
because it enforces the leakage-prevention discipline that we discussed
after the equity-side 0.967 → 0.510 disaster.

Two functions:
  load_panel(as_of, target, horizon, ...) -> DataFrame
      Loads features as they were known at `as_of`. Used in walk-forward
      backtests (called once per fold's training cutoff) and live inference.

  load_targets(target, horizon) -> Series
      Loads target labels for supervised training. Joined to feature panel
      by observation_month with horizon shift.

Design rules (all five decisions enforced here):
  - Decision 1: Hamilton/YoY transforms applied via transforms.py (caller
    decides; this module just loads raw values).
  - Decision 2: At-risk dummies applied by builder.py (this module loads
    raw values that builder transforms).
  - Decision 3: PCA applied by builder.py.
  - Decision 4 (CRITICAL): `as_of` is REQUIRED. Function fails loud if None.
    Every row returned has vintage_date <= as_of, no exceptions.
  - Decision 5: per-model feature subsets via `feature_subset` parameter
    plus `min_history_year` filter using features_registry.available_from.

What this module does NOT do:
  - Apply transforms (yoy, hamilton, etc.) — that's transforms.py / builder.py
  - Compute PCA — that's pca.py
  - Compute breadth or freshness — those are breadth.py / freshness.py
  - Decide which features go to which model — caller passes feature_subset

This module's contract: return a clean rectangular DataFrame of raw values,
PIT-honest, with NaN where data wasn't yet available.
"""
from __future__ import annotations

import sqlite3
from datetime import date as _date
from pathlib import Path
from typing import Literal, Optional

import pandas as pd


# =============================================================================
# Constants
# =============================================================================

DEFAULT_DB_PATH = Path.cwd() / "recession.db"

ValidHorizon = Literal["h=0", "h=1", "h=3", "h=6", "h=12"]
ValidTarget = Literal["T1", "T2", "T5"]   # T3 deferred; T4 is composite


# =============================================================================
# As-of validation
# =============================================================================

def _validate_as_of(as_of: object) -> str:
    """Validate as_of and normalize to YYYY-MM-DD string.

    Decision 4: this is fail-loud. We DELIBERATELY do not provide a default
    value. A None or missing as_of raises ValueError immediately.

    Accepted forms:
        'today'             -> use today's date (live mode)
        '2008-09-30'        -> ISO date string
        date(2008, 9, 30)   -> python date object

    Returns:
        ISO date string YYYY-MM-DD.
    """
    if as_of is None:
        raise ValueError(
            "as_of is required. Pass an ISO date string ('2008-09-30') or "
            "'today' for live mode. Never None — that would silently use "
            "latest data and create look-ahead leakage."
        )

    if isinstance(as_of, str):
        if as_of.lower() == "today":
            return _date.today().isoformat()
        # Validate ISO format by parsing
        try:
            parsed = _date.fromisoformat(as_of)
            return parsed.isoformat()
        except ValueError as e:
            raise ValueError(
                f"as_of='{as_of}' is not a valid ISO date. "
                f"Use 'YYYY-MM-DD' format or 'today'. ({e})"
            )

    if isinstance(as_of, _date):
        return as_of.isoformat()

    raise TypeError(
        f"as_of must be str or date, got {type(as_of).__name__}: {as_of!r}"
    )


# =============================================================================
# Feature panel loader
# =============================================================================

def load_panel(
    as_of: object,                      # str or date — REQUIRED, fails loud
    *,                                  # force keyword args after this
    target: ValidTarget = "T1",
    horizon: ValidHorizon = "h=12",
    min_history_year: Optional[int] = None,
    feature_subset: Optional[list[str]] = None,
    obs_month_min: Optional[str] = None,
    obs_month_max: Optional[str] = None,
    db_path: Path = DEFAULT_DB_PATH,
) -> pd.DataFrame:
    """Load a point-in-time feature panel.

    Returns a DataFrame indexed by observation_month (DatetimeIndex), with
    one column per feature. All values are AS THEY WERE KNOWN on `as_of`.

    Args:
        as_of: ISO date string, 'today', or date object. Required.
            Every row returned has vintage_date <= as_of.
        target: 'T1', 'T2', or 'T5'. Used to apply target-specific
            eligibility filters (e.g., T5 only meaningful from 1986+ when
            BAA10Y starts).
        horizon: 'h=0', 'h=1', 'h=3', 'h=6', 'h=12'. Used to filter
            features by their `eligible_horizons` attribute (Decision 5
            implementation: labor features excluded from h=12 fits).
        min_history_year: if set, drop features whose `available_from`
            is later than this year. Default None = keep all features
            regardless of start year. Use this when a model needs
            long-history features only (e.g., M5 needs 1960+ for
            Stock-Watson; pass min_history_year=1960).
        feature_subset: explicit list of feature names. If provided,
            overrides automatic eligibility filtering. Use for model-
            specific subsets (e.g., M1 = ['T10Y3M'] only).
        obs_month_min, obs_month_max: filter observation_month range.
            Default = full available history.
        db_path: SQLite path. Default = cwd/recession.db.

    Returns:
        DataFrame:
            index = DatetimeIndex of observation months
            columns = feature names
            values = float, NaN where data wasn't yet known at as_of

    Raises:
        ValueError if as_of is missing or invalid.
        FileNotFoundError if db_path doesn't exist.
    """
    # === Decision 4: Validate as_of ===
    as_of_iso = _validate_as_of(as_of)

    if not Path(db_path).exists():
        raise FileNotFoundError(f"DB not found: {db_path}")

    # === Decision 5: Determine eligible features ===
    eligible_features = _determine_eligible_features(
        db_path=db_path,
        target=target,
        horizon=horizon,
        min_history_year=min_history_year,
        feature_subset=feature_subset,
    )

    if not eligible_features:
        raise ValueError(
            f"No features are eligible for target={target}, horizon={horizon}, "
            f"min_history_year={min_history_year}. Check features_registry."
        )

    # === Build the SQL query ===
    # Strategy: for each feature × month, take the row with the LATEST
    # vintage_date <= as_of. This is the value as known on as_of.
    placeholders = ",".join("?" * len(eligible_features))
    sql = f"""
        WITH eligible AS (
            SELECT
                feature_name,
                observation_month,
                value,
                vintage_date,
                ROW_NUMBER() OVER (
                    PARTITION BY feature_name, observation_month
                    ORDER BY vintage_date DESC
                ) AS rn
            FROM features_monthly
            WHERE feature_name IN ({placeholders})
              AND vintage_date <= ?
              {"AND observation_month >= ?" if obs_month_min else ""}
              {"AND observation_month <= ?" if obs_month_max else ""}
        )
        SELECT feature_name, observation_month, value
        FROM eligible
        WHERE rn = 1
        ORDER BY observation_month, feature_name
    """

    params: list = list(eligible_features) + [as_of_iso]
    if obs_month_min:
        params.append(obs_month_min)
    if obs_month_max:
        params.append(obs_month_max)

    conn = sqlite3.connect(db_path)
    try:
        long_df = pd.read_sql_query(sql, conn, params=params)
    finally:
        conn.close()

    if long_df.empty:
        # Return empty DataFrame with feature columns set, so downstream
        # code that expects specific columns doesn't crash.
        return pd.DataFrame(columns=eligible_features)

    # Pivot: long → wide. Rows = months, columns = features.
    wide = long_df.pivot(
        index="observation_month",
        columns="feature_name",
        values="value",
    )
    # Convert index to DatetimeIndex for compatibility with transforms.py
    wide.index = pd.to_datetime(wide.index)
    wide.index.name = "observation_month"
    # Ensure all eligible_features appear as columns even if a feature
    # had zero data points (e.g. all values had vintage_date > as_of).
    for f in eligible_features:
        if f not in wide.columns:
            wide[f] = pd.NA
    # Sort columns alphabetically for stable test behaviour
    wide = wide[sorted(wide.columns)]
    return wide


# =============================================================================
# Eligibility logic — Decision 5
# =============================================================================

def _determine_eligible_features(
    db_path: Path,
    target: str,
    horizon: str,
    min_history_year: Optional[int],
    feature_subset: Optional[list[str]],
) -> list[str]:
    """Determine which features to load given target/horizon/history filters.

    Logic:
    1. If feature_subset is provided explicitly, use that (validated against
       the registry to avoid typos).
    2. Otherwise:
       - Start from all active features in registry
       - If min_history_year is set, drop features whose available_from
         year > min_history_year. Default None means keep all.
       - Drop features ineligible for the requested horizon (per
         series_specs.eligible_horizons — labor features excluded from
         h=12)
       - Apply target-specific exclusions (e.g., SP500 raw level dropped
         in favor of SP500_RET_12M and SP500_DRAWDOWN_12M for modeling)
    """
    conn = sqlite3.connect(db_path)
    try:
        cur = conn.execute(
            """SELECT feature_name, available_from, tier
               FROM features_registry
               WHERE is_active = 1"""
        )
        rows = cur.fetchall()
    finally:
        conn.close()

    all_active = {name: (af, tier) for name, af, tier in rows}

    # Case 1: explicit subset
    if feature_subset is not None:
        unknown = set(feature_subset) - set(all_active)
        if unknown:
            raise ValueError(
                f"Unknown features in feature_subset: {sorted(unknown)}. "
                f"Check features_registry; available active features: "
                f"{sorted(all_active.keys())}"
            )
        return list(feature_subset)

    # Case 2: automatic filtering
    eligible: list[str] = []
    for name, (af, tier) in all_active.items():
        # History filter: only apply if min_history_year is explicitly set.
        # Default (None) keeps all features regardless of start year.
        if min_history_year is not None and af is not None:
            try:
                af_year = int(af[:4])
            except (TypeError, ValueError):
                af_year = 9999   # defensive: if unparseable, drop it
            if af_year > min_history_year:
                continue

        # Apply horizon eligibility (Decision 5):
        # Labor tier (2) features should not appear at h=6 or h=12 unless
        # explicitly subsetted. This matches series_specs.eligible_horizons
        # which we set in Step 3.5. But registry doesn't have that field —
        # it's in series_specs.py. So replicate the rule here:
        if tier == 2 and horizon in ("h=6", "h=12"):
            # Tier 2 = labor. Coincident-by-design. Skip for long horizons.
            continue

        # Drop raw SP500 level — modelers should use SP500_RET_12M or
        # SP500_DRAWDOWN_12M, not the non-stationary level.
        if name == "SP500":
            continue

        # Drop COVID_DUMMY at non-h=0 horizons — it's a coincident control,
        # not a leading indicator.
        if name == "COVID_DUMMY" and horizon != "h=0":
            continue

        eligible.append(name)

    return sorted(eligible)


# =============================================================================
# Targets loader
# =============================================================================

def load_targets(
    target: ValidTarget,
    horizon: ValidHorizon,
    *,
    obs_month_min: Optional[str] = None,
    obs_month_max: Optional[str] = None,
    db_path: Path = DEFAULT_DB_PATH,
) -> pd.Series:
    """Load target labels, shifted forward by `horizon`.

    Example: target='T1', horizon='h=12' returns a Series where row at
    month M has the label of T1 at month M+12. This is the value the
    model is trying to PREDICT given features known at M.

    Note: targets are NOT filtered by as_of here. Targets are only used
    in TRAINING, where you fit on past targets and predict future. The
    caller (walk-forward harness) is responsible for not feeding the
    model labels from after its training cutoff.

    Args:
        target: 'T1', 'T2', or 'T5'.
        horizon: 'h=0' (no shift), 'h=1' (1 month ahead), etc.
        obs_month_min/max: range filter on observation_month BEFORE shift.
        db_path: SQLite path.

    Returns:
        Series indexed by observation_month, values in {0, 1, NaN}.
        NaN appears at the tail where the future label is unknown.
    """
    conn = sqlite3.connect(db_path)
    try:
        cur = conn.execute(
            """SELECT t.observation_month, t.label
               FROM targets_monthly t
               INNER JOIN (
                   SELECT target_id, observation_month, MAX(announcement_date) AS max_ann
                   FROM targets_monthly
                   WHERE target_id = ?
                   GROUP BY target_id, observation_month
               ) latest
                 ON t.target_id = latest.target_id
                 AND t.observation_month = latest.observation_month
                 AND t.announcement_date = latest.max_ann
               WHERE t.target_id = ?""",
            (target, target),
        )
        rows = cur.fetchall()
    finally:
        conn.close()

    if not rows:
        raise ValueError(f"No target rows found for target_id={target!r}")

    s = pd.Series(
        data=[r[1] for r in rows],
        index=pd.to_datetime([r[0] for r in rows]),
        name=f"{target}_label",
    )
    s = s.sort_index()

    # Apply horizon shift: row at M holds label of M+h
    h_months = int(horizon.split("=")[1])
    if h_months > 0:
        s = s.shift(-h_months)

    # Apply observation_month filters AFTER shift (so callers see the
    # full range that has both features and labels)
    if obs_month_min is not None:
        s = s[s.index >= pd.to_datetime(obs_month_min)]
    if obs_month_max is not None:
        s = s[s.index <= pd.to_datetime(obs_month_max)]

    return s


# =============================================================================
# Convenience wrappers
# =============================================================================

def load_panel_latest(
    target: ValidTarget = "T1",
    horizon: ValidHorizon = "h=12",
    **kwargs,
) -> pd.DataFrame:
    """Convenience wrapper: load_panel(as_of='today', ...).

    Use for live inference where you want the most recent available data
    for each feature. NEVER use for backtest training — backtests must
    use the actual fold cutoff date as as_of.
    """
    return load_panel(as_of="today", target=target, horizon=horizon, **kwargs)
