"""
analysis/walk_forward.py — Walk-forward backtest + leakage diagnostic.

Solves the AUC 0.967 (training) vs 0.510 (live) puzzle by running strict
purged k-fold cross-validation with embargo on the SAME features your live
ensemble uses. Three diagnostic modes:

    backtest    — Fresh model trained with proper walk-forward; OOS AUC vs
                  training AUC (the gap = your overfitting / leakage)
    ablation    — Drop each feature one at a time; identify critical and
                  useless features (huge importance = potential leaker)
    leak_audit  — Per-feature correlation with target; flag features whose
                  correlation is too high to be honest (>0.3 = suspicious,
                  >0.5 = almost certainly a leak)

Drop in: ml_quant_fund/analysis/walk_forward.py

Usage:
    # Fast diagnostic, no training (3 minutes):
    python -m analysis.walk_forward --mode leak_audit

    # Full walk-forward backtest (10–30 minutes depending on data size):
    python -m analysis.walk_forward --mode backtest --horizon 1

    # All three diagnostics + write CSVs:
    python -m analysis.walk_forward --mode all --csv-prefix diag

    # Per-horizon backtest (separate model for each horizon):
    python -m analysis.walk_forward --mode backtest

How to read the output:

    train_auc=0.97, test_auc=0.51 → MASSIVE LEAK. Find it via ablation/audit.
    train_auc=0.65, test_auc=0.55 → Mild overfit, normal. AUC 0.55 is real.
    train_auc=0.55, test_auc=0.53 → No leak. AUC ~0.53 is your true ceiling.
                                    Pivot strategy: build new alpha sources.

Schema requirements:
    accuracy.db must have:
      prediction_features(ticker, prediction_date, horizon, [feature columns])
      outcomes(ticker, prediction_date, horizon, actual_return, actual_up)

    They join on (ticker, prediction_date, horizon).
"""

from __future__ import annotations

import argparse
import math
import sqlite3
import warnings
from pathlib import Path
from typing import Iterator, List, Optional, Tuple, Dict

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

from sklearn.metrics import roc_auc_score, brier_score_loss


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

EMBARGO_DAYS = 5
N_FOLDS = 5
RANDOM_STATE = 42
MIN_TRAIN_ROWS = 200
MIN_TEST_ROWS = 50

DEFAULT_XGB_PARAMS = dict(
    n_estimators=200,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="binary:logistic",
    eval_metric="logloss",
    random_state=RANDOM_STATE,
    verbosity=0,
)

# Skip these columns when auto-detecting feature columns.
NON_FEATURE_COLS = {
    "ticker", "prediction_date", "horizon",
    "actual_return", "actual_up", "prob_up",
    "id", "rowid", "created_at", "updated_at",
}


# ---------------------------------------------------------------------------
# Purged k-fold splitter (Lopez de Prado, Chapter 7)
# ---------------------------------------------------------------------------

def purged_kfold_indices(
    dates: pd.Series,
    n_folds: int = N_FOLDS,
    embargo: int = EMBARGO_DAYS,
) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
    """Time-ordered k-fold with embargo.

    Splits the rows into k temporal folds (sorted by date). For each fold:
      - test = that fold's rows
      - train = all OTHER folds, MINUS rows whose date is within
                `embargo` days of any test date.

    The embargo prevents serial-correlation leakage when target horizons
    overlap with adjacent training samples.
    """
    dates = pd.to_datetime(dates).reset_index(drop=True)
    sorted_pos = dates.argsort().values  # positions sorted by date
    n = len(dates)
    fold_size = n // n_folds
    embargo_td = pd.Timedelta(days=embargo)

    for fold in range(n_folds):
        start = fold * fold_size
        end = (fold + 1) * fold_size if fold < n_folds - 1 else n
        test_pos = sorted_pos[start:end]
        if len(test_pos) == 0:
            continue

        test_dates = dates.iloc[test_pos]
        embargo_lo = test_dates.min() - embargo_td
        embargo_hi = test_dates.max() + embargo_td

        # Train = all rows whose date is BEFORE embargo_lo OR AFTER embargo_hi
        train_mask = (dates < embargo_lo) | (dates > embargo_hi)
        train_pos = np.where(train_mask.values)[0]

        if len(train_pos) >= MIN_TRAIN_ROWS and len(test_pos) >= MIN_TEST_ROWS:
            yield train_pos, test_pos


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_panel(
    db_path: Path,
    horizon: Optional[int] = None,
    since: Optional[str] = None,
) -> pd.DataFrame:
    """Load prediction_features × outcomes joined panel."""
    conn = sqlite3.connect(str(db_path))
    cur = conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = {r[0] for r in cur.fetchall()}

    if "prediction_features" not in tables:
        conn.close()
        raise RuntimeError(
            f"Table 'prediction_features' not found in {db_path}.\n"
            f"Available tables: {sorted(tables)}\n"
            f"This module requires the prediction_features table you started "
            f"populating Mar 24 2026. If it doesn't exist yet, you can't run "
            f"the walk-forward backtest from saved features — you'd need to "
            f"recompute features inside your training loop instead."
        )
    if "outcomes" not in tables:
        conn.close()
        raise RuntimeError(f"Table 'outcomes' not found in {db_path}")

    query = """
        SELECT pf.*, o.actual_return, o.actual_up
        FROM prediction_features pf
        JOIN outcomes o
          ON pf.ticker = o.ticker
         AND pf.prediction_date = o.prediction_date
         AND pf.horizon = o.horizon
        WHERE o.actual_return IS NOT NULL
          AND o.actual_up IS NOT NULL
    """
    params: list = []
    if horizon is not None:
        query += " AND pf.horizon = ?"
        params.append(horizon)
    if since:
        query += " AND pf.prediction_date >= ?"
        params.append(since)
    query += " ORDER BY pf.prediction_date, pf.ticker"

    df = pd.read_sql(query, conn, params=params)
    conn.close()

    if df.empty:
        return df

    df["prediction_date"] = pd.to_datetime(df["prediction_date"])
    return df.reset_index(drop=True)


def load_panel_pit(
    db_path: Path,
    horizon: Optional[int] = None,
    since: Optional[str] = None,
    limit: Optional[int] = None,
    verbose: bool = True,
    progress_every: int = 50,
) -> pd.DataFrame:
    """Load outcomes joined with FRESHLY-BUILT point-in-time features.

    For each row in the outcomes table, this calls:
        build_feature_dataframe(ticker, end_date=prediction_date,
                                 training_mode=True)
    and takes the LAST row (= features as of prediction_date with
    no future leak from sentiment/insider/earnings).

    This is the HONEST panel — replaces the prediction_features-table
    query in load_panel(), which was scope-limited to ~27 features
    captured at prediction time.

    SLOW. Each call to build_feature_dataframe is ~10-20 seconds.
    For 7,500 outcomes × 18s = ~37 hours runtime.

    Args:
        db_path:        accuracy.db path
        horizon:        filter to specific horizon (1, 3, or 5)
        since:          ISO date string — only use predictions on/after
        limit:          for testing — only process first N outcomes
        verbose:        print progress
        progress_every: print progress line every N rows

    Returns:
        DataFrame with same shape as load_panel():
        ticker, prediction_date, horizon, actual_return, actual_up,
        and ~80 feature columns from build_feature_dataframe.
    """
    from features.builder import build_feature_dataframe
    import time as _time

    conn = sqlite3.connect(str(db_path))

    # Load outcomes only — features come from build_feature_dataframe
    query = """
        SELECT ticker, prediction_date, horizon, actual_return, actual_up
        FROM outcomes
        WHERE actual_return IS NOT NULL
          AND actual_up IS NOT NULL
    """
    params: list = []
    if horizon is not None:
        query += " AND horizon = ?"
        params.append(horizon)
    if since:
        query += " AND prediction_date >= ?"
        params.append(since)
    query += " ORDER BY ticker, prediction_date, horizon"

    if limit is not None:
        query += f" LIMIT {int(limit)}"

    outcomes = pd.read_sql(query, conn, params=params)
    conn.close()

    if outcomes.empty:
        return outcomes

    if verbose:
        print(f"  PIT panel: processing {len(outcomes):,} outcomes...")
        n_unique_pairs = outcomes[["ticker", "prediction_date"]].drop_duplicates().shape[0]
        print(f"  Unique (ticker, prediction_date) pairs: {n_unique_pairs:,}")
        print(f"  Estimated runtime: ~{n_unique_pairs * 18 / 3600:.1f} hours")

    rows = []
    fail_count = 0
    t0 = _time.time()

    # Group by (ticker, prediction_date) so we only build features once
    # even if there are multiple horizons for the same date.
    grouped = outcomes.groupby(["ticker", "prediction_date"])

    for i, ((ticker, pred_date), group) in enumerate(grouped, 1):
        try:
            df_features = build_feature_dataframe(
                ticker,
                end_date=str(pred_date),
                training_mode=True,
            )
            if df_features.empty:
                fail_count += 1
                continue
            # Take features at the prediction_date (last row of as-of panel)
            feat_row = df_features.iloc[-1].to_dict()
            # Strip non-feature metadata that build_feature_dataframe injects
            feat_row.pop("date", None)
            feat_row.pop("ticker", None)
            # Add one row per horizon for this (ticker, pred_date)
            for _, outcome_row in group.iterrows():
                row = {
                    "ticker": ticker,
                    "prediction_date": pred_date,
                    "horizon": int(outcome_row["horizon"]),
                    "actual_return": float(outcome_row["actual_return"]),
                    "actual_up": int(outcome_row["actual_up"]),
                    **feat_row,
                }
                rows.append(row)
        except Exception as e:
            fail_count += 1
            if verbose and fail_count < 10:
                print(f"  ! {ticker} {pred_date}: {e}")
            continue

        if verbose and i % progress_every == 0:
            elapsed = _time.time() - t0
            rate = i / elapsed
            remaining = (len(grouped) - i) / rate if rate > 0 else 0
            print(
                f"  [{i:,}/{len(grouped):,}] "
                f"({rate:.2f}/s, {remaining/3600:.1f}h remaining, "
                f"{fail_count} failed)"
            )

    if not rows:
        if verbose:
            print(f"  PIT panel: 0 successful rows ({fail_count} failures)")
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df["prediction_date"] = pd.to_datetime(df["prediction_date"])
    df = df.reset_index(drop=True)

    if verbose:
        elapsed = _time.time() - t0
        print(
            f"  PIT panel: {len(df):,} rows, {fail_count} failed, "
            f"elapsed {elapsed/3600:.2f}h"
        )

    return df


def detect_feature_cols(df: pd.DataFrame) -> List[str]:
    """Return numeric columns that look like features (not metadata)."""
    return [
        c for c in df.columns
        if c not in NON_FEATURE_COLS
        and pd.api.types.is_numeric_dtype(df[c])
    ]


# ---------------------------------------------------------------------------
# Walk-forward backtest
# ---------------------------------------------------------------------------

def walk_forward_backtest(
    df: pd.DataFrame,
    feature_cols: List[str],
    n_folds: int = N_FOLDS,
    embargo: int = EMBARGO_DAYS,
    model_params: Optional[dict] = None,
    target_col: str = "actual_up",
    verbose: bool = True,
) -> Tuple[pd.DataFrame, Dict]:
    """Run purged k-fold walk-forward; return per-fold and pooled metrics."""
    if not HAS_XGB:
        raise RuntimeError("xgboost not installed. pip install xgboost")

    params = {**DEFAULT_XGB_PARAMS, **(model_params or {})}

    X = df[feature_cols].fillna(0.0).values.astype(np.float32)
    y = df[target_col].astype(int).values
    dates = df["prediction_date"]

    fold_results = []
    pooled_y_true: List[int] = []
    pooled_y_pred: List[float] = []

    for fold_i, (train_idx, test_idx) in enumerate(
        purged_kfold_indices(dates, n_folds=n_folds, embargo=embargo)
    ):
        X_tr, y_tr = X[train_idx], y[train_idx]
        X_te, y_te = X[test_idx], y[test_idx]

        # Need both classes in train and test to compute AUC
        if len(np.unique(y_tr)) < 2 or len(np.unique(y_te)) < 2:
            continue

        model = XGBClassifier(**params)
        model.fit(X_tr, y_tr)

        y_tr_pred = model.predict_proba(X_tr)[:, 1]
        y_te_pred = model.predict_proba(X_te)[:, 1]

        train_auc = roc_auc_score(y_tr, y_tr_pred)
        test_auc = roc_auc_score(y_te, y_te_pred)
        test_acc = float(((y_te_pred > 0.5).astype(int) == y_te).mean())
        test_brier = brier_score_loss(y_te, y_te_pred)

        fold_results.append({
            "fold": fold_i + 1,
            "n_train": len(train_idx),
            "n_test": len(test_idx),
            "train_auc": round(train_auc, 4),
            "test_auc": round(test_auc, 4),
            "test_acc": round(test_acc, 4),
            "test_brier": round(test_brier, 4),
            "auc_gap": round(train_auc - test_auc, 4),
            "train_min": dates.iloc[train_idx].min().date(),
            "train_max": dates.iloc[train_idx].max().date(),
            "test_min": dates.iloc[test_idx].min().date(),
            "test_max": dates.iloc[test_idx].max().date(),
        })

        pooled_y_true.extend(y_te.tolist())
        pooled_y_pred.extend(y_te_pred.tolist())

        if verbose:
            print(f"  fold {fold_i + 1}: train n={len(train_idx)}, "
                  f"test n={len(test_idx)}, "
                  f"train_auc={train_auc:.3f}, test_auc={test_auc:.3f}, "
                  f"gap={train_auc - test_auc:+.3f}")

    folds_df = pd.DataFrame(fold_results)

    overall: Dict = {}
    if pooled_y_true:
        yt = np.array(pooled_y_true)
        yp = np.array(pooled_y_pred)
        overall["n_oos"] = int(len(yt))
        overall["pooled_oos_auc"] = round(roc_auc_score(yt, yp), 4)
        overall["pooled_oos_acc"] = round(((yp > 0.5).astype(int) == yt).mean(), 4)
        overall["pooled_oos_brier"] = round(brier_score_loss(yt, yp), 4)
        overall["mean_train_auc"] = round(folds_df["train_auc"].mean(), 4)
        overall["mean_test_auc"] = round(folds_df["test_auc"].mean(), 4)
        overall["auc_gap"] = round(
            overall["mean_train_auc"] - overall["mean_test_auc"], 4)

    return folds_df, overall


# ---------------------------------------------------------------------------
# Feature ablation
# ---------------------------------------------------------------------------

def feature_ablation(
    df: pd.DataFrame,
    feature_cols: List[str],
    n_folds: int = 3,  # Fewer folds for speed
    embargo: int = EMBARGO_DAYS,
    sample_size: Optional[int] = 20000,
    top_n_features: Optional[int] = None,
    verbose: bool = True,
) -> Tuple[pd.DataFrame, float]:
    """Drop each feature and measure OOS AUC delta.

    importance = baseline_AUC - ablated_AUC
        > 0   feature helps (positive importance)
        ≈ 0   feature is useless (or correlated with another)
        < 0   feature hurts (you should drop it)
        >> 0  potential leaker (one feature carrying too much weight)
    """
    if sample_size and len(df) > sample_size:
        df = df.sample(sample_size, random_state=RANDOM_STATE).reset_index(drop=True)

    features_to_test = (feature_cols if top_n_features is None
                        else feature_cols[:top_n_features])

    if verbose:
        print(f"\nAblation: baseline AUC over {len(df)} rows × "
              f"{len(feature_cols)} features...")
    _, baseline = walk_forward_backtest(
        df, feature_cols, n_folds=n_folds, embargo=embargo, verbose=False
    )
    baseline_auc = baseline.get("pooled_oos_auc", float("nan"))
    if verbose:
        print(f"  baseline pooled OOS AUC: {baseline_auc:.4f}")

    rows = []
    for i, feat in enumerate(features_to_test):
        reduced = [c for c in feature_cols if c != feat]
        _, summary = walk_forward_backtest(
            df, reduced, n_folds=n_folds, embargo=embargo, verbose=False
        )
        ablated_auc = summary.get("pooled_oos_auc", float("nan"))
        rows.append({
            "feature": feat,
            "auc_with": round(baseline_auc, 4),
            "auc_without": round(ablated_auc, 4),
            "importance": round(baseline_auc - ablated_auc, 4),
        })
        if verbose and (i + 1) % 10 == 0:
            print(f"  {i + 1}/{len(features_to_test)} features ablated")

    out = pd.DataFrame(rows).sort_values("importance", ascending=False)
    return out, baseline_auc


# ---------------------------------------------------------------------------
# Leak audit
# ---------------------------------------------------------------------------

def leak_audit(
    df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str = "actual_up",
) -> pd.DataFrame:
    """Per-feature correlation with target. Flag suspiciously-high values.

    Heuristics:
      |corr| < 0.10  expected for an honest equity-prediction feature
      |corr| 0.10–0.30  worth attention but plausible
      |corr| 0.30–0.50  suspicious — verify how feature is computed
      |corr| > 0.50  almost certainly a leak

    For binary targets we use the point-biserial correlation (which equals
    Pearson on a numeric 0/1 column).
    """
    target = df[target_col].astype(float)
    target_ret = df["actual_return"].astype(float) if "actual_return" in df.columns else None

    rows = []
    for feat in feature_cols:
        x = df[feat].astype(float)

        if x.notna().sum() < 100 or x.std() == 0 or x.std() != x.std():
            continue

        c_class = x.corr(target)
        c_ret = x.corr(target_ret) if target_ret is not None else float("nan")

        abs_class = abs(c_class) if not (c_class != c_class) else 0.0
        if abs_class > 0.50:
            severity = "LEAK_LIKELY"
        elif abs_class > 0.30:
            severity = "SUSPICIOUS"
        elif abs_class > 0.10:
            severity = "ELEVATED"
        else:
            severity = "OK"

        rows.append({
            "feature": feat,
            "corr_with_actual_up": round(c_class, 4),
            "abs_corr_class": round(abs_class, 4),
            "corr_with_actual_return": round(c_ret, 4) if c_ret == c_ret else None,
            "n_obs": int(x.notna().sum()),
            "severity": severity,
        })

    out = pd.DataFrame(rows).sort_values("abs_corr_class", ascending=False)
    return out


# ---------------------------------------------------------------------------
# Reporter
# ---------------------------------------------------------------------------

def print_backtest_report(folds_df: pd.DataFrame, overall: Dict) -> None:
    print("\n" + "=" * 72)
    print("WALK-FORWARD BACKTEST RESULTS")
    print("=" * 72)
    if folds_df.empty:
        print("No folds completed. Likely insufficient data.")
        return
    pd.set_option("display.float_format", "{:.4f}".format)
    print("\nPer-fold:")
    print(folds_df[[
        "fold", "n_train", "n_test", "train_auc", "test_auc",
        "auc_gap", "test_acc", "test_brier",
    ]].to_string(index=False))

    print("\nOverall:")
    for k, v in overall.items():
        print(f"  {k:24s} {v}")

    print("\n--- Diagnosis ---")
    gap = overall.get("auc_gap", 0)
    test_auc = overall.get("mean_test_auc", 0.5)
    if gap > 0.20:
        print(f"  ⚠ MASSIVE train→test AUC gap of {gap:.3f}.")
        print(f"    Training AUC ({overall['mean_train_auc']:.3f}) is fitting noise.")
        print(f"    Run --mode ablation and --mode leak_audit to find the source.")
    elif gap > 0.10:
        print(f"  ⚠ Significant overfit: gap = {gap:.3f}.")
        print(f"    Reduce model complexity (max_depth, n_estimators) or add regularization.")
    elif gap > 0.03:
        print(f"  ✓ Mild overfit ({gap:.3f}). Normal for boosting models.")
    else:
        print(f"  ✓ No overfit. train and test AUC track within {gap:.3f}.")

    if test_auc < 0.52:
        print(f"  ⚠ OOS AUC ({test_auc:.3f}) at coin flip. No detectable signal.")
        print(f"    Pivot: build new alpha sources rather than tune current model.")
    elif test_auc < 0.55:
        print(f"  ⚠ OOS AUC ({test_auc:.3f}) is real but weak.")
        print(f"    Strategy: combine many weak alphas (Finding Alphas Ch. 10).")
    elif test_auc < 0.60:
        print(f"  ✓ OOS AUC ({test_auc:.3f}) is meaningful.")
        print(f"    Strategy: refine + diversify; current architecture viable.")
    else:
        print(f"  ⚠ OOS AUC ({test_auc:.3f}) suspiciously high.")
        print(f"    Audit features for residual leakage before trusting.")


def print_ablation_report(ablation_df: pd.DataFrame, baseline_auc: float) -> None:
    print("\n" + "=" * 72)
    print("FEATURE ABLATION RESULTS")
    print("=" * 72)
    print(f"\nBaseline OOS AUC: {baseline_auc:.4f}")
    print(f"\nTop 15 most-important features (positive = helps OOS AUC):")
    print(ablation_df.head(15).to_string(index=False))
    print(f"\nBottom 15 (low importance — candidates to drop):")
    print(ablation_df.tail(15).to_string(index=False))

    massive = ablation_df[ablation_df["importance"] > 0.05]
    if not massive.empty:
        print("\n⚠ Features with ABNORMALLY HIGH importance (>0.05 AUC drop):")
        print(massive.to_string(index=False))
        print("  In honest 79-feature equity setups, no single feature should be "
              "worth >0.03 AUC.\n  Manually verify how these are computed — they "
              "may be leaking future info.")

    useless = ablation_df[ablation_df["importance"].between(-0.005, 0.005)]
    if len(useless) > 5:
        print(f"\n{len(useless)} features have ~zero importance "
              f"(±0.005 AUC). Candidates to drop:")
        print("  " + ", ".join(useless["feature"].tolist()[:20])
              + (" ..." if len(useless) > 20 else ""))


def print_leak_audit_report(audit_df: pd.DataFrame) -> None:
    print("\n" + "=" * 72)
    print("LEAK AUDIT RESULTS")
    print("=" * 72)

    by_severity = audit_df.groupby("severity").size()
    print("\nFeatures by severity:")
    for sev in ["LEAK_LIKELY", "SUSPICIOUS", "ELEVATED", "OK"]:
        n = by_severity.get(sev, 0)
        marker = "⚠" if sev in ("LEAK_LIKELY", "SUSPICIOUS") else " "
        print(f"  {marker} {sev:14s} {n}")

    suspect = audit_df[audit_df["severity"].isin(["LEAK_LIKELY", "SUSPICIOUS"])]
    if not suspect.empty:
        print(f"\n⚠ Features with suspicious target correlation:")
        print(suspect[[
            "feature", "corr_with_actual_up", "corr_with_actual_return",
            "n_obs", "severity",
        ]].to_string(index=False))
        print("\n  In honest equity prediction, no feature should correlate with")
        print("  next-period direction at |r| > 0.10. Higher correlations suggest")
        print("  the feature was computed using post-target information.")
    else:
        print("\n✓ No features with |corr| > 0.30. No obvious leak signatures.")

    print("\nTop 10 features by |correlation|:")
    print(audit_df.head(10)[[
        "feature", "corr_with_actual_up", "abs_corr_class", "severity",
    ]].to_string(index=False))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Walk-forward backtest + leak audit for ML Quant Fund.",
    )
    parser.add_argument("--db", default="accuracy.db", type=Path)
    parser.add_argument("--mode", default="backtest",
                        choices=["backtest", "ablation", "leak_audit", "all"])
    parser.add_argument("--horizon", type=int, default=None,
                        help="Filter to specific horizon (default: all)")
    parser.add_argument("--since", default=None,
                        help="ISO date — only use predictions on/after")
    parser.add_argument("--folds", type=int, default=N_FOLDS)
    parser.add_argument("--embargo", type=int, default=EMBARGO_DAYS,
                        help="Embargo days between train and test (default: 5)")
    parser.add_argument("--top-features", type=int, default=None,
                        help="Limit ablation to top N features (by variance)")
    parser.add_argument("--sample", type=int, default=20000,
                        help="Subsample size for ablation (default: 20000)")
    parser.add_argument("--csv-prefix", default=None,
                        help="If set, write CSVs with this prefix")
    parser.add_argument("--pit", action="store_true",
                        help="Use point-in-time honest features via "
                             "build_feature_dataframe per outcome (SLOW: 37h+)")
    parser.add_argument("--limit", type=int, default=None,
                        help="(Testing) only process first N outcomes")
    args = parser.parse_args()

    if args.pit:
        print(f"Loading PIT panel from {args.db} (this is the slow honest one)...")
        df = load_panel_pit(
            args.db,
            horizon=args.horizon,
            since=args.since,
            limit=args.limit,
        )
    else:
        print(f"Loading panel from {args.db}...")
        df = load_panel(args.db, horizon=args.horizon, since=args.since)
    if df.empty:
        print("No data after filters. Exiting.")
        return

    feature_cols = detect_feature_cols(df)
    print(f"Loaded {len(df)} rows × {len(feature_cols)} features")
    print(f"Date range: {df['prediction_date'].min().date()} → "
          f"{df['prediction_date'].max().date()}")
    if args.horizon:
        print(f"Horizon filter: {args.horizon}")
    print(f"Target: actual_up ({df['actual_up'].mean()*100:.1f}% positive class)")

    if args.mode in ("backtest", "all"):
        print(f"\nRunning walk-forward backtest "
              f"(n_folds={args.folds}, embargo={args.embargo}d)...")
        folds_df, overall = walk_forward_backtest(
            df, feature_cols, n_folds=args.folds, embargo=args.embargo,
        )
        print_backtest_report(folds_df, overall)
        if args.csv_prefix:
            folds_df.to_csv(f"{args.csv_prefix}_folds.csv", index=False)
            print(f"\nWrote {args.csv_prefix}_folds.csv")

    if args.mode in ("leak_audit", "all"):
        print(f"\nRunning leak audit on {len(feature_cols)} features...")
        audit_df = leak_audit(df, feature_cols)
        print_leak_audit_report(audit_df)
        if args.csv_prefix:
            audit_df.to_csv(f"{args.csv_prefix}_leak_audit.csv", index=False)
            print(f"\nWrote {args.csv_prefix}_leak_audit.csv")

    if args.mode in ("ablation", "all"):
        print(f"\nRunning feature ablation "
              f"(this is the slow one — sample={args.sample})...")
        # Pick top features by variance for ablation (to keep runtime sensible)
        if args.top_features:
            variances = df[feature_cols].var().sort_values(ascending=False)
            ranked_features = variances.index.tolist()
        else:
            ranked_features = feature_cols
        ablation_df, baseline_auc = feature_ablation(
            df, ranked_features,
            n_folds=3,
            embargo=args.embargo,
            sample_size=args.sample,
            top_n_features=args.top_features,
        )
        print_ablation_report(ablation_df, baseline_auc)
        if args.csv_prefix:
            ablation_df.to_csv(f"{args.csv_prefix}_ablation.csv", index=False)
            print(f"\nWrote {args.csv_prefix}_ablation.csv")

    print("\n" + "=" * 72)
    print("DONE")
    print("=" * 72)


if __name__ == "__main__":
    main()
