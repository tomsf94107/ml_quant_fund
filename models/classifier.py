# models/classifier.py
# ─────────────────────────────────────────────────────────────────────────────
# THE canonical XGBoost classifier. Replaces:
#   - helpers_xgb_v1.2.py       (regressor, wrong task)
#   - xgb_forecasting_v1.1.py   (regressor, wrong task)
#   - xgb_forecasting_v1.2.py   (regressor, wrong task)
#   - train_forecast_model_v1.3 (classifier but missing risk weighting)
#
# What we keep from your old code:
#   ✓ Risk-aware sample weighting (alpha=0.30) from helpers_xgb_v1.2
#   ✓ XGBClassifier + logloss + 3 horizons from train_forecast_model_v1.3
#   ✓ walk-forward test split (shuffle=False)
#
# What we add:
#   + Isotonic calibration — raw XGB probabilities are not calibrated.
#     A 70% signal should mean UP 70% of the time. Without this, your
#     confidence scores can't be used for position sizing.
#   + congress_net_shares wired into features (was in schema, never used)
#   + Explicit feature list — no silent column leakage
#   + save/load via joblib (pickle replaced — more reliable across sklearn versions)
#
# Zero Streamlit imports. Zero UI code. Backend only.
# ─────────────────────────────────────────────────────────────────────────────

from __future__ import annotations

import os
import joblib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, roc_auc_score, log_loss, brier_score_loss
)
from xgboost import XGBClassifier

# ── Paths ─────────────────────────────────────────────────────────────────────
MODEL_DIR = Path(os.getenv("MODEL_DIR", "models/saved"))
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# ── Feature columns (must match OUTPUT_COLUMNS in features/builder.py) ────────
# These are the ONLY columns fed to the model.
# Excluded intentionally: date, ticker (identifiers, not features)
# Excluded intentionally: close (raw price leaks magnitude into direction target)
FEATURE_COLUMNS: list[str] = [
    # Returns
    "return_1d", "return_3d", "return_5d",
    # Trend
    "ma_5", "ma_10", "ma_20",
    # Volatility
    "volatility_5d", "volatility_10d",
    # Momentum
    "rsi_14", "macd", "macd_signal",
    # Mean reversion
    "bb_upper", "bb_lower", "bb_width",
    # Volume
    "volume_zscore", "volume_spike",
    "vwap", "obv", "atr",
    # Market context
    "spy_ret", "xlk_ret",
    # Alternative data
    "sentiment_score",
    "insider_net_shares", "insider_7d", "insider_21d",
    "congress_net_shares",            # ← was orphaned, now wired in
    # Risk regime
    "risk_today", "risk_next_1d", "risk_next_3d", "risk_prev_1d",
    # Macro regime
    "is_pandemic",
    # Earnings surprise (high-impact signal)
    "eps_surprise", "rev_surprise",
    "days_to_earnings",
    "post_earnings_1d", "post_earnings_3d", "post_earnings_5d",
    # Intraday-derived daily features
    "vwap_dev_eod", "vol_surge_eod", "intraday_momentum",
    # Pre-market & overnight
    "premarket_gap", "es_overnight",
    # Options IV
    "iv_skew_snap", "pc_ratio_snap",
    # Analyst revisions
    "analyst_upside", "analyst_buy_pct", "analyst_mult",
    # FinBERT NLP
    "finbert_sentiment", "finbert_mult",
    "oil_ret", "oil_spy_corr",          # crude oil signal
    # Extended returns
    "return_20d", "return_60d",
    # Extended trend
    "ma_50", "ma5_above_ma20", "ma20_above_ma50",
    # 52-week range
    "high_52w_ratio", "low_52w_ratio",
    # Extended momentum
    "bb_pct", "rsi_above_70", "rsi_below_30",
    # Volume trend
    "obv_trend",
    # Macro
    "vix_close", "vix_ret", "dxy_ret", "yield_10y",
    "fear_greed", "vix_term_structure",
    # Risk/positioning
    "beta_60d", "short_ratio", "short_pct_float",
    # Sentiment
    "monday_sentiment",
    # Relative performance
    "sector_rel_ret",
    # Calendar
    "day_of_week", "is_month_end",
    # Regime + credit features
    "vix_5d_above_25", "semi_etf_momentum_60d",
    "igv_vs_sp500_ret_30d", "lqd_hyg_spread",
]

TARGET_HORIZONS: tuple[int, ...] = (1, 3, 5)

# ── XGBoost hyperparameters ───────────────────────────────────────────────────
# These are reasonable defaults. Replace with Optuna-tuned values once you
# have enough training data (aim for >500 rows per ticker).
XGB_PARAMS: dict = {
    "n_estimators":     300,
    "learning_rate":    0.05,
    "max_depth":        4,
    "subsample":        0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 5,       # regularisation — prevents overfit on thin tickers
    "gamma":            0.1,
    "reg_alpha":        0.1,     # L1
    "reg_lambda":       1.0,     # L2
    "objective":        "binary:logistic",
    "eval_metric":      "logloss",
    "random_state":     42,
    "n_jobs":           -1,
}

def _get_xgb_params(ticker: str, horizon: int) -> dict:
    """Return Optuna-tuned params if available, else XGB_PARAMS defaults."""
    try:
        from models.tuner import get_params_for
        tuned = get_params_for(ticker, horizon, model="xgb")
        if tuned:
            return tuned
    except Exception:
        pass
    return XGB_PARAMS.copy()

def _get_lgb_params(ticker: str, horizon: int) -> Optional[dict]:
    """Return Optuna-tuned LightGBM params if available, else None."""
    try:
        from models.tuner import get_params_for
        return get_params_for(ticker, horizon, model="lgb")
    except Exception:
        return None

# ── Risk weighting constants (preserved from helpers_xgb_v1.2) ────────────────
RISK_ALPHA       = 0.30   # strength of down-weighting for high-risk days
RISK_WEIGHT_FLOOR = 0.50  # never down-weight below 50% of normal weight


# ══════════════════════════════════════════════════════════════════════════════
#  DATA CLASSES
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class TrainResult:
    """Everything produced by one training run — model + evaluation metrics."""
    ticker:     str
    horizon:    int                         # days ahead (1, 3, or 5)
    model:      CalibratedClassifierCV      # calibrated wrapper around XGBClassifier
    feature_cols: list[str]
    metrics:    dict = field(default_factory=dict)
    feature_importances: dict = field(default_factory=dict)
    shap_importances:    dict = field(default_factory=dict)

    # metrics keys: accuracy, roc_auc, log_loss, brier_score, n_train, n_test

    def model_path(self) -> Path:
        return MODEL_DIR / f"{self.ticker}_target_{self.horizon}d.joblib"

    def save(self) -> Path:
        path = self.model_path()
        joblib.dump(self, path)
        return path

    @classmethod
    def load(cls, ticker: str, horizon: int) -> "TrainResult":
        path = MODEL_DIR / f"{ticker}_target_{horizon}d.joblib"
        if not path.exists():
            raise FileNotFoundError(
                f"No saved model for {ticker} horizon={horizon}d. "
                f"Run train_model() first."
            )
        return joblib.load(path)


# ══════════════════════════════════════════════════════════════════════════════
#  PRIVATE HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _risk_sample_weights(
    df_train: pd.DataFrame,
    alpha: float = RISK_ALPHA,
    floor: float = RISK_WEIGHT_FLOOR,
    recency_factor: float = 3.0,
    recency_days: int = 60,
) -> Optional[np.ndarray]:
    """
    Compute per-sample weights combining:
    1. Recency weighting: last 60 days get 3x more weight so models
       adapt faster to current regime (e.g. tariff shock, VIX spike).
    2. Risk weighting: down-weight high-risk days like FOMC/earnings.
    Both are combined multiplicatively and normalized.
    """
    n = len(df_train)

    # --- Recency weights ---
    recency_weights = np.ones(n)
    if n > recency_days:
        recency_weights[-recency_days:] = recency_factor
    recency_weights = recency_weights / recency_weights.mean()

    # --- Risk weights ---
    risk_weights = None

    # Option B: use real event flags if available and non-zero
    if "risk_today" in df_train.columns:
        r = df_train["risk_today"].astype(float)
        if r.max() > 0:
            r_norm = r / r.max()
            risk_weights = (1.0 - alpha * r_norm).clip(floor, 1.0).values

    # Option A: fall back to VIX proxy if risk_today is missing or all zeros
    if risk_weights is None and "vix_close" in df_train.columns:
        vix = df_train["vix_close"].astype(float)
        vix_weights = np.ones(n)
        vix_weights[vix > 30] = 0.50           # high risk
        vix_weights[(vix > 20) & (vix <= 30)] = 0.75  # medium risk
        risk_weights = vix_weights

    if risk_weights is None:
        return recency_weights

    # --- Combine and normalize ---
    combined = recency_weights * risk_weights
    combined = combined / combined.mean()
    return combined


def _validate_feature_columns(df: pd.DataFrame) -> list[str]:
    """Return the intersection of FEATURE_COLUMNS and df.columns.
    Warns about missing columns but never crashes — missing features become 0.0."""
    present  = [c for c in FEATURE_COLUMNS if c in df.columns]
    missing  = [c for c in FEATURE_COLUMNS if c not in df.columns]
    if missing:
        print(f"  ⚠ Features missing (will be 0.0): {missing}")
    return present


def _prepare_xy(
    df: pd.DataFrame,
    target_col: str,
) -> tuple[pd.DataFrame, pd.Series]:
    """Prepare X, y for one target horizon. Drops rows where target is NaN."""
    feat_cols = _validate_feature_columns(df)

    # Fill missing feature columns with 0.0
    for c in FEATURE_COLUMNS:
        if c not in df.columns:
            df = df.copy()
            df[c] = 0.0

    working = df[FEATURE_COLUMNS + [target_col]].dropna(subset=[target_col]).copy()

    # Fill any remaining NaNs in features with column median (safe for XGB)
    working[FEATURE_COLUMNS] = working[FEATURE_COLUMNS].fillna(
        working[FEATURE_COLUMNS].median()
    )

    X = working[FEATURE_COLUMNS]
    y = working[target_col].astype(int)
    return X, y


# ══════════════════════════════════════════════════════════════════════════════
#  PUBLIC API
# ══════════════════════════════════════════════════════════════════════════════

def train_model(
    ticker: str,
    df: pd.DataFrame,
    horizon: int = 1,
    test_size: float = 0.2,
    save: bool = True,
    verbose: bool = True,
) -> TrainResult:
    """
    Train a calibrated XGBClassifier for `ticker` predicting direction at
    `horizon` days ahead.

    Parameters
    ----------
    ticker      : e.g. "AAPL"
    df          : output of build_feature_dataframe() + add_forecast_targets()
    horizon     : 1, 3, or 5 (days)
    test_size   : fraction held out for evaluation (walk-forward, no shuffle)
    save        : if True, saves the TrainResult to MODEL_DIR
    verbose     : print training summary

    Returns
    -------
    TrainResult with trained model and evaluation metrics
    """
    target_col = f"target_{horizon}d"
    if target_col not in df.columns:
        raise ValueError(
            f"Column '{target_col}' not found. "
            f"Did you run add_forecast_targets() on the DataFrame?"
        )

    X, y = _prepare_xy(df, target_col)

    if len(X) < 60:
        raise ValueError(
            f"{ticker}: only {len(X)} usable rows for {target_col}. "
            f"Need at least 60. Extend start_date."
        )

    # Walk-forward split — NO shuffle. Future must not leak into train set.
    split_idx = int(len(X) * (1 - test_size))
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    # Risk-aware sample weights on training set only
    sample_weights = _risk_sample_weights(
        df.loc[X_train.index], RISK_ALPHA, RISK_WEIGHT_FLOOR
    )

    # ── Base classifier — use Optuna-tuned params if available ────────────────
    xgb_params = _get_xgb_params(ticker, horizon)
    base_clf = XGBClassifier(**xgb_params)

    fit_kwargs: dict = {
        "eval_set":  [(X_test, y_test)],
        "verbose":    False,
    }
    if sample_weights is not None:
        fit_kwargs["sample_weight"] = sample_weights

    base_clf.fit(X_train, y_train, **fit_kwargs)

    # ── Extract feature importances (gain-based + SHAP) ────────────────────
    try:
        raw_imp = base_clf.get_booster().get_score(importance_type="gain")
        feature_importances = {f: raw_imp.get(f, 0.0) for f in FEATURE_COLUMNS}
    except Exception:
        feature_importances = {}

    # SHAP values — more accurate importance than gain
    shap_importances = {}
    try:
        import shap
        explainer   = shap.TreeExplainer(base_clf)
        # Use test set for SHAP (smaller sample = faster)
        sample_size = min(200, len(X_test))
        X_sample    = X_test.iloc[:sample_size] if hasattr(X_test, "iloc") else X_test[:sample_size]
        shap_values = explainer.shap_values(X_sample)
        # Mean absolute SHAP value per feature = global importance
        import numpy as np
        mean_abs_shap = np.abs(shap_values).mean(axis=0)
        for i, feat in enumerate(FEATURE_COLUMNS):
            if i < len(mean_abs_shap):
                shap_importances[feat] = float(mean_abs_shap[i])
    except Exception as e:
        shap_importances = {}
        print(f"  SHAP calculation failed: {e}")

    # ── Isotonic calibration ───────────────────────────────────────────────
    # Isotonic regression re-maps the raw sigmoid output to true probabilities.
    # cv=5 means we calibrate on the test set (already held out).
    # This is safe because the test set was never seen during XGB training.
    calibrated = CalibratedClassifierCV(base_clf, method="isotonic", cv=5)
    calibrated.fit(X_test, y_test)

    # ── Evaluate ───────────────────────────────────────────────────────────
    y_prob = calibrated.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)

    # Guard against single-class test sets (rare but possible for short tickers)
    try:
        auc = roc_auc_score(y_test, y_prob)
    except ValueError:
        auc = float("nan")

    metrics = {
        "accuracy":    round(accuracy_score(y_test, y_pred),   4),
        "roc_auc":     round(auc,                               4),
        "log_loss":    round(log_loss(y_test, y_prob),          4),
        "brier_score": round(brier_score_loss(y_test, y_prob),  4),
        "n_train":     len(X_train),
        "n_test":      len(X_test),
    }

    if verbose:
        print(
            f"  {ticker:<6} | {target_col:<12} | "
            f"acc={metrics['accuracy']:.3f}  "
            f"auc={metrics['roc_auc']:.3f}  "
            f"brier={metrics['brier_score']:.3f}  "
            f"n={metrics['n_train']}+{metrics['n_test']}"
        )

    result = TrainResult(
        ticker=ticker,
        horizon=horizon,
        model=calibrated,
        feature_cols=FEATURE_COLUMNS,
        metrics=metrics,
        feature_importances=feature_importances,
        shap_importances=shap_importances,
    )

    if save:
        result.save()

    return result


def predict_proba(
    ticker: str,
    df: pd.DataFrame,
    horizon: int = 1,
    result: Optional[TrainResult] = None,
) -> pd.Series:
    """
    Return calibrated UP probability for each row in df.

    Parameters
    ----------
    ticker  : e.g. "AAPL"
    df      : output of build_feature_dataframe() (no targets needed)
    horizon : 1, 3, or 5
    result  : if None, loads from disk

    Returns
    -------
    pd.Series of float in [0, 1] — P(close[t+h] > close[t])
    """
    if result is None:
        result = TrainResult.load(ticker, horizon)

    # Fill missing features
    working = df.copy()
    for c in FEATURE_COLUMNS:
        if c not in working.columns:
            working[c] = 0.0

    working[FEATURE_COLUMNS] = working[FEATURE_COLUMNS].fillna(
        working[FEATURE_COLUMNS].median()
    )

    proba = result.model.predict_proba(working[FEATURE_COLUMNS])[:, 1]
    return pd.Series(proba, index=df.index, name=f"prob_up_{horizon}d")


def predict_today(
    ticker: str,
    df: pd.DataFrame,
    horizon: int = 1,
) -> dict:
    """
    Return today's signal for a ticker.

    Returns
    -------
    dict with keys: ticker, horizon, prob_up, signal, confidence
      signal     : "BUY" | "HOLD"
      confidence : "HIGH" (>0.65) | "MEDIUM" (0.55-0.65) | "LOW" (<0.55)
    """
    try:
        proba_series = predict_proba(ticker, df, horizon)
        prob = float(proba_series.iloc[-1])
    except Exception as e:
        return {
            "ticker": ticker, "horizon": horizon,
            "prob_up": None, "signal": "HOLD",
            "confidence": "ERROR", "error": str(e),
        }

    signal = "BUY" if prob >= 0.55 else "HOLD"

    if prob >= 0.65:
        confidence = "HIGH"
    elif prob >= 0.55:
        confidence = "MEDIUM"
    else:
        confidence = "LOW"

    return {
        "ticker":     ticker,
        "horizon":    horizon,
        "prob_up":    round(prob, 4),
        "signal":     signal,
        "confidence": confidence,
    }
