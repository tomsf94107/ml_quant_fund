# models/regime_models.py
# ─────────────────────────────────────────────────────────────────────────────
# Regime-specific XGBoost classifiers.
# Trains a separate model for each market regime (BULL/BEAR/VOLATILE/NEUTRAL).
#
# Why this works better than one model + multiplier:
#   - BULL regime: momentum patterns dominate (buy strength, trend following)
#   - BEAR regime: mean reversion + defensive patterns dominate
#   - VOLATILE regime: short-term reversal patterns dominate
#   - Each model learns the right patterns for its environment
#
# Usage:
#   python -m models.regime_models --tickers AAPL NVDA --horizon 1
#
# At inference time:
#   from models.regime_models import predict_proba_regime
#   prob = predict_proba_regime("AAPL", features_row, horizon=1)
# ─────────────────────────────────────────────────────────────────────────────

from __future__ import annotations

import argparse
import os
import time
import warnings
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score, accuracy_score
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

MODEL_DIR  = Path(os.getenv("MODEL_DIR", "models/saved"))
MODEL_DIR.mkdir(parents=True, exist_ok=True)

REGIMES    = ["BULL", "BEAR", "VOLATILE", "NEUTRAL"]
MIN_ROWS   = 80    # minimum rows per regime to train a model

# ── XGBoost params per regime ─────────────────────────────────────────────────
# Each regime has different optimal hyperparams based on its characteristics
REGIME_XGB_PARAMS = {
    "BULL": {
        # Bull: longer trees, capture momentum trends
        "n_estimators": 300, "learning_rate": 0.05, "max_depth": 5,
        "subsample": 0.85, "colsample_bytree": 0.85,
        "min_child_weight": 3, "gamma": 0.05,
        "reg_alpha": 0.05, "reg_lambda": 1.0,
    },
    "BEAR": {
        # Bear: shallower trees, avoid overfitting on scarce data
        "n_estimators": 200, "learning_rate": 0.05, "max_depth": 3,
        "subsample": 0.8, "colsample_bytree": 0.8,
        "min_child_weight": 5, "gamma": 0.2,
        "reg_alpha": 0.2, "reg_lambda": 2.0,
    },
    "VOLATILE": {
        # Volatile: heavy regularization, mean reversion focus
        "n_estimators": 200, "learning_rate": 0.04, "max_depth": 3,
        "subsample": 0.75, "colsample_bytree": 0.75,
        "min_child_weight": 8, "gamma": 0.3,
        "reg_alpha": 0.3, "reg_lambda": 2.5,
    },
    "NEUTRAL": {
        # Neutral: balanced defaults
        "n_estimators": 250, "learning_rate": 0.05, "max_depth": 4,
        "subsample": 0.8, "colsample_bytree": 0.8,
        "min_child_weight": 5, "gamma": 0.1,
        "reg_alpha": 0.1, "reg_lambda": 1.0,
    },
}
# Add common params to all
for _r in REGIMES:
    REGIME_XGB_PARAMS[_r].update({
        "objective": "binary:logistic", "eval_metric": "logloss",
        "random_state": 42, "n_jobs": -1,
    })


# ── Model path ────────────────────────────────────────────────────────────────
def regime_model_path(ticker: str, regime: str, horizon: int) -> Path:
    return MODEL_DIR / f"{ticker}_regime_{regime.lower()}_{horizon}d.joblib"


# ══════════════════════════════════════════════════════════════════════════════
#  LABEL HISTORICAL DATA WITH REGIMES
# ══════════════════════════════════════════════════════════════════════════════

def label_regimes(df: pd.DataFrame) -> pd.Series:
    """
    Label each row of a feature DataFrame with its market regime.
    Uses SPY/VIX rolling window approach from regime_classifier.

    Returns a Series of regime labels aligned to df.index.
    """
    import yfinance as yf
    from models.regime_classifier import _compute_regime_features, _classify_regime

    # Get date range from df
    dates = pd.to_datetime(df.index if df.index.name == "date"
                           else df.get("date", df.index))
    start = (dates.min() - pd.Timedelta(days=90)).strftime("%Y-%m-%d")
    end   = dates.max().strftime("%Y-%m-%d")

    # Download SPY + VIX
    try:
        raw = yf.download(["SPY", "^VIX", "TLT"], start=start, end=end,
                           auto_adjust=True, progress=False)
        if isinstance(raw.columns, pd.MultiIndex):
            close = raw["Close"].copy()
        else:
            return pd.Series("NEUTRAL", index=df.index)

        close.columns = [c.replace("^", "") for c in close.columns]
        close = close.dropna(how="all")
        close.index = pd.to_datetime(close.index).date
    except Exception:
        return pd.Series("NEUTRAL", index=df.index)

    # Label each date by computing regime from trailing 60-day window
    labels = {}
    close_dates = list(close.index)

    for i, d in enumerate(close_dates):
        if i < 60:
            labels[d] = "NEUTRAL"
            continue
        window = close.iloc[max(0, i-120):i+1]
        try:
            feats  = _compute_regime_features(window)
            regime = _classify_regime(feats)
            labels[d] = regime.label
        except Exception:
            labels[d] = "NEUTRAL"

    # Align to df dates
    df_dates = pd.to_datetime(
        df["date"] if "date" in df.columns else df.index
    ).dt.date if hasattr(pd.to_datetime(df.index if "date" not in df.columns
                                         else df["date"]).iloc[0], 'date') \
        else pd.to_datetime(
        df["date"] if "date" in df.columns else df.index
    )

    result = pd.Series(index=df.index, dtype=str)
    for idx in df.index:
        try:
            d = pd.to_datetime(df.loc[idx, "date"] if "date" in df.columns
                               else idx).date()
            result[idx] = labels.get(d, "NEUTRAL")
        except Exception:
            result[idx] = "NEUTRAL"

    return result.fillna("NEUTRAL")


# ══════════════════════════════════════════════════════════════════════════════
#  TRAIN REGIME-SPECIFIC MODELS
# ══════════════════════════════════════════════════════════════════════════════

def train_regime_models(
    ticker:   str,
    df:       pd.DataFrame,
    horizon:  int = 1,
    verbose:  bool = True,
) -> dict[str, dict]:
    """
    Train a separate XGBoost model for each regime.

    Parameters
    ----------
    ticker  : e.g. "AAPL"
    df      : output of build_feature_dataframe() + add_forecast_targets()
    horizon : 1, 3, or 5

    Returns
    -------
    dict mapping regime → {model, metrics, n_rows}
    """
    from models.classifier import FEATURE_COLUMNS
    from features.builder import add_forecast_targets

    if f"target_{horizon}d" not in df.columns:
        from features.builder import add_forecast_targets as aft
        df = aft(df, horizons=(horizon,))

    target_col = f"target_{horizon}d"
    feat_cols  = [c for c in FEATURE_COLUMNS if c in df.columns]

    # Label regimes
    if verbose:
        print(f"  [{ticker}] Labeling regimes...", end=" ", flush=True)
    t0 = time.time()
    regime_labels = label_regimes(df)
    df = df.copy()
    df["_regime"] = regime_labels.values

    if verbose:
        counts = df["_regime"].value_counts().to_dict()
        print(f"{time.time()-t0:.0f}s  {counts}")

    results = {}

    for regime in REGIMES:
        regime_df = df[df["_regime"] == regime].dropna(
            subset=[target_col] + feat_cols
        )
        n = len(regime_df)

        if n < MIN_ROWS:
            if verbose:
                print(f"  [{ticker}] {regime}: only {n} rows — skipping")
            continue

        X = regime_df[feat_cols].values.astype(np.float32)
        y = regime_df[target_col].values.astype(int)

        # Walk-forward split (preserve time order)
        split = int(len(X) * 0.80)
        X_tr, X_te = X[:split], X[split:]
        y_tr, y_te = y[:split], y[split:]

        if len(X_te) < 10 or len(np.unique(y_te)) < 2:
            # Not enough test data — train on all, no eval
            X_tr, y_tr = X, y
            X_te, y_te = X[-20:], y[-20:]

        try:
            params  = REGIME_XGB_PARAMS[regime]
            base    = XGBClassifier(**params)
            base.fit(X_tr, y_tr,
                     eval_set=[(X_te, y_te)], verbose=False)

            cal = CalibratedClassifierCV(base, method="isotonic", cv=5)
            cal.fit(X_tr, y_tr)

            prob_te = cal.predict_proba(X_te)[:, 1]
            auc = roc_auc_score(y_te, prob_te) if len(np.unique(y_te)) > 1 else 0.5
            acc = accuracy_score(y_te, (prob_te >= 0.5).astype(int))

            result = {
                "model":       cal,
                "feature_cols": feat_cols,
                "metrics":     {"auc": round(auc, 4), "acc": round(acc, 4),
                                 "n_train": len(X_tr), "n_test": len(X_te)},
                "regime":      regime,
                "ticker":      ticker,
                "horizon":     horizon,
            }

            # Save
            path = regime_model_path(ticker, regime, horizon)
            joblib.dump(result, path)
            results[regime] = result

            if verbose:
                print(f"  [{ticker}] {regime:8s} h={horizon}d  "
                      f"n={n:4d}  auc={auc:.3f}  acc={acc:.3f}")

        except Exception as e:
            if verbose:
                print(f"  [{ticker}] {regime}: FAILED — {e}")

    return results


def load_regime_model(
    ticker: str, regime: str, horizon: int
) -> Optional[dict]:
    """Load a saved regime model. Returns None if not found."""
    path = regime_model_path(ticker, regime, horizon)
    return joblib.load(path) if path.exists() else None


# ══════════════════════════════════════════════════════════════════════════════
#  INFERENCE — regime-aware prediction
# ══════════════════════════════════════════════════════════════════════════════

def predict_proba_regime(
    ticker:       str,
    features_row: pd.Series,
    horizon:      int = 1,
    current_regime: Optional[str] = None,
    fallback_prob:  float = 0.5,
) -> tuple[float, str]:
    """
    Predict probability using the regime-specific model.

    Parameters
    ----------
    ticker         : e.g. "AAPL"
    features_row   : single row of feature DataFrame (pd.Series)
    horizon        : 1, 3, or 5
    current_regime : pass directly to avoid re-fetching
    fallback_prob  : return this if regime model not available

    Returns
    -------
    (probability, regime_used)
    """
    # Get current regime
    if current_regime is None:
        try:
            from models.regime_classifier import get_current_regime
            current_regime = get_current_regime(use_cache=True).label
        except Exception:
            current_regime = "NEUTRAL"

    # Try regime-specific model first
    model_data = load_regime_model(ticker, current_regime, horizon)

    # Fallback chain: try NEUTRAL, then return fallback_prob
    if model_data is None:
        model_data = load_regime_model(ticker, "NEUTRAL", horizon)
    if model_data is None:
        return fallback_prob, current_regime

    model     = model_data["model"]
    feat_cols = model_data["feature_cols"]

    X = np.array([[features_row.get(c, 0.0) for c in feat_cols]],
                  dtype=np.float32)
    prob = float(model.predict_proba(X)[0, 1])
    return round(prob, 4), current_regime


def predict_proba_regime_series(
    ticker:         str,
    df:             pd.DataFrame,
    horizon:        int = 1,
) -> pd.Series:
    """
    Generate regime-aware probability series for a full DataFrame.
    Uses each row's historical regime label for backtesting.
    """
    from models.classifier import FEATURE_COLUMNS

    feat_cols    = [c for c in FEATURE_COLUMNS if c in df.columns]
    regime_labels = label_regimes(df)
    probs        = pd.Series(index=df.index, dtype=float)

    # Cache loaded models
    loaded = {}

    for idx in df.index:
        regime = regime_labels.get(idx, "NEUTRAL")
        key    = (regime, horizon)

        if key not in loaded:
            m = load_regime_model(ticker, regime, horizon)
            if m is None:
                m = load_regime_model(ticker, "NEUTRAL", horizon)
            loaded[key] = m

        model_data = loaded[key]
        if model_data is None:
            probs[idx] = 0.5
            continue

        row = df.loc[idx]
        X   = np.array([[row.get(c, 0.0) for c in model_data["feature_cols"]]],
                        dtype=np.float32)
        try:
            probs[idx] = float(model_data["model"].predict_proba(X)[0, 1])
        except Exception:
            probs[idx] = 0.5

    return probs


# ══════════════════════════════════════════════════════════════════════════════
#  BATCH TRAINING CLI
# ══════════════════════════════════════════════════════════════════════════════

def train_all_regime_models(
    tickers:  list[str],
    horizons: tuple[int, ...] = (1, 3, 5),
    verbose:  bool = True,
):
    from features.builder import build_feature_dataframe, add_forecast_targets

    TRAIN_START = "2018-01-01"
    total_ok = 0

    print(f"\n{'═'*60}")
    print(f"  Regime-Specific Model Training")
    print(f"  Tickers : {len(tickers)}")
    print(f"  Regimes : {REGIMES}")
    print(f"  Horizons: {horizons}")
    print(f"{'═'*60}")

    for ticker in tickers:
        print(f"\n{'─'*60}\n  {ticker}\n{'─'*60}")
        try:
            df = build_feature_dataframe(ticker, start_date=TRAIN_START)
            df = add_forecast_targets(df, horizons=horizons)
        except Exception as e:
            print(f"  ✗ Feature build failed: {e}")
            continue

        for h in horizons:
            results = train_regime_models(ticker, df, horizon=h, verbose=verbose)
            total_ok += len(results)

    print(f"\n{'═'*60}")
    print(f"  DONE — {total_ok} regime models trained")
    print(f"  Saved → {MODEL_DIR}")
    print(f"{'═'*60}\n")


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))

    parser = argparse.ArgumentParser()
    parser.add_argument("--tickers",  nargs="+")
    parser.add_argument("--horizon",  type=int)
    parser.add_argument("--horizons", nargs="+", type=int, default=[1, 3, 5])
    args = parser.parse_args()

    if args.tickers:
        tickers = [t.upper() for t in args.tickers]
    else:
        p = Path("tickers.txt")
        tickers = [t.strip().upper() for t in p.read_text().splitlines()
                   if t.strip()] if p.exists() else ["AAPL", "NVDA", "TSLA"]

    horizons = (args.horizon,) if args.horizon else tuple(args.horizons)
    train_all_regime_models(tickers, horizons)
