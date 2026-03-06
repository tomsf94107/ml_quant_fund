# models/tuner.py
# ─────────────────────────────────────────────────────────────────────────────
# Optuna hyperparameter tuning for XGBoost classifiers.
# Finds optimal params per ticker — replaces hardcoded XGB_PARAMS defaults.
#
# Run once after you have enough data:
#   python -m models.tuner --tickers AAPL NVDA TSLA --horizon 1
#   python -m models.tuner --all --horizon 1   # tune all tickers
#
# Results saved to models/saved/tuned_params.json
# train_all.py automatically uses tuned params if available.
# ─────────────────────────────────────────────────────────────────────────────

from __future__ import annotations

import json
import os
import warnings
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

MODEL_DIR   = Path(os.getenv("MODEL_DIR", "models/saved"))
PARAMS_FILE = MODEL_DIR / "tuned_params.json"
N_TRIALS    = 50    # 50 trials per ticker/horizon — good balance of speed vs quality


# ══════════════════════════════════════════════════════════════════════════════
#  OPTUNA OBJECTIVE
# ══════════════════════════════════════════════════════════════════════════════

def _objective(trial, X_train, y_train, X_val, y_val, sample_weights=None):
    """Optuna objective — maximize ROC-AUC on validation set."""
    from xgboost import XGBClassifier
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.metrics import roc_auc_score

    params = {
        "n_estimators":     trial.suggest_int("n_estimators", 100, 500),
        "learning_rate":    trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "max_depth":        trial.suggest_int("max_depth", 3, 7),
        "subsample":        trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "gamma":            trial.suggest_float("gamma", 0.0, 1.0),
        "reg_alpha":        trial.suggest_float("reg_alpha", 0.0, 2.0),
        "reg_lambda":       trial.suggest_float("reg_lambda", 0.0, 2.0),
        "objective":        "binary:logistic",
        "eval_metric":      "logloss",
        "use_label_encoder": False,
        "random_state":     42,
        "n_jobs":           -1,
    }

    try:
        clf = XGBClassifier(**params)
        fit_kwargs = {"eval_set": [(X_val, y_val)], "verbose": False}
        if sample_weights is not None:
            fit_kwargs["sample_weight"] = sample_weights
        clf.fit(X_train, y_train, **fit_kwargs)

        cal = CalibratedClassifierCV(clf, method="isotonic", cv="prefit")
        cal.fit(X_val, y_val)

        y_prob = cal.predict_proba(X_val)[:, 1]
        return roc_auc_score(y_val, y_prob)
    except Exception:
        return 0.5


# ══════════════════════════════════════════════════════════════════════════════
#  TUNE ONE TICKER
# ══════════════════════════════════════════════════════════════════════════════

def tune_ticker(
    ticker:   str,
    df:       pd.DataFrame,
    horizon:  int = 1,
    n_trials: int = N_TRIALS,
    verbose:  bool = True,
) -> dict:
    """
    Run Optuna on one ticker/horizon. Returns best params dict.
    """
    try:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
    except ImportError:
        print("  ⚠ Optuna not installed. Run: pip install optuna")
        return {}

    from models.classifier import FEATURE_COLUMNS, _prepare_xy, _risk_sample_weights

    target_col = f"target_{horizon}d"
    if target_col not in df.columns:
        print(f"  ⚠ {ticker}: no {target_col} column")
        return {}

    # Fill missing features
    working = df.copy()
    for c in FEATURE_COLUMNS:
        if c not in working.columns:
            working[c] = 0.0

    X, y = _prepare_xy(working, target_col)
    if len(X) < 100:
        print(f"  ⚠ {ticker}: only {len(X)} rows — skipping tune")
        return {}

    split = int(len(X) * 0.8)
    X_train, X_val = X.iloc[:split], X.iloc[split:]
    y_train, y_val = y.iloc[:split], y.iloc[split:]

    sw = _risk_sample_weights(working.loc[X_train.index])

    if verbose:
        print(f"  {ticker} horizon={horizon}d — running {n_trials} trials...")

    study = optuna.create_study(direction="maximize",
                                 sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(
        lambda t: _objective(t, X_train, y_train, X_val, y_val, sw),
        n_trials=n_trials,
        show_progress_bar=False,
    )

    best = study.best_params
    best_auc = study.best_value

    # Add fixed params
    best["objective"]         = "binary:logistic"
    best["eval_metric"]       = "logloss"
    best["use_label_encoder"] = False
    best["random_state"]      = 42
    best["n_jobs"]            = -1

    if verbose:
        print(f"    ✓ Best AUC={best_auc:.4f}  depth={best['max_depth']}  "
              f"lr={best['learning_rate']:.4f}  n={best['n_estimators']}")

    return best


# ══════════════════════════════════════════════════════════════════════════════
#  SAVE / LOAD TUNED PARAMS
# ══════════════════════════════════════════════════════════════════════════════

def save_tuned_params(ticker: str, horizon: int, params: dict,
                       path: Path = PARAMS_FILE) -> None:
    existing = load_all_tuned_params(path)
    key = f"{ticker}_{horizon}d"
    existing[key] = {"params": params, "tuned_at": datetime.utcnow().isoformat()}
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(existing, f, indent=2)


def load_all_tuned_params(path: Path = PARAMS_FILE) -> dict:
    if not path.exists():
        return {}
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return {}


def get_params_for(ticker: str, horizon: int,
                    path: Path = PARAMS_FILE) -> Optional[dict]:
    """Return tuned params for ticker/horizon, or None if not tuned yet."""
    data = load_all_tuned_params(path)
    key  = f"{ticker}_{horizon}d"
    entry = data.get(key)
    return entry["params"] if entry else None


# ══════════════════════════════════════════════════════════════════════════════
#  BATCH TUNER
# ══════════════════════════════════════════════════════════════════════════════

def tune_all(
    tickers:  list[str],
    horizons: list[int] = [1, 3, 5],
    n_trials: int = N_TRIALS,
    verbose:  bool = True,
) -> dict:
    """Tune all tickers and save params. Returns summary dict."""
    from features.builder import build_feature_dataframe, add_forecast_targets

    results = {}

    for ticker in tickers:
        try:
            df = build_feature_dataframe(ticker)
            df = add_forecast_targets(df)
        except Exception as e:
            print(f"  ⚠ {ticker}: feature build failed — {e}")
            continue

        for horizon in horizons:
            key = f"{ticker}_{horizon}d"
            params = tune_ticker(ticker, df, horizon=horizon,
                                  n_trials=n_trials, verbose=verbose)
            if params:
                save_tuned_params(ticker, horizon, params)
                results[key] = params

    return results


# ══════════════════════════════════════════════════════════════════════════════
#  CLI
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse
    from models.train_all import DEFAULT_TICKERS

    parser = argparse.ArgumentParser(description="Optuna hyperparameter tuning")
    parser.add_argument("--tickers",  nargs="+", default=None)
    parser.add_argument("--all",      action="store_true",
                        help="Tune all tickers in DEFAULT_TICKERS")
    parser.add_argument("--horizons", nargs="+", type=int, default=[1, 3, 5])
    parser.add_argument("--trials",   type=int, default=N_TRIALS)
    args = parser.parse_args()

    tickers = DEFAULT_TICKERS if args.all else (args.tickers or DEFAULT_TICKERS[:5])

    print(f"\nOptuna tuning: {len(tickers)} tickers × {args.horizons} horizons")
    print(f"Trials per model: {args.trials}")
    print(f"Estimated time: ~{len(tickers) * len(args.horizons) * args.trials * 0.3 / 60:.0f} minutes\n")

    results = tune_all(tickers, horizons=args.horizons, n_trials=args.trials)
    print(f"\nDone. {len(results)} models tuned → saved to {PARAMS_FILE}")
