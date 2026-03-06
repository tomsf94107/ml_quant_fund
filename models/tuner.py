# models/tuner.py
# ─────────────────────────────────────────────────────────────────────────────
# Optuna hyperparameter tuner for the ML Quant Fund XGBoost + LightGBM models.
#
# Usage:
#   python -m models.tuner                         # tune all tickers
#   python -m models.tuner --tickers AAPL NVDA     # tune specific tickers
#   python -m models.tuner --trials 50             # more trials = better params
#   python -m models.tuner --horizon 1             # tune only 1d horizon
#   python -m models.tuner --retune                # re-tune already-tuned tickers
#
# Install: pip install optuna lightgbm
# ─────────────────────────────────────────────────────────────────────────────

from __future__ import annotations

import argparse
import json
import time
import warnings
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

MODEL_DIR   = Path("models/saved")
PARAMS_FILE = MODEL_DIR / "best_params.json"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_TRIALS  = 40
DEFAULT_TIMEOUT = 120
CV_SPLITS       = 5


def load_best_params() -> dict:
    if PARAMS_FILE.exists():
        with open(PARAMS_FILE) as f:
            return json.load(f)
    return {}


def save_best_params(params: dict):
    params["_updated"] = datetime.utcnow().isoformat()
    with open(PARAMS_FILE, "w") as f:
        json.dump(params, f, indent=2)


def get_params_for(ticker: str, horizon: int, model: str = "xgb") -> Optional[dict]:
    """Return best tuned params or None if not yet tuned."""
    all_params = load_best_params()
    key   = f"{ticker}_{horizon}d"
    entry = all_params.get(key, {})
    return entry.get(f"{model}_params", None)


def tune_ticker_horizon(
    ticker:   str,
    df:       pd.DataFrame,
    horizon:  int,
    n_trials: int = DEFAULT_TRIALS,
    timeout:  int = DEFAULT_TIMEOUT,
    verbose:  bool = True,
) -> dict:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    from sklearn.model_selection import TimeSeriesSplit
    from sklearn.metrics import roc_auc_score
    from xgboost import XGBClassifier
    from features.builder import add_forecast_targets
    from models.classifier import FEATURE_COLUMNS

    target_col = f"target_{horizon}d"
    df_h = add_forecast_targets(df.copy(), horizons=(horizon,))
    df_h = df_h.dropna(subset=[target_col] + [c for c in FEATURE_COLUMNS if c in df_h.columns])

    feat_cols = [c for c in FEATURE_COLUMNS if c in df_h.columns]
    X = df_h[feat_cols].values.astype(np.float32)
    y = df_h[target_col].values.astype(int)

    if len(X) < 150:
        if verbose:
            print(f"  ⚠ {ticker} h={horizon}d: only {len(X)} rows, skipping")
        return {}

    tscv = TimeSeriesSplit(n_splits=CV_SPLITS)
    result = {}

    # ── XGBoost ───────────────────────────────────────────────────────────────
    def xgb_obj(trial):
        p = {
            "n_estimators":     trial.suggest_int("n_estimators", 100, 600),
            "learning_rate":    trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
            "max_depth":        trial.suggest_int("max_depth", 3, 7),
            "subsample":        trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
            "gamma":            trial.suggest_float("gamma", 0.0, 1.0),
            "reg_alpha":        trial.suggest_float("reg_alpha", 0.0, 2.0),
            "reg_lambda":       trial.suggest_float("reg_lambda", 0.5, 5.0),
            "objective": "binary:logistic", "eval_metric": "logloss",
            "use_label_encoder": False, "random_state": 42, "n_jobs": -1, "verbosity": 0,
        }
        aucs = []
        for tr, val in tscv.split(X):
            clf = XGBClassifier(**p)
            clf.fit(X[tr], y[tr], eval_set=[(X[val], y[val])], verbose=False)
            prob = clf.predict_proba(X[val])[:, 1]
            if len(np.unique(y[val])) == 2:
                aucs.append(roc_auc_score(y[val], prob))
        return np.mean(aucs) if aucs else 0.5

    if verbose:
        print(f"  XGB ({n_trials} trials)...", end=" ", flush=True)
    t0 = time.time()
    study = optuna.create_study(direction="maximize",
                                 sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(xgb_obj, n_trials=n_trials, timeout=timeout, show_progress_bar=False)
    best_xgb = study.best_params
    best_xgb.update({"objective": "binary:logistic", "eval_metric": "logloss",
                      "use_label_encoder": False, "random_state": 42, "n_jobs": -1})
    result["xgb_params"]   = best_xgb
    result["best_xgb_auc"] = round(study.best_value, 4)
    if verbose:
        print(f"AUC={study.best_value:.4f} ({time.time()-t0:.0f}s)")

    # ── LightGBM ──────────────────────────────────────────────────────────────
    try:
        import lightgbm as lgb  # noqa: F401

        def lgb_obj(trial):
            p = {
                "n_estimators":      trial.suggest_int("n_estimators", 100, 600),
                "learning_rate":     trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
                "max_depth":         trial.suggest_int("max_depth", 3, 8),
                "num_leaves":        trial.suggest_int("num_leaves", 15, 127),
                "subsample":         trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree":  trial.suggest_float("colsample_bytree", 0.5, 1.0),
                "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
                "reg_alpha":         trial.suggest_float("reg_alpha", 0.0, 2.0),
                "reg_lambda":        trial.suggest_float("reg_lambda", 0.0, 5.0),
                "objective": "binary", "metric": "auc",
                "random_state": 42, "n_jobs": -1, "verbose": -1,
            }
            aucs = []
            for tr, val in tscv.split(X):
                clf = lgb.LGBMClassifier(**p)
                clf.fit(X[tr], y[tr], eval_set=[(X[val], y[val])])
                prob = clf.predict_proba(X[val])[:, 1]
                if len(np.unique(y[val])) == 2:
                    aucs.append(roc_auc_score(y[val], prob))
            return np.mean(aucs) if aucs else 0.5

        if verbose:
            print(f"  LGB ({n_trials} trials)...", end=" ", flush=True)
        t0 = time.time()
        study2 = optuna.create_study(direction="maximize",
                                      sampler=optuna.samplers.TPESampler(seed=42))
        study2.optimize(lgb_obj, n_trials=n_trials, timeout=timeout, show_progress_bar=False)
        best_lgb = study2.best_params
        best_lgb.update({"objective": "binary", "metric": "auc",
                          "random_state": 42, "n_jobs": -1, "verbose": -1})
        result["lgb_params"]   = best_lgb
        result["best_lgb_auc"] = round(study2.best_value, 4)
        if verbose:
            print(f"AUC={study2.best_value:.4f} ({time.time()-t0:.0f}s)")
    except ImportError:
        if verbose:
            print("  LightGBM not installed, skipping")

    return result


def tune_all(
    tickers:       list[str],
    horizons:      tuple[int, ...] = (1, 3, 5),
    n_trials:      int = DEFAULT_TRIALS,
    timeout:       int = DEFAULT_TIMEOUT,
    skip_existing: bool = True,
):
    from features.builder import build_feature_dataframe

    all_params = load_best_params()
    total = len(tickers) * len(horizons)
    done  = 0

    print(f"\n{'═'*60}")
    print(f"  Optuna Tuning — {len(tickers)} tickers × {len(horizons)} horizons")
    print(f"  Trials: {n_trials} per combo  |  TimeSeriesSplit CV={CV_SPLITS}")
    print(f"{'═'*60}")

    for ticker in tickers:
        print(f"\n{'─'*60}\n  {ticker}\n{'─'*60}")
        try:
            df = build_feature_dataframe(ticker, start_date="2018-01-01",
                                          include_sentiment=False)
        except Exception as e:
            print(f"  ✗ Feature build failed: {e}")
            continue

        for h in horizons:
            key = f"{ticker}_{h}d"
            if skip_existing and key in all_params:
                xgb_auc = all_params[key].get("best_xgb_auc", "?")
                print(f"  ↷ {ticker} h={h}d already tuned (XGB={xgb_auc}), skipping")
                continue

            try:
                res = tune_ticker_horizon(ticker, df, h,
                                           n_trials=n_trials, timeout=timeout)
                if res:
                    all_params[key] = res
                    save_best_params(all_params)
                    print(f"  ✓ saved  XGB={res.get('best_xgb_auc','?')}  "
                          f"LGB={res.get('best_lgb_auc','?')}")
            except Exception as e:
                print(f"  ✗ failed: {e}")
            done += 1

    print(f"\n{'═'*60}")
    print(f"  DONE — {done}/{total} combos tuned")
    print(f"  Saved → {PARAMS_FILE}")
    print(f"{'═'*60}\n")

    rows = [{"key": k,
             "XGB AUC": v.get("best_xgb_auc", "—"),
             "LGB AUC": v.get("best_lgb_auc", "—")}
            for k, v in all_params.items() if not k.startswith("_")]
    if rows:
        print(pd.DataFrame(rows).sort_values("XGB AUC", ascending=False).to_string(index=False))


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))

    parser = argparse.ArgumentParser()
    parser.add_argument("--tickers", nargs="+")
    parser.add_argument("--horizon", type=int)
    parser.add_argument("--trials",  type=int, default=DEFAULT_TRIALS)
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT)
    parser.add_argument("--retune",  action="store_true")
    args = parser.parse_args()

    if args.tickers:
        tickers = [t.upper() for t in args.tickers]
    else:
        p = Path("tickers.txt")
        tickers = [t.strip().upper() for t in p.read_text().splitlines() if t.strip()] \
                  if p.exists() else ["AAPL", "NVDA", "TSLA", "AMD"]

    horizons = (args.horizon,) if args.horizon else (1, 3, 5)
    tune_all(tickers, horizons, args.trials, args.timeout, not args.retune)
