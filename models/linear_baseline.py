"""
Linear baseline — Alpha Hunt Queue #1 (Jun 1 2026).

Answers ONE question: is the per-ticker direction model's ~0.50 test AUC from the
TREE OVERFITTING, or from the SIGNAL being genuinely absent at this target? Linear
(L2-logistic) cannot memorize noise like XGBoost, so:
  - linear test AUC > 0.52 with SMALL train-test gap -> tree was overfitting.
  - linear test AUC also ~0.50                       -> signal absent at this target;
    overfit ruled out -> move to Hunt Queue #2 (pairs/cointegration).

Reuses the EXACT proven harness (same features, same purged expanding folds,
embargo=horizon, same per-ticker direction target). ONLY the estimator changes:
XGBClassifier -> StandardScaler + LogisticRegression(L2). Reports TRAIN and TEST
auc per fold so the gap is visible (XGB was train 0.66-0.79 / test ~0.50).

NOT expected to resurrect the dead 1-5d target — only localizes WHY, cheaply.

    python -m models.linear_baseline --tickers AAPL,MU,NVDA,AMD,MSFT --horizon 5
    python -m models.linear_baseline --all --horizon 5
"""
from __future__ import annotations
import argparse
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from features.builder import build_feature_dataframe, add_forecast_targets
from models.classifier import FEATURE_COLUMNS, _prepare_xy
from models.walk_forward import _make_folds


def _fit_eval_linear(X, y, train_end, test_start, test_end, horizon, C):
    X_tr, y_tr = X.iloc[:train_end], y.iloc[:train_end]
    X_te, y_te = X.iloc[test_start:test_end], y.iloc[test_start:test_end]
    if y_tr.nunique() < 2 or y_te.nunique() < 2:
        return None
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_te_s = scaler.transform(X_te)
    clf = LogisticRegression(penalty="l2", C=C, max_iter=1000)
    clf.fit(X_tr_s, y_tr)
    p_tr = clf.predict_proba(X_tr_s)[:, 1]
    p_te = clf.predict_proba(X_te_s)[:, 1]
    try:
        tr_auc = roc_auc_score(y_tr, p_tr)
        te_auc = roc_auc_score(y_te, p_te)
    except ValueError:
        return None
    return {"train_auc": tr_auc, "test_auc": te_auc, "n_test": len(y_te)}


def baseline_eval(ticker, df, horizon=5, C=0.1,
                  min_train=504, test_window=63, step=63, verbose=True):
    target_col = f"target_{horizon}d"
    if target_col not in df.columns:
        raise ValueError(f"{target_col} missing — run add_forecast_targets().")
    X, y = _prepare_xy(df, target_col)
    n = len(X)
    if n < min_train + test_window + horizon:
        raise ValueError(f"{ticker} h={horizon}: only {n} rows.")
    folds = _make_folds(n, min_train, test_window, step, purge=horizon)
    rows = []
    for i, (tr_s, tr_e, te_s, te_e) in enumerate(folds):
        m = _fit_eval_linear(X, y, tr_e, te_s, te_e, horizon, C)
        if m:
            m.update({"ticker": ticker, "fold": i})
            rows.append(m)
            if verbose:
                print(f"  fold {i:2d}  train_auc={m['train_auc']:.3f}  "
                      f"test_auc={m['test_auc']:.3f}  gap={m['train_auc']-m['test_auc']:+.3f}")
    if not rows:
        return None
    pf = pd.DataFrame(rows)
    return {
        "ticker": ticker, "n_folds": len(pf),
        "train_auc": round(pf["train_auc"].mean(), 4),
        "test_auc":  round(pf["test_auc"].mean(), 4),
        "gap":       round((pf["train_auc"] - pf["test_auc"]).mean(), 4),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tickers", type=str, default=None, help="comma-separated")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--tickers-file", type=str, default="tickers.txt")
    ap.add_argument("--horizon", type=int, default=5, choices=[1, 3, 5])
    ap.add_argument("--C", type=float, default=0.1, help="L2 strength (smaller=stronger)")
    ap.add_argument("--start", type=str, default="2018-01-01")
    args = ap.parse_args()

    if args.all:
        with open(args.tickers_file) as f:
            tickers = [t.strip() for t in f if t.strip() and not t.startswith("#")]
    elif args.tickers:
        tickers = [t.strip().upper() for t in args.tickers.split(",")]
    else:
        ap.error("pass --tickers A,B,C or --all")

    summaries = []
    for t in tickers:
        print(f"\n-- {t} h={args.horizon}d (L2-logistic, C={args.C}) --")
        try:
            df = add_forecast_targets(build_feature_dataframe(t, start_date=args.start))
            s = baseline_eval(t, df, horizon=args.horizon, C=args.C)
            if s:
                summaries.append(s)
                print(f"  SUMMARY train_auc={s['train_auc']:.3f}  "
                      f"test_auc={s['test_auc']:.3f}  gap={s['gap']:+.3f}  ({s['n_folds']} folds)")
        except Exception as e:
            print(f"  FAILED: {e}")

    if summaries:
        agg = pd.DataFrame(summaries)
        print("\n" + "=" * 60)
        print(f"LINEAR BASELINE -- {len(agg)} tickers, h={args.horizon}d, C={args.C}")
        print(f"  mean TRAIN auc: {agg['train_auc'].mean():.3f}")
        print(f"  mean TEST  auc: {agg['test_auc'].mean():.3f}")
        print(f"  mean GAP:       {agg['gap'].mean():+.3f}")
        print("-" * 60)
        print("READ: XGB was train 0.66-0.79 / test ~0.50 (gap ~0.25).")
        print("  - linear test >0.52 + small gap -> tree OVERFIT was the problem")
        print("  - linear test ~0.50             -> signal ABSENT at this target;")
        print("    overfit ruled out -> Hunt Queue #2 (pairs/cointegration)")
        print("=" * 60)


if __name__ == "__main__":
    main()
