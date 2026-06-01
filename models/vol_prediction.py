"""
Volatility prediction — Alpha Hunt Queue #4 (Jun 1 2026).

DIFFERENT QUESTION than every killed signal: not "will price go UP" but "will the
next h days be HIGH-vol or LOW-vol". Vol clusters/persists, so it's genuinely
predictable (research: AUC>0.6 achievable). NOT a return signal by itself — value is
POSITION SIZING + regime gating. Bar = vol-prediction AUC, not return-Sharpe.

Target (strictly forward, same NaN-last-h discipline as add_forecast_targets):
  fwd_vol[t] = std(daily returns over t+1..t+h); target=1 if fwd_vol > trailing
  per-ticker median else 0.

LEAK WATCH (Rule 1): forward target must not overlap backward features; WF purge=h
separates train/test by h. If AUC>0.75 suspect overlap/autocorr leak and attack it.

    python -m models.vol_prediction --tickers AAPL,MU,NVDA,AMD,MSFT --horizon 5
    python -m models.vol_prediction --all --horizon 5
"""
from __future__ import annotations
import argparse
import numpy as np, pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from features.builder import build_feature_dataframe
from models.classifier import FEATURE_COLUMNS, _prepare_xy
from models.walk_forward import _make_folds


def add_vol_target(df, horizon, med_window=126):
    df = df.copy()
    r = df["close"].pct_change()
    # forward realized vol over the next h days [t+1..t+h], aligned to row t.
    # reverse-rolling trick: reverse, rolling-std, reverse back, then shift -1 so it
    # starts at t+1 (strictly forward, no inclusion of day t's own return).
    fwd = r[::-1].rolling(horizon).std()[::-1].shift(-1)
    df["_fwd_vol"] = fwd
    med = fwd.rolling(med_window, min_periods=med_window // 2).median()
    df["target_vol"] = (fwd > med).astype(float)
    df.loc[df.index[-horizon:], "target_vol"] = np.nan
    df.loc[df["_fwd_vol"].isna(), "target_vol"] = np.nan
    df.loc[med.isna(), "target_vol"] = np.nan
    return df


def _prepare_xy_vol(df):
    feat = [c for c in FEATURE_COLUMNS if c in df.columns]
    sub = df.dropna(subset=["target_vol"]).copy()
    X = sub[feat].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    y = sub["target_vol"].astype(int)
    return X, y


def _fit_eval(X, y, tr_e, te_s, te_e, C):
    X_tr, y_tr = X.iloc[:tr_e], y.iloc[:tr_e]
    X_te, y_te = X.iloc[te_s:te_e], y.iloc[te_s:te_e]
    if y_tr.nunique() < 2 or y_te.nunique() < 2:
        return None
    sc = StandardScaler()
    clf = LogisticRegression(penalty="l2", C=C, max_iter=1000)
    clf.fit(sc.fit_transform(X_tr), y_tr)
    p_tr = clf.predict_proba(sc.transform(X_tr))[:, 1]
    p_te = clf.predict_proba(sc.transform(X_te))[:, 1]
    try:
        return {"train_auc": roc_auc_score(y_tr, p_tr),
                "test_auc": roc_auc_score(y_te, p_te), "n": len(y_te)}
    except ValueError:
        return None


def vol_eval(ticker, df, horizon=5, C=0.1, min_train=504, test_window=63, step=63, verbose=True):
    df = add_vol_target(df, horizon)
    X, y = _prepare_xy_vol(df)
    n = len(X)
    if n < min_train + test_window + horizon:
        raise ValueError(f"{ticker}: only {n} rows")
    folds = _make_folds(n, min_train, test_window, step, purge=horizon)
    rows = []
    for i, (tr_s, tr_e, te_s, te_e) in enumerate(folds):
        m = _fit_eval(X, y, tr_e, te_s, te_e, C)
        if m:
            m.update({"fold": i}); rows.append(m)
            if verbose:
                print(f"  fold {i:2d}  train_auc={m['train_auc']:.3f}  test_auc={m['test_auc']:.3f}")
    if not rows:
        return None
    pf = pd.DataFrame(rows)
    return {"ticker": ticker, "n_folds": len(pf),
            "train_auc": round(pf["train_auc"].mean(), 4),
            "test_auc": round(pf["test_auc"].mean(), 4),
            "pos_rate": round(float(y.mean()), 3)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tickers", default=None)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--tickers-file", default="tickers.txt")
    ap.add_argument("--horizon", type=int, default=5, choices=[1, 3, 5])
    ap.add_argument("--C", type=float, default=0.1)
    ap.add_argument("--start", default="2018-01-01")
    args = ap.parse_args()
    if args.all:
        with open(args.tickers_file) as f:
            tickers = [t.strip() for t in f if t.strip() and not t.startswith("#")]
    elif args.tickers:
        tickers = [t.strip().upper() for t in args.tickers.split(",")]
    else:
        ap.error("pass --tickers or --all")
    out = []
    for t in tickers:
        print(f"\n-- {t} h={args.horizon}d VOL-prediction --")
        try:
            df = build_feature_dataframe(t, start_date=args.start)
            s = vol_eval(t, df, horizon=args.horizon, C=args.C)
            if s:
                out.append(s)
                print(f"  SUMMARY train_auc={s['train_auc']:.3f}  test_auc={s['test_auc']:.3f}  "
                      f"pos_rate={s['pos_rate']}  ({s['n_folds']} folds)")
        except Exception as e:
            print(f"  FAILED: {e}")
    if out:
        agg = pd.DataFrame(out)
        print("\n" + "=" * 60)
        print(f"VOL PREDICTION -- {len(agg)} tickers, h={args.horizon}d")
        print(f"  mean TRAIN auc: {agg['train_auc'].mean():.3f}")
        print(f"  mean TEST  auc: {agg['test_auc'].mean():.3f}")
        print("-" * 60)
        print("READ (direction was ~0.50; vol should be EASIER):")
        print("  - test >0.58 across tickers -> REAL, usable for SIZING / regime gate")
        print("  - test ~0.50                -> even the easy target is dead here")
        print("  - test >0.75                -> SUSPECT overlap/autocorr leak, attack it")
        print("=" * 60)


if __name__ == "__main__":
    main()
