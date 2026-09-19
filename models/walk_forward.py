"""
Walk-forward cross-validation — honest accuracy without leakage.

Fixes 3 leaks in train_model():
  1. Calibration fit on test set → calibrate on held-out train slice instead.
  2. No purge between train/test → drop last `horizon` train rows.
  3. Single fold → expanding-window multi-fold.

Usage (module):
    from models.walk_forward import walk_forward_eval
    metrics_df, summary = walk_forward_eval(ticker, df, horizon=1)

Usage (CLI):
    python -m models.walk_forward --ticker AAPL --horizon 1
    python -m models.walk_forward --all --horizon 1
"""
from __future__ import annotations

# Faulthandler — dump all thread tracebacks every 60 sec to find hangs.
# Added May 6 2026 per ChatGPT walk-forward debug consult. If walk_forward
# hangs, logs/walk_forward_stacks.log shows where it's stuck.
# Trigger manually: kill -USR1 <PID>
import faulthandler
import os
import signal
from pathlib import Path as _Path
_Path("logs").mkdir(exist_ok=True)
_STACK_LOG = open("logs/walk_forward_stacks.log", "a", buffering=1)
faulthandler.enable(file=_STACK_LOG, all_threads=True)
faulthandler.dump_traceback_later(60, repeat=True, file=_STACK_LOG)
if hasattr(signal, "SIGUSR1"):
    faulthandler.register(signal.SIGUSR1, file=_STACK_LOG, all_threads=True)

import argparse
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import (
    accuracy_score, roc_auc_score, log_loss, brier_score_loss,
)
from xgboost import XGBClassifier

from features.builder import build_feature_dataframe, add_forecast_targets
from models.classifier import (
    FEATURE_COLUMNS,
    _prepare_xy, _get_xgb_params, _risk_sample_weights,
    RISK_ALPHA, RISK_WEIGHT_FLOOR,
)

REPORT_DIR = Path("reports")
REPORT_DIR.mkdir(exist_ok=True)


# DB persistence (May 6 2026) — write walk-forward results to accuracy.db
import sqlite3 as _sql_wf
from datetime import date as _date_wf
_DB_PATH = Path(__file__).parent.parent / "accuracy.db"


def _write_summary_to_db(df, horizon: int) -> None:
    """Persist walk-forward summary to accuracy.db.walk_forward_history."""
    if df is None or df.empty:
        return
    run_date = _date_wf.today().strftime("%Y-%m-%d")
    rows = []
    for _, r in df.iterrows():
        buy_hit = r.get("buy_hit_55_mean", None)
        if pd.isna(buy_hit):
            buy_hit = None
        rows.append((
            run_date, r["ticker"], horizon,
            float(r.get("accuracy_mean", 0.0)),
            float(r.get("auc_mean", 0.0)),
            float(buy_hit) if buy_hit is not None else None,
            int(r.get("buy_n_55_total", 0)),
            float(r.get("pos_rate_mean", 0.0)),
            int(r.get("n_folds", r.get("folds", 0))),
        ))
    with _sql_wf.connect(str(_DB_PATH)) as conn:
        conn.executemany("""
            INSERT OR REPLACE INTO walk_forward_history
                (run_date, ticker, horizon, accuracy, auc, buy_hit_55,
                 buy_n_55, pos_rate, folds)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, rows)
        conn.commit()
    print(f"  → DB: wrote {len(rows)} rows for h={horizon} (run_date={run_date})")



def _make_folds(
    n: int,
    min_train: int,
    test_window: int,
    step: int,
    purge: int,
) -> list[tuple[int, int, int, int]]:
    """
    Return list of (train_start, train_end, test_start, test_end) tuples.
    Expanding window. Purge = `horizon` rows dropped from train tail.
    """
    folds = []
    test_start = min_train
    while test_start + test_window <= n:
        train_start = 0
        train_end   = test_start - purge       # purge gap
        test_end    = test_start + test_window
        if train_end - train_start >= 100:      # min viable train
            folds.append((train_start, train_end, test_start, test_end))
        test_start += step
    return folds


def _fit_and_eval_fold(
    X: pd.DataFrame, y: pd.Series, w: Optional[np.ndarray],
    train_end: int, test_start: int, test_end: int,
    ticker: str, horizon: int, calib_frac: float = 0.2,
    df_full: Optional[pd.DataFrame] = None,
) -> dict:
    """Fit on inner-train, calibrate on held-out slice, evaluate on test."""
    X_tr_all, y_tr_all = X.iloc[:train_end], y.iloc[:train_end]
    X_te,     y_te     = X.iloc[test_start:test_end], y.iloc[test_start:test_end]

    # Inner split for calibration (last calib_frac of train)
    n_tr      = len(X_tr_all)
    split_in  = int(n_tr * (1 - calib_frac))
    X_in, y_in = X_tr_all.iloc[:split_in], y_tr_all.iloc[:split_in]
    X_cal, y_cal = X_tr_all.iloc[split_in:], y_tr_all.iloc[split_in:]

    # WEIGHTS ON THE INNER SLICE, COMPUTED FROM IT (fixed 2026-09-19).
    #
    # This used to slice a pre-computed vector: w[:split_in]. _risk_sample_weights
    # boosts the LAST 60 rows 3x, and the inner split drops the last 20% of the
    # training slice for calibration -- so the boosted rows were ALWAYS in the
    # calibration slice, never in the fit. Simulated across fold sizes 600,
    # 1200 and 1800: 0 of 60 boosted rows inside inner-train every time, and
    # w_in came out a single repeated value.
    #
    # The recency component therefore never reached a walk-forward fit, for any
    # ticker, any fold, ever. classifier.train_model does not have this problem
    # -- it computes weights on X_train and fits the same rows -- so PRODUCTION
    # gets the boost and the VALIDATOR did not. Every walk_forward number before
    # today describes a differently-fitted model than the one predicting.
    #
    # Recomputing on X_in makes "last 60 rows" mean the last 60 rows OF THE FIT.
    if df_full is not None:
        w_in = _risk_sample_weights(df_full.loc[X_in.index],
                                    RISK_ALPHA, RISK_WEIGHT_FLOOR)
        w_in = np.asarray(w_in) if w_in is not None else None
    else:
        w_in = w[:split_in] if w is not None else None

    # Fit base XGB
    params   = _get_xgb_params(ticker, horizon)
    base_clf = XGBClassifier(**params)
    fit_kw: dict = {"verbose": False}
    if w_in is not None:
        fit_kw["sample_weight"] = w_in
    base_clf.fit(X_in, y_in, **fit_kw)

    # Fit isotonic on calibration slice
    p_cal = base_clf.predict_proba(X_cal)[:, 1]
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(p_cal, y_cal)

    # Predict on test
    p_test_raw = base_clf.predict_proba(X_te)[:, 1]
    p_test     = iso.transform(p_test_raw)
    yhat       = (p_test >= 0.5).astype(int)

    # Metrics
    try:
        auc = roc_auc_score(y_te, p_test)
    except ValueError:
        auc = float("nan")

    # BUY hit rate at multiple thresholds
    buy_stats = {}
    for thr in (0.50, 0.55, 0.60):
        mask = p_test >= thr
        n_buy = int(mask.sum())
        if n_buy > 0:
            hit = float(y_te[mask].mean())
        else:
            hit = float("nan")
        buy_stats[f"buy_hit_{int(thr*100)}"] = round(hit, 4)
        buy_stats[f"buy_n_{int(thr*100)}"]   = n_buy

    return {
        "n_train":   n_tr - (n_tr - split_in),   # inner train size
        "n_calib":   len(X_cal),
        "n_test":    len(X_te),
        "accuracy":  round(accuracy_score(y_te, yhat), 4),
        "roc_auc":   round(auc, 4),
        "log_loss":  round(log_loss(y_te, p_test, labels=[0, 1]), 4),
        "brier":     round(brier_score_loss(y_te, p_test), 4),
        "pos_rate":  round(float(y_te.mean()), 4),
        **buy_stats,
    }


def walk_forward_eval(
    ticker: str,
    df: pd.DataFrame,
    horizon: int = 1,
    min_train: int = 504,
    test_window: int = 63,
    step: int = 63,
    verbose: bool = True,
) -> Tuple[pd.DataFrame, dict]:
    """
    Walk-forward CV for one ticker × horizon.

    Returns (per_fold_df, summary_dict).
    """
    target_col = f"target_{horizon}d"
    if target_col not in df.columns:
        raise ValueError(f"{target_col} missing — run add_forecast_targets().")

    X, y = _prepare_xy(df, target_col)
    n = len(X)
    if n < min_train + test_window + horizon:
        raise ValueError(
            f"{ticker} h={horizon}: only {n} rows, need "
            f">= {min_train + test_window + horizon}."
        )

    # WEIGHTS ARE COMPUTED PER FOLD, NOT ONCE (fixed 2026-09-19).
    #
    # This used to compute _risk_sample_weights over the WHOLE frame and pass
    # the same vector to every fold, where _fit_and_eval_fold takes w[:split_in]
    # for the inner fit. _risk_sample_weights puts a 3x boost on the LAST 60
    # rows -- "so models adapt faster to current regime", per its own docstring
    # at classifier.py:321 -- and those rows sit at the very end of the frame.
    #
    # Measured on a 2,170-row ticker: the boost covers indices 2110-2169, the
    # inner-train slice ends at 1735, and ZERO boosted rows land inside it.
    # w_in came out a single repeated value, 0.9476. The recency component has
    # never reached a walk-forward fit, for any ticker, any fold.
    #
    # classifier.train_model does NOT have this problem: it computes weights on
    # X_train itself (line 494) and fits on the same rows (511), so the boost
    # applies. PRODUCTION GETS THE BOOST, THE VALIDATOR DID NOT -- so every
    # walk_forward number before today describes a differently-fitted model
    # than the one making predictions. The 2026-09-13 baseline is not
    # comparable to runs after this fix.
    #
    # Passing df through and slicing it per fold is what makes the validator
    # measure what production does.
    w = None  # computed per fold below, from df

    folds = _make_folds(n, min_train, test_window, step, purge=horizon)
    if not folds:
        raise ValueError(f"{ticker} h={horizon}: no valid folds from n={n}.")

    rows = []
    for i, (tr_s, tr_e, te_s, te_e) in enumerate(folds):
        try:
            # Weights for THIS fold's training rows only, so the recency
            # boost lands on the newest rows of the slice being fitted.
            # df_full lets the fold weight its own INNER slice, so the 3x
            # recency boost lands on the newest rows actually fitted rather
            # than in the calibration holdout.
            m = _fit_and_eval_fold(X, y, None, tr_e, te_s, te_e,
                                   ticker, horizon, df_full=df)
            m.update({"ticker": ticker, "horizon": horizon, "fold": i,
                      "train_end_idx": tr_e, "test_start_idx": te_s})
            rows.append(m)
            if verbose:
                print(f"  fold {i:2d}  acc={m['accuracy']:.3f}  "
                      f"auc={m['roc_auc']:.3f}  "
                      f"buy55_hit={m['buy_hit_55']}  n_buy55={m['buy_n_55']}")
        except Exception as e:
            if verbose:
                print(f"  fold {i:2d}  FAILED: {e}")

    per_fold = pd.DataFrame(rows)
    if per_fold.empty:
        raise RuntimeError(f"{ticker} h={horizon}: all folds failed.")

    # Aggregate (pooled metrics are more honest than mean-of-folds for small samples)
    summary = {
        "ticker":        ticker,
        "horizon":       horizon,
        "n_folds":       len(per_fold),
        "accuracy_mean": round(per_fold["accuracy"].mean(), 4),
        "accuracy_std":  round(per_fold["accuracy"].std(),  4),
        "auc_mean":      round(per_fold["roc_auc"].mean(),  4),
        "brier_mean":    round(per_fold["brier"].mean(),    4),
        "buy_hit_55_mean": round(per_fold["buy_hit_55"].mean(skipna=True), 4),
        "buy_n_55_total":  int(per_fold["buy_n_55"].sum()),
        "pos_rate_mean": round(per_fold["pos_rate"].mean(), 4),
    }
    return per_fold, summary


def _load_df_for_ticker(ticker: str, start_date: str = "2018-01-01") -> pd.DataFrame:
    """Mirror train_all's data prep. Adjust if train_one_ticker differs."""
    df = build_feature_dataframe(ticker, start_date=start_date)
    df = add_forecast_targets(df)
    return df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ticker", type=str, default=None)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--horizon", type=int, default=1, choices=[1, 3, 5])
    ap.add_argument("--tickers-file", type=str, default="tickers.txt")
    ap.add_argument("--start", type=str, default="2018-01-01")
    ap.add_argument("--start-from", type=str, default=None,
                    help="batch slice start ticker (subprocess batching)")
    ap.add_argument("--end-at", type=str, default=None,
                    help="batch slice end ticker (inclusive, subprocess batching)")
    args = ap.parse_args()

    if args.all:
        with open(args.tickers_file) as f:
            tickers = [t.strip() for t in f if t.strip() and not t.startswith("#")]
    elif args.ticker:
        tickers = [args.ticker]
    else:
        ap.error("pass --ticker TICKER or --all")

    # Subprocess batching slice (May 6 2026)
    if args.start_from:
        if args.start_from in tickers:
            tickers = tickers[tickers.index(args.start_from):]
            print(f"Starting from {args.start_from}")
    if args.end_at:
        if args.end_at in tickers:
            tickers = tickers[:tickers.index(args.end_at) + 1]
            print(f"Ending at {args.end_at} (inclusive). Slice: {len(tickers)} tickers")

    summaries, per_folds = [], []
    for t in tickers:
        print(f"\n── {t} h={args.horizon}d ──")
        try:
            df = _load_df_for_ticker(t, start_date=args.start)
            pf, s = walk_forward_eval(t, df, horizon=args.horizon, verbose=True)
            per_folds.append(pf)
            summaries.append(s)
            print(f"  SUMMARY acc={s['accuracy_mean']:.3f}±{s['accuracy_std']:.3f} "
                  f"auc={s['auc_mean']:.3f} buy55_hit={s['buy_hit_55_mean']} "
                  f"buy55_n={s['buy_n_55_total']} folds={s['n_folds']}")
        except Exception as e:
            print(f"  SKIP {t}: {e}")

    if summaries:
        summary_df = pd.DataFrame(summaries)
        out_s = REPORT_DIR / f"walkforward_summary_h{args.horizon}.csv"
        write_header = not out_s.exists()
        summary_df.to_csv(out_s, mode='a' if not write_header else 'w',
                          header=write_header, index=False)
        # Persist to DB (May 6 2026)
        try:
            _write_summary_to_db(summary_df, args.horizon)
        except Exception as e:
            print(f"  ⚠ DB write failed: {e}")
        print(f"\nSummary → {out_s}")
        print(f"  Mean accuracy across tickers: {summary_df['accuracy_mean'].mean():.4f}")
        print(f"  Mean AUC across tickers:      {summary_df['auc_mean'].mean():.4f}")
        print(f"  Mean BUY@0.55 hit:            {summary_df['buy_hit_55_mean'].mean():.4f}")
        print(f"  Total BUY@0.55 signals:       {int(summary_df['buy_n_55_total'].sum())}")

    if per_folds:
        pf_all = pd.concat(per_folds, ignore_index=True)
        out_p = REPORT_DIR / f"walkforward_folds_h{args.horizon}.csv"
        write_header = not out_p.exists()
        pf_all.to_csv(out_p, mode='a' if not write_header else 'w',
                      header=write_header, index=False)
        print(f"Per-fold → {out_p}")


if __name__ == "__main__":
    main()
