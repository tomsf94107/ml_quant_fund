#!/usr/bin/env python3
"""
analysis/wf_returns_dump.py -- persist PER-ROW out-of-sample predictions + RETURNS.

WHY THIS EXISTS
    models/walk_forward.py collapses each fold to hit-counts (buy_hit_55, buy_n_55)
    and DISCARDS the per-row predictions. You therefore cannot compute a return
    spread, a beta-strip, or a cost-net -- the information was never written down.
    A hit rate has no magnitude: 52.9% right is compatible with losing money.

WHAT IT DOES
    Re-runs the IDENTICAL fold loop -- same _make_folds, same _prepare_xy, same
    _get_xgb_params (seed 42), same isotonic calibration on the last 20% of train,
    same risk sample weights -- and keeps every out-of-sample row:

        ticker, horizon, fold, date, prob_up, y_true, fwd_ret, spy_fwd_ret

    fwd_ret     = close[t+h] / close[t] - 1
    spy_fwd_ret = cum[t+h]   / cum[t]   - 1     where cum = (1 + spy_ret).cumprod()

    Both use the SAME window as target_{h}d (close.shift(-h) > close), so the
    label, the stock return, and the market return are aligned to the bar.

    prob_up here MUST reproduce the buy_hit_55 numbers from walk_forward_history.
    If it does not, something in this file has diverged and nothing downstream is
    trustworthy -- run the parity check that wf_returns_test.py prints.

USAGE
    Driver (batches 10 tickers per subprocess, like walk_forward_batched):
        python -m analysis.wf_returns_dump --horizon 5 --start 2018-01-01

    Worker (a specific slice):
        python -m analysis.wf_returns_dump --horizon 5 --start 2018-01-01 --tickers AAPL,MSFT

OUTPUT
    reports/wf_rows_h{H}.csv    (truncated by the driver, appended by workers)
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from features.builder import build_feature_dataframe, add_forecast_targets
from models.classifier import (
    _prepare_xy, _get_xgb_params, _risk_sample_weights,
    RISK_ALPHA, RISK_WEIGHT_FLOOR,
)
from models.walk_forward import _make_folds

# These MUST match walk_forward_eval's defaults or the folds will not line up.
MIN_TRAIN    = 504
TEST_WINDOW  = 63
STEP         = 63
CALIB_FRAC   = 0.2

BATCH_SIZE   = 10
TICKERS_FILE = ROOT / "tickers.txt"
OUT_DIR      = ROOT / "reports"

COLS = ["ticker", "horizon", "fold", "date", "prob_up",
        "y_true", "fwd_ret", "spy_fwd_ret"]


def dump_one(ticker: str, horizon: int, start: str) -> pd.DataFrame:
    df = add_forecast_targets(build_feature_dataframe(ticker, start_date=start))

    target_col = f"target_{horizon}d"
    for need in (target_col, "close", "spy_ret", "date"):
        if need not in df.columns:
            raise ValueError(f"{ticker}: panel missing '{need}'")

    # Forward returns on the FULL panel, same shift as the label.
    close        = df["close"].astype(float)
    fwd_ret_full = close.shift(-horizon) / close - 1.0
    cum          = (1.0 + df["spy_ret"].astype(float).fillna(0.0)).cumprod()
    spy_fwd_full = cum.shift(-horizon) / cum - 1.0

    # _prepare_xy drops ONLY the tail-h rows (target NaN), so X.index is a
    # contiguous label subset of df's RangeIndex. .loc alignment is exact.
    X, y = _prepare_xy(df, target_col)
    n = len(X)
    if n < MIN_TRAIN + TEST_WINDOW + horizon:
        raise ValueError(f"{ticker} h={horizon}: only {n} rows")

    w = _risk_sample_weights(df.loc[X.index], RISK_ALPHA, RISK_WEIGHT_FLOOR)
    w = np.asarray(w) if w is not None else None

    folds = _make_folds(n, MIN_TRAIN, TEST_WINDOW, STEP, purge=horizon)
    if not folds:
        raise ValueError(f"{ticker} h={horizon}: no folds")

    out = []
    for i, (tr_s, tr_e, te_s, te_e) in enumerate(folds):
        X_tr, y_tr = X.iloc[:tr_e], y.iloc[:tr_e]
        X_te, y_te = X.iloc[te_s:te_e], y.iloc[te_s:te_e]

        n_tr       = len(X_tr)
        split_in   = int(n_tr * (1 - CALIB_FRAC))
        X_in, y_in = X_tr.iloc[:split_in], y_tr.iloc[:split_in]
        X_cal, y_cal = X_tr.iloc[split_in:], y_tr.iloc[split_in:]
        w_in       = w[:split_in] if w is not None else None

        clf = XGBClassifier(**_get_xgb_params(ticker, horizon))
        fit_kw: dict = {"verbose": False}
        if w_in is not None:
            fit_kw["sample_weight"] = w_in
        clf.fit(X_in, y_in, **fit_kw)

        iso = IsotonicRegression(out_of_bounds="clip")
        iso.fit(clf.predict_proba(X_cal)[:, 1], y_cal)
        p_test = iso.transform(clf.predict_proba(X_te)[:, 1])

        idx = X_te.index          # original df labels
        out.append(pd.DataFrame({
            "ticker":      ticker,
            "horizon":     horizon,
            "fold":        i,
            "date":        df.loc[idx, "date"].values,
            "prob_up":     p_test,
            "y_true":      y_te.values.astype(int),
            "fwd_ret":     fwd_ret_full.loc[idx].values,
            "spy_fwd_ret": spy_fwd_full.loc[idx].values,
        }))

    return pd.concat(out, ignore_index=True)


def load_tickers() -> list[str]:
    with open(TICKERS_FILE) as f:
        return [ln.strip().upper() for ln in f
                if ln.strip() and not ln.strip().startswith("#")]


def run_worker(tickers: list[str], horizon: int, start: str, out_path: Path):
    for t in tickers:
        try:
            rows = dump_one(t, horizon, start)
            hdr = not out_path.exists()
            rows[COLS].to_csv(out_path, mode="a", header=hdr, index=False)
            print(f"  {t:<6} {len(rows):>5} rows  folds={rows.fold.nunique()}",
                  flush=True)
        except Exception as e:
            print(f"  {t:<6} FAILED: {str(e)[:70]}", flush=True)


def run_driver(horizon: int, start: str, out_path: Path):
    tickers = load_tickers()
    if out_path.exists():
        out_path.unlink()
        print(f"removed stale {out_path}")

    batches = [tickers[i:i + BATCH_SIZE]
               for i in range(0, len(tickers), BATCH_SIZE)]
    print(f"{len(tickers)} tickers, h={horizon}, {len(batches)} batches "
          f"(BATCH_SIZE={BATCH_SIZE})", flush=True)

    env = os.environ.copy()
    env["ML_QUANT_NO_TAIL_REFETCH"] = "1"   # the panel builder must not 429 us

    t0 = time.time()
    for bi, chunk in enumerate(batches, 1):
        print(f"\nBATCH {bi}/{len(batches)}: {chunk[0]} -> {chunk[-1]}", flush=True)
        args = [sys.executable, "-m", "analysis.wf_returns_dump",
                "--horizon", str(horizon), "--start", start,
                "--tickers", ",".join(chunk), "--out", str(out_path)]
        r = subprocess.run(args, env=env, cwd=str(ROOT))
        print(f"BATCH {bi}/{len(batches)}: exit={r.returncode}", flush=True)

    el = time.time() - t0
    print(f"\nDONE in {el/60:.1f} min -> {out_path}", flush=True)
    if out_path.exists():
        df = pd.read_csv(out_path)
        print(f"rows={len(df):,}  tickers={df.ticker.nunique()}  "
              f"folds={sorted(df.fold.unique())}", flush=True)
    else:
        print("NO OUTPUT WRITTEN -- every ticker failed.", flush=True)
        sys.exit(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizon", type=int, required=True)
    ap.add_argument("--start", default="2018-01-01")
    ap.add_argument("--tickers", default=None,
                    help="comma-separated; worker mode. omit for driver mode.")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    OUT_DIR.mkdir(exist_ok=True)
    out_path = Path(a.out) if a.out else OUT_DIR / f"wf_rows_h{a.horizon}.csv"

    if a.tickers:
        run_worker([t.strip().upper() for t in a.tickers.split(",")],
                   a.horizon, a.start, out_path)
    else:
        run_driver(a.horizon, a.start, out_path)


if __name__ == "__main__":
    main()
