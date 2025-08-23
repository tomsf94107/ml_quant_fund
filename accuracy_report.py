# accuracy_report.py — robust rolling accuracy CLI
# Usage:
#   export NO_SENTIMENT=1   # optional to skip FinBERT during training
#   python accuracy_report.py AAPL --start 2025-01-01 --end 2025-08-01 --window 30

from __future__ import annotations
import argparse, os, sys, inspect
from datetime import date, timedelta
import numpy as np
import pandas as pd

# Make parent (package root) importable
_THIS   = os.path.abspath(__file__)
_REPO   = os.path.dirname(_THIS)
_PARENT = os.path.dirname(_REPO)
if _PARENT not in sys.path:
    sys.path.insert(0, _PARENT)

from ml_quant_fund.forecast_utils import build_feature_dataframe
try:
    from ml_quant_fund.forecast_utils import compute_rolling_accuracy as _cra
except Exception:
    _cra = None
from ml_quant_fund.core.helpers_xgb import train_xgb_predict

CAND_DATE = ["date","Date","ds","index","level_0"]

def _pick_col(df: pd.DataFrame, candidates) -> str | None:
    if df is None or df.empty: return None
    for c in candidates:
        if c in df.columns: return c
    lower = {c.lower(): c for c in df.columns}
    for c in candidates:
        if c.lower() in lower: return lower[c.lower()]
    return None

def _try_compute_rolling_accuracy(tkr: str, start: str, end: str, window_days: int):
    """Try multiple signatures for compute_rolling_accuracy; normalize to ['date','accuracy']."""
    if _cra is None:
        return None
    attempts = [
        lambda: _cra(tkr, start_date=start, end_date=end, window_days=window_days),
        lambda: _cra(tkr, start, end, window_days),
        lambda: _cra(tkr, start, end),
        lambda: _cra(tkr, window_days=window_days),
        lambda: _cra(tkr),
    ]
    for fn in attempts:
        try:
            res = fn()
            if res is None:
                continue
            if isinstance(res, pd.Series):
                ser = res.copy()
                idx_name = ser.index.name or "date"
                df = ser.reset_index().rename(columns={idx_name: "date", ser.name or "value": "accuracy"})
            else:
                df = res.copy()
                dcol = _pick_col(df, ["date","Date","ds"])
                vcol = _pick_col(df, ["accuracy","acc","rolling_accuracy","hit_rate","value"])
                if dcol: df = df.rename(columns={dcol:"date"})
                else:    df.insert(0, "date", pd.to_datetime(df.index, errors="coerce"))
                if vcol: df = df.rename(columns={vcol:"accuracy"})
                else:
                    last = df.columns[-1]
                    if last != "date":
                        df = df.rename(columns={last:"accuracy"})
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
            df = df.dropna(subset=["date"]).sort_values("date")
            if "accuracy" in df.columns:
                return df[["date","accuracy"]]
        except TypeError:
            continue
        except Exception:
            continue
    return None

def _fallback_from_model(tkr: str, start: str, end: str, window_days: int) -> pd.DataFrame:
    """Train once and compute rolling directional accuracy over time (NumPy-safe alignment)."""
    # Skip slow sentiment during training unless explicitly enabled
    os.environ.setdefault("NO_SENTIMENT", os.getenv("NO_SENTIMENT", "1"))

    df = build_feature_dataframe(tkr, start_date=start, end_date=end)
    if df is None or df.empty:
        raise ValueError("Empty feature dataframe")

    model, X_test, y_test, y_pred, y_prob = train_xgb_predict(df)
    if y_test is None or y_pred is None or len(y_test) == 0 or len(y_pred) == 0:
        raise ValueError("Model returned no predictions")

    n = min(len(y_test), len(y_pred))
    y_true = np.asarray(y_test[:n], dtype=float)
    y_hat  = np.asarray(y_pred[:n], dtype=float)

    # Directional hits on arrays (avoid pandas index alignment)
    hit = (np.sign(np.diff(y_true)) == np.sign(np.diff(y_hat))).astype(float)  # length n-1

    # Dates aligned to the last n points, then drop first to match diff
    dcol = _pick_col(df, CAND_DATE)
    if dcol and dcol in df.columns:
        dates_n = pd.to_datetime(df[dcol].tail(n), errors="coerce").to_numpy()
    else:
        # fallback date range if no date column available
        dates_n = pd.date_range(start=pd.Timestamp(start), periods=n, freq="B").to_numpy()
    dates = dates_n[1:]  # align with diff (n-1)

    # Rolling accuracy (require a few points)
    win = max(3, int(window_days or 30))
    acc_roll = pd.Series(hit).rolling(win, min_periods=max(3, win//3)).mean().to_numpy()

    out = pd.DataFrame({"date": dates, "accuracy": acc_roll})
    out = out[~pd.isna(out["date"])].sort_values("date")
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ticker")
    ap.add_argument("--start", default=(date.today()-timedelta(days=365)).isoformat())
    ap.add_argument("--end",   default=date.today().isoformat())
    ap.add_argument("--window", type=int, default=30)
    args = ap.parse_args()

    tkr = args.ticker.upper()
    start, end, win = args.start, args.end, int(args.window)

    acc_ts = _try_compute_rolling_accuracy(tkr, start, end, win)
    if acc_ts is None or acc_ts.empty:
        acc_ts = _fallback_from_model(tkr, start, end, win)

    acc_ts["date"] = pd.to_datetime(acc_ts["date"], errors="coerce")
    acc_ts = acc_ts.dropna(subset=["date"]).sort_values("date")
    outpath = f"{tkr}_accuracy_{win}d.csv"
    acc_ts.to_csv(outpath, index=False)
    print("Saved", outpath, "| rows:", len(acc_ts))

if __name__ == "__main__":
    main()
