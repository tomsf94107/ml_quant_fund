# ────────────────────────────────────────────────────────────────────────────
# train_forecast_model.py — robust training script
#   • Flattens MultiIndex columns/indices
#   • Standardizes 'date' / 'ticker' / 'close' columns
#   • Safely merges risk flags on a plain 'date' column (fills zeros on failure)
#   • Uses helpers_xgb._pick_feature_columns (includes DB rollups: 7d/21d)
#   • Trains XGBoost per ticker, with clear logging + robust accuracy calc
# ────────────────────────────────────────────────────────────────────────────

from __future__ import annotations

import os, sys
from datetime import date, timedelta
from typing import List

import numpy as np
import pandas as pd
from dotenv import load_dotenv

# Path bootstrap: make parent importable as package root
_THIS   = os.path.abspath(__file__)
_REPO   = os.path.dirname(_THIS)
_PARENT = os.path.dirname(_REPO)
if _PARENT not in sys.path:
    sys.path.insert(0, _PARENT)

# Project imports
from ml_quant_fund.forecast_utils import build_feature_dataframe
from ml_quant_fund.core.helpers_xgb import train_xgb_predict, _pick_feature_columns  # noqa: F401

# Optional: risk module (best-effort)
try:
    import events_risk as _risk
except Exception:
    _risk = None

load_dotenv()

# ────────────────────────────────────────────────────────────────────────────
# Column hygiene & feature picking
# ────────────────────────────────────────────────────────────────────────────
EXCLUDE_EXACT = {
    "ticker","date","ds","actual","y","yhat","yhat_lower","yhat_upper",
    "Signal","Prob","Prob_eff","GateBlock",
}
EXCLUDE_PREFIXES = ("ret_","return_","Gate","Signal")
INSIDER_DB_FEATURES = ["ins_net_shares_7d_db","ins_net_shares_21d_db"]

CAND_DATE  = ["date","Date","ds","index","level_0"]
CAND_TKR   = ["ticker","Ticker","symbol","Symbol"]
CAND_CLOSE = ["close","Close","Adj Close","AdjClose","adj_close","close__Adj Close"]


def _flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = ["__".join(str(x) for x in tup if x is not None) for tup in df.columns]
    else:
        df.columns = [str(c) for c in df.columns]
    return df


def _ensure_date_ticker_close(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or len(df) == 0:
        return df
    # bring any index levels back as columns
    if not isinstance(df.index, pd.RangeIndex):
        df = df.reset_index()
    df = _flatten_columns(df)

    # date
    if "date" not in df.columns:
        for c in CAND_DATE:
            if c in df.columns:
                df = df.rename(columns={c: "date"})
                break
    df["date"] = pd.to_datetime(df.get("date", pd.NaT), errors="coerce")

    # ticker
    if "ticker" not in df.columns:
        for c in CAND_TKR:
            if c in df.columns:
                df = df.rename(columns={c: "ticker"})
                break
    if "ticker" in df.columns:
        df["ticker"] = df["ticker"].astype(str).str.upper()

    # close
    if "close" not in df.columns:
        for c in CAND_CLOSE:
            if c in df.columns:
                df = df.rename(columns={c: "close"})
                break
    if "close" in df.columns:
        df["close"] = pd.to_numeric(df["close"], errors="coerce")

    return df


def _attach_risk_flags(df: pd.DataFrame, start_dt: date, end_dt: date) -> pd.DataFrame:
    """Try to build/merge risk flags by 'date'. Fill zeros if not available."""
    df = _ensure_date_ticker_close(df)
    risk_df = pd.DataFrame()

    try:
        if _risk is not None:
            # Try a few likely function names from events_risk.py.
            if hasattr(_risk, "build_risk_calendar"):
                risk_df = _risk.build_risk_calendar(start_dt, end_dt)
            elif hasattr(_risk, "load_risk_flags"):
                risk_df = _risk.load_risk_flags(start_dt, end_dt)
            elif hasattr(_risk, "get_risk_frame"):
                risk_df = _risk.get_risk_frame(start_dt, end_dt)
    except Exception as e:
        print(f"⚠️ Risk provider failed: {e}")
        risk_df = pd.DataFrame()

    try:
        risk_df = _ensure_date_ticker_close(risk_df)
        need = [c for c in ["date","risk_today","risk_next_1d","risk_next_3d","risk_prev_1d"] if c in risk_df.columns]
        if "date" in df.columns and "date" in need:
            df = df.merge(risk_df[need], on="date", how="left")
    except Exception as e:
        print(f"⚠️ Risk flags unavailable (using zeros): {e}")

    for c in ["risk_today","risk_next_1d","risk_next_3d","risk_prev_1d"]:
        if c not in df.columns:
            df[c] = 0
    return df


# ────────────────────────────────────────────────────────────────────────────
# Accuracy helper (robust to regression/classification outputs)
# ────────────────────────────────────────────────────────────────────────────
def _safe_accuracy(y_true, y_pred) -> float:
    """Robust accuracy:
       - If y_true looks binary (0/1), compute classification accuracy (threshold 0.5 if needed).
       - Else treat as regression and compute directional accuracy on diffs.
    """
    import numpy as np, pandas as pd
    if y_true is None or y_pred is None:
        return float("nan")

    yt = pd.Series(y_true).astype(float)
    yp = pd.Series(y_pred).astype(float)

    # Align lengths
    n = min(len(yt), len(yp))
    if n < 2:
        return float("nan")
    yt = yt.iloc[:n].copy()
    yp = yp.iloc[:n].copy()

    # Classification case: y_true mostly {0,1}
    uniq = yt.dropna().unique()
    uniq_set = set(np.round(uniq, 6))
    if uniq_set.issubset({0.0, 1.0}) or (yt.nunique(dropna=True) <= 3 and yt.dropna().isin([0, 1]).mean() > 0.95):
        # If preds are probabilities or continuous, threshold at 0.5
        if not set(np.round(yp.dropna().unique(), 6)).issubset({0.0, 1.0}):
            yp = (yp > 0.5).astype(int)
        else:
            yp = yp.astype(int)
        return float((yp.values == yt.astype(int).values).mean())

    # Regression case: directional accuracy on diffs
    dir_true = np.sign(yt.diff()).iloc[1:]
    dir_pred = np.sign(yp.diff()).iloc[1:]
    m = min(len(dir_true), len(dir_pred))
    if m == 0:
        return float("nan")
    return float((dir_true.iloc[:m].values == dir_pred.iloc[:m].values).mean())


# ────────────────────────────────────────────────────────────────────────────
# Ticker utilities
# ────────────────────────────────────────────────────────────────────────────
def load_tickers_list(path: str = "tickers.csv") -> List[str]:
    if os.path.exists(path):
        return [t.strip().upper() for t in open(path).read().splitlines() if t.strip()]
    return ["AAPL","MSFT","TSLA","NVDA"]


# ────────────────────────────────────────────────────────────────────────────
# Training per ticker
# ────────────────────────────────────────────────────────────────────────────

def train_model_for_ticker(ticker: str, start_dt: date, end_dt: date):
    print(f"\n=== {ticker} ===")
    df = build_feature_dataframe(ticker, start_date=start_dt.isoformat(), end_date=end_dt.isoformat())
    if df is None or len(df) == 0:
        raise ValueError("Empty feature dataframe")

    df = _ensure_date_ticker_close(df)
    df = _attach_risk_flags(df, start_dt, end_dt)

    if "close" not in df.columns or df["close"].isna().all():
        raise KeyError("close")

    model, X_test, y_test, y_pred, y_prob = train_xgb_predict(df)

    # log shapes
    print(
        "lens →",
        "X_test:", (0 if X_test is None else len(X_test)),
        "y_test:", (0 if y_test is None else len(y_test)),
        "y_pred:", (0 if y_pred is None else len(y_pred)),
    )

    acc = _safe_accuracy(y_test, y_pred)
    print(f"Accuracy (dir): {acc:.3f}" if acc == acc else "Accuracy (dir): —")
    return acc  # <-- return for aggregation


# ────────────────────────────────────────────────────────────────────────────
# Entrypoint
# ────────────────────────────────────────────────────────────────────────────

def train_all_models():
    end_dt = date.today()
    start_dt = end_dt - timedelta(days=365*3)

    tickers = load_tickers_list()
    accs = []  # collect (ticker, acc)

    for t in tickers:
        try:
            acc = train_model_for_ticker(t, start_dt, end_dt)
            if acc == acc:  # not NaN
                accs.append((t, float(acc)))
        except KeyError as e:
            print(f"⚠️  Skipping {t}: {e!s}")
        except Exception as e:
            print(f"⚠️  {t}: {e}")

    # ---- summary ----
    good = [(t, a) for t, a in accs if a == a]
    if good:
        good.sort(key=lambda x: x[1], reverse=True)
        print("\nTop 8 tickers by accuracy:")
        for t, a in good[:8]:
            print(f"  {t}: {a:.3f}")
        vals = [a for _, a in good]
        print(f"\nMean accuracy: {np.mean(vals):.3f}  |  Median: {np.median(vals):.3f}")


if __name__ == "__main__":
    train_all_models()
