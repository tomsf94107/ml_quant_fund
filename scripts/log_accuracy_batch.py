# scripts/log_accuracy_batch.py

# --- path bootstrap: make parent-of-repo importable as a package root ---
import os, sys
_THIS   = os.path.abspath(__file__)
_REPO   = os.path.dirname(os.path.dirname(_THIS))   # …/ml_quant_fund
_PARENT = os.path.dirname(_REPO)                    # …/
if _PARENT not in sys.path:
    sys.path.insert(0, _PARENT)
# ------------------------------------------------------------------------

import numpy as np
from inspect import signature
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from ml_quant_fund.forecast_utils import build_feature_dataframe
from ml_quant_fund.core.helpers_xgb import train_xgb_predict
from ml_quant_fund.accuracy_sink import log_accuracy

# Keep runtime light by default
os.environ.setdefault("NO_SENTIMENT", "1")
os.environ.setdefault("NO_INSIDERS", "1")

TICKERS = [
    "AAPL","MSFT","NVDA","AMZN","GOOG","TSLA",
    "CRWD","DDOG","PLTR","UNH","NVO","MRNA",
    "FIG","OPEN","CRCL","TSM","SNOW","PFE","MP","DUOL","FTNT","SMCI","NTDOY"
    # removed duplicate/typo: CRWV
]

def bf_compat(t, start="2024-01-01", end=None):
    p = set(signature(build_feature_dataframe).parameters)
    if {"start_date","end_date"} & p:
        return build_feature_dataframe(t, start_date=start, end_date=end)
    if {"start","end"} & p:
        return build_feature_dataframe(t, start=start, end=end)
    try:
        return build_feature_dataframe(t, start, end)  # positional
    except TypeError:
        return build_feature_dataframe(t)

if __name__ == "__main__":
    logged = 0
    for t in TICKERS:
        try:
            df = bf_compat(t, "2024-01-01")
            m, Xt, yt, yp, _ = train_xgb_predict(df)
            if yt is None or yp is None or len(yt) != len(yp):
                print(f"skip {t}: no comparable predictions"); continue

            yt, yp = np.asarray(yt), np.asarray(yp)
            mae = float(mean_absolute_error(yt, yp))
            mse = float(mean_squared_error(yt, yp))
            try:
                r2 = float(r2_score(yt, yp))
            except Exception:
                r2 = float("nan")

            log_accuracy(t, mae, mse, r2)  # writes to Neon via ACCURACY_DSN
            logged += 1
            print(f"logged {t}: MAE={mae:.3f} MSE={mse:.3f} R2={r2:.3f}")
        except Exception as e:
            print(f"skip {t}: {e}")

    print(f"Done. Logged {logged} tickers.")
