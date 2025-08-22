# scripts/retrain_if_degraded.py
import os, pandas as pd
from sqlalchemy import create_engine
from datetime import datetime, timedelta

from ml_quant_fund.forecast_utils import auto_retrain_forecast_model

DSN = os.environ["ACCURACY_DSN"]
eng = create_engine(DSN, pool_pre_ping=True)

TICKERS = ["AAPL","MSFT","NVDA","AMZN","GOOG","TSLA","CRWD","DDOG","PLTR","UNH","NVO","MRNA"]
LOOKBACK_DAYS = 30
R2_MIN = 0.15         # trigger if mean R² < 0.15 over lookback
MAE_WORSE_Q = 0.80     # also trigger if MAE is above 80th pct of its history

now = pd.Timestamp.utcnow()
since = now - pd.Timedelta(days=LOOKBACK_DAYS)

df = pd.read_sql("""
  select ts, ticker, mae, mse, r2
  from metrics
  where ts >= now() - interval '365 days'
""", eng)

df['ts'] = pd.to_datetime(df['ts'], utc=True)

to_retrain = []
for t in TICKERS:
    d = df[df.ticker == t].copy()
    if d.empty: 
        continue
    recent = d[d.ts >= since]
    if recent.empty:
        continue
    mean_r2 = recent['r2'].dropna().mean()
    mae_q80 = d['mae'].dropna().quantile(0.80) if d['mae'].notna().any() else None
    mae_recent = recent['mae'].dropna().mean() if recent['mae'].notna().any() else None

    if (mean_r2 is not None and mean_r2 < R2_MIN) or (mae_q80 and mae_recent and mae_recent > mae_q80):
        to_retrain.append(t)

for t in to_retrain:
    try:
        auto_retrain_forecast_model(t)  # your existing retrain helper
        print(f"Retrained {t}")
    except Exception as e:
        print(f"Retrain failed for {t}: {e}")
