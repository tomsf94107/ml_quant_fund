#forecast v1.3 with loading logs, comparing predictions vs actuals, and auto-retraining 

import yfinance as yf
import pandas as pd
from prophet import Prophet
from datetime import datetime, timedelta
import numpy as np
import os
import glob

# Path for forecast log
LOG_DIR = "forecast_logs"
os.makedirs(LOG_DIR, exist_ok=True)

# ---- Function: 3-Month Forecast using Prophet ----
def forecast_price_trend(ticker: str, start_date=None, end_date=None, period_months=3, log_results=True):
    if end_date is None:
        end_date = datetime.today()
    if start_date is None:
        start_date = end_date - timedelta(days=5 * 365)

    df = yf.download(ticker, start=start_date, end=end_date, auto_adjust=True)
    if df.empty or 'Close' not in df:
        return None, f"No data found for {ticker}"

    df_prophet = df[['Close']].reset_index()
    df_prophet.columns = ['ds', 'y']

    model = Prophet(daily_seasonality=True)
    model.fit(df_prophet)

    future = model.make_future_dataframe(periods=period_months * 30)
    forecast = model.predict(future)

    result_df = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]

    # Log forecast
    if log_results:
        log_path = os.path.join(LOG_DIR, f"forecast_{ticker}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
        result_df.to_csv(log_path, index=False)

    return result_df, None


# ---- Function: Intraday Heuristic Movement Forecast ----
def forecast_today_movement(ticker: str, log_results=True):
    df = yf.download(ticker, period="7d", interval="1h", auto_adjust=True)
    if df.empty or 'Close' not in df:
        return None, "No intraday data available"

    df['Return'] = df['Close'].pct_change()
    df['Trend'] = df['Return'].rolling(window=3).mean()
    latest_trend = df['Trend'].iloc[-1]
    latest_pct = df['Return'].iloc[-1] * 100 if not np.isnan(df['Return'].iloc[-1]) else 0

    if latest_trend > 0.001:
        signal = f"📈 Likely Uptrend Today ({latest_pct:.2f}%)"
    elif latest_trend < -0.001:
        signal = f"📉 Likely Downtrend Today ({latest_pct:.2f}%)"
    else:
        signal = f"🔄 Flat or Unclear Trend ({latest_pct:.2f}%)"

    # Log intraday signal
    if log_results:
        log_path = os.path.join(LOG_DIR, f"intraday_{ticker}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
        df.to_csv(log_path)

    return signal, None


# ---- Function: Auto Retrain Forecast Model ----
def auto_retrain_forecast_model(ticker: str):
    forecast_files = sorted(glob.glob(os.path.join(LOG_DIR, f"forecast_{ticker}_*.csv")))
    if not forecast_files:
        return None

    actual_df = yf.download(ticker, period="1y", interval="1d", auto_adjust=True)
    if actual_df.empty:
        return None

    actual_df = actual_df[['Close']].reset_index()
    actual_df.columns = ['ds', 'actual']
    eval_df = pd.DataFrame()

    for file in forecast_files:
        forecast_df = pd.read_csv(file)
        joined = pd.merge(forecast_df, actual_df, on='ds', how='inner')
        joined['error'] = joined['yhat'] - joined['actual']
        eval_df = pd.concat([eval_df, joined[['ds', 'yhat', 'actual', 'error']]])

    if eval_df.empty:
        return None

    mae = eval_df['error'].abs().mean()
    rmse = np.sqrt((eval_df['error'] ** 2).mean())
    accuracy = (np.sign(eval_df['yhat'].diff()) == np.sign(eval_df['actual'].diff())).mean()

    print(f"Retrain Evaluation — Ticker: {ticker}")
    print(f"MAE: {mae:.2f}, RMSE: {rmse:.2f}, Directional Accuracy: {accuracy:.2%}")

    # Retrain updated model and overwrite oldest file if needed (placeholder)
    # Could extend this to update params or select better model
    return eval_df


# ---- Self-triggering: Auto-retrain on every app run ----
def run_auto_retrain_all():
    tickers = set()
    for file in os.listdir(LOG_DIR):
        if file.startswith("forecast_"):
            parts = file.split("_")
            if len(parts) > 1:
                tickers.add(parts[1])

    for tkr in tickers:
        auto_retrain_forecast_model(tkr)


# ---- Run Auto Retrain at Import ----
run_auto_retrain_all()
