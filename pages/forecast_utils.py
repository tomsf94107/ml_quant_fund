#forecast v1.4 with the datetime fix applied to prevent the merge error, and all recent logic 
# including logging, forecasting, intraday prediction, and auto-retraining

import os
import pandas as pd
import numpy as np
from prophet import Prophet
from datetime import datetime, timedelta
import yfinance as yf
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Setup logs directory
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

# ---- Function: Auto Retrain Forecast Model (evaluate accuracy on past predictions) ----
def auto_retrain_forecast_model(ticker: str):
    # Load the latest forecast log for this ticker
    logs = [f for f in os.listdir(LOG_DIR) if f.startswith(f"forecast_{ticker}_")]
    if not logs:
        return

    latest_log = sorted(logs)[-1]
    forecast_df = pd.read_csv(os.path.join(LOG_DIR, latest_log))

    # Ensure ds is datetime for both forecast and actuals
    forecast_df['ds'] = pd.to_datetime(forecast_df['ds'])

    # Get actuals for overlapping range
    start = forecast_df['ds'].min()
    end = forecast_df['ds'].max()
    actuals = yf.download(ticker, start=start, end=end, auto_adjust=True)
    if actuals.empty or 'Close' not in actuals:
        return

    actual_df = actuals[['Close']].reset_index()
    actual_df.columns = ['ds', 'actual']
    actual_df['ds'] = pd.to_datetime(actual_df['ds'])

    # Merge forecast and actuals
    joined = pd.merge(forecast_df, actual_df, on='ds', how='inner')
    if joined.empty:
        return

    # Evaluate accuracy
    y_true = joined['actual']
    y_pred = joined['yhat']
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    print(f"[{ticker}] Forecast Evaluation — MAE: {mae:.2f}, MSE: {mse:.2f}, R²: {r2:.2f}")

# ---- Batch Auto Retrain (Optional for startup) ----
def run_auto_retrain_all(tickers=None):
    if tickers is None:
        tickers = ["AAPL", "MSFT", "GOOG", "TSLA"]
    for tkr in tickers:
        auto_retrain_forecast_model(tkr)
