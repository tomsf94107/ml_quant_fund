# forecast_utils.py — v1.7
import os
import pandas as pd
import numpy as np
from prophet import Prophet
from datetime import datetime, timedelta
import yfinance as yf
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

LOG_DIR = "forecast_logs"
EVAL_DIR = "forecast_eval"
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(EVAL_DIR, exist_ok=True)

# ---- Forecast 3-Month Trend ----
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

    actuals = df[['Close']].reset_index()
    actuals.columns = ['ds', 'actual']
    actuals['ds'] = pd.to_datetime(actuals['ds'])
    result_df = pd.merge(result_df, actuals, on='ds', how='left')

    if log_results:
        log_path = os.path.join(LOG_DIR, f"forecast_{ticker}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
        result_df.to_csv(log_path, index=False)

    return result_df, None

# ---- Intraday Movement Signal ----
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

    if log_results:
        log_path = os.path.join(LOG_DIR, f"intraday_{ticker}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
        df.to_csv(log_path)

    return signal, None

# ---- Forecast Accuracy Evaluation ----
def auto_retrain_forecast_model(ticker: str):
    logs = [f for f in os.listdir(LOG_DIR) if f.startswith(f"forecast_{ticker}_")]
    if not logs:
        print(f"⏭️ Skipping {ticker} — no forecast logs.")
        return

    latest_log = sorted(logs)[-1]
    forecast_df = pd.read_csv(os.path.join(LOG_DIR, latest_log))
    forecast_df['ds'] = pd.to_datetime(forecast_df['ds'])

    start = forecast_df['ds'].min()
    end = forecast_df['ds'].max()
    actuals = yf.download(ticker, start=start, end=end, auto_adjust=True)
    if actuals.empty or 'Close' not in actuals:
        print(f"⚠️ Skipping {ticker} — no actual price data.")
        return

    actual_df = actuals[['Close']].reset_index()
    actual_df.columns = ['ds', 'actual']
    actual_df['ds'] = pd.to_datetime(actual_df['ds'])

    joined = pd.merge(forecast_df, actual_df, on='ds', how='inner')
    if joined.empty:
        print(f"⚠️ Skipping {ticker} — no overlap between forecast and actuals.")
        return

    y_true = joined['actual']
    y_pred = joined['yhat']
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    print(f"[{ticker}] Forecast Evaluation — MAE: {mae:.2f}, MSE: {mse:.2f}, R²: {r2:.2f}")

    eval_path = os.path.join(EVAL_DIR, "forecast_evals.csv")
    row = pd.DataFrame([{
        "ticker": ticker,
        "timestamp": datetime.now(),
        "mae": mae,
        "mse": mse,
        "r2": r2
    }])

    if os.path.exists(eval_path):
        row.to_csv(eval_path, mode="a", header=False, index=False)
    else:
        row.to_csv(eval_path, index=False)

# ---- Rolling Direction Accuracy ----
def compute_rolling_accuracy(log_path):
    df = pd.read_csv(log_path, parse_dates=['ds']).sort_values('ds')
    df['pred_direction'] = df['yhat'].diff().apply(lambda x: 1 if x > 0 else -1)
    df['actual_direction'] = df['actual'].diff().apply(lambda x: 1 if x > 0 else -1)
    df['correct'] = df['pred_direction'] == df['actual_direction']
    df['7d_accuracy'] = df['correct'].rolling(window=7).mean()
    df['30d_accuracy'] = df['correct'].rolling(window=30).mean()
    return df[['ds', '7d_accuracy', '30d_accuracy', 'correct']]

# ---- Batch Retrain All ----
def run_auto_retrain_all(ticker_list=None):
    if ticker_list is None:
        ticker_list = ["AAPL", "MSFT"]
    for tkr in ticker_list:
        try:
            print(f"🔁 Retraining: {tkr}")
            auto_retrain_forecast_model(tkr)
        except Exception as e:
            print(f"❌ Error retraining {tkr}: {e}")
