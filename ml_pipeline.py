import yfinance as yf
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt

# Step 1: Download and flatten data
df = yf.download("AAPL", start="2018-01-01", end="2024-12-31")
df.columns = df.columns.droplevel(1)  # Remove ticker from MultiIndex

# Step 2: Feature engineering
df["Return_1D"] = df["Close"].pct_change()
df["Return_5D"] = df["Close"].pct_change(5)
df["SMA_10"] = df["Close"].rolling(10).mean()
df["SMA_50"] = df["Close"].rolling(50).mean()
df["SMA_ratio"] = df["SMA_10"] / df["SMA_50"]
df["Volatility"] = df["Return_1D"].rolling(10).std()
df["Volume_Change"] = df["Volume"].pct_change()

# RSI calculation
def compute_rsi(series, window=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window).mean()
    avg_loss = loss.rolling(window).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

df["RSI_14"] = compute_rsi(df["Close"])

# MACD calculation
ema_12 = df["Close"].ewm(span=12, adjust=False).mean()
ema_26 = df["Close"].ewm(span=26, adjust=False).mean()
df["MACD"] = ema_12 - ema_26
df["MACD_signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
df["MACD_hist"] = df["MACD"] - df["MACD_signal"]

# Step 3: Label creation
df["Target"] = (df["Return_1D"].shift(-1) > 0).astype(int)
df.dropna(inplace=True)

# Step 4: Prepare features and model
features = [
    "Return_5D", "SMA_ratio", "Volatility", "Volume_Change",
    "RSI_14", "MACD", "MACD_signal", "MACD_hist"
]
X = df[features]
y = df["Target"]
X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False, test_size=0.3)

model = XGBClassifier(use_label_encoder=False, eval_metric="logloss")
model.fit(X_train, y_train)

# Step 5: Evaluate performance
y_pred = model.predict(X_test)
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Accuracy Score:", accuracy_score(y_test, y_pred))

# Step 6: Backtest simulation
df_bt = df.iloc[-len(y_test):].copy()
df_bt["Predicted"] = y_pred
df_bt["Strategy_Return"] = df_bt["Return_1D"] * df_bt["Predicted"]
df_bt["Cumulative_Market"] = (1 + df_bt["Return_1D"]).cumprod()
df_bt["Cumulative_Strategy"] = (1 + df_bt["Strategy_Return"]).cumprod()

# Step 7: Plot
plt.figure(figsize=(12, 6))
plt.plot(df_bt.index, df_bt["Cumulative_Market"], label="Market")
plt.plot(df_bt.index, df_bt["Cumulative_Strategy"], label="Strategy")
plt.title("Equity Curve: Market vs ML Strategy")
plt.xlabel("Date")
plt.ylabel("Cumulative Return")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
