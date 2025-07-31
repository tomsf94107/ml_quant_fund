# ────────────────────────────────────────────────────────────────────────────
#  v17.9  •  Added sys.path fix for Streamlit import issues
# ────────────────────────────────────────────────────────────────────────────
import os, sys, io, zipfile, base64, tempfile
from dotenv import load_dotenv
load_dotenv()

# 🔧 NEW FIX — ensure root path is added for module imports in Streamlit
dir_above = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
if dir_above not in sys.path:
    sys.path.append(dir_above)

# ----- numeric / plotting ----------------------------------------------------
import numpy as np
# ----- compatibility shims for NumPy ≥2.0 -------------------------------
if not hasattr(np, "bool"): np.bool = np.bool_   # already in your file
if not hasattr(np, "int"):  np.int  = int        # ← add this line

import pandas as pd
import matplotlib.pyplot as plt
import altair as alt
import shap
from sklearn.metrics import accuracy_score
from datetime import datetime, date
from core.helpers_xgb import train_xgb_predict  

# ----- web / app -------------------------------------------------------------
import streamlit as st
from streamlit_autorefresh import st_autorefresh

# ----- email -----------------------------------------------------------------
import smtplib
from email.mime.text import MIMEText

# ----- G-Sheets --------------------------------------------------------------
import gspread
from oauth2client.service_account import ServiceAccountCredentials

# ----- your core utils -------------------------------------------------------
from forecast_utils import (
    build_feature_dataframe,
    forecast_price_trend,
    forecast_today_movement,
    auto_retrain_forecast_model,
    compute_rolling_accuracy,
    get_latest_forecast_log,
    run_auto_retrain_all,
)

# ──────────────────────────  AUTH  ───────────────────────────────────────────
def check_login():
    pwd = st.text_input("Enter password:", type="password")
    if pwd != st.secrets.get("app_password", "MlQ@nt@072025"):
        st.stop()
check_login()

# ──────────────────────────  EMAIL  ──────────────────────────────────────────
def send_alert_email(ticker: str, prob: float):
    try:
        msg = MIMEText(f"High-confidence BUY signal for {ticker} (p={prob:.2f})")
        msg["Subject"] = f"Trading Alert · {ticker}"
        msg["From"]    = os.getenv("EMAIL_SENDER")
        msg["To"]      = os.getenv("EMAIL_RECEIVER")
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as s:
            s.login(os.getenv("EMAIL_SENDER"), os.getenv("EMAIL_PASSWORD"))
            s.send_message(msg)
    except Exception as e:
        st.error(f"Email failed: {e}")


# ──────────────────────────  SHAP VIS  ───────────────────────────────────────
def plot_shap(model, X):
    """
    Safe SHAP plot for XGBoost models.

    • works with XGBRegressor / XGBClassifier
    • avoids utf-8 decode errors by passing a callable (model.predict)
    """
    try:
        # ── 0. sanity checks ────────────────────────────────────────────
        if X is None or len(X) == 0:
            st.warning("⚠️ No rows for SHAP.")
            return

        # keep only numeric, finite values
        X_num = (
            X.select_dtypes(include=[np.number])
              .replace([np.inf, -np.inf], np.nan)
              .dropna()
              .astype("float64")
        )
        if X_num.empty:
            st.warning("⚠️ No valid numeric features for SHAP.")
            return

        # ── 1. build explainer with callable ----------------------------
        # use up to 100 rows of background to keep it fast
        bg = shap.sample(X_num, min(100, len(X_num)), random_state=0)

        explainer   = shap.Explainer(model.predict, bg)   # <– note the callable
        shap_values = explainer(X_num)

        # ── 2. bar summary plot ----------------------------------------
        st.subheader("🔍 SHAP Feature Importance")
        fig = plt.figure()
        shap.plots.bar(shap_values, max_display=10, show=False)
        st.pyplot(fig)
        plt.clf()

    except Exception as e:
        st.error(f"❌ SHAP failed: {e}")

# ──────────────────────────  TICKER LIST  ────────────────────────────────────
def load_forecast_tickers():
    return (open("tickers.csv").read().splitlines()
            if os.path.exists("tickers.csv") else ["AAPL", "MSFT"])

def save_forecast_tickers(lst):
    with open("tickers.csv", "w") as f:
        for t in lst:
            f.write(t.strip().upper() + "\n")

# ──────────────────────────  G-SHEET ACCURACY  ───────────────────────────────
def load_accuracy_log_from_gsheet():
    try:
        scope = [
            "https://spreadsheets.google.com/feeds",
            "https://www.googleapis.com/auth/drive",
        ]
        creds = ServiceAccountCredentials.from_json_keyfile_dict(
            st.secrets["gcp_service_account"], scope
        )
        sheet = gspread.authorize(creds).open("forecast_evaluation_log").sheet1
        return pd.DataFrame(sheet.get_all_records())
    except Exception as e:
        st.error(f"⚠️ G-Sheet load failed: {e}")
        return pd.DataFrame()

# ──────────────────────────  UI Config   ────────────────────────────────────

st_autorefresh(interval=5 * 60 * 1000, key="auto-refresh")
st.title("📈 ML-Based Stock Strategy Dashboard")
st.caption(f"🕒 Last updated {datetime.now():%Y-%m-%d %H:%M:%S}")

# ──────────────────────────  SIDEBAR  ────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📆 Date Range")
    start_date = st.date_input("Start date", value=date(2025, 3, 1))
    end_date   = st.date_input("End date",   value=date(2025, 7, 20))

    st.markdown("## 🧠 Forecasting Model")
    model_choice = st.radio(
        "Select Model",
        ["XGBoost (Short Term) [Recommended]", "Prophet (Long Term)"], 0
    )

    st.markdown("## 🛠️ Data Tools")
    if st.button("⚙️ Populate All Forecast Logs"):
        run_auto_retrain_all()
        st.success("✅ Logs populated!")

    tickers              = st.text_input("Tickers (comma-sep)", "AAPL,MSFT").upper().split(",")
    confidence_threshold = st.slider("Confidence threshold", 0.5, 0.99, 0.7)
    enable_email         = st.toggle("📧 Email alerts", True)
    enable_shap          = st.toggle("🔍 SHAP explainability", True)
    enable_zip_download  = st.toggle("📦 ZIP of results", True)

    with st.expander("📋 Manage ticker list"):
        txt = st.text_area("One ticker per line", "\n".join(load_forecast_tickers()), height=140)
        if st.button("💾 Save tickers"):
            save_forecast_tickers(txt.splitlines())
            st.success("Saved.")

# ═════════════════════════  FORECAST SECTION  ════════════════════════════════

with st.expander("🗕️ Forecast Price Trends"):
    tkr_in        = st.text_input("Enter a ticker", "AAPL")
    forecast_days = st.slider("📅 Horizon (days)", 1, 90, 15)
    use_prophet   = model_choice.startswith("Prophet") or forecast_days > 30

    if tkr_in and st.button("Run Forecast"):
        tkr = tkr_in.upper()
        err = None                      # <- always defined

        # ── 1. Generate forecast_df ─────────────────────────────────────────
        if use_prophet:
            forecast_df, err   = forecast_price_trend(
                tkr, period_months=int(forecast_days / 30)
            )
            model_used = "Prophet"

        else:   # XGBoost short-term path
            base_df = build_feature_dataframe(tkr, start=start_date, end=end_date)
            try:
                _, _, _, y_pred, _ = train_xgb_predict(
                    base_df, horizon_days=forecast_days
                )

                # recent 60-day history for context
                recent = (
                    base_df[["Close"]]
                    .tail(60)
                    .reset_index()
                    .rename(columns={"Date": "ds", "Close": "actual"})
                )

                futr = pd.DataFrame(
                    {
                        "ds":   pd.date_range(datetime.today(), periods=len(y_pred)),
                        "yhat": y_pred,
                        "yhat_lower": np.nan,
                        "yhat_upper": np.nan,
                        "actual":     np.nan,
                    }
                )

                forecast_df = pd.concat([recent, futr], ignore_index=True)
                model_used  = "XGBoost"

            except Exception as e:
                st.error(f"❌ XGBoost failed: {e}")
                st.stop()

        # ── 2. Display / chart ─────────────────────────────────────────────
        if err:
            st.warning(err)

        elif forecast_df.empty:
            st.warning("⚠️ Empty forecast dataframe.")

        else:
            st.subheader(
                f"📊 {forecast_days}-Day Price Forecast  •  {model_used}"
            )

            # split for styling
            hist = forecast_df[forecast_df["actual"].notna()]
            futr = forecast_df[forecast_df["actual"].isna()]

            hist_line = (
                alt.Chart(hist)
                .mark_line(color="lightgrey", strokeDash=[4, 4])
                .encode(x="ds:T", y="actual:Q", tooltip=["ds:T", "actual:Q"])
            )

            fut_line = (
                alt.Chart(futr)
                .mark_line(color="#1f77b4", size=2.5)
                .encode(x="ds:T", y="yhat:Q", tooltip=["ds:T", "yhat:Q"])
            )

            conf_band = (
                alt.Chart(futr)
                .mark_area(color="lightsteelblue", opacity=0.25)
                .encode(x="ds:T", y="yhat_lower:Q", y2="yhat_upper:Q")
            )

            st.altair_chart(
                (hist_line + conf_band + fut_line)
                .properties(title=f"{tkr} – {forecast_days}-Day Projection")
                .interactive(),
                use_container_width=True,
            )

            # ── 3. Intraday / ML movement ────────────────────────────────
            
            st.subheader("🗓️ Today's Movement Prediction")
            move_msg, move_err = forecast_today_movement(
                tkr, start=start_date, end=end_date
            )

            if move_err:               # ✅ explicit branch – nothing returned
                st.warning(move_err)
            else:
                st.success(move_msg)            

# ═════════════════════════  STRATEGY SECTION  ════════════════════════════════
if "live_signals" not in st.session_state:
    st.session_state["live_signals"] = {}

if st.button("🚀 Run Strategy"):

    # ---- dashboard of last run ---------------------------------------------
    st.subheader("📱 Live Signals Dashboard")
    for k,v in st.session_state["live_signals"].items():
        sig = "🟢 BUY" if v["signal"] else "🔴 HOLD"
        st.markdown(f"**{k}** → {sig} ({v['confidence']*100:.1f}%)")

    csv_buffers = []
    # ─── per-ticker loop ──────────────────────────────────────────────
    for raw in tickers:
        tkr = raw.strip().upper()
        if not tkr: continue
        st.subheader(f"📊 {tkr} Strategy")

        try:
            # -------- data + model -----------------------------------
            df = build_feature_dataframe(tkr, start=start_date, end=end_date)
            model, X_test, y_test, y_pred, y_prob = train_xgb_predict(df)
            if y_prob is None: y_prob = [np.nan]*len(y_pred)
            if y_test is None or len(y_test)==0:
                st.warning("⚠️ Model returned no predictions."); continue

            # -------- results frame ----------------------------------    
            df_test = df.iloc[-len(y_test):].copy()
            df_test["Prob"]   = y_prob
            df_test["Signal"] = (df_test["Prob"] > confidence_threshold).astype(int)
            df_test["Strategy"] = df_test["Signal"].shift(1) * df_test["Return_1D"]
            df_test["Market"]   = df_test["Return_1D"]
            df_test.dropna(subset=["Strategy","Market"], inplace=True)
            df_test[["Strategy","Market"]] = (1+df_test[["Strategy","Market"]]).cumprod()

            # ---- metrics ---------------------------------------------------
            y_dir_true = (y_test.diff()>0).astype(int).iloc[1:]
            y_dir_pred = (pd.Series(y_pred,index=y_test.index).diff()>0).astype(int).iloc[1:]
            acc  = accuracy_score(y_dir_true, y_dir_pred)
            strat_ret = df_test["Strategy"].pct_change()
            sharpe = np.sqrt(252)*strat_ret.mean()/strat_ret.std() if strat_ret.std() else np.nan
            mdd   = ((df_test["Strategy"]/df_test["Strategy"].cummax())-1).min()
            cagr  = (df_test["Strategy"].iloc[-1]/df_test["Strategy"].iloc[0])**(252/len(df_test))-1

            st.metric("Accuracy", f"{acc:.2f}")
            st.metric("Sharpe",   f"{sharpe:.2f}" if not np.isnan(sharpe) else "nan")
            st.metric("Max DD",   f"{mdd:.2%}")
            st.metric("CAGR",     f"{cagr:.2%}")

            # ---- plot ------------------------------------------------------
            if {"Strategy","Market"}.issubset(df_test.columns) and not df_test.empty:
                st.line_chart(df_test[["Strategy","Market"]])
            else:
                st.warning("⚠️ Strategy series empty – insufficient rows.")

            # ---- downloads -------------------------------------------------
            csv_bytes = df_test.to_csv(index=False).encode()
            st.download_button(f"🗅 CSV – {tkr}", csv_bytes,
                               file_name=f"{tkr}_strategy.csv", mime="text/csv")
            csv_buffers.append((f"{tkr}_strategy.csv", csv_bytes))

            # ---- email -----------------------------------------------------
            last = df_test.iloc[-1]
            if enable_email and last["Signal"]==1 and pd.notna(last["Prob"]) \
               and last["Prob"] > confidence_threshold:
                send_alert_email(tkr, float(last["Prob"]))

            # ---- SHAP ------------------------------------------------------
            if enable_shap: plot_shap(model, X_test)

            # ---- live dashboard state -------------------------------------
            st.session_state["live_signals"][tkr] = {
                "signal": int(last["Signal"]),
                "confidence": float(last["Prob"]) if pd.notna(last["Prob"]) else 0.0
            }

        except Exception as e:
            st.error(f"⚠️ {tkr}: {e}")

    # ---- ZIP download ------------------------------------------------------
    if enable_zip_download and csv_buffers:
        zbuf = io.BytesIO()
        with zipfile.ZipFile(zbuf, "w", zipfile.ZIP_DEFLATED) as zf:
            for fname, data in csv_buffers: zf.writestr(fname, data)
        st.download_button("📦 Download ALL as ZIP",
                           zbuf.getvalue(),
                           file_name="strategy_exports.zip",
                           mime="application/zip")

# ═════════════════════════  ACCURACY DASHBOARD  ═════════════════════════════
st.subheader("📊 Forecast Accuracy Dashboard")
acc_df = load_accuracy_log_from_gsheet()
if acc_df.empty:
    st.warning("No accuracy data found.")
else:
    acc_df["timestamp"] = pd.to_datetime(acc_df["timestamp"])
    st.dataframe(acc_df.sort_values("timestamp", ascending=False))
    st.line_chart(acc_df.set_index("timestamp")[["mae","mse","r2"]])
