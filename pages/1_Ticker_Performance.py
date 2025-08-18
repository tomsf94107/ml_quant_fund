# ────────────────────────────────────────────────────────────────────────────
#  v18.5  •  added risk event calendar + risk-aware signals
# ────────────────────────────────────────────────────────────────────────────

# ── Path bootstrap (ensure parent of repo is importable) ────────────────────
import os, sys
_THIS   = os.path.abspath(__file__)
_REPO   = os.path.dirname(os.path.dirname(_THIS))   # …/ml_quant_fund
_PARENT = os.path.dirname(_REPO)                    # …/
if _PARENT not in sys.path:
    sys.path.insert(0, _PARENT)

# stdlib / third-party
import io, zipfile, glob
from dotenv import load_dotenv
load_dotenv()

import numpy as np
# NumPy ≥2.0 shims (safe no-ops on <2.0)
if not hasattr(np, "bool"): np.bool = np.bool_
if not hasattr(np, "int"):  np.int  = int

import pandas as pd
import matplotlib.pyplot as plt
import altair as alt
from sklearn.metrics import accuracy_score
from datetime import datetime, date, timedelta

import streamlit as st
from streamlit_autorefresh import st_autorefresh
import smtplib
from email.mime.text import MIMEText

# ── Project imports (package style ONLY) ─────────────────────────────────────
from ml_quant_fund.forecast_utils import (
    build_feature_dataframe,
    forecast_price_trend,
    forecast_today_movement,
    plot_shap,
    auto_retrain_forecast_model,
    compute_rolling_accuracy,
    get_latest_forecast_log,
    run_auto_retrain_all,
    load_forecast_accuracy,
)
from ml_quant_fund.core.helpers_xgb import train_xgb_predict, RISK_ALPHA


# ──────────────────────────  IMPORTANCES TAB  ───────────────────────────────
importances_dir = "charts"
models_dir      = "models"

def show_importances_tab():
    st.title("📈 Feature Importances Over Time")
    img_path = os.path.join(importances_dir, "importances_over_time.png")

    if os.path.exists(img_path):
        st.image(img_path, caption="7-Day Rolling Feature Importances", use_column_width=True)
        last_mod = os.path.getmtime(img_path)
        st.markdown(f"**Last updated:** {datetime.fromtimestamp(last_mod):%Y-%m-%d %H:%M}")
    else:
        st.warning(f"No chart found at `{img_path}`.")
        uploaded_file = st.file_uploader("Upload a feature importances chart (PNG/JPG):",
                                         type=["png", "jpg", "jpeg"])
        if uploaded_file and st.button("Save chart to disk"):
            os.makedirs(importances_dir, exist_ok=True)
            with open(img_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            st.success(f"Chart saved to `{img_path}`!")

    # surface top-5 features from latest summary
    summary_files = sorted(glob.glob(os.path.join(models_dir, "feature_importances_summary_*.csv")))
    root_summary  = os.path.join(models_dir, "feature_importances_summary.csv")
    if not summary_files and os.path.exists(root_summary):
        summary_files = [root_summary]

    if summary_files:
        latest = summary_files[-1]
        try:
            summary_df = pd.read_csv(latest, index_col=0)
            mean_imp   = summary_df.mean(axis=0).sort_values(ascending=False)
            top5       = mean_imp.head(5)
            st.markdown("**Top 5 Features (by mean importance):**")
            cols = st.columns(len(top5))
            for col, (feat, val) in zip(cols, top5.items()):
                col.metric(label=feat, value=f"{val:.3f}")
        except Exception:
            st.warning("Could not load feature-importances summary for top features.")


# ──────────────────────────  AUTH (optional)  ───────────────────────────────
def check_login():
    if st.session_state.get("auth_ok"):
        return
    pwd = st.text_input("Enter password:", type="password")
    if pwd == "":
        st.stop()
    if pwd != st.secrets.get("app_password", "MlQ@nt@072025"):
        st.error("❌ Wrong password")
        st.stop()
    st.session_state["auth_ok"] = True


# ──────────────────────────  EMAIL  ─────────────────────────────────────────
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


# ──────────────────────────  SHAP VIS  ──────────────────────────────────────
def plot_shap(model, X):
    try:
        import shap
        if X is None or len(X) == 0:
            st.warning("⚠️ No rows for SHAP.")
            return
        X_num = (
            X.select_dtypes(include=[np.number])
             .replace([np.inf, -np.inf], np.nan)
             .dropna()
             .astype("float64")
        )
        if X_num.empty:
            st.warning("⚠️ No valid numeric features for SHAP.")
            return
        bg = shap.sample(X_num, min(100, len(X_num)), random_state=0)
        explainer   = shap.Explainer(model.predict, bg)
        shap_values = explainer(X_num)
        st.subheader("🔍 SHAP Feature Importance")
        fig = plt.figure()
        shap.plots.bar(shap_values, max_display=10, show=False)
        st.pyplot(fig)
        plt.clf()
    except Exception as e:
        st.error(f"❌ SHAP failed: {e}")


# ──────────────────────────  TICKER LIST  ───────────────────────────────────
def load_forecast_tickers():
    return (open("tickers.csv").read().splitlines()
            if os.path.exists("tickers.csv") else ["AAPL", "MSFT"])

def save_forecast_tickers(lst):
    with open("tickers.csv", "w") as f:
        for t in lst:
            f.write(t.strip().upper() + "\n")


# ──────────────────────────  UI CONFIG  ─────────────────────────────────────
st_autorefresh(interval=5 * 60 * 1000, key="auto-refresh")

pages = ["Dashboard", "Importances Over Time"]
page = st.sidebar.radio("Go to", pages)
if page == "Importances Over Time":
    show_importances_tab()
    st.stop()

st.title("📈 ML-Based Stock Strategy Dashboard")
st.caption(f"🕒 Last updated {datetime.now():%Y-%m-%d %H:%M:%S}")

# ──────────────────────────  SIDEBAR  ───────────────────────────────────────
with st.sidebar:
    if page == "Dashboard":
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
            run_auto_retrain_all(load_forecast_tickers())
            st.success("✅ Logs populated!")

        tickers              = st.text_input("Tickers (comma-sep)", "AAPL,MSFT").upper().split(",")
        confidence_threshold = st.slider("Confidence threshold", 0.5, 0.99, 0.79)
        enable_email         = st.toggle("📧 Email alerts", True)
        enable_shap          = st.toggle("🔍 SHAP explainability", True)
        enable_zip_download  = st.toggle("📦 ZIP of results", True)

        with st.expander("📋 Manage ticker list"):
            txt = st.text_area("One ticker per line", "\n".join(load_forecast_tickers()), height=140)
            if st.button("💾 Save tickers"):
                save_forecast_tickers(txt.splitlines())
                st.success("Saved.")

        st.markdown("## 📈 Accuracy Filters")
        options  = load_forecast_tickers()
        selected = st.multiselect("Filter by ticker", options, default=options)
        st.session_state["acc_ticker_filter"] = selected

        st.markdown("## 🛡️ Risk Gate")
        block_tau = st.slider("Block entries when risk_next_3d ≥", 0, 6, 3, 1)

# ──────────────────────────  RISK BADGE (from calendar page)  ───────────────
risk_info = st.session_state.get("event_risk_next72")
if risk_info:
    st.metric("Next 72h Event Risk", f'{risk_info["label"]} ({risk_info["score"]})')
risk_mult = {"Low": 1.00, "Medium": 0.92, "High": 0.85}.get(
    risk_info["label"], 1.00
) if risk_info else 1.00

# ──────────────────────────  FORECAST SECTION  ───────────────────────────────
with st.expander("🗕️ Forecast Price Trends"):
    tkr_in        = st.text_input("Enter a ticker", "AAPL")
    forecast_days = st.slider("📅 Horizon (days)", 1, 90, 15)
    use_prophet   = model_choice.startswith("Prophet") or forecast_days > 30

    if tkr_in and st.button("Run Forecast"):
        tkr = tkr_in.upper()
        err = None

        if use_prophet:
            forecast_df, err = forecast_price_trend(
                tkr, period_months=int(forecast_days / 30)
            )
            model_used = "Prophet"
        else:
            base_df = build_feature_dataframe(tkr, start=start_date, end=end_date)
            try:
                _, _, _, y_pred, _ = train_xgb_predict(
                    base_df, horizon_days=forecast_days
                )
                recent = (
                    base_df[["Close"]]
                    .tail(60)
                    .reset_index()
                    .rename(columns={"Date": "ds", "Close": "actual"})
                )
                futr = pd.DataFrame({
                    "ds":          pd.date_range(datetime.today(), periods=len(y_pred)),
                    "yhat":        y_pred,
                    "yhat_lower":  np.nan,
                    "yhat_upper":  np.nan,
                    "actual":      np.nan,
                })
                forecast_df = pd.concat([recent, futr], ignore_index=True)
                model_used  = "XGBoost"
            except Exception as e:
                st.error(f"❌ XGBoost failed: {e}")
                st.stop()

        if err:
            st.warning(err)
        elif forecast_df.empty:
            st.warning("⚠️ Empty forecast dataframe.")
        else:
            st.subheader(f"📊 {forecast_days}-Day Price Forecast  •  {model_used}")
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

            st.subheader("🗓️ Today's Movement Prediction")
            move_msg, move_err = forecast_today_movement(tkr, start=start_date, end=end_date)
            if move_err:
                st.warning(move_err)
            else:
                st.success(move_msg)

# ──────────────────────────  STRATEGY SECTION  ───────────────────────────────
if "live_signals" not in st.session_state:
    st.session_state["live_signals"] = {}

if st.button("🚀 Run Strategy"):
    st.subheader("📱 Live Signals Dashboard")
    for k, v in st.session_state["live_signals"].items():
        sig = "🟢 BUY" if v["signal"] else "🔴 HOLD"
        st.markdown(f"**{k}** → {sig} ({v['confidence']*100:.1f}%)")
    csv_buffers = []

    for raw in tickers:
        tkr = raw.strip().upper()
        if not tkr:
            continue

        st.subheader(f"📊 {tkr} Strategy")
        try:
            # -------- data --------------------------------------------
            df = build_feature_dataframe(tkr, start=start_date, end=end_date)

            # ---- risk diagnostics -----------------------------------
            risk_cols = [c for c in ["risk_today","risk_next_1d","risk_next_3d","risk_prev_1d"] if c in df.columns]
            with st.expander("🛡️ Risk diagnostics", expanded=False):
                if risk_cols:
                    nz = (df[risk_cols] != 0).mean().rename("nonzero_frac").round(3)
                    c1, c2 = st.columns([1,1])
                    c1.write("Non-zero fraction by column")
                    c1.dataframe(nz.to_frame(), use_container_width=True)
                    c2.write("Last 10 risk rows")
                    c2.dataframe(df[risk_cols].tail(10), use_container_width=True)
                else:
                    st.info("No risk columns present on the training dataframe.")
                st.caption(f"Training down-weight alpha: {RISK_ALPHA}")

            # -------- model -------------------------------------------
            model, X_test, y_test, y_pred, y_prob = train_xgb_predict(df)

            # Build confidence proxy if helper didn't return one
            if y_prob is None and y_pred is not None and X_test is not None and len(y_pred) == len(X_test):
                pred = pd.Series(y_pred, index=X_test.index)
                close_now = X_test["Close"]
                pred_ret  = (pred - close_now) / close_now
                atr = df.loc[X_test.index, "ATR"] if "ATR" in df.columns else pd.Series(0.02, index=X_test.index)
                vol = (atr / close_now).replace([np.inf, -np.inf], np.nan)
                vol = vol.fillna(atr.median() / max(1e-8, float(close_now.median())))
                k = 3.0
                z = pred_ret / vol.replace(0, np.nan).fillna(vol.median())
                y_prob = (1.0 / (1.0 + np.exp(-k * z))).clip(0.0, 1.0).values

            if y_prob is None:
                y_prob = np.full(len(y_pred), np.nan)

            if y_test is None or len(y_test) == 0:
                st.warning("⚠️ Model returned no predictions.")
                continue

            # -------- results frame ----------------------------------
            df_test = df.iloc[-len(y_test):].copy()

            # optional block gate by event risk
            gate = (df_test.get("risk_next_3d", pd.Series(0, index=df_test.index)) >= block_tau)
            df_test["GateBlock"] = gate.astype(int)

            df_test["Prob"]     = y_prob
            df_test["Prob_eff"] = df_test["Prob"] * risk_mult  # global 72h multiplier
            # block entries on high upcoming risk
            df_test["Signal"]   = ((df_test["Prob_eff"] > confidence_threshold) & (~gate)).astype(int)

            df_test["Strategy"] = df_test["Signal"].shift(1) * df_test["Return_1D"]
            df_test["Market"]   = df_test["Return_1D"]
            df_test.dropna(subset=["Strategy","Market"], inplace=True)
            df_test[["Strategy","Market"]] = (1 + df_test[["Strategy","Market"]]).cumprod()

            # ---- metrics --------------------------------------------
            y_dir_true = (y_test.diff() > 0).astype(int).iloc[1:]
            y_dir_pred = (pd.Series(y_pred, index=y_test.index).diff() > 0).astype(int).iloc[1:]
            acc  = accuracy_score(y_dir_true, y_dir_pred)
            strat_ret = df_test["Strategy"].pct_change()
            sharpe = np.sqrt(252) * strat_ret.mean() / strat_ret.std() if strat_ret.std() else np.nan
            mdd   = ((df_test["Strategy"] / df_test["Strategy"].cummax()) - 1).min()
            cagr  = (df_test["Strategy"].iloc[-1] / df_test["Strategy"].iloc[0]) ** (252 / len(df_test)) - 1

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Accuracy", f"{acc:.2f}")
            c2.metric("Sharpe",   f"{sharpe:.2f}" if not np.isnan(sharpe) else "nan")
            c3.metric("Max DD",   f"{mdd:.2%}")
            c4.metric("CAGR",     f"{cagr:.2%}")

            # ---- plot ------------------------------------------------
            if {"Strategy","Market"}.issubset(df_test.columns) and not df_test.empty:
                st.line_chart(df_test[["Strategy","Market"]])
            else:
                st.warning("⚠️ Strategy series empty – insufficient rows.")

            # ---- downloads ------------------------------------------
            csv_bytes = df_test.to_csv(index=False).encode()
            st.download_button(f"🗅 CSV – {tkr}", csv_bytes,
                               file_name=f"{tkr}_strategy.csv", mime="text/csv")
            csv_buffers.append((f"{tkr}_strategy.csv", csv_bytes))

            # ---- email ----------------------------------------------
            if (
                enable_email
                and df_test.iloc[-1]["Signal"] == 1
                and pd.notna(df_test.iloc[-1]["Prob_eff"])
                and df_test.iloc[-1]["Prob_eff"] > confidence_threshold
            ):
                send_alert_email(tkr, float(df_test.iloc[-1]["Prob_eff"]))

            # ---- SHAP -----------------------------------------------
            if enable_shap:
                plot_shap(model, X_test)

            # ---- live dashboard state -------------------------------
            eff_conf = df_test.iloc[-1].get("Prob_eff", df_test.iloc[-1]["Prob"])
            st.session_state["live_signals"][tkr] = {
                "signal": int(df_test.iloc[-1]["Signal"]),
                "confidence": float(eff_conf) if pd.notna(eff_conf) else 0.0
            }

        except Exception as e:
            st.error(f"⚠️ {tkr}: {e}")

    # ---- ZIP download ------------------------------------------------------
    if enable_zip_download and csv_buffers:
        zbuf = io.BytesIO()
        with zipfile.ZipFile(zbuf, "w", zipfile.ZIP_DEFLATED) as zf:
            for fname, data in csv_buffers:
                zf.writestr(fname, data)
        st.download_button("📦 Download ALL as ZIP",
                           zbuf.getvalue(),
                           file_name="strategy_exports.zip",
                           mime="application/zip")

# ═════════════════════════  📊 ACCURACY DASHBOARD  ═════════════════════════════

st.subheader("📊 Forecast Accuracy Dashboard")

# Option 1: pass DB URL from Streamlit into the util (no Streamlit import in forecast_utils.py)
db_url = st.secrets.get("accuracy_db_url", "sqlite:///forecast_accuracy.db")
acc_df = load_forecast_accuracy(db_url)

if acc_df.empty:
    st.info("No accuracy data found yet.")
else:
    # --- Normalize dtypes & sort ---
    acc_df["timestamp"] = pd.to_datetime(acc_df["timestamp"], errors="coerce")
    for c in ["mae", "mse", "r2"]:
        if c in acc_df.columns:
            acc_df[c] = pd.to_numeric(acc_df[c], errors="coerce")
    acc_df = acc_df.sort_values("timestamp")

    # --- Ticker filter with session persistence ---
    default_sel = st.session_state.get("acc_ticker_filter", [])
    sel = st.multiselect(
        "Filter tickers",
        options=sorted(acc_df["ticker"].dropna().unique().tolist()),
        default=default_sel,
    )
    if sel:
        acc_df = acc_df[acc_df["ticker"].isin(sel)]
        st.session_state["acc_ticker_filter"] = sel

    # --- Summary metrics ---
    c1, c2, c3 = st.columns(3)
    c1.metric("Avg MAE", f"{acc_df['mae'].mean(skipna=True):.3f}")
    c2.metric("Avg MSE", f"{acc_df['mse'].mean(skipna=True):.3f}")
    c3.metric("Avg R²",  f"{acc_df['r2'].mean(skipna=True):.3f}")

    # --- Line chart (guard against empty) ---
    chart_df = acc_df.set_index("timestamp")[["mae", "mse", "r2"]].dropna(how="all")
    if not chart_df.empty:
        st.line_chart(chart_df)
    else:
        st.warning("No numeric accuracy data to plot yet.")
