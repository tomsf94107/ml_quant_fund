# ────────────────────────────────────────────────────────────────────────────
#  v19.4  •  More robust insider UI:
#           - yfinance MultiIndex flattening + slicing by ticker
#           - Safer truth-value checks for Altair layers (None checks)
#           - Guarded forecast concat to quiet pandas FutureWarning
#  v19.3  •  Robust price normalization (handles MultiIndex columns from yf)
#           - Normalizes price df to ['ticker','date','close','shares_outstanding','market_cap']
#           - Fixes insider section crashing when px had ('Close','TICKER') style columns
#  v19.2  •  Accuracy Over Time section (rolling) + robust insider (SQLite)
#           - Local DB rollups merged: insider_7d/21d → ins_net_shares_7d_db/_21d_db
#           - KPI cards + tooltips show DB values; safer insider loading
#           - Minor fixes: MIMEText import, defensive col-picking
#  v19.1  •  Insider data from local SQLite (loader.insider_loader)
#  v19.0  •  Insider Signals integrated (loaders + features + charts)
#  v18.7  •  robust col picking, compat wrappers, cached accuracy loader
# ────────────────────────────────────────────────────────────────────────────

# ── Path bootstrap (ensure parent of repo is importable) ────────────────────
import os, sys
_THIS   = os.path.abspath(__file__)
_REPO   = os.path.dirname(os.path.dirname(_THIS))   # …/ml_quant_fund
_PARENT = os.path.dirname(_REPO)                    # …/
if _PARENT not in sys.path:
    sys.path.insert(0, _PARENT)

# stdlib / third-party
import io, zipfile, glob, inspect
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
from email.mime.text import MIMEText  # FIXED import path

# ── Project imports (package style ONLY) ─────────────────────────────────────
from ml_quant_fund.forecast_utils import (
    build_feature_dataframe,
    forecast_price_trend,
    forecast_today_movement,
    auto_retrain_forecast_model,
    compute_rolling_accuracy,
    get_latest_forecast_log,
    run_auto_retrain_all,
)
from ml_quant_fund.core.helpers_xgb import train_xgb_predict, RISK_ALPHA
from ml_quant_fund.accuracy_sink import load_accuracy_any  # import only (no top-level DB calls)

# Insider feature builders
try:
    from ml_quant_fund.insider_features import (
        build_daily_insider_features,
        add_rolling_insider_features,
    )
except Exception:
    from insider_features import (
        build_daily_insider_features,
        add_rolling_insider_features,
    )

# Loaders (LOCAL SQLITE insiders + yfinance price)
try:
    from loader import insider_loader as db_insider_loader, price_loader as yf_price_loader, insider_source_label
except Exception:
    db_insider_loader = None
    yf_price_loader   = None
    def insider_source_label(): return "Unknown"

# ──────────────────────────  HELPERS  ────────────────────────────────────────
def pick_col(df: pd.DataFrame, candidates) -> str | None:
    """Return first matching column name (case-insensitive)."""
    if df is None or df.empty:
        return None
    cols = list(df.columns)
    for c in candidates:
        if c in df.columns:
            return c
    lower_map = {c.lower(): c for c in cols}
    for c in candidates:
        cl = c.lower()
        if cl in lower_map:
            return lower_map[cl]
    return None

def _as_ymd(d):
    return pd.to_datetime(d).date().isoformat() if pd.notnull(pd.to_datetime(d, errors="coerce")) else None

def _normalize_price_df(px: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """
    Normalize a price df to columns:
      ['ticker','date','close','shares_outstanding','market_cap']
    Works even if `px` has MultiIndex columns like ('Close','AAPL') or mixed/blank
    second levels (e.g., ('date','')).
    """
    import pandas as pd
    if px is None or len(px) == 0:
        return pd.DataFrame(columns=["ticker","date","close","shares_outstanding","market_cap"])

    df = px.copy()
    T = str(ticker).upper()

    # 1) Flatten / slice MultiIndex
    if isinstance(df.columns, pd.MultiIndex):
        # Try slicing by any level that contains the ticker
        sliced = False
        for lev in range(df.columns.nlevels):
            try:
                if any(str(v).upper() == T for v in df.columns.get_level_values(lev)):
                    try:
                        tmp = df.xs(T, axis=1, level=lev, drop_level=True)
                        if tmp.shape[1] > 0:
                            df = tmp
                            sliced = True
                            break
                    except Exception:
                        pass
            except Exception:
                pass
        # If still MultiIndex (or slice failed), flatten safely
        if isinstance(df.columns, pd.MultiIndex):
            flat = []
            for tpl in df.columns.to_flat_index():
                parts = [str(p) for p in tpl if str(p) not in ("", "None", "nan")]
                if len(parts) >= 2:
                    # keep first part; append suffix only if it’s not the same ticker
                    name = f"{parts[0]}_{parts[1]}" if parts[1].upper() != T else parts[0]
                else:
                    name = parts[0]
                flat.append(name)
            df.columns = flat

    # 2) Ensure a date column
    if "date" not in df.columns:
        if isinstance(df.index, pd.DatetimeIndex):
            df = df.reset_index().rename(columns={df.index.name or "index": "date"})
        elif "Date" in df.columns:
            df = df.rename(columns={"Date": "date"})
        else:
            df = df.reset_index().rename(columns={"index": "date"})

    # 3) Find/rename close (accept many variants, including suffixed ones)
    def _find_first(names):
        lower_map = {c.lower(): c for c in df.columns}
        for n in names:
            if n in lower_map:           # direct lower-case match
                return lower_map[n]
        # Try title-cased with spaces
        for n in names:
            pretty = n.replace("_"," ").title()
            if pretty in df.columns:
                return pretty
        return None

    close_candidates = [
        "adj close", "adj_close", "adjclose", "close",
        f"adj close_{T.lower()}", f"adj_close_{T.lower()}", f"adjclose_{T.lower()}",
        f"close_{T.lower()}", f"adj close_{T}", f"adj_close_{T}", f"adjclose_{T}", f"close_{T}",
    ]
    # Build the lowercase map once
    df.columns = [str(c) for c in df.columns]
    lower_map = {c.lower(): c for c in df.columns}
    ccol = _find_first(close_candidates)
    if ccol:
        df = df.rename(columns={ccol: "close"})
    elif "close" not in df.columns:
        # last resort: first numeric column
        numcols = df.select_dtypes(include="number").columns.tolist()
        if numcols:
            df = df.rename(columns={numcols[0]: "close"})
        else:
            df["close"] = pd.NA

    # 4) Ensure ticker column
    df["ticker"] = T

    # 5) shares_outstanding / market_cap (accept suffixed forms)
    for base in ("shares_outstanding", "market_cap"):
        if base not in df.columns:
            cand = _find_first([base, f"{base}_{T.lower()}", f"{base}_{T}"])
            if cand:
                df = df.rename(columns={cand: base})
            else:
                df[base] = pd.NA

    # 6) Final tidy
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date
    keep = [c for c in ["ticker","date","close","shares_outstanding","market_cap"] if c in df.columns]
    out = df[keep].copy().dropna(subset=["date"])
    return out

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
        uploaded_file = st.file_uploader(
            "Upload a feature importances chart (PNG/JPG):",
            type=["png", "jpg", "jpeg"]
        )
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
def plot_shap_local(model, X):
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
st.button("🔄 Refresh accuracy cache", on_click=st.cache_data.clear)

# ──────────────────────────  SIDEBAR  ───────────────────────────────────────
with st.sidebar:
    if page == "Dashboard":
        st.markdown("## 📆 Date Range")
        start_date = st.date_input("Start date", value=date(2025, 3, 1))
        end_date   = st.date_input("End date",   value=date.today())

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
        show_insider_ui      = st.toggle("🕵️ Show Insider Signals section", True)

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

# ──────────────────────────  COMPAT WRAPPERS  ───────────────────────────────
def build_features_compat(tkr, start, end):
    """Call build_feature_dataframe with whatever signature it has."""
    try:
        params = set(inspect.signature(build_feature_dataframe).parameters.keys())
        if {"price_loader","insider_loader"} <= params:
            from loader import price_loader as _pl, insider_loader as _il
            return build_feature_dataframe(
                tkr, start_date=start, end_date=end, price_loader=_pl, insider_loader=_il
            )
        if {"start_date", "end_date"} & params:
            return build_feature_dataframe(tkr, start_date=start, end_date=end)
        if {"start", "end"} & params:
            return build_feature_dataframe(tkr, start=start, end=end)
    except Exception:
        pass
    try:
        return build_feature_dataframe(tkr, start, end)  # positional
    except TypeError:
        try:
            return build_feature_dataframe(tkr, start_date=start, end_date=end)
        except TypeError:
            return build_feature_dataframe(tkr, start=start, end=end)

def today_move_compat(tkr, start, end):
    """Call forecast_today_movement with flexible args."""
    try:
        params = set(inspect.signature(forecast_today_movement).parameters.keys())
        if {"start_date", "end_date"} & params:
            return forecast_today_movement(tkr, start_date=start, end_date=end)
        if {"start", "end"} & params:
            return forecast_today_movement(tkr, start=start, end=end)
    except Exception:
        pass
    try:
        return forecast_today_movement(tkr, start, end)  # positional
    except TypeError:
        try:
            return forecast_today_movement(tkr, start_date=start, end_date=end)
        except TypeError:
            return forecast_today_movement(tkr)

# ──────────────────────────  INSIDER UI SECTION  ────────────────────────────
def _safe_date(obj):
    try:
        return pd.to_datetime(obj, errors="coerce").date()
    except Exception:
        return None

def _load_insider_features_for_range(tkr: str, start, end):
    """
    Fetch insider 'final_daily' from local SQLite via loader.insider_loader(),
    build rolling features, and merge DB rollups (insider_7d/insider_21d).
    """
    if db_insider_loader is None or yf_price_loader is None:
        return None, None, "Loaders unavailable"

    try:
        t = tkr.upper().strip()
        sd = _safe_date(start)
        ed = _safe_date(end)
        if sd is None or ed is None:
            return None, None, "Invalid dates"

        # Price calendar (for rolling windows)
        px_raw = yf_price_loader(t, sd, ed)
        if px_raw is None or len(px_raw) == 0:
            return None, None, "No price data"
        # Normalize price frame (handles MultiIndex like ('Close','TICKER'))
        px = _normalize_price_df(px_raw, t)
        if px.empty or "close" not in px.columns:
            return None, None, "Price data missing 'close' after normalization"

        # Insider 'final_daily' from local DB (normalized schema)
        final_daily = db_insider_loader(t, sd, ed)
        if final_daily is None or final_daily.empty:
            return px[["ticker","date","close"]].copy(), pd.DataFrame(), "No insider entries in range (SQLite)"

        # Build features from daily insiders
        daily = build_daily_insider_features(final_daily)
        feats = add_rolling_insider_features(
            px[["ticker","date","close","shares_outstanding","market_cap"]],
            daily
        )

        # Merge DB-provided rollups if present: insider_7d/21d → ins_net_shares_7d_db/_21d_db
        try:
            have_cols = set(final_daily.columns)
            if {"insider_7d","insider_21d","filed_date","ticker"} <= have_cols:
                roll = (
                    final_daily[["ticker","filed_date","insider_7d","insider_21d"]]
                    .rename(columns={"filed_date":"date"})
                    .copy()
                )
                roll["date"] = pd.to_datetime(roll["date"], errors="coerce").dt.date
                feats = feats.merge(roll, on=["ticker","date"], how="left")
                feats = feats.rename(columns={
                    "insider_7d":  "ins_net_shares_7d_db",
                    "insider_21d": "ins_net_shares_21d_db",
                })
                for c in ["ins_net_shares_7d_db","ins_net_shares_21d_db"]:
                    if c not in feats.columns:
                        feats[c] = 0.0
                    else:
                        feats[c] = feats[c].fillna(0.0)
            else:
                for c in ["ins_net_shares_7d_db","ins_net_shares_21d_db"]:
                    if c not in feats.columns:
                        feats[c] = 0.0
        except Exception:
            for c in ["ins_net_shares_7d_db","ins_net_shares_21d_db"]:
                if c not in feats.columns:
                    feats[c] = 0.0

        return px, feats, None
    except Exception as e:
        return None, None, f"Insider load failed: {e}"

def render_insider_section(default_ticker: str, start, end, expanded=False):
    tkr = (default_ticker or "AAPL").strip().upper()
    with st.expander("🕵️ Insider Signals (Daily)", expanded=expanded):
        st.caption(f"Source: {insider_source_label()}  → table: insider_flows")
        px, insf, err = _load_insider_features_for_range(tkr, start, end)

        if err:
            st.info(f"{tkr}: {err}")

        if insf is None or insf.empty:
            st.warning("No insider signals to display for the selected range.")
            return

        # Compute quick metrics (last available day with features)
        insf = insf.copy()
        insf["date"] = pd.to_datetime(insf["date"], errors="coerce")
        insf = insf.dropna(subset=["date"]).sort_values("date")

        last_row = insf.iloc[-1]
        z30   = float(last_row.get("ins_pressure_30d_z", 0))
        n7    = float(last_row.get("ins_large_or_exec_7d", 0))
        ns30  = float(last_row.get("ins_net_shares_30d",
                                   insf.get("ins_net_shares", pd.Series(0)).rolling(30, min_periods=1).sum().iloc[-1]
                                   if "ins_net_shares" in insf else 0))
        ns7_db  = float(last_row.get("ins_net_shares_7d_db", 0))
        ns21_db = float(last_row.get("ins_net_shares_21d_db", 0))

        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Insider Pressure (30d, z)", f"{z30:+.2f}")
        c2.metric("Exec/Large trades (7d)",    f"{int(n7)}")
        c3.metric("Net Shares (30d, calc)",    f"{ns30:,.0f}")
        c4.metric("Net Shares (7d, DB)",       f"{ns7_db:,.0f}")
        c5.metric("Net Shares (21d, DB)",      f"{ns21_db:,.0f}")

        # Charts
        chart_cols = ["date","close","ins_pressure_30d_z","ins_large_or_exec_7d","ins_net_shares",
                      "ins_net_shares_7d_db","ins_net_shares_21d_db"]
        chart_df = insf[[c for c in chart_cols if c in insf.columns]].copy()
        chart_df["date"] = pd.to_datetime(chart_df["date"])

        base = alt.Chart(chart_df).encode(x=alt.X("date:T", title="Date"))

        # Price line with rich tooltips including DB rollups
        tooltip_base = ["date:T"]
        if "close" in chart_df.columns:
            tooltip_base.append(alt.Tooltip("close:Q", format=",.2f"))
        if "ins_pressure_30d_z" in chart_df.columns:
            tooltip_base.append(alt.Tooltip("ins_pressure_30d_z:Q", format=",.2f", title="Insider z (30d)"))
        if "ins_net_shares_7d_db" in chart_df.columns:
            tooltip_base.append(alt.Tooltip("ins_net_shares_7d_db:Q", format=",.0f", title="Net Shares 7d (DB)"))
        if "ins_net_shares_21d_db" in chart_df.columns:
            tooltip_base.append(alt.Tooltip("ins_net_shares_21d_db:Q", format=",.0f", title="Net Shares 21d (DB)"))

        price_line = base.mark_line().encode(
            y=alt.Y("close:Q", title="Close") if "close" in chart_df.columns else alt.Y("date:T"),
            tooltip=tooltip_base
        )

        press_line = base.mark_line(strokeDash=[4,2]).encode(
            y=alt.Y("ins_pressure_30d_z:Q", title="Insider Pressure (z)"),
            tooltip=["date:T","ins_pressure_30d_z:Q","ins_large_or_exec_7d:Q"] \
                    if "ins_pressure_30d_z" in chart_df.columns else ["date:T"]
        ) if "ins_pressure_30d_z" in chart_df.columns else None

        exec_pts = base.mark_circle(size=40).encode(
            y="ins_pressure_30d_z:Q",
            color=alt.condition("datum.ins_large_or_exec_7d > 0", alt.value("red"), alt.value("transparent"))
            if "ins_large_or_exec_7d" in chart_df.columns else alt.value("transparent"),
            tooltip=[
                "date:T",
                alt.Tooltip("ins_large_or_exec_7d:Q", title="Exec/Large (7d)") if "ins_large_or_exec_7d" in chart_df.columns else alt.Tooltip("date:T"),
                alt.Tooltip("ins_net_shares_7d_db:Q", title="Net Shares 7d (DB)", format=",.0f")
                    if "ins_net_shares_7d_db" in chart_df.columns else alt.Tooltip("date:T"),
            ]
        ) if "ins_pressure_30d_z" in chart_df.columns else None

        pr = price_line
        if press_line is not None:
            pr = pr + press_line
        if exec_pts is not None:
            pr = pr + exec_pts

        st.altair_chart(pr.resolve_scale(y="independent").properties(
            title=f"{tkr} — Price vs. Insider Pressure (30d z-score)"
        ), use_container_width=True)

        # Daily net shares bars (calc) with DB rollup tooltips
        if "ins_net_shares" in chart_df.columns:
            bars_tt = ["date:T", alt.Tooltip("ins_net_shares:Q", title="Net Shares (daily)", format=",.0f")]
            if "ins_net_shares_7d_db" in chart_df.columns:
                bars_tt.append(alt.Tooltip("ins_net_shares_7d_db:Q", title="Net Shares 7d (DB)", format=",.0f"))
            if "ins_net_shares_21d_db" in chart_df.columns:
                bars_tt.append(alt.Tooltip("ins_net_shares_21d_db:Q", title="Net Shares 21d (DB)", format=",.0f"))

            bars = base.mark_bar(opacity=0.6).encode(
                y=alt.Y("ins_net_shares:Q", title="Net Shares (daily)"),
                tooltip=bars_tt
            ).properties(title=f"{tkr} — Daily Net Shares")
            st.altair_chart(bars, use_container_width=True)

        # Optional raw table
        with st.expander("🧾 View last 15 insider rows"):
            show_cols = [c for c in [
                "date","ins_net_shares","ins_buy_ct","ins_sell_ct","ins_holdings_delta",
                "ins_large_or_exec_7d","ins_pressure_7d","ins_pressure_30d","ins_pressure_30d_z",
                "ins_net_shares_7d_db","ins_net_shares_21d_db"
            ] if c in insf.columns]
            st.dataframe(insf[show_cols].tail(15))

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
                tkr, period_months=int(max(1, forecast_days // 30))
            )
            model_used = "Prophet"
        else:
            base_df = build_features_compat(
                tkr, _as_ymd(start_date), _as_ymd(end_date)
            )
            try:
                _, _, _, y_pred, _ = train_xgb_predict(
                    base_df, horizon_days=forecast_days
                )
                # choose columns robustly
                date_col  = pick_col(base_df, ["date", "Date", "ds"])
                close_col = pick_col(base_df, ["close", "Close", "Adj Close", "AdjClose", "adj_close"])
                if date_col and close_col:
                    recent = base_df[[date_col, close_col]].tail(60).rename(
                        columns={date_col: "ds", close_col: "actual"}
                    )
                else:
                    temp = base_df.reset_index()
                    idx  = pick_col(temp, ["date", "Date", "index"])
                    ccol = pick_col(temp, ["close", "Close"])
                    recent = temp[[idx, ccol]].tail(60).rename(
                        columns={idx: "ds", ccol: "actual"}
                    )
                futr = pd.DataFrame({
                    "ds":          pd.date_range(datetime.today(), periods=len(y_pred)),
                    "yhat":        y_pred,
                    "yhat_lower":  np.nan,
                    "yhat_upper":  np.nan,
                    "actual":      np.nan,
                })
                parts = [x for x in [recent, futr] if x is not None and not getattr(x, "empty", False)]
                forecast_df = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
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
                .mark_line()
                .encode(x="ds:T", y="actual:Q", tooltip=["ds:T", "actual:Q"])
                .properties(title=f"{tkr} – {forecast_days}-Day Projection")
            )
            fut_line = (
                alt.Chart(futr)
                .mark_line()
                .encode(x="ds:T", y="yhat:Q", tooltip=["ds:T", "yhat:Q"])
            )
            conf_band = (
                alt.Chart(futr)
                .mark_area(opacity=0.25)
                .encode(x="ds:T", y="yhat_lower:Q", y2="yhat_upper:Q")
            )
            st.altair_chart(
                (hist_line + conf_band + fut_line).interactive(),
                use_container_width=True,
            )

            st.subheader("🗓️ Today's Movement Prediction")
            move_msg, move_err = today_move_compat(
                tkr, _as_ymd(start_date), _as_ymd(end_date)
            )
            if move_err:
                st.warning(move_err)
            else:
                st.success(move_msg)

# ──────────────────────────  INSIDER SECTION (UI)  ──────────────────────────
try:
    if show_insider_ui:
        # Show insider section for the main forecast ticker if provided, otherwise first in the list
        default_tkr = (locals().get("tkr_in") or tickers[0]).strip().upper()
        render_insider_section(default_tkr, _as_ymd(start_date), _as_ymd(end_date), expanded=False)
except Exception as e:
    st.info(f"Insider section disabled: {e}")

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
            df = build_features_compat(
                tkr, _as_ymd(start_date), _as_ymd(end_date)
            )

            # ---- Insider quick badges inside strategy block (optional) ----
            try:
                if {"ins_pressure_30d_z","ins_large_or_exec_7d"} <= set(df.columns):
                    last = df[["ins_pressure_30d_z","ins_large_or_exec_7d"]].dropna(how="all").tail(1)
                    if len(last):
                        z30 = float(last["ins_pressure_30d_z"].values[-1])
                        n7  = int(last["ins_large_or_exec_7d"].values[-1])
                        c1, c2 = st.columns(2)
                        c1.metric("Insider Pressure 30d (z)", f"{z30:+.2f}")
                        c2.metric("Exec/Large (7d)", f"{n7}")
            except Exception:
                pass

            # ---- Risk diagnostics (UI) ----
            with st.expander("🛡️ Risk diagnostics", expanded=False):
                if df is None or df.empty:
                    st.info("No price/feature data loaded.")
                else:
                    risk_cols = ["risk_today", "risk_next_1d", "risk_next_3d", "risk_prev_1d"]
                    avail = [c for c in risk_cols if c in df.columns]
                    if not avail:
                        st.warning("Risk columns are missing from the feature frame.")
                    else:
                        nz = (df[avail].fillna(0) != 0).mean().rename("nonzero_frac").to_frame()
                        last = df[avail].dropna(how="all").tail(10)
                        c1, c2 = st.columns(2)
                        with c1:
                            st.dataframe(nz.style.format({"nonzero_frac": "{:.2%}"}))
                        with c2:
                            st.dataframe(last)

            # -------- model -------------------------------------------
            model, X_test, y_test, y_pred, y_prob = train_xgb_predict(df)

            # Build confidence proxy if helper didn't return one
            if y_prob is None and y_pred is not None and X_test is not None and len(y_pred) == len(X_test):
                pred = pd.Series(y_pred, index=X_test.index)
                close_col = pick_col(X_test, ["Close", "close"])
                close_now = X_test[close_col] if close_col else pd.Series(1.0, index=X_test.index)
                pred_ret  = (pred - close_now) / close_now.replace(0, np.nan)
                atr_col   = pick_col(df, ["ATR", "atr"])
                atr       = df.loc[X_test.index, atr_col] if atr_col else pd.Series(0.02, index=X_test.index)
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

            # optional gate by event risk
            gate = (df_test.get("risk_next_3d", pd.Series(0, index=df_test.index)) >= block_tau)
            df_test["GateBlock"] = gate.astype(int)

            df_test["Prob"]     = y_prob
            df_test["Prob_eff"] = df_test["Prob"] * risk_mult  # global 72h multiplier

            # trade signal (enter long only when confident and not gated)
            df_test["Signal"] = ((df_test["Prob_eff"] > confidence_threshold) & (~gate)).astype(int)

            # ---- RETURNS (use these for metrics) ----
            ret_col  = pick_col(df_test, ["return_1d", "Return_1D", "ret_1d"])
            ret_mkt  = df_test[ret_col].fillna(0) if ret_col else pd.Series(0, index=df_test.index)
            ret_strat = (df_test["Signal"].shift(1).fillna(0) * ret_mkt)  # avoid look-ahead

            # ---- EQUITY (use for plotting, DD, CAGR) ----
            eq_mkt   = (1 + ret_mkt).cumprod()
            eq_strat = (1 + ret_strat).cumprod()

            # ---- metrics --------------------------------------------
            y_dir_true = (y_test.diff() > 0).astype(int).iloc[1:]
            y_dir_pred = (pd.Series(y_pred, index=y_test.index).diff() > 0).astype(int).iloc[1:]
            acc = accuracy_score(y_dir_true, y_dir_pred) if len(y_dir_true) == len(y_dir_pred) and len(y_dir_true) > 0 else float("nan")

            ann = 252
            mu  = ret_strat.mean()
            sd  = ret_strat.std(ddof=1)
            sharpe = np.sqrt(ann) * mu / sd if sd and not np.isnan(sd) else np.nan

            mdd = (eq_strat / eq_strat.cummax() - 1).min()
            n_days = max(1, len(eq_strat))
            cagr = (eq_strat.iloc[-1]) ** (ann / n_days) - 1

            trades = ((df_test["Signal"] == 1) & (df_test["Signal"].shift(1) != 1)).sum()
            exposure = float(df_test["Signal"].mean())
            wins = ret_strat[ret_strat > 0].sum()
            loss = -ret_strat[ret_strat < 0].sum()
            profit_factor = (wins / loss) if loss > 0 else np.nan

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Accuracy", f"{acc:.2f}" if not np.isnan(acc) else "—")
            c2.metric("Sharpe",   f"{sharpe:.2f}" if not np.isnan(sharpe) else "—")
            c3.metric("Max DD",   f"{mdd:.2%}")
            c4.metric("CAGR",     f"{cagr:.2%}")
            st.caption(f"Trades: {int(trades)} • Exposure: {exposure:.1%} • Profit factor: {profit_factor:.2f}")

            plot_df = pd.DataFrame({"Strategy": eq_strat, "Market": eq_mkt})
            st.line_chart(plot_df)

            csv_bytes = pd.concat(
                [df_test[["Signal","Prob","Prob_eff","GateBlock"]],
                 ret_strat.rename("StrategyRet"),
                 eq_strat.rename("Strategy")], axis=1
            ).to_csv(index=True).encode()
            st.download_button(
                f"🗅 CSV – {tkr}",
                csv_bytes,
                file_name=f"{tkr}_strategy.csv",
                mime="text/csv"
            )
            csv_buffers.append((f"{tkr}_strategy.csv", csv_bytes))

            if (
                enable_email
                and df_test.iloc[-1]["Signal"] == 1
                and pd.notna(df_test.iloc[-1]["Prob_eff"])
                and df_test.iloc[-1]["Prob_eff"] > confidence_threshold
            ):
                send_alert_email(tkr, float(df_test.iloc[-1]["Prob_eff"]))

            if enable_shap:
                plot_shap_local(model, X_test)

            eff_conf = df_test.iloc[-1].get("Prob_eff", df_test.iloc[-1]["Prob"])
            st.session_state["live_signals"][tkr] = {
                "signal": int(df_test.iloc[-1]["Signal"]),
                "confidence": float(eff_conf) if pd.notna(eff_conf) else 0.0
            }

        except Exception as e:
            st.error(f"⚠️ {tkr}: {e}")

    if enable_zip_download and csv_buffers:
        zbuf = io.BytesIO()
        with zipfile.ZipFile(zbuf, "w", zipfile.ZIP_DEFLATED) as zf:
            for fname, data in csv_buffers:
                zf.writestr(fname, data)
        st.download_button(
            "📦 Download ALL as ZIP",
            zbuf.getvalue(),
            file_name="strategy_exports.zip",
            mime="application/zip"
        )

# ═════════ ACCURACY DASHBOARD (cached loader) ═════════
st.subheader("📊 Forecast Accuracy Dashboard")

@st.cache_data(ttl=300)
def get_acc_df():
    # Prefer Postgres via ACCURACY_DSN
    src = "Neon/Postgres" if os.getenv("ACCURACY_DSN") else "SQLite"
    df = pd.DataFrame()
    try:
        df = load_accuracy_any()  # uses ACCURACY_DSN if set
    except Exception as e:
        st.warning(f"Postgres load failed: {e}")
        df = pd.DataFrame()

    # Fallback to local SQLite if Postgres returned nothing
    if df.empty:
        try:
            from loader import load_eval_logs_from_forecast_db
            df = load_eval_logs_from_forecast_db(os.getenv("FORECAST_ACCURACY_DB"))
            src = "SQLite"
        except Exception:
            src = "None"
            df = pd.DataFrame(columns=["date","ticker","mae","mse","r2","model","confidence"])
    return df, src

acc_df, source = get_acc_df()

with st.expander("🔧 Accuracy datasource debug", expanded=False):
    st.write("Source:", source, "| rows:", len(acc_df))
    if not acc_df.empty:
        st.write("Date range:", acc_df["date"].min(), "→", acc_df["date"].max())
        st.dataframe(acc_df.sort_values("date")["date ticker mae mse r2".split()].tail(5))

if acc_df.empty:
    st.info("No accuracy data found yet.")
else:
    acc_df = acc_df.copy()
    acc_df["date"] = pd.to_datetime(acc_df["date"], errors="coerce")
    for c in ["mae", "mse", "r2", "confidence"]:
        if c in acc_df.columns:
            acc_df[c] = pd.to_numeric(acc_df[c], errors="coerce")
    acc_df["ticker"] = acc_df["ticker"].astype(str).str.upper()
    acc_df = acc_df.dropna(subset=["date"]).sort_values("date")

    options = sorted(acc_df["ticker"].dropna().unique().tolist())
    prev = [t for t in st.session_state.get("acc_ticker_filter", []) if t in options]
    sel = st.multiselect("Filter tickers", options=options, default=prev)
    st.session_state["acc_ticker_filter"] = sel
    if sel:
        acc_df = acc_df[acc_df["ticker"].isin(sel)]

    c1, c2, c3 = st.columns(3)
    c1.metric("Avg MAE", f"{acc_df['mae'].mean(skipna=True):.3f}" if "mae" in acc_df else "—")
    c2.metric("Avg MSE", f"{acc_df['mse'].mean(skipna=True):.3f}" if "mse" in acc_df else "—")
    c3.metric("Avg R²",  f"{acc_df['r2'].mean(skipna=True):.3f}"  if "r2"  in acc_df else "—")

    chart_cols = [c for c in ["mae","mse","r2"] if c in acc_df.columns]
    chart_df = acc_df.set_index("date")[chart_cols].dropna(how="all") if chart_cols else pd.DataFrame()
    if not chart_df.empty:
        st.line_chart(chart_df)
    else:
        st.warning("No numeric accuracy data to plot yet.")

# ═════════ ACCURACY OVER TIME (per ticker) ═════════
st.subheader("🎯 Accuracy Over Time (per ticker)")

def rolling_acc_compat(tkr, start=None, end=None, window_days=30):
    """Call compute_rolling_accuracy with flexible signatures."""
    try:
        params = set(inspect.signature(compute_rolling_accuracy).parameters.keys())
        if {"start_date","end_date","window_days"} <= params:
            return compute_rolling_accuracy(tkr, start_date=start, end_date=end, window_days=window_days), None
        if {"window_days"} <= params:
            return compute_rolling_accuracy(tkr, window_days=window_days), None
        if {"start_date","end_date"} <= params:
            return compute_rolling_accuracy(tkr, start_date=start, end_date=end), None
        return compute_rolling_accuracy(tkr, start, end, window_days), None
    except TypeError:
        try:
            return compute_rolling_accuracy(tkr, start, end), None
        except Exception as e:
            return None, f"{e}"
    except Exception as e:
        return None, f"{e}"

acc_tickers = sorted(list(set(load_forecast_tickers() + (acc_df["ticker"].astype(str).str.upper().unique().tolist() if not acc_df.empty else []))))
sel_tkr = st.selectbox("Select ticker", acc_tickers, index=acc_tickers.index(acc_tickers[0]) if acc_tickers else 0)
win = st.slider("Rolling window (days)", 5, 120, 30, step=5)
start_for_acc = _as_ymd(locals().get("start_date", date(2024,1,1)))
end_for_acc   = _as_ymd(locals().get("end_date", date.today()))

acc_ts, err = rolling_acc_compat(sel_tkr, start=start_for_acc, end=end_for_acc, window_days=win)

if err:
    st.info(f"Could not compute rolling accuracy via compute_rolling_accuracy(): {err}")
else:
    if acc_ts is None or (hasattr(acc_ts, "empty") and acc_ts.empty):
        st.warning("No accuracy time series available for this ticker/date range.")
    else:
        # Normalize shape → want columns: date, accuracy
        if isinstance(acc_ts, pd.Series):
            acc_ts = acc_ts.reset_index()
            date_col = pick_col(acc_ts, ["date","Date","ds","index"]) or acc_ts.columns[0]
            val_col  = pick_col(acc_ts, ["accuracy","acc","value"]) or acc_ts.columns[-1]
            acc_ts = acc_ts.rename(columns={date_col:"date", val_col:"accuracy"})
        else:
            dcol = pick_col(acc_ts, ["date","Date","ds"]) or acc_ts.columns[0]
            vcol = pick_col(acc_ts, ["accuracy","acc","rolling_accuracy","hit_rate"]) or acc_ts.columns[-1]
            acc_ts = acc_ts.rename(columns={dcol:"date", vcol:"accuracy"})

        acc_ts["date"] = pd.to_datetime(acc_ts["date"], errors="coerce")
        acc_ts = acc_ts.dropna(subset=["date"]).sort_values("date")

        last_acc = acc_ts["accuracy"].dropna().iloc[-1] if acc_ts["accuracy"].notna().any() else np.nan
        st.metric(f"{sel_tkr} rolling accuracy ({win}d)", f"{last_acc:.3f}" if pd.notna(last_acc) else "—")

        chart_df = acc_ts[["date","accuracy"]].dropna()
        if not chart_df.empty:
            st.line_chart(chart_df.set_index("date"))
        else:
            st.info("No numeric accuracy values to plot.")

        acc_ts["month"] = acc_ts["date"].dt.to_period("M").dt.to_timestamp()
        monthly = acc_ts.groupby("month", as_index=False)["accuracy"].mean().dropna()
        if not monthly.empty:
            st.bar_chart(monthly.set_index("month")["accuracy"])

        csv = acc_ts.to_csv(index=False).encode()
        st.download_button(
            "⬇️ Download accuracy time series (CSV)",
            data=csv,
            file_name=f"{sel_tkr}_accuracy_{win}d.csv",
            mime="text/csv"
        )
