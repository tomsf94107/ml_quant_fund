# ui/1_Dashboard.py
# ─────────────────────────────────────────────────────────────────────────────
# Main dashboard. This file contains ZERO business logic.
# Every computation is delegated to the backend modules.
#
# What was removed from v19.3:
#   ✗ All compat wrappers (build_features_compat, today_move_compat, etc.)
#   ✗ MAE/MSE/R² metrics — replaced by accuracy/ROC-AUC/Brier from sink.py
#   ✗ Inline signal computation inside button handler
#   ✗ Uncalibrated sigmoid confidence proxy
#   ✗ Rolling accuracy via compute_rolling_accuracy (now from sink.py)
#   ✗ Duplicate gspread auth
#
# What we kept:
#   ✓ Password auth
#   ✓ BUY/HOLD signal cards with confidence badges
#   ✓ Sharpe/MaxDD/CAGR/profit_factor backtest metrics
#   ✓ Equity curve chart
#   ✓ Email alerts on high-confidence BUY signals
#   ✓ Event risk badge from calendar page session state
#   ✓ block_tau and confidence threshold sliders
#   ✓ ZIP download of all signal CSVs
#   ✓ Insider signals section
# ─────────────────────────────────────────────────────────────────────────────

import os, sys
import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.timezone import now_et
from datetime import date, datetime, timedelta

# ── Path bootstrap ────────────────────────────────────────────────────────────
_HERE = os.path.abspath(os.path.dirname(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, os.pardir))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import io
import zipfile
import smtplib
from email.mime.text import MIMEText

import numpy as np
import pandas as pd
import altair as alt
import streamlit as st
from streamlit_autorefresh import st_autorefresh

# ── Backend imports ───────────────────────────────────────────────────────────
from features.builder import build_feature_dataframe
from signals.generator import (
    generate_signals, signals_to_dataframe,
    DEFAULT_CONFIDENCE_THRESHOLD, DEFAULT_BLOCK_TAU,
)
from data.etl_sentiment import run_sentiment_etl, get_sentiment_score, _current_time_slot
from accuracy.sink import (
    log_predictions_batch, load_accuracy,
    load_prediction_history, reconcile_outcomes,
)

# ══════════════════════════════════════════════════════════════════════════════
#  PAGE CONFIG
# ══════════════════════════════════════════════════════════════════════════════

# ── Startup: train models if missing (Streamlit Cloud first deploy) ──────────
import sys as _sys
import os as _os
_sys.path.insert(0, _os.path.abspath(_os.path.join(_os.path.dirname(__file__), "..")))
try:
    from startup import models_are_trained, run_startup
    if not models_are_trained():
        with st.spinner("⚙️ First launch — training models (10-15 min)... Please wait."):
            run_startup(verbose=True)
except Exception as _e:
    st.warning(f"Startup check failed: {_e}")


import json as _json_cache
from pathlib import Path as _Path_cache
import pytz as _pytz_cache

_CACHE_PATH = os.path.join(_ROOT, "data", "signals_cache.json")

# Load ticker metadata for bucket/tier display
def _load_meta() -> dict:
    try:
        import pandas as _pm
        _mp = os.path.join(_ROOT, "tickers_metadata.csv")
        if os.path.exists(_mp):
            _df = _pm.read_csv(_mp)
            return _df.set_index("ticker").to_dict("index")
    except Exception:
        pass
    return {}
_TICKER_META = _load_meta()

def _et_now():
    return _datetime.datetime.now(_pytz_cache.timezone("America/New_York")).strftime("%Y-%m-%dT%H:%M:%S")

st.set_page_config(
    page_title="ML Quant Fund",
    page_icon="📈",
    layout="wide",
)
st_autorefresh(interval=5 * 60 * 1000, key="auto-refresh")


# ══════════════════════════════════════════════════════════════════════════════
#  AUTH
# ══════════════════════════════════════════════════════════════════════════════

def _check_login():
    if st.session_state.get("auth_ok"):
        return
    pwd = st.text_input("Password:", type="password", key="login_pwd")
    if not pwd:
        st.stop()
    if pwd != st.secrets.get("app_password", "MlQ@nt@072025"):
        st.error("❌ Wrong password")
        st.stop()
    st.session_state["auth_ok"] = True

_check_login()


# ══════════════════════════════════════════════════════════════════════════════
#  HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _send_alert(ticker: str, prob: float, horizon: int):
    try:
        msg = MIMEText(
            f"High-confidence BUY signal\n"
            f"Ticker  : {ticker}\n"
            f"Horizon : {horizon}d\n"
            f"Prob(up): {prob:.1%}\n"
            f"Time    : {now_et().strftime('%Y-%m-%d %H:%M ET')}"
        )
        msg["Subject"] = f"🟢 BUY signal · {ticker}"
        msg["From"]    = os.getenv("EMAIL_SENDER", "")
        msg["To"]      = os.getenv("EMAIL_RECEIVER", "")
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as s:
            s.login(os.getenv("EMAIL_SENDER", ""), os.getenv("EMAIL_PASSWORD", ""))
            s.send_message(msg)
    except Exception as e:
        st.warning(f"Email failed: {e}")


def _confidence_badge(confidence: str) -> str:
    return {"HIGH": "🟢", "MEDIUM": "🟡", "LOW": "🔴"}.get(confidence, "⚪")


def _load_tickers() -> list[str]:
    path = os.path.join(_ROOT, "tickers.txt")
    if os.path.exists(path):
        return [t.strip().upper() for t in open(path).read().splitlines() if t.strip()]
    return ["AAPL", "NVDA", "MSFT", "TSLA", "AMD", "META", "GOOG", "AMZN"]


def _save_tickers(lst: list[str]):
    with open(os.path.join(_ROOT, "tickers.txt"), "w") as f:
        f.write("\n".join(lst))


# ══════════════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("## 📆 Date Range")
    start_date = st.date_input("Start", value=date(2022, 1, 1))
    end_date   = st.date_input("End",   value=date.today())

    st.markdown("## 🎯 Signal Settings")
    horizon              = st.selectbox("Horizon", [1, 3, 5], index=0,
                                        format_func=lambda x: f"{x}d")
    confidence_threshold = st.slider("Confidence threshold",
                                     0.50, 0.95, DEFAULT_CONFIDENCE_THRESHOLD, 0.01)
    block_tau            = st.slider("Block when risk_next_3d ≥",
                                     0, 6, DEFAULT_BLOCK_TAU, 1)

    st.markdown("## 💼 Portfolio")
    portfolio_value = st.number_input(
        "Portfolio value ($)",
        min_value=10000,
        max_value=10000000,
        value=300000,
        step=10000,
        help="Your Fidelity account value — used for position sizing"
    )
    st.caption(f"Max position: ${portfolio_value * 0.30:,.0f} (30%) | Min: ${portfolio_value * 0.05:,.0f} (5%)")

    st.markdown("## 📧 Alerts")
    enable_email = st.toggle("Email alerts on BUY", value=False)

    st.markdown("## 📦 Export")
    enable_zip = st.toggle("ZIP download", value=False)

    st.markdown("## 🗂 Tickers")
    all_tickers = _load_tickers()

    col_a, col_b = st.columns(2)
    if col_a.button("✅ Select All"):
        st.session_state["selected_tickers"] = all_tickers
    if col_b.button("❌ Clear"):
        st.session_state["selected_tickers"] = []

    # Filter saved tickers to only those still in tickers.txt
    saved = st.session_state.get("selected_tickers", all_tickers)
    valid_saved = [t for t in saved if t in all_tickers]
    tickers = st.multiselect(
        "Select tickers to run",
        options=all_tickers,
        default=valid_saved,
        key="selected_tickers",
    )

    with st.expander("✏️ Edit master list"):
        raw = st.text_area(
            "One per line",
            "\n".join(all_tickers),
            height=200,
        )
        if st.button("💾 Save list"):
            _save_tickers([t.strip().upper() for t in raw.splitlines() if t.strip()])
            st.success("Saved.")
            st.rerun()

    if st.button("🔄 Refresh accuracy cache"):
        st.cache_data.clear()
        st.rerun()

    st.markdown("---")
    st.markdown("## 📰 Sentiment")
    sent_tickers = st.multiselect(
        "Tickers to refresh",
        options=all_tickers,
        default=[],
        placeholder="Select tickers...",
        key="sent_refresh_tickers",
    )
    if st.button("🔁 Refresh Sentiment Now",
                 disabled=not sent_tickers,
                 help="Run FinBERT on selected tickers. ~4s each."):
        with st.spinner(f"Running sentiment on {len(sent_tickers)} ticker(s)..."):
            try:
                run_sentiment_etl(
                    tickers=sent_tickers,
                    time_slot="intraday",
                    force=True,
                    verbose=False,
                )
                st.success(f"✓ Sentiment updated for {', '.join(sent_tickers)}")
            except Exception as e:
                st.error(f"Sentiment failed: {e}")


# ══════════════════════════════════════════════════════════════════════════════
#  HEADER
# ══════════════════════════════════════════════════════════════════════════════

st.title("📈 ML Quant Fund")
st.caption(f"🕒 {now_et().strftime('%Y-%m-%d %H:%M:%S ET')}")

# ── Event risk badge (set by Page 7 — calendar) ──────────────────────────────
risk_info  = st.session_state.get("event_risk_next72")
risk_label = risk_info["label"] if risk_info else None
risk_mult  = {"Low": 1.00, "Medium": 0.92, "High": 0.85}.get(risk_label, 1.00)

if risk_info:
    col1, col2 = st.columns([1, 5])
    col1.metric("Next 72h Risk", f"{risk_info['label']} ({risk_info['score']})",
                help="Event risk score for the next 72 hours from earnings, Fed/CPI, and FDA calendars. Higher = more uncertainty around upcoming dates.")

# ── Macro regime badge ────────────────────────────────────────────────────────
try:
    import sys, os
    _ui_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
    if _ui_root not in sys.path:
        sys.path.insert(0, _ui_root)
    from ui.components.regime_widget import render_regime_widget
    render_regime_widget()
except Exception as e:
    st.caption(f"Regime widget unavailable: {e}")


# ══════════════════════════════════════════════════════════════════════════════
#  CACHE LOADING HELPERS
# ══════════════════════════════════════════════════════════════════════════════

import json as _json
from types import SimpleNamespace as _NS

# ══════════════════════════════════════════════════════════════════════════════
#  CACHE HELPERS — read/write data/signals_cache.json
#  Refresh Live writes here. Run Strategy reads here. Same file, single source of truth.
#  _CACHE_PATH is defined once at the top of this file (around line 81).
# ══════════════════════════════════════════════════════════════════════════════
import json as _jc, pytz as _ptz

def _read_cache():
    try:
        if not os.path.exists(_CACHE_PATH): return None
        with open(_CACHE_PATH) as _f: return _jc.load(_f)
    except: return None

def _write_cache(sigs):
    try:
        os.makedirs(os.path.dirname(_CACHE_PATH), exist_ok=True)
        ts = datetime.now(_ptz.timezone("America/New_York")).strftime("%Y-%m-%dT%H:%M:%S")
        with open(_CACHE_PATH, "w") as _f:
            _jc.dump({"generated_at": ts, "signals": sigs}, _f, indent=2)
    except: pass

# ══════════════════════════════════════════════════════════════════════════════
#  RUN STRATEGY
# ══════════════════════════════════════════════════════════════════════════════

# ── Cache status + buttons ───────────────────────────────────────────────────
_cache = _read_cache()
_c_left, _c_mid, _c_right = st.columns([3, 1, 1])
if _cache:
    _ts = _cache.get("generated_at", "?").replace("T", " ")
    _c_left.info(f"📦 Cached signals from **{_ts} ET** — showing pre-computed results")
else:
    _c_left.warning("⚠️ No cache found — click **Refresh Live** to generate signals")

_run_cache   = _c_mid.button("📦 Run Strategy", type="secondary",
    disabled=not bool(_cache))
_refresh_live = _c_right.button("🔄 Refresh Live", type="primary")

# ── Decide mode ───────────────────────────────────────────────────────────────
_auto_load = st.session_state.pop("auto_load_cache", False)

if not _run_cache and not _refresh_live and not _auto_load:
    if _cache:
        # Auto-load cache on page open — no button click needed
        _run_cache = True
    else:
        st.info("No cache yet. Click **🔄 Refresh Live** to generate signals.")
        st.stop()

# Auto-load after Refresh Live completes
if _auto_load:
    _run_cache = True

_use_cache = _run_cache and bool(_cache)
csv_buffers = []

if _use_cache:
    from types import SimpleNamespace as _NS
    signal_summary = []
    for s in _cache.get("signals", []):
        if s.get("horizon") != horizon: continue
        if tickers and s.get("ticker") not in tickers: continue
        _peff = s.get("prob_eff", 0.0)
        _sig = s.get("signal","HOLD") if _peff >= confidence_threshold else "HOLD"
        _nt = s.get("n_trades") or 0
        # sharpe_reliable: use cached flag if present (new caches); else fall back
        # to the n_trades>=30 guard so OLD caches are still handled honestly.
        _rel = s.get("sharpe_reliable")
        if _rel is None:
            _rel = (_nt >= 30)
        _m = _NS(sharpe=float(s.get("sharpe") or "nan"),
                 max_drawdown=float(s.get("max_drawdown") or "nan"),
                 cagr=float(s.get("cagr") or "nan"),
                 accuracy=float(s.get("accuracy") or "nan"),
                 n_trades=_nt,
                 profit_factor=float(s.get("profit_factor") or "nan"),
                 exposure=float(s.get("exposure") or "nan"),
                 psr=s.get("psr"),
                 sharpe_reliable=bool(_rel))
        signal_summary.append(_NS(
            ticker=s["ticker"], horizon=horizon,
            today_signal=_sig, today_prob=s.get("prob",0),
            today_prob_eff=_peff,
            current_price=s.get("current_price"),
            price_date=s.get("price_date"),
            price_target_up=s.get("price_target_up"),
            price_target_dn=s.get("price_target_dn"),
            expected_return=s.get("expected_return"),
            atr=s.get("atr"), metrics=_m, error=None, signal_df=None,
        ))
    if not signal_summary:
        st.warning("No signals in cache for selected filters. Click Refresh Live.")
        st.stop()

elif _refresh_live:
    # REFACTORED May 7 2026: Refresh Live now spawns daily_runner_batched
    # subprocess (same as runfund) — single source of truth for signal
    # generation. All today's fixes (pre-check, Massive-only, batching)
    # automatically apply. UI uses existing auto_load_cache path on completion.
    import subprocess as _sub
    import os as _os
    import re as _re
    import time as _time
    _ROOT_DIR = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
    _PYTHON   = "/Users/atomnguyen/.pyenv/versions/ml_quant_310/bin/python"

    progress = st.progress(0.0, text="Starting daily_runner_batched (3 batches x 42 tickers)...")
    status_box = st.empty()
    log_box = st.empty()
    log_lines = []

    try:
        proc = _sub.Popen(
            [_PYTHON, "-m", "scripts.daily_runner_batched"],
            cwd=_ROOT_DIR,
            stdout=_sub.PIPE,
            stderr=_sub.STDOUT,
            text=True,
            bufsize=1,
        )

        ticker_re   = _re.compile(r"\u2500\u2500\s+([A-Z]{1,5})\s+h=(\d+)d\s+\u2500\u2500")
        batch_re    = _re.compile(r"BATCH\s+(\d+)/(\d+)")
        exit_re     = _re.compile(r"BATCH\s+(\d+)/(\d+):\s+exit=(\d+),\s+elapsed=([\d.]+)s")
        done_re     = _re.compile(r"DONE\s+\u2014\s+(\d+)\s+signals,\s+(\d+)\s+BUY,\s+(\d+)\s+failed")
        all_done_re = _re.compile(r"ALL BATCHES COMPLETE")

        current_batch = 0
        total_batches = 3
        tickers_in_batch = 0

        for line in proc.stdout:
            line = line.rstrip()
            log_lines.append(line)
            if len(log_lines) > 15:
                log_lines = log_lines[-15:]

            m = exit_re.search(line)
            if m:
                bn = m.group(1)
                progress.progress(float(bn) / total_batches,
                    text=f"Batch {bn}/{total_batches} done (exit={m.group(3)}, {float(m.group(4)):.0f}s)")
                continue

            m = batch_re.search(line)
            if m and "exit=" not in line:
                current_batch = int(m.group(1))
                progress.progress((current_batch - 1) / total_batches,
                    text=f"Batch {current_batch}/{total_batches} starting...")
                continue

            m = ticker_re.search(line)
            if m:
                tickers_in_batch += 1
                base = (current_batch - 1) / total_batches
                within = min(tickers_in_batch / 42.0, 1.0) / total_batches
                progress.progress(min(base + within, 0.99),
                    text=f"Batch {current_batch}/{total_batches} - {m.group(1)}")
                continue

            m = done_re.search(line)
            if m:
                status_box.info(
                    f"Batch {current_batch}: {m.group(1)} signals, "
                    f"{m.group(2)} BUY, {m.group(3)} failed"
                )
                tickers_in_batch = 0
                continue

            if all_done_re.search(line):
                progress.progress(1.0, text="ALL BATCHES COMPLETE")

            log_box.code("\n".join(log_lines), language="text")

        proc.wait()
        if proc.returncode != 0:
            st.error(f"daily_runner_batched exited with code {proc.returncode}")
            st.stop()

    except Exception as _e:
        st.error(f"Subprocess failed: {_e}")
        st.stop()

    progress.progress(1.0, text="Done. Reloading cache...")
    status_box.success("Signals refreshed via daily_runner_batched. Reloading dashboard...")
    _time.sleep(1)
    st.session_state["auto_load_cache"] = True
    st.rerun()

    # ─────────────────────────────────────────────────────────────────────────

else:
    st.info("No cached signals yet. Click **Refresh Live** to generate signals.")
    st.stop()

# DISPLAY RESULTS
# ─────────────────────────────────────────────────────────────────────────

# ── Price sanity check ───────────────────────────────────────────────────
# Fetch ground truth prices and flag stale cache prices (live mode only)
if not _use_cache:
    try:
        import yfinance as yf
        all_syms = [r.ticker for r in signal_summary]
        raw_px = yf.download(all_syms, period="2d", auto_adjust=True, progress=False)
        if hasattr(raw_px.columns, "levels"):
            raw_px = raw_px["Close"]
        latest_px = raw_px.iloc[-1].to_dict() if not raw_px.empty else {}
        for r in signal_summary:
            true_price = latest_px.get(r.ticker)
            if true_price and r.current_price:
                diff_pct = abs(true_price - r.current_price) / true_price
                if diff_pct > 0.10:
                    st.warning(f"⚠️ {r.ticker}: price mismatch — model used {r.current_price:.2f}, market says {true_price:.2f}")
    except Exception:
        pass  # price check is best-effort

    
if not signal_summary:
    st.error("No signals generated. Check tickers and date range.")
    st.stop()

# ══════════════════════════════════════════════════════════════════════════
#  VIEW SWITCHER — Momentum / Kill switch ON / Kill switch OFF  (May 31 2026)
# ══════════════════════════════════════════════════════════════════════════
import sqlite3 as _sql_v
import pandas as _pd_v

# email alerts still fire (independent of which view is shown)
for r in signal_summary:
    if enable_email and r.today_signal == "BUY" and r.today_prob_eff >= 0.65:
        _send_alert(r.ticker, r.today_prob_eff, horizon)

# sector/bucket map
@st.cache_data(ttl=3600)
def _bucket_map_ui():
    try:
        m = _pd_v.read_csv("tickers_metadata.csv")
        return dict(zip(m["ticker"].str.upper(), m["bucket"].fillna("UNK")))
    except Exception:
        return {}
_BMAP = _bucket_map_ui()

# momentum shadow picks (latest date)
@st.cache_data(ttl=600)
def _momentum_picks_ui():
    try:
        c = _sql_v.connect("accuracy.db")
        d = _pd_v.read_sql("SELECT MAX(prediction_date) d FROM momentum_shadow_predictions", c)["d"].iloc[0]
        df = _pd_v.read_sql(
            "SELECT ticker, kind, mom_pct_rank FROM momentum_shadow_predictions "
            "WHERE prediction_date=? AND is_buy_candidate=1 ORDER BY mom_pct_rank DESC",
            c, params=[d])
        c.close()
        # Dedupe to one row per ticker. A name can be flagged by BOTH momentum
        # signals (mom_6_1 = 6mo, mom_12_1 = 12mo, both skip last 21d). Show it
        # once: best rank for sort, both individual ranks, and a badge if doubly
        # confirmed. (DB keeps both kind rows for the WF record — this is display only.)
        if df.empty:
            return d, df
        def _agg(g):
            r6  = g.loc[g["kind"]=="mom_6_1",  "mom_pct_rank"]
            r12 = g.loc[g["kind"]=="mom_12_1", "mom_pct_rank"]
            r6v  = float(r6.iloc[0])  if len(r6)  else None
            r12v = float(r12.iloc[0]) if len(r12) else None
            ranks = [x for x in (r6v, r12v) if x is not None]
            return _pd_v.Series({
                "rank_6_1":  r6v,
                "rank_12_1": r12v,
                "best_rank": max(ranks) if ranks else 0.0,
                "n_kinds":   int(g["kind"].nunique()),
            })
        try:
            agg = df.groupby("ticker", as_index=False).apply(_agg, include_groups=False)
        except TypeError:
            # older pandas without include_groups kwarg
            agg = df.groupby("ticker", as_index=False).apply(_agg)
        agg = agg.sort_values("best_rank", ascending=False).reset_index(drop=True)
        return d, agg
    except Exception:
        return None, _pd_v.DataFrame()
_mom_date, _mom_df = _momentum_picks_ui()

_view = st.segmented_control(
    "What do you want to see?",
    options=["🟢 Momentum (TRADE THIS)", "🔴 Kill switch ON", "🟡 Kill switch OFF"],
    default="🟢 Momentum (TRADE THIS)",
    key="view_mode",
)
if _view is None:
    _view = "🟢 Momentum (TRADE THIS)"

# context line
if _view.startswith("🟢"):
    st.success("**Momentum signal.** The real picks you would trade. Shadow mode — building a live record, ~0 resolved yet (first ~late June).")
elif _view.startswith("🔴"):
    st.error("**Kill switch ON (current reality).** Every direction-model BUY is forced to HOLD. Nothing here is traded — this is why the board is all HOLD.")
else:
    st.warning("**Kill switch OFF (preview only).** What the direction model WOULD buy if unblocked. NOT live — shown so you can see its picks. It has no edge (BUY 57.8% < HOLD 59.8%).")

# filters
_fc1, _fc2, _fc3 = st.columns([2, 2, 1])
_f_tk = _fc1.text_input("Filter ticker", placeholder="e.g. MU", key="f_ticker").upper().strip()
_sectors = ["All sectors"] + sorted({v for v in _BMAP.values()})
_f_sec = _fc2.selectbox("Filter sector", _sectors, key="f_sector")

def _match(tk):
    if _f_tk and _f_tk not in tk: return False
    if _f_sec != "All sectors" and _BMAP.get(tk, "UNK") != _f_sec: return False
    return True

# render cards by view
if _view.startswith("🟢"):
    picks = [row for row in _mom_df.itertuples() if _match(row.ticker)]
    _ndouble = sum(1 for r in picks if getattr(r, "n_kinds", 1) >= 2)
    _fc3.caption(f"{len(picks)} picks · {_ndouble} on both signals")
    if not picks:
        st.info("No momentum picks match the filter." if (_f_tk or _f_sec!="All sectors") else "No momentum picks in latest shadow run.")
    else:
        cols = st.columns(4)
        for i, r in enumerate(picks):
            tk = r.ticker
            both = getattr(r, "n_kinds", 1) >= 2
            r6, r12 = getattr(r, "rank_6_1", None), getattr(r, "rank_12_1", None)
            parts = []
            if r6  is not None: parts.append(f"6-1 {r6:.2f}")
            if r12 is not None: parts.append(f"12-1 {r12:.2f}")
            rank_str = " · ".join(parts)
            star    = "⭐ " if both else ""
            buy_lbl = "🟢 BUY ✕2" if both else "🟢 BUY"
            cols[i % 4].markdown(
                f"<div style='padding:8px 0;line-height:1.25'>"
                f"<span style='font-size:1.7rem;font-weight:700'>{star}{tk}</span>"
                f"<span style='font-size:0.8rem;font-weight:600;color:#16a34a;"
                f"margin-left:8px;vertical-align:middle'>{buy_lbl}</span><br>"
                f"<span style='font-size:0.75rem;color:#888'>{rank_str} · {_BMAP.get(tk,'UNK')}</span>"
                f"</div>",
                unsafe_allow_html=True,
            )
elif _view.startswith("🔴"):
    shown = [r for r in signal_summary if _match(r.ticker)]
    _fc3.caption(f"{len(shown)} · all HOLD")
    cols = st.columns(4)
    for i, r in enumerate(shown):
        cols[i % 4].metric(label=r.ticker, value="🔴 HOLD",
                           delta=f"p={r.today_prob_eff:.1%} · {_BMAP.get(r.ticker,'UNK')}")
else:  # Kill switch OFF preview
    shown = [r for r in signal_summary if _match(r.ticker)]
    nb = sum(1 for r in shown if r.today_prob_eff >= 0.60)
    _fc3.caption(f"{len(shown)} · {nb} would BUY")
    cols = st.columns(4)
    for i, r in enumerate(shown):
        would = r.today_prob_eff >= 0.60
        cols[i % 4].metric(label=r.ticker,
                           value=("🟢 BUY" if would else "🔴 HOLD"),
                           delta=f"p={r.today_prob_eff:.1%} · {_BMAP.get(r.ticker,'UNK')}")


# ══════════════════════════════════════════════════════════════════════════
#  DETAIL TABS — Forecast / Intraday / Watchlist / Per-ticker / Accuracy
# ══════════════════════════════════════════════════════════════════════════
tab_forecast, tab_intraday, tab_watch, tab_detail, tab_accuracy = st.tabs([
    "🎯 Forecast", "⚡ Intraday + EOD", "👀 Watchlist", "📑 Per-ticker detail", "📊 Accuracy"
])

with tab_forecast:

    import pandas as pd

    # Pre-compute long-only conviction weights for BUY signals (May 27 2026)
    # Formula matches portfolio/neutralizer.py long_only mode:
    #   signal_value_i = max(prob_raw_i - 0.5, 0) for each BUY ticker
    #   weight_i = signal_value_i / sum(signal_value for BUYs)
    # Informational only — does NOT auto-execute. See docs/neutralizer_backtest_findings_may27.md
    _buy_signal_values = {}
    for r in signal_summary:
        if r.today_signal == "BUY":
            _buy_signal_values[r.ticker] = max((r.today_prob or 0.0) - 0.5, 0.0)
    _buy_sum = sum(_buy_signal_values.values()) or 1.0  # avoid div by zero
    _rec_weights = {t: v / _buy_sum for t, v in _buy_signal_values.items()}

    # ── Phase 2H — A8 prob_top_decile + Blend Score (May 27 2026) ─────────────────
    # Compute cross-sectional z-scored blend: 0.3 × prob_raw_z + 0.7 × a8_z
    # Backtest (33 days, Apr-May 2026) showed +54pp cumulative vs production
    # with stable winning weights across H1/H2 splits (+68% H1, +72% H2 each).
    # See: docs/phase_2H_blend_findings_may27.md (to be written)
    import os as _os_blend
    import numpy as _np_blend
    _a8_lookup = {}  # ticker -> a8_prob for latest date
    _blend_scores = {}  # ticker -> blend score
    _a8_panel_path = _os_blend.path.join(
        _os_blend.path.dirname(_os_blend.path.dirname(_os_blend.path.abspath(__file__))),
        "data", "a8_oos_panel.parquet"
    )
    try:
        if _os_blend.path.exists(_a8_panel_path):
            _a8_panel = pd.read_parquet(_a8_panel_path)
            _a8_panel["date"] = pd.to_datetime(_a8_panel["date"])
            # Get the most recent A8 date
            _a8_latest = _a8_panel["date"].max()
            _a8_today = _a8_panel[_a8_panel["date"] == _a8_latest].set_index("ticker")["a8_prob"]
            _a8_lookup = _a8_today.to_dict()
        
            # Compute cross-sectional z-scores across signal_summary
            _all_probs = [(r.ticker, r.today_prob, _a8_lookup.get(r.ticker)) 
                          for r in signal_summary if r.today_prob is not None]
            if _all_probs:
                _probs_arr = _np_blend.array([p for _, p, _ in _all_probs])
                _a8_arr = _np_blend.array([a if a is not None else _np_blend.nan 
                                           for _, _, a in _all_probs])
                # z-score, treating NaN as 0
                _prob_mean, _prob_std = _probs_arr.mean(), _probs_arr.std()
                _a8_mean = _np_blend.nanmean(_a8_arr)
                _a8_std = _np_blend.nanstd(_a8_arr) if _np_blend.isfinite(_np_blend.nanstd(_a8_arr)) else 1.0
                for tkr, p, a in _all_probs:
                    _pz = (p - _prob_mean) / _prob_std if _prob_std > 0 else 0.0
                    _az = ((a - _a8_mean) / _a8_std) if (a is not None and _a8_std > 0) else 0.0
                    # Optimal blend weights from stability test
                    _blend_scores[tkr] = 0.3 * _pz + 0.7 * _az
    except Exception as _blend_e:
        # Fail loud per Rule #1 (b)
        import logging as _lg
        _lg.getLogger(__name__).error(f"blend_score computation failed: {_blend_e}")
        _a8_lookup = {}
        _blend_scores = {}

    # Rank BLEND scores among today's BUYs for tier display (top-5 highlighted)
    _buy_blend_sorted = sorted(
        [(t, _blend_scores.get(t, -999)) for t in _rec_weights.keys()],
        key=lambda x: -x[1]
    )
    _blend_tier = {t: i+1 for i, (t, _) in enumerate(_buy_blend_sorted)}

    # ── PRICE STALENESS GUARD (always-on: cache + live) ──────────────
    # The silent-stale-price bug: if the pipeline ran before the latest daily bar
    # posted, current_price is a prior session close (NVDA showed 211.14 = May 29
    # close while June 1 closed 224.36). Make staleness LOUD, not silent.
    import pandas as _pd_stale
    _pdates = [r.price_date for r in signal_summary if getattr(r, "price_date", None)]
    if _pdates:
        _maxpd = max(_pdates)
        _today_n = _pd_stale.Timestamp.now(tz="America/New_York").normalize().tz_localize(None)
        _last_bday = _today_n if _today_n.weekday() < 5 else _today_n - _pd_stale.offsets.BDay(1)
        try:
            _days_behind = int(_pd_stale.bdate_range(_pd_stale.Timestamp(_maxpd), _last_bday).size) - 1
        except Exception:
            _days_behind = 0
        if _days_behind >= 1:
            st.warning(
                f"⚠️ Forecast prices are as of **{_maxpd}** — markets have traded "
                f"{_days_behind}+ session(s) since; prices & targets below are STALE. "
                f"Click **🔄 Refresh Live** for current prices."
            )
        else:
            st.caption(f"💲 Prices as of close {_maxpd}")

    forecast_rows = []
    for r in signal_summary:
        exp_ret = r.expected_return or 0.0
        # Lean: direction implied by prob_up regardless of BUY/HOLD/SELL
        prob = r.today_prob_eff
        if prob >= 0.65:   lean = "⬆️ Strong UP"
        elif prob >= 0.55: lean = "⬆️ Weak UP"
        elif prob >= 0.45: lean = "⬇️ Weak DOWN"
        else:              lean = "⬇️ Strong DOWN"

        # Rec Weight — only for BUYs, blank for HOLD/SELL
        _rec_w = _rec_weights.get(r.ticker)
        rec_weight_str = f"{_rec_w:.1%}" if _rec_w and r.today_signal == "BUY" else "—"
    
        # Phase 2H — A8 prob + Blend Score + Tier (May 27 2026)
        _a8_val = _a8_lookup.get(r.ticker)
        a8_str = f"{_a8_val:.1%}" if _a8_val is not None else "—"
        _blend = _blend_scores.get(r.ticker)
        blend_str = f"{_blend:+.2f}" if _blend is not None and r.today_signal == "BUY" else "—"
        _tier = _blend_tier.get(r.ticker)
        if _tier is not None and r.today_signal == "BUY":
            if _tier <= 5:
                tier_str = f"🥇 #{_tier}"
            elif _tier <= 10:
                tier_str = f"🥈 #{_tier}"
            else:
                tier_str = f"#{_tier}"
        else:
            tier_str = "—"

        forecast_rows.append({
            "Ticker":       r.ticker,
            "Signal":       r.today_signal,
            "Lean":         lean,
            "Price":        f"${r.current_price:.2f}"    if r.current_price   else "—",
            "Prob Raw":     f"{r.today_prob:.1%}",
            "Prob Eff":     f"{r.today_prob_eff:.1%}",
            "Rec Weight":   rec_weight_str,
            "A8":           a8_str,
            "Blend":        blend_str,
            "Rank":         tier_str,
            "Target ▲":     f"${r.price_target_up:.2f}"  if r.price_target_up else "—",
            "Target ▼":     f"${r.price_target_dn:.2f}"  if r.price_target_dn else "—",
            "Exp Return":   f"{exp_ret:+.2%}"             if r.expected_return is not None else "—",
            "ATR":          f"${r.atr:.2f}"               if r.atr             else "—",
            "Sharpe":       (("⚠ " + f"{r.metrics.sharpe:.2f}") if (not np.isnan(r.metrics.sharpe) and not getattr(r.metrics, "sharpe_reliable", False)) else (f"{r.metrics.sharpe:.2f}" if not np.isnan(r.metrics.sharpe) else "—")),
            "Bucket":       _TICKER_META.get(r.ticker, {}).get("bucket", "—"),
            "Tier":         _TICKER_META.get(r.ticker, {}).get("tier", "—"),
        })

    fdf = pd.DataFrame(forecast_rows)

    def _color_signal(val):
        if val == "BUY":  return "color: #22c55e; font-weight: bold"
        return "color: #94a3b8"

    def _color_exp(val):
        if val == "—": return ""
        try:
            n = float(val.replace("%","").replace("+",""))
            return "color: #22c55e" if n >= 0 else "color: #ef4444"
        except: return ""

    # ── Styled forecast table via HTML component ─────────────────────────────
    import json
    signals_json = json.dumps(forecast_rows)
    html = f"""
    <style>
      @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&display=swap');
      *{{box-sizing:border-box;margin:0;padding:0;}}
      .ft{{font-family:'IBM Plex Mono',monospace;background:#0a0a0f;border:1px solid #1e1e2e;border-radius:8px;overflow:clip;}}
      .ft-head{{position:sticky;top:0;z-index:5;display:grid;grid-template-columns:7% 6% 8% 11% 7% 6% 7% 7% 9% 9% 8% 7% 8%;padding:8px 14px;background:#0d0d18;font-size:10px;color:#4a5568;letter-spacing:.08em;border-bottom:1px solid #1e1e2e;}}
      .ft-head span{{text-align:right;cursor:pointer;user-select:none;}} .ft-head span:first-child,.ft-head span:nth-child(2){{text-align:left;}}
      .ft-head span:hover{{color:#94a3b8;}}
      .ft-head span.sort-asc::after{{content:" ▲";font-size:8px;}}
      .ft-head span.sort-desc::after{{content:" ▼";font-size:8px;}}
      .ft-row{{display:grid;grid-template-columns:7% 6% 8% 11% 7% 6% 7% 7% 9% 9% 8% 7% 8%;padding:11px 14px;border-bottom:1px solid #0f0f1a;transition:background .12s;}}
      .ft-row:hover{{background:#13131f;}}
      .ft-row span{{font-size:12px;color:#cbd5e1;display:flex;align-items:center;justify-content:flex-end;}}
      .ft-row span:first-child{{font-weight:600;color:#f8fafc;font-size:13px;justify-content:flex-start;}}
      .ft-row span:nth-child(2){{justify-content:flex-start;}}
      .badge{{font-size:10px;font-weight:600;padding:2px 7px;border-radius:3px;}}
      .buy{{color:#22c55e;background:#052e16;border:1px solid #166534;}}
      .hold{{color:#f59e0b;background:#1a1008;border:1px solid #7c4a00;}}
      .sell{{color:#ef4444;background:#1c0a0a;border:1px solid #7f1d1d;}}
      .up{{color:#22c55e !important;}}
      .dn{{color:#ef4444 !important;}}
      .dim{{color:#64748b !important;}}
      .prob-col{{display:flex;flex-direction:column;gap:3px;align-items:flex-end;}}
      .bar-bg{{width:70px;height:3px;background:#1e1e2e;border-radius:2px;overflow:hidden;}}
      .bar-fill{{height:100%;border-radius:2px;}}
      .legend{{margin-top:10px;padding:10px 14px;background:#0d0d18;border:1px solid #1e1e2e;border-radius:6px;font-size:11px;color:#4a5568;line-height:1.8;}}
    </style>
    <div class="ft">
      <div class="ft-head">
        <span>TICKER</span><span>SIGNAL</span><span>PRICE</span>
        <span>PROB EFF</span><span>REC % <span style="cursor:help;color:#3b82f6;font-size:11px;" title="Recommended Weight — % of trading budget to allocate to this BUY relative to other BUYs today.&#10;&#10;Formula: (prob_raw - 0.5) / sum_of_all_BUY_convictions&#10;&#10;Interpretation:&#10;  &gt; 15% = top conviction, strongest BUY of the day&#10;  8-15% = high conviction, well above average&#10;  4-8%  = average conviction&#10;  &lt; 4% = low conviction, barely above 0.5&#10;&#10;Example: $10,000 budget, ADSK at 25.8% = $2,580&#10;&#10;Note: No max-weight cap. Your risk tolerance overrides.&#10;Most retail traders cap at 10-15% per single position.&#10;&#10;A/B test result (May 27 2026, 89 buckets):&#10;Conviction-weight ≈ equal-weight on real BUY portfolios.&#10;Diff: -1pp to +1pp over ~30 days. Treat as guidance, not edge.">&#9432;</span></span><span>RANK <span style="cursor:help;color:#3b82f6;font-size:11px;" title="Phase 2H Rank — ⚠ MODEL KILLED May 31 2026 (ranker leak + A8 ceiling, 5 tests). MONITORING ONLY, not traded. Position in blend score ranking among today's BUYs.&#10;&#10;🥇 #1-5  = top 5 BUYs (highest combined conviction)&#10;🥈 #6-10 = next 5 BUYs&#10;#11+    = lower-tier BUYs&#10;&#10;Backtest (33 days, Apr-May 2026):&#10;Top-5 by blend score: +174% cum return vs +117% baseline&#10;w_prob=0.3, w_a8=0.7, stable across H1/H2 splits">&#9432;</span></span><span>A8 <span style="cursor:help;color:#3b82f6;font-size:11px;" title="A8 prob — ⚠ KILLED/at-ceiling (5 overlay tests failed, struck off 1.8). MONITORING ONLY, not traded. A8 model's prob(top decile cross-sectional return).&#10;&#10;Where ticker ranks in universe by 5-day fwd return.&#10;&#10;Interpretation:&#10;  &gt; 20% = strong cross-sectional standout&#10;  10-20% = above-average universe rank&#10;  &lt; 10% = below-average rank&#10;&#10;IC = 0.111 (real cross-sectional alpha)">&#9432;</span></span><span>BLEND <span style="cursor:help;color:#3b82f6;font-size:11px;" title="Blend score — ⚠ KILLED May 31 2026. The H1/H2 stability claims below were refuted by five later tests (A8 ceiling). MONITORING ONLY, not traded. Cross-sectional z-scored combination.&#10;&#10;Formula: 0.3 × prob_raw_z + 0.7 × a8_z&#10;&#10;Higher = better candidate among today's BUYs&#10;&#10;Optimal weights validated by stability test:&#10;  H1 (Mar-Apr): +68% cum&#10;  H2 (Apr-May): +72% cum">&#9432;</span></span>
        <span>TARGET ▲</span><span>TARGET ▼</span><span>EXP RETURN</span><span>ATR</span><span>SHARPE <span style="cursor:help;color:#3b82f6;font-size:11px;" title="Sharpe ratio — annualized risk-adjusted return (return &#247; volatility, scaled to 252 trading days).&#10;&#10;Ranges:&#10;  &lt; 0   = losing money (avg return negative)&#10;  0-1   = mediocre&#10;  1-2   = good&#10;  2-3   = very good&#10;  &gt; 3   = excellent — but verify (suspiciously high can mean overfit or leakage)&#10;&#10;IMPORTANT: high probability + low/negative Sharpe = wins often but the few big losses make it net-negative. Trust prob and Sharpe TOGETHER, not probability alone.">&#9432;</span></span>
      </div>
      <div id="tbody"></div>
    </div>
    <div class="legend">
      📖 &nbsp;
      <span style="color:#22c55e">▲ Target = price + ATR</span> &nbsp;·&nbsp;
      <span style="color:#ef4444">▼ Target = price − ATR</span> &nbsp;·&nbsp;
      <span style="color:#94a3b8">Exp Return = prob-weighted gain/loss</span> &nbsp;·&nbsp;
      <span style="color:#3b82f6">Bar = prob vs threshold</span>
    </div>
    <script>
      let data = {signals_json};
      let sortCol = null;
      let sortDir = 1;
      const tbody = document.getElementById('tbody');
      const headers = document.querySelectorAll('.ft-head span');
      const colKeys = ['Ticker','Signal','Price','Prob Eff','Rec Weight','Rank','A8','Blend','Target ▲','Target ▼','Exp Return','ATR','Sharpe'];

      function parseVal(v) {{
        if (!v || v === '—') return -Infinity;
        const n = parseFloat(String(v).replace(/[%$+]/g,''));
        return isNaN(n) ? String(v) : n;
      }}

      function renderRows() {{
        tbody.innerHTML = '';
        data.forEach(r => {{
        const prob = parseFloat(r['Prob Eff']);
        const exp  = parseFloat(r['Exp Return']);
        const sh   = parseFloat(r['Sharpe']);
        const sig  = r['Signal'];
        const bc   = prob >= 65 ? '#22c55e' : prob >= 55 ? '#f59e0b' : '#3b82f6';
        const sc   = sh >= 2 ? '#22c55e' : sh >= 1 ? '#f59e0b' : '#ef4444';
        const badgeClass = sig==='BUY' ? 'buy' : sig==='SELL' ? 'sell' : 'hold';
        const row = document.createElement('div');
        row.className = 'ft-row';
        row.innerHTML = `
          <span>${{r.Ticker}}</span>
          <span><span class="badge ${{badgeClass}}">${{sig}}</span></span>
          <span>${{r.Price}}</span>
          <span>
            <div class="prob-col">
              <span style="color:#94a3b8;font-size:12px">${{r['Prob Eff']}}</span>
              <div class="bar-bg"><div class="bar-fill" style="width:${{Math.min(prob,100)}}%;background:${{bc}}"></div></div>
              <span style="font-size:9px;color:#2d3748">threshold: 65%</span>
            </div>
          </span>
          <span style="color:${{sig==='BUY'?'#22c55e':'#475569'}};font-weight:600">${{r['Rec Weight']}}</span>
          <span style="color:${{r.Rank && r.Rank.indexOf('🥇')>=0?'#fbbf24':r.Rank && r.Rank.indexOf('🥈')>=0?'#94a3b8':'#475569'}};font-weight:500">${{r.Rank}}</span>
          <span style="color:#a78bfa;font-weight:500">${{r.A8}}</span>
          <span style="color:${{parseFloat(r.Blend)>=0?'#22c55e':parseFloat(r.Blend)<0?'#ef4444':'#475569'}};font-weight:500">${{r.Blend}}</span>
          <span class="up">${{r['Target ▲']}}</span>
          <span class="dn">${{r['Target ▼']}}</span>
          <span style="color:${{exp>=0?'#22c55e':'#ef4444'}};font-weight:500">${{r['Exp Return']}}</span>
          <span class="dim">${{r.ATR}}</span>
          <span style="color:${{sc}};font-weight:500">${{r.Sharpe}}</span>
        `;
        tbody.appendChild(row);
        }});
      }}

      headers.forEach((h, i) => {{
        h.style.cursor = 'pointer';
        h.addEventListener('click', () => {{
          headers.forEach(x => x.classList.remove('sort-asc','sort-desc'));
          if (sortCol === i) {{ sortDir *= -1; }} else {{ sortCol = i; sortDir = 1; }}
          h.classList.add(sortDir === 1 ? 'sort-asc' : 'sort-desc');
          data.sort((a, b) => {{
            const va = parseVal(a[colKeys[i]]);
            const vb = parseVal(b[colKeys[i]]);
            if (va < vb) return -1 * sortDir;
            if (va > vb) return 1 * sortDir;
            return 0;
          }});
          renderRows();
        }});
      }});

      renderRows();
    </script>
    """
    st.components.v1.html(html, height=min(80 + len(forecast_rows) * 44, 800), scrolling=True)

with tab_watch:
    # ── Watchlist Section ─────────────────────────────────────────────────────
    import json as _json
    from pathlib import Path as _Path
    _wl_cache = _Path(_ROOT) / "data" / "watchlist_cache.json"
    if _wl_cache.exists():
        try:
            _wl_data = _json.loads(_wl_cache.read_text())
            _wl_sigs = [s for s in _wl_data.get("signals", []) if s.get("horizon") == 1]
            if _wl_sigs:
                st.caption("Predictions only — excluded from accuracy scoring. Volatile tickers for monitoring.")
                _wl_rows = []
                for s in _wl_sigs:
                    _wl_rows.append({
                        "Ticker":   s.get("ticker", ""),
                        "Signal":   s.get("signal", ""),
                        "Prob Raw": f"{s.get('prob', 0)*100:.1f}%",
                        "Prob Eff": f"{s.get('prob_eff', 0)*100:.1f}%",
                        "Price":    f"${s.get('current_price', 0):.2f}" if s.get("current_price") else "—",
                        "Target ▲": f"${s.get('price_target_up', 0):.2f}" if s.get("price_target_up") else "—",
                        "Target ▼": f"${s.get('price_target_dn', 0):.2f}" if s.get("price_target_dn") else "—",
                        "Exp Ret":  f"{s.get('expected_return', 0)*100:+.2f}%" if s.get("expected_return") else "—",
                    })
                _wldf = pd.DataFrame(_wl_rows)
                def _wl_color(val):
                    if val == "BUY":  return "color: #22c55e; font-weight: bold"
                    if val == "SELL": return "color: #ef4444; font-weight: bold"
                    return "color: #94a3b8"
                st.dataframe(
                    _wldf.style.applymap(_wl_color, subset=["Signal"]),
                    use_container_width=True, hide_index=True
                )
                st.caption(f"Last updated: {_wl_data.get('generated_at', '—')}")
        except Exception as _e:
            pass

with tab_intraday:
    st.caption("Compares EOD model signal with intraday 1hr/2hr/4hr momentum · tickers from your strategy run")

    try:
        from features.intraday_builder import get_all_intraday_signals, is_market_open
        import json
        from pathlib import Path
        from datetime import date as _date

        intraday_tickers = [r.ticker for r in signal_summary]

        # Read latest intraday snapshot from cron (22:00 VN / 11:00 ET)
        # Use most recent file, not today's — handles VN/ET date crossover
        _snap_dir = Path("data/intraday_history")
        _snap_files = sorted(_snap_dir.glob("*.json")) if _snap_dir.exists() else []
        _snap_path = _snap_files[-1] if _snap_files else None
        intra_sigs = []
        _snap_ts = None
        if _snap_path and _snap_path.exists():
            try:
                with open(_snap_path) as _f:
                    _snap = json.load(_f)
                intra_sigs = _snap.get("signals", []) if isinstance(_snap, dict) else _snap
                if isinstance(_snap, dict):
                    _snap_ts = _snap.get("generated_at", "—")
                else:
                    from datetime import datetime as _dt
                    _snap_ts = _dt.fromtimestamp(_snap_path.stat().st_mtime).strftime("%H:%M VN")
            except Exception:
                intra_sigs = []

        if intra_sigs:
            st.caption(f"📦 Loaded from snapshot: {_snap_ts} · {len(intra_sigs)} tickers (refresh dashboard cron at 22:00 VN to update)")
        else:
            with st.spinner("No snapshot — fetching live intraday signals (slow)..."):
                intra_sigs = get_all_intraday_signals(intraday_tickers)

        sig_lkp  = {s["ticker"]: s for s in intra_sigs}
        eod_lkp  = {r.ticker: r for r in signal_summary}

        def _isig_fmt(s, p):
            if s == "UP":   return f"🟢 UP ({p:.0%})"
            if s == "DOWN": return f"🔴 DOWN ({p:.0%})"
            return f"⚪ NTRL ({p:.0%})"

        def _alignment(eod_sig, i1, i2, i4):
            up = [i1,i2,i4].count("UP")
            dn = [i1,i2,i4].count("DOWN")
            if eod_sig == "BUY"  and up >= 2: return "🔥 BOTH BULLISH"
            if eod_sig == "SELL" and dn >= 2: return "🔥 BOTH BEARISH"
            if eod_sig == "BUY"  and dn >= 2: return "⚠️ CONFLICT"
            if eod_sig == "SELL" and up >= 2: return "⚠️ CONFLICT"
            if up >= 2: return "📈 INTRA BULL"
            if dn >= 2: return "📉 INTRA BEAR"
            return "➖ NEUTRAL"

        # Intraday price sanity check
        intra_price_issues = []
        for t in intraday_tickers:
            s = sig_lkp.get(t)
            e = eod_lkp.get(t)
            if s and e and s.get("current_price") and e.current_price:
                diff = abs(s["current_price"] - e.current_price) / e.current_price
                if diff > 0.10:
                    intra_price_issues.append(f"{t} (intraday ${s['current_price']:.2f} vs EOD ${e.current_price:.2f})")
        if intra_price_issues:
            st.warning(f"⚠️ Intraday price mismatch on: {', '.join(intra_price_issues)}")

        align_rows = []
        for t in intraday_tickers:
            s = sig_lkp.get(t)
            e = eod_lkp.get(t)
            if not s or not e or not s.get("current_price"):
                continue
            i1 = s["signal_1hr"]; i2 = s["signal_2hr"]; i4 = s["signal_4hr"]
            align_rows.append({
                "Ticker":    t,
                "Price":     f"${s['current_price']:.2f}",
                "EOD Signal": e.today_signal,
                "EOD Prob":  f"{e.today_prob_eff:.0%}",
                "1hr":       _isig_fmt(i1, s["prob_1hr"]),
                "2hr":       _isig_fmt(i2, s["prob_2hr"]),
                "4hr":       _isig_fmt(i4, s["prob_4hr"]),
                "Alignment": _alignment(e.today_signal, i1, i2, i4),
            })

        if align_rows:
            adf = pd.DataFrame(align_rows)

            # Sort: BOTH BULLISH first, then INTRA BULL, NEUTRAL, CONFLICT, INTRA BEAR
            sort_order = {"🔥 BOTH BULLISH": 0, "📈 INTRA BULL": 1, "➖ NEUTRAL": 2,
                          "⚠️ CONFLICT": 3, "📉 INTRA BEAR": 4, "🔥 BOTH BEARISH": 5}
            adf["_sort"] = adf["Alignment"].map(sort_order).fillna(9)
            adf = adf.sort_values("_sort").drop(columns=["_sort"])

            st.dataframe(adf, use_container_width=True, hide_index=True)

            # Summary counts
            counts = adf["Alignment"].value_counts()
            summary_parts = [f"{v}× {k}" for k,v in counts.items()]
            st.caption("  ·  ".join(summary_parts))

            with st.expander("📖 How to read Alignment", expanded=False):
                st.markdown("""
    | Alignment | Meaning | Action |
    |-----------|---------|--------|
    | 🔥 BOTH BULLISH | EOD=BUY + Intraday UP | Highest conviction entry |
    | 🔥 BOTH BEARISH | EOD=SELL + Intraday DOWN | Highest conviction avoid |
    | 📈 INTRA BULL | EOD=HOLD but intraday momentum UP | Watch — may break out |
    | 📉 INTRA BEAR | EOD=HOLD but intraday momentum DOWN | Avoid short-term |
    | ⚠️ CONFLICT | EOD and intraday disagree | Wait for clarity |
    | ➖ NEUTRAL | No strong signal in either direction | Hold current position |
    """)

            # ── Live interpretation ───────────────────────────────────────────
            st.subheader("🧠 Live Interpretation")

            both_bull = [r["Ticker"] for r in align_rows if r["Alignment"] == "🔥 BOTH BULLISH"]
            both_bear = [r["Ticker"] for r in align_rows if r["Alignment"] == "🔥 BOTH BEARISH"]
            intra_bull = [r["Ticker"] for r in align_rows if r["Alignment"] == "📈 INTRA BULL"]
            intra_bear = [r["Ticker"] for r in align_rows if r["Alignment"] == "📉 INTRA BEAR"]
            conflict   = [r["Ticker"] for r in align_rows if r["Alignment"] == "⚠️ CONFLICT"]
            eod_buys   = [r for r in align_rows if r["EOD Signal"] == "BUY"]

            if both_bull:
                tickers_str = ", ".join(both_bull)
                st.success(f"🔥 **Highest conviction BUY:** {tickers_str} — EOD model AND intraday both bullish. Strongest entry signal.")
                try:
                    from signals.position_sizer import get_position_size
                    for ticker_b in both_bull:
                        row = next((r for r in results if r.get("ticker") == ticker_b), None)
                        if row:
                            pos = get_position_size(
                                ticker=ticker_b,
                                prob_eff=float(row.get("prob_eff", 0.7)),
                                confidence=row.get("confidence", "HIGH"),
                                portfolio_value=portfolio_value,
                                current_price=row.get("current_price"),
                            )
                            if pos.final_pct > 0:
                                shares_str = f" (~{pos.shares} shares)" if pos.shares else ""
                                st.info(f"📐 **{ticker_b} suggested size:** {pos.final_pct*100:.1f}% = ${pos.dollars:,.0f}{shares_str}")
                except Exception:
                    pass

            if both_bear:
                tickers_str = ", ".join(both_bear)
                st.error(f"🔥 **Highest conviction AVOID:** {tickers_str} — EOD model AND intraday both bearish.")

            if eod_buys and not both_bull:
                for r in eod_buys:
                    t = r["Ticker"]
                    p = r["EOD Prob"]
                    al = r["Alignment"]
                    if al == "➖ NEUTRAL":
                        st.info(f"✅ **{t} EOD BUY ({p})** — Model confident but intraday neutral. Valid entry, no intraday confirmation yet. Consider waiting for intraday to turn UP.")
                    elif al == "⚠️ CONFLICT":
                        st.warning(f"⚠️ **{t} EOD BUY ({p}) but intraday bearish** — Conflicting signals. Reduce position size or wait.")

            if intra_bull and not both_bull:
                tickers_str = ", ".join(intra_bull)
                st.info(f"📈 **Watch list:** {tickers_str} — Intraday momentum bullish but EOD model cautious. If EOD prob rises above threshold tomorrow, these become BUY candidates.")

            if intra_bear:
                tickers_str = ", ".join(intra_bear)
                st.warning(f"📉 **Avoid short-term:** {tickers_str} — Intraday momentum bearish. Even if EOD signal fires BUY, day traders are selling. Wait for intraday to recover.")

            if conflict:
                tickers_str = ", ".join(conflict)
                st.warning(f"⚠️ **Conflicting signals:** {tickers_str} — EOD and intraday disagree. Hold off until signals align.")

            if not both_bull and not eod_buys and not intra_bull:
                st.info("No strong directional signals right now. Market is in a wait-and-see mode.")
        else:
            st.info("No intraday data available.")

    except Exception as e:
        st.warning(f"Intraday signals unavailable: {e}")

with tab_detail:
    # ── How to read this table ────────────────────────────────────────────────
    with st.expander("📖 How to read the Forecast Table", expanded=False):
        st.markdown("""
    **SIGNAL** — BUY or HOLD.
    - **BUY** fires when `Prob Eff` exceeds the confidence threshold (default 65% in VOLATILE regime, 55% in NEUTRAL).
    - **HOLD** means wait — either probability is too low or the regime is suppressing the signal.

    **PRICE** — Today's closing price. This is the price the model used to generate the forecast.

    **PROB EFF** — The model's confidence that this stock goes UP over the next `horizon` days, after all signal adjustments:
    ```
    Raw ML prob × regime multiplier × sentiment × options flow × short interest
    ```
    - Above threshold → BUY
    - Below threshold → HOLD
    - Currently in VOLATILE regime, threshold = 65%

    **TARGET ▲** — Where the stock is likely to go if it moves UP.
    Calculated as: `current price + ATR × √horizon`
    Example: NVDA at $121 with ATR $5.35 → Target ▲ = $126.35

    **TARGET ▼** — Where it goes if it moves DOWN.
    Calculated as: `current price − ATR × √horizon`
    This is your downside risk if the signal is wrong.

    **EXP RETURN** — Probability-weighted expected return. The single most actionable number:
    ```
    = (Prob_eff × upside%) − ((1 − Prob_eff) × downside%)
    ```
    - **Positive (green)** = model expects to make money → worth watching
    - **Negative (red)** = model expects to lose money → stay out
    - Right now all negative because VOLATILE regime is suppressing prob_eff below 50%

    **ATR** — Average True Range. How much this stock moves on a typical day in dollars.
    - Low ATR (e.g. NVO $3.73) = stable, lower risk
    - High ATR (e.g. TSLA $13.09) = volatile, bigger swings both ways

    **SHARPE** — Historical risk-adjusted return from backtesting.
    - Above 2.0 = excellent (green)
    - 1.0–2.0 = good (yellow)
    - Below 1.0 = poor (red)

    ---
    **When to act:**
    1. Exp Return turns **green** on a ticker
    2. Prob Eff crosses the threshold → **BUY fires**
    3. Regime shifts from VOLATILE → NEUTRAL/BULL (threshold drops from 65% → 55%)
    """)

    # ── Per-ticker detail ─────────────────────────────────────────────────────
    for result in signal_summary:
        with st.expander(f"📊 {result.ticker} — Detail", expanded=False):
            m = result.metrics

            # Backtest KPIs
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Sharpe",   f"{m.sharpe:.2f}"       if not np.isnan(m.sharpe)       else "—",
                      help="Annualized risk-adjusted return. Returns ÷ volatility, scaled to 252 trading days. Sharpe > 1 is good; > 2 is excellent.")
            c2.metric("Max DD",   f"{m.max_drawdown:.1%}"  if not np.isnan(m.max_drawdown) else "—",
                      help="Maximum peak-to-trough drawdown over the backtest period. Smaller is better. -20% means down 20% at the worst point before recovery.")
            c3.metric("CAGR",     f"{m.cagr:.1%}"          if not np.isnan(m.cagr)         else "—",
                      help="Compound Annual Growth Rate. The single yearly return that, compounded, produces the backtest's total return.")
            c4.metric("Accuracy", f"{m.accuracy:.1%}"      if not np.isnan(m.accuracy)     else "—",
                      help="Fraction of predictions where direction (up/down) matched actual outcome. 50% = coin flip. Current realistic ceiling for retail equity ML is ~55%.")

            st.caption(
                f"Trades: {m.n_trades} · "
                f"Exposure: {m.exposure:.1%} · "
                f"Profit factor: {m.profit_factor:.2f}"
                if not np.isnan(m.profit_factor) else
                f"Trades: {m.n_trades} · Exposure: {m.exposure:.1%}"
            )

            # Equity curve
            if not hasattr(result, 'signal_df') or result.signal_df is None:
                st.info("Signal detail not available in cached mode. Click Refresh Live for full detail.")
                continue
            sdf = result.signal_df.copy()
            sdf["date"] = pd.to_datetime(sdf["date"])
            sdf = sdf.sort_values("date")

            ret_strat = (sdf["signal"] * sdf["return_1d"]).fillna(0)
            ret_mkt   = sdf["return_1d"].fillna(0)
            eq = pd.DataFrame({
                "Strategy": (1 + ret_strat).cumprod(),
                "Market":   (1 + ret_mkt).cumprod(),
            }, index=sdf["date"]).dropna()

            st.line_chart(eq)

            # Signal table (last 20 rows)
            show_cols = [c for c in
                ["date", "close", "prob", "prob_eff", "signal_raw", "gate_block"]
                if c in sdf.columns]
            st.dataframe(
                sdf[show_cols].tail(20).style.format({
                    "close":    "{:.2f}",
                    "prob":     "{:.1%}",
                    "prob_eff": "{:.1%}",
                }),
                use_container_width=True,
            )

            # CSV buffer for ZIP
            csv_bytes = sdf.to_csv(index=False).encode()
            st.download_button(
                f"⬇️ CSV — {result.ticker}",
                csv_bytes,
                file_name=f"{result.ticker}_signals.csv",
                mime="text/csv",
                key=f"csv_{result.ticker}",
            )
            csv_buffers.append((f"{result.ticker}_signals.csv", csv_bytes))

    # ── ZIP download ──────────────────────────────────────────────────────────
    if enable_zip and csv_buffers:
        zbuf = io.BytesIO()
        with zipfile.ZipFile(zbuf, "w", zipfile.ZIP_DEFLATED) as zf:
            for fname, data in csv_buffers:
                zf.writestr(fname, data)
        st.download_button(
            "📦 Download ALL as ZIP",
            zbuf.getvalue(),
            file_name="signals_export.zip",
            mime="application/zip",
            key="zip_all",
        )


    # ══════════════════════════════════════════════════════════════════════════════
    #  ACCURACY SECTION
    # ══════════════════════════════════════════════════════════════════════════════

with tab_accuracy:

    # ── EOD + Intraday Accuracy Tables ───────────────────────────────────────────
    acc_tab1, acc_tab2 = st.tabs(["📅 EOD Model Accuracy", "⚡ Intraday Accuracy"])

    with acc_tab1:
        try:
            from accuracy.sink import get_eod_accuracy_summary, get_spy_relative_accuracy
        
            # BUY/SELL accuracy
            eod_acc = get_eod_accuracy_summary()
            if eod_acc:
                edf = pd.DataFrame(eod_acc)
                edf["accuracy"]   = edf["accuracy"].apply(lambda x: f"{x:.1%}" if x is not None else "N/A")
                edf["avg_return"] = edf["avg_return"].apply(lambda x: f"{x:+.2%}" if x is not None else "N/A")
                edf.columns = ["Ticker", "# Outcomes", "Accuracy (BUY/SELL)", "Avg Return"]
                st.dataframe(edf, use_container_width=True, hide_index=True)
                valid = [r for r in eod_acc if r["accuracy"] is not None]
                if valid:
                    avg = sum(r["accuracy"] for r in valid) / len(valid)
                    st.caption(f"Overall BUY/SELL accuracy: {avg:.1%} · Only {sum(r['n'] for r in eod_acc if r['accuracy'] is not None)} BUY/SELL signals so far — need 60+ for statistical significance")
            else:
                st.info("No EOD accuracy data yet.")

            # SPY-relative accuracy
            st.markdown("---")
            st.markdown("**📊 Daily Performance vs SPY**")
            st.caption("Are our tickers outperforming the market each day? More meaningful than BUY accuracy with small sample.")
            spy_acc = get_spy_relative_accuracy()
            if spy_acc:
                sdf = pd.DataFrame(spy_acc)
                sdf["spy_ret"]      = sdf["spy_ret"].apply(lambda x: f"{x:+.2%}")
                sdf["avg_ret"]      = sdf["avg_ret"].apply(lambda x: f"{x:+.2%}")
                sdf["avg_vs_spy"]   = sdf["avg_vs_spy"].apply(lambda x: f"{x:+.2%}")
                sdf["pct_beat_spy"] = sdf["pct_beat_spy"].apply(lambda x: f"{x:.0%}")
                sdf["buy_acc"]      = sdf["buy_acc"].apply(lambda x: f"{x:.0%}" if x is not None else "—")
                sdf.columns = ["Date","SPY Return","Avg Return","Avg vs SPY","% Beat SPY","# BUYs","BUY Acc"]
                st.dataframe(sdf, use_container_width=True, hide_index=True)
                avg_vs_spy = sum(r["avg_vs_spy"] for r in spy_acc) / len(spy_acc)
                avg_beat   = sum(r["pct_beat_spy"] for r in spy_acc) / len(spy_acc)
                if avg_vs_spy > 0:
                    st.success(f"✅ On average our tickers beat SPY by {avg_vs_spy:+.2%} per day · {avg_beat:.0%} of tickers beat SPY")
                else:
                    st.warning(f"⚠️ On average our tickers underperform SPY by {avg_vs_spy:+.2%} per day")
            else:
                st.info("No SPY comparison data yet.")
        except Exception as e:
            st.warning(f"EOD accuracy unavailable: {e}")

    with acc_tab2:
        try:
            from accuracy.sink import get_intraday_accuracy_summary, reconcile_intraday_outcomes
            reconcile_intraday_outcomes()
            intra_acc = get_intraday_accuracy_summary()
            if intra_acc:
                idf = pd.DataFrame(intra_acc)
                idf["accuracy"] = idf["accuracy"].apply(lambda x: f"{x:.1%}" if x is not None else "N/A")
                idf["horizon_hr"] = idf["horizon_hr"].apply(lambda x: f"{x}hr")
                idf = idf.drop(columns=["computed_at"])
                idf.columns = ["Ticker", "Horizon", "Accuracy", "# Predictions"]
                st.dataframe(idf, use_container_width=True, hide_index=True)
                valid = [r for r in intra_acc if r["accuracy"] is not None]
                if valid:
                    avg = sum(r["accuracy"] for r in valid) / len(valid)
                    st.caption(f"Overall intraday accuracy: {avg:.1%} · Needs 5+ outcomes per ticker to be meaningful")
            else:
                st.info("No intraday accuracy data yet — check back after market hours once outcomes are reconciled.")
        except Exception as e:
            st.warning(f"Intraday accuracy unavailable: {e}")

    @st.cache_data(ttl=300)
    def _load_accuracy(horizon_filter):
        try:
            return load_accuracy(horizon=horizon_filter, window_days=90)
        except Exception:
            return pd.DataFrame()

    acc_horizon = st.selectbox("Accuracy horizon", [1, 3, 5],
                                format_func=lambda x: f"{x}d", key="acc_hz")
    acc_df = _load_accuracy(acc_horizon)

    if acc_df.empty:
        st.info(
            "No accuracy data yet. Predictions are logged each time you run the strategy. "
            "Accuracy is computed after the horizon passes and outcomes can be verified."
        )
    else:
        c1, c2, c3 = st.columns(3)
        c1.metric("Avg Accuracy",  f"{acc_df['accuracy'].mean():.1%}",
                  help="Mean accuracy across all tickers in the universe. Walk-forward baseline (May 9): h=1 0.5172, h=3 0.4994, h=5 0.5071.")
        c2.metric("Avg ROC-AUC",   f"{acc_df['roc_auc'].mean():.3f}",
                  help="Receiver Operating Characteristic Area Under Curve. Measures ranking quality: 0.5 = random, 1.0 = perfect. Walk-forward baseline: 0.52.")
        c3.metric("Avg Brier",     f"{acc_df['brier_score'].mean():.3f}",
                  help="Brier score = mean squared error between predicted probability and actual outcome. Lower is better. 0.25 = uninformative.")

        st.dataframe(
            acc_df[["ticker", "horizon", "accuracy", "roc_auc",
                    "brier_score", "n_predictions"]]
            .style.format({
                "accuracy":    lambda x: f"{x:.1%}" if x is not None and str(x) != 'nan' else "N/A",
                "roc_auc":     lambda x: f"{x:.3f}" if x is not None and str(x) != 'nan' else "N/A",
                "brier_score": lambda x: f"{x:.3f}" if x is not None and str(x) != 'nan' else "N/A",
            }),
            use_container_width=True,
        )

        # Accuracy trend for selected ticker
        acc_tickers = acc_df["ticker"].tolist()
        if acc_tickers:
            sel = st.selectbox("Trend for ticker", acc_tickers, key="acc_trend_tkr")
            hist = load_prediction_history(sel, horizon=acc_horizon, days=90)
            if not hist.empty:
                hist["rolling_acc"] = hist["correct"].rolling(20, min_periods=5).mean()
                chart = (
                    alt.Chart(hist)
                    .mark_line()
                    .encode(
                        x=alt.X("prediction_date:T", title="Date"),
                        y=alt.Y("rolling_acc:Q", title="Rolling Accuracy (20d)",
                                scale=alt.Scale(domain=[0, 1])),
                        tooltip=["prediction_date:T", "rolling_acc:Q",
                                 "signal:N", "actual_up:Q"],
                    )
                    .properties(title=f"{sel} — Rolling 20-day Accuracy")
                )
                st.altair_chart(chart, use_container_width=True)

        # Reconcile button (run manually or set on a schedule)
        if not os.environ.get("STREAMLIT_SHARING_MODE") and st.button("🔁 Reconcile outcomes now"):
            with st.spinner("Fetching actual outcomes..."):
                try:
                    n = reconcile_outcomes()
                    st.success(f"✓ {n} new outcomes recorded")
                    st.cache_data.clear()
                    st.rerun()
                except Exception as e:
                    st.error(f"Reconciliation failed: {e}")
