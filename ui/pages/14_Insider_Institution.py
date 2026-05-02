# ui/pages/14_Insider_Institution.py
# ─────────────────────────────────────────────────────────────────────────────
# Insider & Institutional Flow page.
#
# Tab 1: Insider trades — reads existing insider_trades.db (insider_filings_raw
#        table populated by data.etl_insider scraper). Refresh button calls
#        run_insider_etl() in a background thread.
# Tab 2: Institutional flow — reads institutional_trades.db (populated by
#        features.institutional_ingest scraper). Async refresh w/ progress bar.
# ─────────────────────────────────────────────────────────────────────────────
from __future__ import annotations

import sqlite3
import sys
import threading
import time
from datetime import date, datetime, timedelta, timezone
from functools import reduce
from pathlib import Path

import pandas as pd
import streamlit as st

# Make repo root importable so `data.etl_insider` and `features.*` resolve
ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# AgGrid for BI-style filterable tables
try:
    from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode, JsCode
    AGGRID_OK = True
except ImportError:
    AGGRID_OK = False

st.set_page_config(page_title="Insider & Institutional", layout="wide")

# ─── Paths ───────────────────────────────────────────────────────────────────
INSIDER_DB = ROOT / "insider_trades.db"
INSTITUTIONAL_DB = ROOT / "institutional_trades.db"
TICKERS_TXT = ROOT / "tickers.txt"

# ─── Module-level shared state for background ingest ────────────────────────
# Threads write here, Streamlit main thread reads. No session_state coupling
# from threads (avoids context warnings).
_INSIDER_STATE = {"status": "idle", "progress": 0.0, "message": ""}
_INST_STATE = {"status": "idle", "progress": 0.0, "message": ""}
_STATE_LOCK = threading.Lock()


def _set_insider(**kwargs):
    with _STATE_LOCK:
        _INSIDER_STATE.update(kwargs)


def _set_inst(**kwargs):
    with _STATE_LOCK:
        _INST_STATE.update(kwargs)


def _get_insider() -> dict:
    with _STATE_LOCK:
        return dict(_INSIDER_STATE)


def _get_inst() -> dict:
    with _STATE_LOCK:
        return dict(_INST_STATE)


# ─── Helpers ─────────────────────────────────────────────────────────────────

def vn_now() -> str:
    vn = timezone(timedelta(hours=7))
    return datetime.now(vn).strftime("%Y-%m-%d %H:%M VN")


def utc_to_vn(iso_ts: str) -> str:
    if not iso_ts:
        return "Never"
    try:
        dt = datetime.fromisoformat(iso_ts.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone(timedelta(hours=7))).strftime("%Y-%m-%d %H:%M VN")
    except Exception:
        return iso_ts


def load_tickers() -> list[str]:
    if not TICKERS_TXT.exists():
        return []
    return [
        line.strip().upper()
        for line in TICKERS_TXT.read_text().splitlines()
        if line.strip() and not line.startswith("#")
    ]


# ─── Data loading ────────────────────────────────────────────────────────────

@st.cache_data(ttl=30)
def load_insider_filings(start_date: str, end_date: str) -> pd.DataFrame:
    if not INSIDER_DB.exists():
        return pd.DataFrame()
    conn = sqlite3.connect(INSIDER_DB)
    try:
        df = pd.read_sql(
            """
            SELECT id, ticker, trade_date, filing_date,
                   insider_name, insider_title, role_weight, is_csuite,
                   transaction_code, shares, price_per_share, notional_usd,
                   acquired_disposed, accession, fetched_at
              FROM insider_filings_raw
             WHERE trade_date >= ? AND trade_date <= ?
             ORDER BY trade_date DESC, fetched_at DESC
            """,
            conn,
            params=(start_date, end_date),
        )
    finally:
        conn.close()
    return df


@st.cache_data(ttl=30)
def load_alerts_7d() -> pd.DataFrame:
    if not INSIDER_DB.exists():
        return pd.DataFrame()
    conn = sqlite3.connect(INSIDER_DB)
    try:
        cutoff = (datetime.utcnow() - timedelta(days=7)).isoformat()
        df = pd.read_sql(
            """
            SELECT ticker, signal, rationale, notional_usd, insider_title, sent_at
              FROM insider_alerts
             WHERE signal = 'GREEN_STRONG' AND sent_at >= ?
             ORDER BY sent_at DESC
             LIMIT 30
            """,
            conn,
            params=(cutoff,),
        )
    finally:
        conn.close()
    return df


def get_insider_state_db() -> dict:
    """
    Returns last refresh metadata. Prefers insider_raw_scraper_state (populated
    by the new per-transaction scraper); falls back to insider_scraper_state
    (populated by the legacy aggregator / alerts classifier) if the raw table
    doesn't exist yet.
    """
    if not INSIDER_DB.exists():
        return {}
    conn = sqlite3.connect(INSIDER_DB)
    try:
        # Preferred: raw scraper state
        try:
            row = conn.execute(
                """SELECT last_poll_at, last_row_count, last_ticker_count
                     FROM insider_raw_scraper_state WHERE id = 1"""
            ).fetchone()
            if row and row[0]:
                return {
                    "last_poll_at": row[0],
                    "last_row_count": row[1],
                    "last_ticker_count": row[2],
                    "source": "raw_scraper",
                }
        except sqlite3.OperationalError:
            pass

        # Fallback: legacy table
        try:
            row = conn.execute(
                "SELECT last_poll_at FROM insider_scraper_state WHERE id = 1"
            ).fetchone()
            if row and row[0]:
                return {"last_poll_at": row[0], "source": "legacy"}
        except sqlite3.OperationalError:
            pass
    finally:
        conn.close()
    return {}


@st.cache_data(ttl=30)
def load_institutional(start_date: str, end_date: str, min_notional: float) -> pd.DataFrame:
    if not INSTITUTIONAL_DB.exists():
        return pd.DataFrame()
    conn = sqlite3.connect(INSTITUTIONAL_DB)
    try:
        df = pd.read_sql(
            """
            SELECT id, ticker, trade_ts, trade_date, side, shares, price, notional_usd,
                   exchange_code, exchange_name,
                   is_dark_pool, is_block, is_sweep, is_cross,
                   COALESCE(is_algo, 0) as is_algo,
                   COALESCE(is_closing_auction, 0) as is_closing_auction,
                   provider
              FROM institutional_trades
             WHERE trade_date >= ? AND trade_date <= ?
               AND notional_usd >= ?
             ORDER BY trade_ts DESC
             LIMIT 100000
            """,
            conn,
            params=(start_date, end_date, min_notional),
        )
    finally:
        conn.close()
    return df


def get_inst_state_db() -> dict:
    if not INSTITUTIONAL_DB.exists():
        return {}
    conn = sqlite3.connect(INSTITUTIONAL_DB)
    try:
        row = conn.execute(
            """
            SELECT last_poll_at, last_provider, last_row_count, last_ticker_count, last_error
              FROM institutional_scraper_state WHERE id = 1
            """
        ).fetchone()
    except sqlite3.OperationalError:
        return {}
    finally:
        conn.close()
    if not row:
        return {}
    return {
        "last_poll_at": row[0],
        "last_provider": row[1],
        "last_row_count": row[2],
        "last_ticker_count": row[3],
        "last_error": row[4],
    }


# ─── Background ingest workers ───────────────────────────────────────────────

def _worker_insider(days_back: int):
    _set_insider(status="running", progress=0.0, message="Starting insider raw scraper...")
    try:
        from data.etl_insider_raw import run_insider_raw_etl
        tickers = load_tickers()
        if not tickers:
            _set_insider(status="error", message="No tickers loaded from tickers.txt")
            return

        n_tickers = len(tickers)

        def cb(ticker: str, idx: int, total: int, frac: float):
            overall = (idx + frac) / total if total else 0.0
            _set_insider(
                status="running",
                progress=min(overall, 1.0),
                message=f"[{idx+1}/{total}] {ticker} ({frac*100:.0f}%)"
            )

        results = run_insider_raw_etl(tickers, days_back=days_back, progress_cb=cb)
        total = sum(results.values())
        active = sum(1 for v in results.values() if v > 0)
        _set_insider(
            status="done", progress=1.0,
            message=f"✓ Refreshed: {total:,} new transactions across {active}/{n_tickers} tickers"
        )
    except Exception as e:
        _set_insider(status="error", message=f"✗ Error: {e}")


def _worker_institutional(days_back: int, use_cursor: bool):
    _set_inst(status="running", progress=0.0, message=f"Starting institutional ingest ({days_back}d)...")
    try:
        from features.institutional_ingest import run_institutional_ingest
        tickers = load_tickers()
        if not tickers:
            _set_inst(status="error", message="No tickers loaded from tickers.txt")
            return

        n_tickers = len(tickers)

        def cb(ticker: str, idx: int, total: int, frac: float):
            overall = (idx + frac) / total if total else 0.0
            _set_inst(
                status="running",
                progress=min(overall, 1.0),
                message=f"[{idx+1}/{total}] {ticker} ({frac*100:.0f}%)"
            )

        results = run_institutional_ingest(
            tickers, days_back=days_back, use_cursor=use_cursor, progress_cb=cb
        )
        total = sum(results.values())
        active = sum(1 for v in results.values() if v > 0)
        _set_inst(
            status="done", progress=1.0,
            message=f"✓ Refreshed: {total:,} prints across {active}/{n_tickers} tickers"
        )
    except Exception as e:
        _set_inst(status="error", message=f"✗ Error: {e}")


def _start_thread(target, *args):
    t = threading.Thread(target=target, args=args, daemon=True)
    t.start()
    return t


# ─── UI: Header + Refresh cards ──────────────────────────────────────────────

st.title("💼 Insider & Institutional Flow")
st.caption(
    "Per-transaction insider trades (SEC Form 4) and institutional flow (Massive). "
    "Manual refresh — no cron."
)

card_ins, card_inst = st.columns(2)

with card_ins:
    with st.container(border=True):
        ins_db_state = get_insider_state_db()
        last_poll = utc_to_vn(ins_db_state.get("last_poll_at"))
        st.markdown(f"**Insider · last refresh**  \n{last_poll}")

        ins_status = _get_insider()
        if ins_status["status"] == "running":
            st.progress(min(ins_status["progress"], 1.0))
            st.caption(ins_status["message"])
        else:
            cols = st.columns([2, 1])
            with cols[0]:
                if st.button("🔄 Refresh insider", key="btn_refresh_insider", use_container_width=True):
                    _start_thread(_worker_insider, 7)
                    time.sleep(0.2)
                    st.rerun()
            with cols[1]:
                with st.popover("Options", use_container_width=True):
                    days_ins = st.number_input("Days back", min_value=1, max_value=365, value=7, key="ins_days")
                    if st.button("Run with custom days", key="btn_ins_custom"):
                        _start_thread(_worker_insider, int(days_ins))
                        st.rerun()

            if ins_status["status"] == "done":
                st.success(ins_status["message"])
            elif ins_status["status"] == "error":
                st.error(ins_status["message"])

with card_inst:
    with st.container(border=True):
        inst_db_state = get_inst_state_db()
        last_poll_inst = utc_to_vn(inst_db_state.get("last_poll_at"))
        n_rows = inst_db_state.get("last_row_count")
        n_tk = inst_db_state.get("last_ticker_count")
        line2 = ""
        if n_rows is not None and n_tk is not None:
            line2 = f"  \n{int(n_rows):,} prints · {int(n_tk)} tickers"
        st.markdown(f"**Institutional · last refresh**  \n{last_poll_inst}{line2}")

        inst_status = _get_inst()
        if inst_status["status"] == "running":
            st.progress(min(inst_status["progress"], 1.0))
            st.caption(inst_status["message"])
        else:
            cols = st.columns([2, 1])
            with cols[0]:
                if st.button("🔄 Refresh (incremental)", key="btn_refresh_inst", use_container_width=True):
                    _start_thread(_worker_institutional, 30, True)
                    time.sleep(0.2)
                    st.rerun()
            with cols[1]:
                with st.popover("Deep backfill", use_container_width=True):
                    deep_days = st.number_input("Days back", min_value=30, max_value=365, value=90, step=30, key="inst_deep_days")
                    no_cursor = st.checkbox("Ignore cursor (full re-pull)", value=False, key="inst_no_cursor")
                    st.caption("⚠ 90+ days × 125 tickers can take 30+ minutes.")
                    if st.button("Run deep backfill", key="btn_inst_deep"):
                        _start_thread(_worker_institutional, int(deep_days), not no_cursor)
                        st.rerun()

            if inst_status["status"] == "done":
                st.success(inst_status["message"])
            elif inst_status["status"] == "error":
                st.error(inst_status["message"])

# Auto-refresh ONLY while a job is running, so progress updates without freezing
# the rest of the UI. Sleep is 3s to keep CPU low.
if _get_insider()["status"] == "running" or _get_inst()["status"] == "running":
    time.sleep(3)
    st.rerun()

st.divider()

# ─── Tabs ────────────────────────────────────────────────────────────────────

tab_insider, tab_inst = st.tabs(["📋 Insider trades", "🐋 Institutional flow"])

# ════════════════════════════════════════════════════════════════════════════
# TAB 1: INSIDER
# ════════════════════════════════════════════════════════════════════════════

with tab_insider:
    # Alerts ribbon
    alerts_df = load_alerts_7d()
    if not alerts_df.empty:
        st.markdown("**🟢 Recent C-suite buys · last 7 days · GREEN_STRONG alerts**")
        chips = alerts_df.head(10)
        chip_cols = st.columns(len(chips))
        for i, (_, row) in enumerate(chips.iterrows()):
            with chip_cols[i]:
                notional_str = ""
                if row.get("notional_usd") and row["notional_usd"] > 0:
                    notional_str = f" · ${row['notional_usd']/1e6:.1f}M"
                if st.button(f"{row['ticker']}{notional_str}", key=f"chip_{i}_{row['ticker']}"):
                    st.session_state["insider_filter_ticker"] = row["ticker"]
                    st.rerun()
        st.markdown("")

    # Filters
    tickers_list = load_tickers()
    f1, f2, f3 = st.columns([2, 2, 1])
    with f1:
        default_tickers = []
        if st.session_state.get("insider_filter_ticker"):
            default_tickers = [st.session_state["insider_filter_ticker"]]
        sel_tickers = st.multiselect(
            "Tickers",
            options=tickers_list,
            default=default_tickers,
            placeholder="All tickers",
            key="insider_tickers",
        )
    with f2:
        date_range = st.date_input(
            "Date range",
            value=(date.today() - timedelta(days=30), date.today()),
            key="insider_dates",
        )
    with f3:
        csuite_only = st.checkbox("✓ C-suite only", key="insider_csuite_only")

    # Form 4 transaction code labels (UI shows label, SQL filters on letter)
    TX_CODE_LABELS = {
        "P": "P · open buy",
        "S": "S · open sell",
        "M": "M · option exercise",
        "A": "A · grant/award",
        "F": "F · tax withholding",
        "G": "G · gift",
        "D": "D · disposed to issuer",
        "C": "C · derivative conversion",
        "X": "X · other",
    }
    AD_LABELS = {"A": "A · acquired", "D": "D · disposed"}

    # Initialize defaults once: P, S, M selected on first render
    if "insider_tx_codes_init" not in st.session_state:
        st.session_state["insider_tx_codes"] = [
            TX_CODE_LABELS["P"], TX_CODE_LABELS["S"], TX_CODE_LABELS["M"],
        ]
        st.session_state["insider_tx_codes_init"] = True

    f4, f5, f6, f6b = st.columns([2, 1.2, 3.5, 0.4])
    with f4:
        role_tier = st.multiselect(
            "Role tier",
            options=["C-Suite (3.0+)", "Executive (1.5–2.9)", "Director (1.0)", "Other (<1.0)"],
            default=[],
            key="insider_role_tier",
        )
    with f5:
        side_filter_labels = st.multiselect(
            "A/D",
            options=list(AD_LABELS.values()),
            default=[],
            key="insider_ad",
        )
        side_filter = [s.split(" · ")[0] for s in side_filter_labels] if side_filter_labels else []
    with f6:
        tx_codes_labels = st.multiselect(
            "Transaction type",
            options=list(TX_CODE_LABELS.values()),
            key="insider_tx_codes",
        )
        tx_codes = [lbl.split(" · ")[0] for lbl in tx_codes_labels] if tx_codes_labels else []
    with f6b:
        st.markdown("""<div style='padding-top: 28px;'></div>""", unsafe_allow_html=True)
        with st.popover("ⓘ"):
            st.markdown("""
**Form 4 transaction codes**

| Code | Meaning |
|---|---|
| **P · open buy** | Open-market purchase. Insider used own cash. **Strongest bullish signal.** |
| **S · open sell** | Open-market sale for cash. Bearish but noisy — often scheduled 10b5-1 plans. |
| **M · option exercise** | Exercised options into shares. Meaningful only when shares are held. |
| **A · grant** | Compensation award. Not an economic decision by the insider. |
| **F · tax withholding** | Shares withheld to cover tax on a vest/exercise. Mechanical. |
| **G · gift** | Transfer to family or charity. Mostly neutral. |
| **D · disposed to issuer** | Sold back to company (e.g. buyback participation). Rare. |
| **C · derivative conversion** | Conversion of derivative (warrants → stock). Mechanical. |
| **X · other** | Non-standard transaction. Read the filing for context. |

*Defaults P, S, M are selected because they capture the economically
meaningful insider trades. A, F, G are typically execution-mechanics noise
but available when needed.*

**Acquired / Disposed (A/D)** is the direction of the share movement,
independent of the transaction code. An option exercise (M code) is always
A (acquired), even if the insider sells immediately after — that follow-on
sale prints as a separate row with code S, A/D=D.
            """)

    f7, f8 = st.columns([2, 2])
    with f7:
        title_search = st.text_input("Title contains", key="insider_title_q")
    with f8:
        min_notional = st.number_input(
            "Min notional ($)", min_value=0, max_value=100_000_000, value=0, step=10_000,
            key="insider_min_not"
        )

    # Date parsing
    if isinstance(date_range, tuple) and len(date_range) >= 2:
        start_str = date_range[0].isoformat()
        end_str = date_range[1].isoformat()
    elif isinstance(date_range, tuple) and len(date_range) == 1:
        start_str = date_range[0].isoformat()
        end_str = date.today().isoformat()
    else:
        start_str = (date.today() - timedelta(days=30)).isoformat()
        end_str = date.today().isoformat()

    df = load_insider_filings(start_str, end_str)

    # Apply filters
    if not df.empty:
        if sel_tickers:
            df = df[df["ticker"].isin(sel_tickers)]
        if csuite_only:
            df = df[df["is_csuite"] == 1]
        if role_tier:
            tier_filters = []
            if "C-Suite (3.0+)" in role_tier:
                tier_filters.append(df["role_weight"] >= 3.0)
            if "Executive (1.5–2.9)" in role_tier:
                tier_filters.append((df["role_weight"] >= 1.5) & (df["role_weight"] < 3.0))
            if "Director (1.0)" in role_tier:
                tier_filters.append(df["role_weight"] == 1.0)
            if "Other (<1.0)" in role_tier:
                tier_filters.append(df["role_weight"] < 1.0)
            if tier_filters:
                df = df[reduce(lambda a, b: a | b, tier_filters)]
        if title_search:
            df = df[df["insider_title"].fillna("").str.contains(title_search, case=False, na=False)]
        if side_filter:
            df = df[df["acquired_disposed"].isin(side_filter)]
        if tx_codes:
            df = df[df["transaction_code"].isin(tx_codes)]
        if min_notional > 0:
            df = df[df["notional_usd"].fillna(0) >= min_notional]

    # Metric cards
    if not df.empty:
        signed = df.apply(
            lambda r: (r["notional_usd"] or 0) if r["acquired_disposed"] == "A"
            else -(r["notional_usd"] or 0), axis=1,
        )
        net = signed.sum()
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Net flow (filtered)", f"${net/1e6:+,.1f}M")
        m2.metric("C-suite trades", f"{int((df['is_csuite']==1).sum())}")
        m3.metric("Total trades", f"{len(df):,}")
        top_by = df.assign(_s=signed).groupby("ticker")["_s"].sum().sort_values()
        if len(top_by) > 0:
            top_pos = top_by.iloc[-1]
            top_pos_t = top_by.index[-1]
            m4.metric(f"Top buy: {top_pos_t}", f"${top_pos/1e6:+,.1f}M")

    st.markdown("")

    # Table
    if df.empty:
        st.info("No insider trades match current filters.")
    else:
        display_df = df.copy()
        display_df["notional_usd"] = display_df["notional_usd"].fillna(0)
        display_df = display_df[
            [
                "trade_date", "ticker", "insider_name", "insider_title",
                "role_weight", "is_csuite",
                "transaction_code", "acquired_disposed",
                "shares", "price_per_share", "notional_usd",
                "filing_date", "accession",
            ]
        ]

        if AGGRID_OK:
            gb = GridOptionsBuilder.from_dataframe(display_df)
            gb.configure_default_column(
                filter=True, sortable=True, resizable=True, floatingFilter=True
            )
            gb.configure_pagination(paginationAutoPageSize=False, paginationPageSize=50)
            gb.configure_column(
                "notional_usd",
                type=["numericColumn", "numberColumnFilter"],
                valueFormatter=JsCode(
                    "function(params){return params.value!=null?'$'+params.value.toLocaleString(undefined,{maximumFractionDigits:0}):'';}"
                ),
                header_name="Notional",
            )
            gb.configure_column(
                "price_per_share",
                type=["numericColumn"],
                valueFormatter=JsCode(
                    "function(params){return params.value!=null?'$'+params.value.toFixed(2):'';}"
                ),
                header_name="Price",
            )
            gb.configure_column(
                "shares",
                type=["numericColumn"],
                valueFormatter=JsCode(
                    "function(params){return params.value!=null?params.value.toLocaleString():'';}"
                ),
            )
            gb.configure_column("is_csuite", header_name="C-suite", width=90)
            gb.configure_column("role_weight", header_name="Role wt", width=90)
            gb.configure_column("transaction_code", header_name="Code", width=80)
            gb.configure_column("acquired_disposed", header_name="A/D", width=70)
            grid_opts = gb.build()
            AgGrid(
                display_df,
                gridOptions=grid_opts,
                height=600,
                theme="streamlit",
                update_mode=GridUpdateMode.NO_UPDATE,
                allow_unsafe_jscode=True,
                fit_columns_on_grid_load=False,
            )
        else:
            st.warning(
                "Install `streamlit-aggrid` for BI-style filters. "
                "`pip install streamlit-aggrid`"
            )
            st.dataframe(display_df, use_container_width=True, height=600)

        csv = display_df.to_csv(index=False).encode()
        st.download_button(
            "⬇ Export filtered CSV",
            data=csv,
            file_name=f"insider_trades_{start_str}_{end_str}.csv",
            mime="text/csv",
            key="dl_insider",
        )

# ════════════════════════════════════════════════════════════════════════════
# TAB 2: INSTITUTIONAL
# ════════════════════════════════════════════════════════════════════════════

with tab_inst:
    # Filters
    g1, g2 = st.columns([2, 2])
    with g1:
        inst_tickers = st.multiselect(
            "Tickers",
            options=tickers_list,
            default=[],
            placeholder="All tickers",
            key="inst_tickers",
        )
    with g2:
        inst_date_range = st.date_input(
            "Date range",
            value=(date.today() - timedelta(days=2), date.today()),
            key="inst_dates",
        )

    g3, g4, g5, g6, g7 = st.columns(5)
    with g3:
        side_filter_inst = st.multiselect(
            "Side",
            options=["BUY", "SELL", "UNKNOWN"],
            default=["BUY", "SELL"],
            key="inst_side",
        )
    with g4:
        dark_only = st.checkbox("Dark only", key="inst_dark_only")
    with g5:
        block_only = st.checkbox("Block only", key="inst_block_only")
    with g6:
        algo_only = st.checkbox("Algo only", key="inst_algo_only")
    with g7:
        hide_close = st.checkbox("Hide closing auction", value=True, key="inst_hide_close",
                                  help="4 PM ET prints are mostly mechanical rebalancing")

    # Notional with presets + slider
    st.markdown("**Min notional**")
    p_cols = st.columns([1, 1, 1, 1, 1, 4])
    presets = [("$1M", 1_000_000), ("$2.5M", 2_500_000),
               ("$5M", 5_000_000), ("$10M", 10_000_000), ("$25M", 25_000_000)]
    for i, (lbl, val) in enumerate(presets):
        if p_cols[i].button(lbl, key=f"preset_{val}", use_container_width=True):
            st.session_state["inst_min_n"] = val
            st.rerun()
    with p_cols[5]:
        min_n = st.slider(
            "Min notional",
            min_value=250_000, max_value=25_000_000,
            value=st.session_state.get("inst_min_n", 1_000_000),
            step=250_000,
            key="inst_min_n_slider",
            label_visibility="collapsed",
            format="$%d",
        )
        st.session_state["inst_min_n"] = min_n

    # Flag legend popover (the i-icon)
    with st.popover("ⓘ What do flags mean?"):
        st.markdown(
            """
| Flag | Meaning |
|---|---|
| **DP** | **Dark pool** — trade executed off-exchange (FINRA TRF/ADF). Institutions use these to hide intent and avoid moving the price. High dark pool volume in a name often signals accumulation or distribution. |
| **BLK** | **Block trade** — single print ≥ 10,000 shares. Indicates a large player. |
| **ALGO** | **Algorithmic execution** — trade flagged with TRF condition codes indicating VWAP/TWAP, prior-reference-price, or contingent execution. Common for institutional algos working a parent order over time. |
| **CLS** | **Closing auction** — trade printed during the 4 PM ET closing window. Mostly mechanical index/ETF rebalancing rather than directional intent — consider filtering these out for signal analysis. |
| ~~SWP~~ | (Lit-tape only — not populated for dark pool data. Will be populated once lit-tape block ingestion is added.) |
| ~~CRS~~ | (Same as SWP.) |

Side (BUY/SELL/UNKNOWN) is inferred via the **Lee-Ready algorithm** using the
NBBO bid/ask carried with each UW print: trade price compared to the midpoint;
falls back to the tick rule (vs. previous trade price) when at midpoint.
Accuracy ~80% in normal markets.
            """
        )

    # Date parsing
    if isinstance(inst_date_range, tuple) and len(inst_date_range) >= 2:
        i_start = inst_date_range[0].isoformat()
        i_end = inst_date_range[1].isoformat()
    else:
        i_start = (date.today() - timedelta(days=2)).isoformat()
        i_end = date.today().isoformat()

    inst_df = load_institutional(i_start, i_end, float(min_n))

    if not inst_df.empty:
        if inst_tickers:
            inst_df = inst_df[inst_df["ticker"].isin(inst_tickers)]
        if side_filter_inst:
            inst_df = inst_df[inst_df["side"].isin(side_filter_inst)]
        if dark_only:
            inst_df = inst_df[inst_df["is_dark_pool"] == 1]
        if block_only:
            inst_df = inst_df[inst_df["is_block"] == 1]
        if algo_only:
            inst_df = inst_df[inst_df["is_algo"] == 1]
        if hide_close:
            inst_df = inst_df[inst_df["is_closing_auction"] == 0]

    # Metric cards
    if not inst_df.empty:
        signed_inst = inst_df.apply(
            lambda r: r["notional_usd"] if r["side"] == "BUY"
            else (-r["notional_usd"] if r["side"] == "SELL" else 0),
            axis=1,
        )
        net_inst = signed_inst.sum()
        n1, n2, n3, n4 = st.columns(4)
        n1.metric("Net flow (filtered)", f"${net_inst/1e6:+,.1f}M")
        n2.metric("Block trades", f"{int((inst_df['is_block']==1).sum()):,}")
        n3.metric("Dark prints", f"{int((inst_df['is_dark_pool']==1).sum()):,}")
        top_inst = (
            inst_df.assign(_s=signed_inst).groupby("ticker")["_s"].sum().sort_values()
        )
        if len(top_inst) > 0:
            top_neg = top_inst.iloc[0]
            top_neg_t = top_inst.index[0]
            top_pos = top_inst.iloc[-1]
            top_pos_t = top_inst.index[-1]
            most = (top_pos_t, top_pos) if abs(top_pos) >= abs(top_neg) else (top_neg_t, top_neg)
            n4.metric(f"Top: {most[0]}", f"${most[1]/1e6:+,.1f}M")

    st.markdown("")

    if inst_df.empty:
        if not INSTITUTIONAL_DB.exists():
            st.info(
                "No data yet. Click **Refresh (incremental)** above to start the first ingest. "
                "First backfill will pull 30 days × all tickers — expect 15–30 minutes."
            )
        else:
            st.info("No institutional trades match current filters.")
    else:
        # Build flag column
        def flag_str(row) -> str:
            flags = []
            if row.get("is_dark_pool"): flags.append("DP")
            if row.get("is_block"): flags.append("BLK")
            if row.get("is_algo"): flags.append("ALGO")
            if row.get("is_closing_auction"): flags.append("CLS")
            if row.get("is_sweep"): flags.append("SWP")
            if row.get("is_cross"): flags.append("CRS")
            return " · ".join(flags)

        display_inst = inst_df.copy()
        display_inst["flags"] = display_inst.apply(flag_str, axis=1)
        display_inst = display_inst[
            [
                "trade_ts", "ticker", "side", "shares", "price",
                "notional_usd", "exchange_name", "flags",
                "is_dark_pool", "is_block", "is_sweep", "is_cross",
                "is_algo", "is_closing_auction",
            ]
        ]

        if AGGRID_OK:
            gb = GridOptionsBuilder.from_dataframe(display_inst)
            gb.configure_default_column(
                filter=True, sortable=True, resizable=True, floatingFilter=True
            )
            gb.configure_pagination(paginationAutoPageSize=False, paginationPageSize=50)
            gb.configure_column(
                "notional_usd",
                type=["numericColumn"],
                valueFormatter=JsCode(
                    "function(params){return params.value!=null?'$'+params.value.toLocaleString(undefined,{maximumFractionDigits:0}):'';}"
                ),
                header_name="Notional",
            )
            gb.configure_column(
                "price",
                type=["numericColumn"],
                valueFormatter=JsCode(
                    "function(params){return params.value!=null?'$'+params.value.toFixed(2):'';}"
                ),
            )
            gb.configure_column(
                "shares",
                type=["numericColumn"],
                valueFormatter=JsCode(
                    "function(params){return params.value!=null?params.value.toLocaleString():'';}"
                ),
            )
            gb.configure_column("trade_ts", header_name="Time (UTC)")
            gb.configure_column("exchange_name", header_name="Venue")
            # Hide raw flag bools (we use the combined `flags` string column)
            gb.configure_column("is_dark_pool", hide=True)
            gb.configure_column("is_block", hide=True)
            gb.configure_column("is_sweep", hide=True)
            gb.configure_column("is_cross", hide=True)
            gb.configure_column("is_algo", hide=True)
            gb.configure_column("is_closing_auction", hide=True)

            # Highlight dark-pool rows
            row_style = JsCode(
                """
                function(params) {
                  if (params.data && params.data.is_dark_pool === 1) {
                    return { background: 'rgba(55,138,221,0.07)' };
                  }
                  return null;
                }
                """
            )
            grid_opts = gb.build()
            grid_opts["getRowStyle"] = row_style

            AgGrid(
                display_inst,
                gridOptions=grid_opts,
                height=600,
                theme="streamlit",
                update_mode=GridUpdateMode.NO_UPDATE,
                allow_unsafe_jscode=True,
                fit_columns_on_grid_load=False,
            )
        else:
            st.warning(
                "Install `streamlit-aggrid` for BI-style filters. "
                "`pip install streamlit-aggrid`"
            )
            st.dataframe(display_inst, use_container_width=True, height=600)

        csv = display_inst.to_csv(index=False).encode()
        st.download_button(
            "⬇ Export filtered CSV",
            data=csv,
            file_name=f"institutional_trades_{i_start}_{i_end}.csv",
            mime="text/csv",
            key="dl_inst",
        )
