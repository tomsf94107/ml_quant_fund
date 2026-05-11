# ui/pages/16_Trade_Log.py
# ─────────────────────────────────────────────────────────────────────────────
# Manual trade log — log executed trades and view recent history.
# Sprint 1 Day 5 (May 11 2026).
#
# Phase A scope:
#   ✓ Form to log a trade (writes to manual_trade_log table)
#   ✓ Table view of recent trades (read-only)
#
# NOT in scope (deferred):
#   ✗ "Mark Executed" button on Dashboard signal cards (Phase B)
#   ✗ P&L computation (Day 6)
#   ✗ Edit/delete UI
# ─────────────────────────────────────────────────────────────────────────────
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import sqlite3
from datetime import datetime, date
from pathlib import Path

import pandas as pd
import streamlit as st

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(page_title="Trade Log — ML Quant Fund", page_icon="📒", layout="wide")
st.title("📒 Manual Trade Log")
st.caption("Log executed trades · view recent history · feeds Day 6 P&L tracker")

# ── DB path ───────────────────────────────────────────────────────────────────
_HERE = Path(__file__).resolve()
_ROOT = _HERE.parents[2]
DB_PATH = str(_ROOT / "accuracy.db")


def _get_conn():
    """Connect to accuracy.db with sane defaults."""
    conn = sqlite3.connect(DB_PATH, timeout=30)
    conn.row_factory = sqlite3.Row
    return conn


def _verify_table_exists() -> bool:
    """Check manual_trade_log table is present (created by migrate_manual_trade_log.py)."""
    with _get_conn() as conn:
        cur = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='manual_trade_log'"
        )
        return cur.fetchone() is not None


def _insert_trade(ticker, side, shares, price, fill_time, notes, signal_id, suggested_price):
    """Insert a trade row. Returns the new row id."""
    with _get_conn() as conn:
        cur = conn.execute("""
            INSERT INTO manual_trade_log
                (ticker, side, shares, price, fill_time, notes, signal_id, suggested_price)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (ticker, side, shares, price, fill_time, notes or None,
              signal_id, suggested_price))
        conn.commit()
        return cur.lastrowid


def _load_recent_trades(limit: int = 50) -> pd.DataFrame:
    """Fetch most recent trades for the table view."""
    with _get_conn() as conn:
        return pd.read_sql_query("""
            SELECT
                id,
                fill_time,
                ticker,
                side,
                shares,
                price,
                suggested_price,
                COALESCE(notes, '') AS notes,
                signal_id,
                created_at
            FROM manual_trade_log
            ORDER BY fill_time DESC, created_at DESC
            LIMIT ?
        """, conn, params=(limit,))


def _load_summary_stats() -> dict:
    """Quick summary stats for the metrics row."""
    with _get_conn() as conn:
        total = conn.execute("SELECT COUNT(*) FROM manual_trade_log").fetchone()[0]
        by_side = dict(conn.execute("""
            SELECT side, COUNT(*) FROM manual_trade_log GROUP BY side
        """).fetchall())
        last7 = conn.execute("""
            SELECT COUNT(*) FROM manual_trade_log
            WHERE fill_time >= datetime('now', '-7 days')
        """).fetchone()[0]
        return {
            "total": total,
            "buys":  by_side.get("BUY", 0),
            "sells": by_side.get("SELL", 0),
            "trims": by_side.get("TRIM", 0),
            "last7": last7,
        }


# ── Safety check: table must exist ───────────────────────────────────────────
if not _verify_table_exists():
    st.error(
        "❌ manual_trade_log table does not exist. "
        "Run `python scripts/migrate_manual_trade_log.py` first."
    )
    st.stop()


# ── Summary metrics ──────────────────────────────────────────────────────────
stats = _load_summary_stats()
m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Total trades",   stats["total"])
m2.metric("BUYs",            stats["buys"])
m3.metric("SELLs",           stats["sells"])
m4.metric("TRIMs",           stats["trims"])
m5.metric("Last 7 days",    stats["last7"])

st.divider()


# ── Log a trade form ─────────────────────────────────────────────────────────
st.subheader("➕ Log a trade")

with st.form("log_trade", clear_on_submit=True):
    c1, c2, c3 = st.columns(3)
    with c1:
        ticker = st.text_input(
            "Ticker", max_chars=8,
            help="e.g. AAPL, NVDA"
        ).strip().upper()
        side = st.selectbox("Side", options=["BUY", "SELL", "TRIM"])
    with c2:
        shares = st.number_input(
            "Shares", min_value=0.0, step=1.0, format="%.4f",
            help="Use decimals for fractional shares"
        )
        price = st.number_input(
            "Fill price ($)", min_value=0.0, step=0.01, format="%.4f"
        )
    with c3:
        fill_date = st.date_input("Fill date", value=date.today())
        fill_time_t = st.time_input("Fill time (ET)", value=datetime.now().time().replace(microsecond=0))

    c4, c5 = st.columns(2)
    with c4:
        suggested_price = st.number_input(
            "Suggested price ($, optional)",
            min_value=0.0, step=0.01, format="%.4f", value=0.0,
            help="Price the system suggested at fill_time (slippage tracking)"
        )
    with c5:
        signal_id = st.number_input(
            "Signal ID (optional)", min_value=0, step=1, value=0,
            help="predictions.id this trade is responding to (0 = no signal)"
        )

    notes = st.text_area("Notes (optional)", max_chars=500, height=80)

    submitted = st.form_submit_button("💾 Log trade", type="primary")

    if submitted:
        # Validation
        errors = []
        if not ticker:
            errors.append("Ticker is required")
        elif not all(c.isalnum() or c in ".-" for c in ticker) or len(ticker) > 8:
            errors.append("Ticker must be 1-8 alphanumeric chars (. and - allowed)")
        if shares <= 0:
            errors.append("Shares must be > 0")
        if price <= 0:
            errors.append("Price must be > 0")

        if errors:
            st.error("Fix these before logging:\n" + "\n".join(f"  • {e}" for e in errors))
        else:
            fill_ts = datetime.combine(fill_date, fill_time_t).isoformat(timespec="seconds")
            new_id = _insert_trade(
                ticker=ticker, side=side,
                shares=float(shares), price=float(price),
                fill_time=fill_ts,
                notes=notes.strip() if notes else None,
                signal_id=int(signal_id) if signal_id > 0 else None,
                suggested_price=float(suggested_price) if suggested_price > 0 else None,
            )
            st.toast(f"✅ Logged trade #{new_id}: {side} {shares} {ticker} @ ${price:.2f}",
                     icon="✅")
            st.rerun()


st.divider()


# ── Recent trades table ──────────────────────────────────────────────────────
st.subheader("📋 Recent trades")

c1, c2 = st.columns([1, 4])
with c1:
    limit = st.selectbox("Show", [10, 25, 50, 100, 250], index=2)

df = _load_recent_trades(limit=limit)

if df.empty:
    st.info("No trades logged yet. Use the form above to log your first trade.")
else:
    # Pretty format for display
    df_display = df.copy()
    df_display["price"] = df_display["price"].apply(lambda v: f"${v:.2f}")
    df_display["suggested_price"] = df_display["suggested_price"].apply(
        lambda v: f"${v:.2f}" if pd.notna(v) else "—"
    )
    df_display["shares"] = df_display["shares"].apply(lambda v: f"{v:g}")
    df_display["signal_id"] = df_display["signal_id"].apply(
        lambda v: int(v) if pd.notna(v) else "—"
    )

    st.dataframe(
        df_display,
        use_container_width=True,
        hide_index=True,
        column_config={
            "id":              st.column_config.NumberColumn("ID", width="small"),
            "fill_time":       st.column_config.TextColumn("Fill time"),
            "ticker":          st.column_config.TextColumn("Ticker", width="small"),
            "side":            st.column_config.TextColumn("Side", width="small"),
            "shares":          st.column_config.TextColumn("Shares", width="small"),
            "price":           st.column_config.TextColumn("Fill $", width="small"),
            "suggested_price": st.column_config.TextColumn("Suggested $", width="small"),
            "notes":           st.column_config.TextColumn("Notes"),
            "signal_id":       st.column_config.TextColumn("Signal", width="small"),
            "created_at":      st.column_config.TextColumn("Logged at"),
        },
    )

st.caption(
    f"Showing most recent {min(len(df), limit)} of {stats['total']} trades. "
    "P&L tracking arrives in Sprint 1 Day 6."
)
