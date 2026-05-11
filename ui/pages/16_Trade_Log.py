# ui/pages/16_Trade_Log.py
# ─────────────────────────────────────────────────────────────────────────────
# Manual trade log — log executed trades and view recent history + P&L.
# Sprint 1 Day 5-6 (May 11 2026).
#
# Day 5: form-based trade entry + recent trades table
# Day 6: FIFO realized P&L tab (compute_fifo_pnl from portfolio/realized_pnl)
#
# NOT in scope (deferred):
#   ✗ "Mark Executed" button on Dashboard signal cards (later sprint)
#   ✗ Edit/delete trade UI
#   ✗ Unrealized P&L (lives on Portfolio page)
# ─────────────────────────────────────────────────────────────────────────────
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import sqlite3
from datetime import datetime, date
from pathlib import Path

import pandas as pd
import streamlit as st

from portfolio.realized_pnl import compute_fifo_pnl, summarize_realized

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(page_title="Trade Log — ML Quant Fund", page_icon="📒", layout="wide")
st.title("📒 Manual Trade Log")
st.caption("Log executed trades · view recent history · realized P&L via FIFO")

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
    """Check manual_trade_log table is present."""
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
                id, fill_time, ticker, side, shares, price, suggested_price,
                COALESCE(notes, '') AS notes, signal_id, created_at
            FROM manual_trade_log
            ORDER BY fill_time DESC, created_at DESC
            LIMIT ?
        """, conn, params=(limit,))


def _load_all_trades() -> pd.DataFrame:
    """Fetch ALL trades for FIFO P&L computation (no limit)."""
    with _get_conn() as conn:
        return pd.read_sql_query("""
            SELECT ticker, side, shares, price, fill_time
            FROM manual_trade_log
            ORDER BY fill_time ASC
        """, conn)


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


# ── Tabs ─────────────────────────────────────────────────────────────────────
tab_log, tab_pnl = st.tabs(["📋 Log & View", "💰 Realized P&L"])


# =============================================================================
# TAB 1 — Log & View
# =============================================================================
with tab_log:
    # ── Summary metrics ──────────────────────────────────────────────────────
    stats = _load_summary_stats()
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Total trades", stats["total"])
    m2.metric("BUYs",         stats["buys"])
    m3.metric("SELLs",        stats["sells"])
    m4.metric("TRIMs",        stats["trims"])
    m5.metric("Last 7 days",  stats["last7"])

    st.divider()

    # ── Log a trade form ─────────────────────────────────────────────────────
    st.subheader("➕ Log a trade")

    with st.form("log_trade", clear_on_submit=True):
        c1, c2, c3 = st.columns(3)
        with c1:
            ticker = st.text_input(
                "Ticker", max_chars=8,
                help="e.g. AAPL, NVDA, BRK.B"
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
            fill_time_t = st.time_input(
                "Fill time (ET)",
                value=datetime.now().time().replace(microsecond=0),
            )

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
                st.toast(
                    f"✅ Logged trade #{new_id}: {side} {shares} {ticker} @ ${price:.2f}",
                    icon="✅",
                )
                st.rerun()

    st.divider()

    # ── Recent trades table ──────────────────────────────────────────────────
    st.subheader("📋 Recent trades")

    c1, c2 = st.columns([1, 4])
    with c1:
        limit = st.selectbox("Show", [10, 25, 50, 100, 250], index=2)

    df = _load_recent_trades(limit=limit)

    if df.empty:
        st.info("No trades logged yet. Use the form above to log your first trade.")
    else:
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
        f"Showing most recent {min(len(df), limit)} of {stats['total']} trades."
    )


# =============================================================================
# TAB 2 — Realized P&L (FIFO)
# =============================================================================
with tab_pnl:
    st.subheader("💰 Realized P&L (FIFO)")
    st.caption(
        "Closed positions matched FIFO — oldest BUY lots are exited first. "
        "TRIM and SELL are treated identically for P&L purposes."
    )

    all_trades = _load_all_trades()

    if all_trades.empty:
        st.info("No trades logged yet. Use the Log & View tab to log your first trade.")
    else:
        closed_df, open_df, warnings = compute_fifo_pnl(all_trades)
        stats_pnl = summarize_realized(closed_df)

        # ── Summary metrics row 1 ────────────────────────────────────────────
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Closed positions", stats_pnl["n_closed"])
        m2.metric(
            "Realized P&L",
            f"${stats_pnl['total_gross_pnl']:+,.2f}",
            f"{stats_pnl['weighted_return_pct']:+.2f}% (weighted)",
        )
        m3.metric("Win rate", f"{stats_pnl['win_rate_pct']:.1f}%")
        m4.metric("Avg hold", f"{stats_pnl['avg_hold_days']:.1f} days")

        # ── Summary metrics row 2 ────────────────────────────────────────────
        m5, m6, m7, m8 = st.columns(4)
        m5.metric("Avg return", f"{stats_pnl['avg_return_pct']:+.2f}%")
        m6.metric("Best trade", f"{stats_pnl['best_trade_pct']:+.2f}%")
        m7.metric("Worst trade", f"{stats_pnl['worst_trade_pct']:+.2f}%")
        m8.metric("Capital risked", f"${stats_pnl['total_cost']:,.0f}")

        # ── Warnings (oversold, sell-without-buy) ────────────────────────────
        if warnings:
            with st.expander(f"⚠ {len(warnings)} warning(s) from FIFO matching"):
                for w in warnings:
                    st.warning(w)

        st.divider()

        # ── Closed positions table ───────────────────────────────────────────
        st.subheader("📈 Closed positions")
        if closed_df.empty:
            st.info("No closed positions yet. Log a SELL or TRIM after a BUY to close a position.")
        else:
            display_closed = closed_df.copy()
            display_closed["entry_price"] = display_closed["entry_price"].apply(lambda v: f"${v:.2f}")
            display_closed["exit_price"]  = display_closed["exit_price"].apply(lambda v: f"${v:.2f}")
            display_closed["gross_pnl"]   = display_closed["gross_pnl"].apply(lambda v: f"${v:+,.2f}")
            display_closed["return_pct"]  = display_closed["return_pct"].apply(lambda v: f"{v:+.2f}%")
            display_closed["shares"]      = display_closed["shares"].apply(lambda v: f"{v:g}")
            display_closed["hold_days"]   = display_closed["hold_days"].apply(lambda v: f"{v:.1f}d")

            display_closed = display_closed[[
                "ticker", "entry_time", "exit_time", "exit_side",
                "shares", "entry_price", "exit_price",
                "return_pct", "gross_pnl", "hold_days",
            ]]

            st.dataframe(
                display_closed,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "ticker":      st.column_config.TextColumn("Ticker", width="small"),
                    "entry_time":  st.column_config.TextColumn("Entered"),
                    "exit_time":   st.column_config.TextColumn("Exited"),
                    "exit_side":   st.column_config.TextColumn("Exit", width="small"),
                    "shares":      st.column_config.TextColumn("Shares", width="small"),
                    "entry_price": st.column_config.TextColumn("Entry $", width="small"),
                    "exit_price":  st.column_config.TextColumn("Exit $", width="small"),
                    "return_pct":  st.column_config.TextColumn("Return %", width="small"),
                    "gross_pnl":   st.column_config.TextColumn("P&L $", width="small"),
                    "hold_days":   st.column_config.TextColumn("Hold", width="small"),
                },
            )

        st.divider()

        # ── Open positions table ─────────────────────────────────────────────
        st.subheader("📦 Open positions (unmatched BUY lots)")
        if open_df.empty:
            st.info("No open positions — all BUY lots fully closed by SELL/TRIM.")
        else:
            display_open = open_df.copy()
            display_open["entry_price"] = display_open["entry_price"].apply(lambda v: f"${v:.2f}")
            display_open["cost_basis"]  = display_open["cost_basis"].apply(lambda v: f"${v:,.2f}")
            display_open["shares"]      = display_open["shares"].apply(lambda v: f"{v:g}")

            st.dataframe(
                display_open,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "ticker":      st.column_config.TextColumn("Ticker", width="small"),
                    "entry_time":  st.column_config.TextColumn("Entered"),
                    "shares":      st.column_config.TextColumn("Shares", width="small"),
                    "entry_price": st.column_config.TextColumn("Entry $", width="small"),
                    "cost_basis":  st.column_config.TextColumn("Cost basis", width="small"),
                },
            )

        st.caption(
            f"FIFO matched {len(closed_df)} closed position(s) from {len(all_trades)} trades. "
            f"{len(open_df)} open lot(s) remaining."
        )
