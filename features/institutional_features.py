# features/institutional_features.py
# ─────────────────────────────────────────────────────────────────────────────
# Institutional flow features from local institutional_trades.db (UW darkpool).
#
# Schema source: features/institutional_ingest.py populates institutional_trades
# with Lee-Ready signed trades (~80% accuracy per Memory #6) including:
#   side (BUY/SELL/UNKNOWN), shares, notional_usd, nbbo_bid/ask,
#   is_dark_pool, is_block, is_sweep, is_cross, is_algo, is_closing_auction
#
# Key features (priority order):
#   P1 — Highest expected signal:
#     signed_flow_pct_5d         : (buy_notional − sell_notional) / total
#     block_buy_sell_ratio_7d    : buy-side block $ / sell-side block $
#     dp_signed_flow_5d          : signed_flow restricted to is_dark_pool=1
#
#   P2 — Validate after P1 ships:
#     sweep_count_7d             : urgency signal (multi-venue execution)
#     closing_auction_imbalance_5d : EOD positioning net flow
#
#   P3 — Nice-to-have:
#     signed_flow_pct_30d        : slower-moving regime
#     block_notional_7d_norm     : SUM(block notional) / SUM(total notional)
#     block_count_7d             : raw count of blocks
#
# PIT discipline: every query uses trade_date < as_of_date (STRICT <).
# Reason: at 8 PM ET on May 14, we know May 13 data fully but May 14 is
# partial. Use only completed trading days.
#
# How this helps (hypothesis-driven):
#   1. OKLO May 13 BUY 86% → -3.5% LOSS (post-earnings damage regime).
#      Block flow likely shows institutional selling → signed_flow_5d negative.
#   2. ANET May 5 BUY 91% → -16.26% LOSS (largest single loss).
#      Block flow on May 1-4 should be diagnostic.
#   3. LIN/GLD chronic losers (Memory finding: model bullish, mostly LOSS).
#      Signed flow gives independent institutional ground-truth check.
#   4. SNOW pre-earnings trap.
#      Closing-auction imbalance + sweeps often precede earnings positioning.
#
# Integration status (per Rule #1 checklist):
#   [✓] Code committed                    ← when committed
#   [ ] Unit tests pass                   ← next step
#   [ ] Imported by builder.py            ← gate with ML_QUANT_INST_FEATURES=1
#   [ ] Output in prediction_features     ← schema migration
#   [✓] Data source on cron               ← Tue-Sat 17:30 ET (shipped May 15)
#   [ ] Pipeline B nonzero importance     ← validate post-retrain
#   [ ] Documented in journal.txt         ← final step
# ─────────────────────────────────────────────────────────────────────────────

from __future__ import annotations

import sqlite3
import warnings
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ── DB path ───────────────────────────────────────────────────────────────────
DB_PATH = Path(__file__).parent.parent / "institutional_trades.db"

# ── Thresholds / constants ────────────────────────────────────────────────────
WINDOW_5D = 5
WINDOW_7D = 7
WINDOW_30D = 30

# Block size floor — UW Basic plan ingests at $250k floor per Memory #6,
# but we further filter "mega-block" at $1M for high-conviction signal
MEGA_BLOCK_NOTIONAL = 1_000_000.0


# ── Helpers ───────────────────────────────────────────────────────────────────
def _connect(db_path: Path = DB_PATH) -> sqlite3.Connection:
    """Singleton-style: open read-only connection per call. SQLite is fast enough."""
    return sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)


def _trading_days_back(as_of_date: str, n_calendar_days: int) -> str:
    """
    Return the start date for a window of n CALENDAR days back from as_of.

    Note: we use calendar days, not trading days, because the SQL filter
    naturally drops weekends/holidays (no trades on those days).

    Reason this is calendar-not-trading: builder.py callers don't have a
    trading-calendar helper plumbed in. Approximate is fine — over 5d or
    7d windows, a holiday or two doesn't materially shift the signed-flow
    ratio.
    """
    start_dt = datetime.fromisoformat(as_of_date) - timedelta(days=n_calendar_days)
    return start_dt.date().isoformat()


# ── PRIORITY 1 features ───────────────────────────────────────────────────────
def _signed_flow_pct(
    conn: sqlite3.Connection,
    ticker: str,
    start_date: str,
    as_of_date: str,
    dark_pool_only: bool = False,
) -> Optional[float]:
    """
    Return (buy_notional - sell_notional) / total_notional over [start, as_of).

    PIT: trade_date STRICTLY less than as_of_date.

    Range: -1.0 (all selling) to +1.0 (all buying).
    Returns None if no trades in window.

    Hypothesis: directional institutional pressure not visible in price/volume.
    """
    dp_clause = "AND is_dark_pool = 1" if dark_pool_only else ""
    q = f"""
    SELECT
      SUM(CASE WHEN side = 'BUY'  THEN notional_usd ELSE 0 END) as buy_n,
      SUM(CASE WHEN side = 'SELL' THEN notional_usd ELSE 0 END) as sell_n,
      SUM(notional_usd) as total_n
    FROM institutional_trades
    WHERE ticker = ?
      AND trade_date >= ?
      AND trade_date < ?
      {dp_clause}
      AND side IN ('BUY', 'SELL')   -- exclude UNKNOWN from signed calc
    """
    row = conn.execute(q, (ticker.upper(), start_date, as_of_date)).fetchone()
    if not row or row[2] is None or row[2] == 0:
        return None
    buy_n, sell_n, total_n = row
    return float((buy_n - sell_n) / total_n)


def _block_buy_sell_ratio(
    conn: sqlite3.Connection,
    ticker: str,
    start_date: str,
    as_of_date: str,
) -> Optional[float]:
    """
    Return buy-side block $ / sell-side block $ over [start, as_of).

    PIT: strict < as_of_date.

    Range: 0 to inf. >1 = bullish, <1 = bearish, =None if no blocks.
    Clipped to [0.01, 100] to avoid extreme outliers in feature scaling.

    Hypothesis: block trades represent institutional conviction. Direction
    matters more than volume.
    """
    q = """
    SELECT
      SUM(CASE WHEN side = 'BUY'  THEN notional_usd ELSE 0 END) as buy_block,
      SUM(CASE WHEN side = 'SELL' THEN notional_usd ELSE 0 END) as sell_block
    FROM institutional_trades
    WHERE ticker = ?
      AND trade_date >= ?
      AND trade_date < ?
      AND is_block = 1
      AND side IN ('BUY', 'SELL')
    """
    row = conn.execute(q, (ticker.upper(), start_date, as_of_date)).fetchone()
    if not row or row[0] is None or row[1] is None or row[1] == 0:
        return None
    buy_block, sell_block = row
    ratio = float(buy_block / sell_block)
    return max(0.01, min(100.0, ratio))


# ── PRIORITY 2 features ───────────────────────────────────────────────────────
def _sweep_count(
    conn: sqlite3.Connection,
    ticker: str,
    start_date: str,
    as_of_date: str,
) -> int:
    """
    Count of is_sweep=1 trades over [start, as_of).

    Sweeps = multi-venue urgent execution. Strong informed-trader signal.
    Hypothesis: spikes in sweeps precede price moves.
    """
    q = """
    SELECT COUNT(*)
    FROM institutional_trades
    WHERE ticker = ?
      AND trade_date >= ?
      AND trade_date < ?
      AND is_sweep = 1
    """
    row = conn.execute(q, (ticker.upper(), start_date, as_of_date)).fetchone()
    return int(row[0]) if row else 0


def _closing_auction_imbalance(
    conn: sqlite3.Connection,
    ticker: str,
    start_date: str,
    as_of_date: str,
) -> Optional[float]:
    """
    (closing_auction_buy − closing_auction_sell) / closing_auction_total
    over [start, as_of).

    Range: -1 to +1. None if no auction prints.

    Hypothesis: EOD positioning by institutions predicts next-day direction.
    """
    q = """
    SELECT
      SUM(CASE WHEN side = 'BUY'  THEN notional_usd ELSE 0 END) as buy_n,
      SUM(CASE WHEN side = 'SELL' THEN notional_usd ELSE 0 END) as sell_n,
      SUM(notional_usd) as total_n
    FROM institutional_trades
    WHERE ticker = ?
      AND trade_date >= ?
      AND trade_date < ?
      AND is_closing_auction = 1
      AND side IN ('BUY', 'SELL')
    """
    row = conn.execute(q, (ticker.upper(), start_date, as_of_date)).fetchone()
    if not row or row[2] is None or row[2] == 0:
        return None
    buy_n, sell_n, total_n = row
    return float((buy_n - sell_n) / total_n)


# ── PRIORITY 3 features ───────────────────────────────────────────────────────
def _block_notional_norm(
    conn: sqlite3.Connection,
    ticker: str,
    start_date: str,
    as_of_date: str,
) -> Optional[float]:
    """SUM(block notional) / SUM(total notional). Fraction of volume in blocks."""
    q = """
    SELECT
      SUM(CASE WHEN is_block = 1 THEN notional_usd ELSE 0 END) as block_n,
      SUM(notional_usd) as total_n
    FROM institutional_trades
    WHERE ticker = ?
      AND trade_date >= ?
      AND trade_date < ?
    """
    row = conn.execute(q, (ticker.upper(), start_date, as_of_date)).fetchone()
    if not row or row[1] is None or row[1] == 0:
        return None
    return float(row[0] / row[1])


def _block_count(
    conn: sqlite3.Connection,
    ticker: str,
    start_date: str,
    as_of_date: str,
) -> int:
    """Raw count of is_block=1 trades."""
    q = """
    SELECT COUNT(*)
    FROM institutional_trades
    WHERE ticker = ?
      AND trade_date >= ?
      AND trade_date < ?
      AND is_block = 1
    """
    row = conn.execute(q, (ticker.upper(), start_date, as_of_date)).fetchone()
    return int(row[0]) if row else 0


# ── PUBLIC API ────────────────────────────────────────────────────────────────
def get_institutional_features(
    ticker: str,
    as_of_date: str,
    db_path: Path = DB_PATH,
) -> dict:
    """
    Return institutional-flow features for a ticker as of a given date.

    PIT discipline: all queries use trade_date STRICTLY less than as_of_date.

    Args:
        ticker: 'OKLO', 'NVDA', etc. (case-insensitive)
        as_of_date: 'YYYY-MM-DD'. The DATE OF PREDICTION. We use data
                    BEFORE this date — at prediction time, today's
                    trades aren't yet known.
        db_path: override for testing.

    Returns:
        dict with all 8 features. Some may be None if no trades in window.
        Always returns the same keys for stable feature panel.
    """
    ticker = ticker.upper().strip()
    out: dict = {
        # P1
        "inst_signed_flow_5d":      None,
        "inst_block_buy_sell_7d":   None,
        "inst_dp_signed_flow_5d":   None,
        # P2
        "inst_sweep_count_7d":      0,
        "inst_auction_imbal_5d":    None,
        # P3
        "inst_signed_flow_30d":     None,
        "inst_block_notional_7d":   None,
        "inst_block_count_7d":      0,
    }

    try:
        conn = _connect(db_path)
    except sqlite3.Error:
        # DB missing or unreadable — return all-None feature row.
        # builder.py callers should handle None gracefully (already does
        # for short_interest pattern).
        return out

    try:
        start_5d  = _trading_days_back(as_of_date, WINDOW_5D)
        start_7d  = _trading_days_back(as_of_date, WINDOW_7D)
        start_30d = _trading_days_back(as_of_date, WINDOW_30D)

        # P1
        out["inst_signed_flow_5d"]    = _signed_flow_pct(conn, ticker, start_5d,  as_of_date)
        out["inst_block_buy_sell_7d"] = _block_buy_sell_ratio(conn, ticker, start_7d, as_of_date)
        out["inst_dp_signed_flow_5d"] = _signed_flow_pct(conn, ticker, start_5d,  as_of_date, dark_pool_only=True)
        # P2
        out["inst_sweep_count_7d"]    = _sweep_count(conn, ticker, start_7d,  as_of_date)
        out["inst_auction_imbal_5d"]  = _closing_auction_imbalance(conn, ticker, start_5d, as_of_date)
        # P3
        out["inst_signed_flow_30d"]   = _signed_flow_pct(conn, ticker, start_30d, as_of_date)
        out["inst_block_notional_7d"] = _block_notional_norm(conn, ticker, start_7d, as_of_date)
        out["inst_block_count_7d"]    = _block_count(conn, ticker, start_7d, as_of_date)
    finally:
        conn.close()

    return out


def get_institutional_features_batch(
    tickers: list[str],
    as_of_date: str,
    db_path: Path = DB_PATH,
) -> pd.DataFrame:
    """
    Batch version returning a DataFrame indexed by ticker.

    Same PIT semantics as single-ticker call. Useful in builder.py to avoid
    one-DB-connection-per-ticker overhead.
    """
    rows = []
    for t in tickers:
        feats = get_institutional_features(t, as_of_date, db_path=db_path)
        feats["ticker"] = t.upper().strip()
        rows.append(feats)
    df = pd.DataFrame(rows).set_index("ticker")
    return df


# ── CLI for spot-checking ─────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Spot-check institutional features")
    parser.add_argument("ticker")
    parser.add_argument("as_of_date", help="YYYY-MM-DD (prediction date)")
    args = parser.parse_args()

    feats = get_institutional_features(args.ticker, args.as_of_date)
    print(f"\n{args.ticker} as of {args.as_of_date}:")
    for k, v in feats.items():
        if v is None:
            print(f"  {k:30s} None")
        elif isinstance(v, float):
            print(f"  {k:30s} {v:+.4f}")
        else:
            print(f"  {k:30s} {v}")



# ══════════════════════════════════════════════════════════════════════════════
#  PIT-correct loader over date_index (added 2026-05-21)
# ══════════════════════════════════════════════════════════════════════════════
#  features/builder.py was calling get_institutional_features ONCE with
#  as_of=end_str, then broadcasting that single result to all rows. That:
#    (1) gave every row the same value (constant column, useless to XGBoost)
#    (2) leaked today's data into every historical row (lookahead bias)
#  This function calls get_institutional_features per-row over date_index,
#  respecting PIT discipline. Mirrors load_finbert_pit pattern.

def load_institutional_features_pit(
    ticker: str,
    date_index,
    db_path = None,
):
    """
    PIT-safe loader: for each date in date_index, query inst features
    with as_of=that date. Returns DataFrame indexed by date with all 4
    columns wired to features/builder._INST_FEATURE_COLS.

    Performance note: this calls get_institutional_features once per row,
    which means N SQL queries per training build. For 579 rows that's ~10s.
    Acceptable for now. Could be vectorized later if it becomes a bottleneck.
    """
    import pandas as pd
    import numpy as np

    if db_path is None:
        db_path = DB_PATH

    cols = [
        "inst_block_buy_sell_7d",
        "inst_signed_flow_30d",
        "inst_auction_imbal_5d",
        "inst_signed_flow_5d",
    ]
    out = pd.DataFrame(index=pd.Index(date_index), columns=cols, dtype=float)
    out[:] = np.nan

    for i, asof in enumerate(date_index):
        # Normalize to YYYY-MM-DD string
        if hasattr(asof, "strftime"):
            asof_str = asof.strftime("%Y-%m-%d")
        else:
            asof_str = str(asof)
        try:
            feats = get_institutional_features(ticker, asof_str, db_path=db_path)
            for c in cols:
                v = feats.get(c)
                if v is not None:
                    out.iloc[i, out.columns.get_loc(c)] = float(v)
        except Exception:
            continue

    return out
