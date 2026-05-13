"""
analysis/drawdown_tracker.py
─────────────────────────────────────────────────────────────────────────────
Portfolio drawdown tracking per AI playbook section 11.1.

Manual-snapshot model: user calls log_snapshot() from Portfolio UI when
entering trade data weekly. Tracks peak portfolio_value and computes
drawdown_pct vs peak. Returns active risk-management rule.

Why manual (not cron): portfolio.json is XOR-encrypted with user password.
Cron has no access to decrypted state. Snapshot fires when user is logged 
in to Streamlit Portfolio page with password unlocked.

Per AI playbook section 11.1, drawdown thresholds trigger:
  0 to -10%:   NORMAL          — full trading
  -10 to -15%: STOP_NEW        — stop adding new positions
  -15 to -20%: EXIT_LOTTO      — close lotto-tier (red) positions
  -20 to -30%: EXIT_TACTICAL   — close tactical (yellow) positions
  -30 to -40%: REDUCE_CORE     — reduce core 40%
  < -40%:      FULL_RISK_OFF   — hold only defensive green
"""
import sqlite3
from datetime import date, datetime
from pathlib import Path
from typing import Optional

_DB_PATH = Path(__file__).parent.parent / "accuracy.db"


# AI playbook section 11.1 thresholds (descending)
# (max_drawdown_pct, status, description)
DRAWDOWN_RULES = [
    (0.00,  "NORMAL",         "Full trading authorized"),
    (-0.10, "STOP_NEW",       "Stop adding new positions"),
    (-0.15, "EXIT_LOTTO",     "Close all lotto-tier (red) positions"),
    (-0.20, "EXIT_TACTICAL",  "Close tactical (yellow) positions"),
    (-0.30, "REDUCE_CORE",    "Reduce core positions by 40%"),
    (-0.40, "FULL_RISK_OFF",  "Hold only defensive (green) positions"),
]


def classify_drawdown(dd_pct: float) -> tuple[str, str]:
    """
    Map drawdown_pct (negative number, e.g. -0.12 for -12%) to (status, rule).
    """
    status = "NORMAL"
    rule   = "Full trading authorized"
    for threshold, s, r in DRAWDOWN_RULES:
        if dd_pct <= threshold:
            status = s
            rule = r
    return status, rule


def get_peak() -> float:
    """Read highest portfolio_value ever logged. Returns 0 if no history."""
    try:
        with sqlite3.connect(f"file:{_DB_PATH}?mode=ro", uri=True) as conn:
            row = conn.execute(
                "SELECT MAX(peak_value) FROM drawdown_history"
            ).fetchone()
            return float(row[0]) if row and row[0] else 0.0
    except Exception:
        return 0.0


def log_snapshot(
    portfolio_value: float,
    cash: float = 0,
    positions_count: int = 0,
    notes: Optional[str] = None,
    snapshot_date: Optional[str] = None,
) -> dict:
    """
    Persist a portfolio MTM snapshot. Computes drawdown vs all-time peak.

    Returns: dict with snapshot_date, portfolio_value, peak_value,
             drawdown_pct, status, rule.
    """
    if portfolio_value <= 0:
        raise ValueError(f"portfolio_value must be > 0 (got {portfolio_value})")

    snapshot_date = snapshot_date or str(date.today())
    prior_peak    = get_peak()
    peak_value    = max(prior_peak, float(portfolio_value))
    drawdown_pct  = (portfolio_value - peak_value) / peak_value if peak_value > 0 else 0.0
    status, rule  = classify_drawdown(drawdown_pct)

    with sqlite3.connect(_DB_PATH) as conn:
        conn.execute("""
            INSERT OR REPLACE INTO drawdown_history
                (snapshot_date, portfolio_value, cash, positions_count,
                 peak_value, drawdown_pct, status, notes, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (snapshot_date, float(portfolio_value), float(cash),
              int(positions_count), float(peak_value), float(drawdown_pct),
              status, notes or "", datetime.now().isoformat()))
        conn.commit()

    return {
        "snapshot_date":   snapshot_date,
        "portfolio_value": portfolio_value,
        "peak_value":      peak_value,
        "drawdown_pct":    drawdown_pct,
        "status":          status,
        "rule":            rule,
    }


def get_current_status() -> dict:
    """
    Read most recent snapshot. Returns dict with all key fields,
    or {"status": "NO_DATA"} if no snapshots yet.
    """
    try:
        with sqlite3.connect(f"file:{_DB_PATH}?mode=ro", uri=True) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute("""
                SELECT * FROM drawdown_history
                ORDER BY snapshot_date DESC, created_at DESC LIMIT 1
            """).fetchone()
            if not row:
                return {"status": "NO_DATA",
                        "rule":   "No snapshots logged yet — visit Portfolio page"}
            return {
                "snapshot_date":   row["snapshot_date"],
                "portfolio_value": row["portfolio_value"],
                "peak_value":      row["peak_value"],
                "drawdown_pct":    row["drawdown_pct"],
                "status":          row["status"],
                "rule":            classify_drawdown(row["drawdown_pct"])[1],
            }
    except Exception as e:
        return {"status": "ERROR", "rule": str(e)}


def get_history(days_back: int = 90) -> list[dict]:
    """Return last N days of snapshots, oldest first, for charting."""
    try:
        from datetime import timedelta
        cutoff = (date.today() - timedelta(days=days_back)).isoformat()
        with sqlite3.connect(f"file:{_DB_PATH}?mode=ro", uri=True) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute("""
                SELECT snapshot_date, portfolio_value, peak_value, 
                       drawdown_pct, status
                FROM drawdown_history
                WHERE snapshot_date >= ?
                ORDER BY snapshot_date ASC
            """, (cutoff,)).fetchall()
            return [dict(r) for r in rows]
    except Exception:
        return []


if __name__ == "__main__":
    # CLI smoke test
    print("Testing drawdown_tracker...")
    r = get_current_status()
    print(f"Current status: {r}")
    
    # Demo: log a test snapshot
    import sys
    if "--test" in sys.argv:
        s = log_snapshot(portfolio_value=300000, cash=300000, 
                         positions_count=0, notes="initial snapshot")
        print(f"Test snapshot: {s}")
