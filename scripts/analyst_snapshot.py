"""
scripts/analyst_snapshot.py - dated weekly snapshots of analyst revisions.
Snapshot-only feed (yfinance) becomes a revisions HISTORY by logging weekly.
Signal (revision momentum, Chan-Jegadeesh-Lakonishok) testable after ~3-6mo accrual.
Writes: accuracy.db analyst_snapshots(ticker, snap_date, payload_json).
Run: python -m scripts.analyst_snapshot   (weekly cron, Sun 07:20 VN)
"""
import json, sqlite3, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

SCHEMA = """
CREATE TABLE IF NOT EXISTS analyst_snapshots (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker TEXT NOT NULL,
    snap_date TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    UNIQUE(ticker, snap_date)
);
"""


def main():
    import sys
    sys.path.insert(0, str(ROOT))
    from data.alpha_sources import get_analyst_revisions

    tickers = [l.strip().upper() for l in open(ROOT / "tickers.txt")
               if l.strip() and not l.startswith("#")]
    conn = sqlite3.connect(str(ROOT / "accuracy.db"), timeout=30)
    conn.executescript(SCHEMA)
    snap_date = time.strftime("%Y-%m-%d")
    ok = err = 0
    for i, t in enumerate(tickers, 1):
        try:
            d = get_analyst_revisions(t)
            conn.execute(
                "INSERT OR IGNORE INTO analyst_snapshots (ticker, snap_date, payload_json) VALUES (?,?,?)",
                (t, snap_date, json.dumps(d, default=str)))
            ok += 1
        except Exception:
            err += 1
        if i % 50 == 0:
            conn.commit()
            print(f"[{i}/{len(tickers)}] ok={ok} err={err}", flush=True)
        time.sleep(0.3)  # yfinance politeness
    conn.commit()
    n = conn.execute("SELECT COUNT(*), COUNT(DISTINCT snap_date) FROM analyst_snapshots").fetchone()
    print(f"DONE {snap_date}: ok={ok} err={err} | table total {n[0]} rows, {n[1]} snapshot dates")


if __name__ == "__main__":
    main()
