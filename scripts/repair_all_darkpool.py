#!/usr/bin/env python3
"""Heal dark-pool partial days for EVERY ticker with recent history.

WHY (Jul 31 2026): repair coverage was DEFAULT_TICKERS = TICKER_CONFIG.keys(),
i.e. the hand-maintained config -- so any ticker reported on but never hand-
configured was invisible to batch heals. AMD, MRVL and RZLV each needed a
manual repair after a report noticed the gap. Coverage now follows the DATA:
anything with prints in the window gets healed.

Budget-aware: UW caps 40K calls/day and a heal costs ~5-150 pages per ticker.
--max-tickers bounds a run; --dry lists targets without fetching.
"""
import argparse
import sqlite3
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=30, help="lookback for activity")
    ap.add_argument("--max-tickers", type=int, default=25)
    ap.add_argument("--dry", action="store_true")
    ap.add_argument("--exclude", default="", help="comma list to skip (already healed)")
    ap.add_argument("--only", default="", help="comma list to heal, ignoring ranking")
    a = ap.parse_args()

    con = sqlite3.connect(f"file:{ROOT / 'earnings_monitor.db'}?mode=ro", uri=True)
    rows = con.execute(
        "SELECT ticker, COUNT(DISTINCT et_date) d, MAX(et_date) last "
        "FROM darkpool_prints WHERE et_date >= date('now', ?) "
        "GROUP BY ticker ORDER BY d DESC", (f"-{a.days} day",)).fetchall()
    con.close()

    _ex = {x.strip().upper() for x in a.exclude.split(",") if x.strip()}
    _only = {x.strip().upper() for x in a.only.split(",") if x.strip()}
    _cand = [r for r in rows if r[0].upper() not in _ex
             and (not _only or r[0].upper() in _only)]
    targets = [r[0] for r in _cand][:a.max_tickers]
    print(f"{len(rows)} ticker(s) with dark-pool history in the last {a.days}d; "
          f"healing {len(targets)} (cap --max-tickers {a.max_tickers})")
    for t, d, last in (_cand if a.dry else _cand[:a.max_tickers]):
        print(f"  {t:<6} {d:>3} distinct day(s), latest {last}")
    if a.dry:
        print("DRY RUN -- no fetches. Drop --dry to heal.")
        return 0

    for t in targets:
        print(f"\n=== {t} ===")
        r = subprocess.run(
            [sys.executable, str(ROOT / "scripts" / "repair_darkpool_days.py"),
             "--ticker", t],
            capture_output=True, text=True)
        tail = [l for l in r.stdout.strip().splitlines() if l.strip()][-1:] or ["(no output)"]
        print(f"  {tail[0]}")
        if r.returncode != 0:
            print(f"  [warn] exit {r.returncode}: {r.stderr.strip()[:200]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
