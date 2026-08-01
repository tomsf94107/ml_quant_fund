#!/usr/bin/env python3
"""Staged dark-pool initiation across the universe, budget-capped per run.

WHY: UW serves ~44 days of history and NOTHING older -- unfetched days are
permanently lost. Tickers never walked still have a full retrievable window
sitting at the vendor right now. This initiates them in daily patches.

BUDGET: pages cap at 500 rows, so calls ~= rows/500 [measured]. The runner
tracks rows upserted per ticker, converts to an estimated call count, and
STOPS when --budget is reached -- it does not start a ticker it cannot
plausibly finish inside the remaining allowance.

Ordering: never-fetched first (their tape is actively expiring), then
stalest-fetched. Progress is journaled so the next day's run resumes.
"""
import argparse
import csv
import json
import re
import sqlite3
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
JOURNAL = ROOT / "logs" / "darkpool_initiation.json"
ROWS_PER_CALL = 500


def load_journal():
    try:
        return json.loads(JOURNAL.read_text())
    except Exception:
        return {"done": {}, "runs": []}


def universe():
    p = ROOT / "tickers_metadata.csv"
    if not p.exists():
        return []
    return [(r.get("ticker") or "").strip().upper()
            for r in csv.DictReader(p.open()) if (r.get("ticker") or "").strip()]


def existing_coverage():
    con = sqlite3.connect(f"file:{ROOT / 'earnings_monitor.db'}?mode=ro", uri=True)
    rows = con.execute(
        "SELECT ticker, COUNT(DISTINCT et_date), MAX(et_date) "
        "FROM darkpool_prints GROUP BY ticker").fetchall()
    con.close()
    return {r[0].upper(): (r[1], r[2]) for r in rows}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=int, default=50000, help="max estimated API calls this run")
    ap.add_argument("--days", type=int, default=45)
    ap.add_argument("--reserve", type=int, default=3000,
                    help="stop this far from budget; a fat ticker can overshoot")
    ap.add_argument("--dry", action="store_true")
    a = ap.parse_args()

    j = load_journal()
    cov = existing_coverage()
    uni = universe()
    if not uni:
        print("No tickers_metadata.csv -- nothing to initiate.")
        return 1

    never = [t for t in uni if t not in cov and t not in j["done"]]
    stale = sorted([t for t in uni if t in cov and t not in j["done"]],
                   key=lambda t: cov[t][1] or "")
    queue = never + stale

    print(f"Universe {len(uni)} | already initiated (journal) {len(j['done'])} | "
          f"never fetched {len(never)} | previously fetched {len(stale)}")
    print(f"Budget {a.budget:,} est. calls (reserve {a.reserve:,}), window {a.days}d")
    if a.dry:
        for t in queue[:40]:
            mark = "NEW" if t in never else f"last {cov.get(t, ('', '?'))[1]}"
            print(f"  {t:<6} {mark}")
        print(f"  ... {len(queue)} queued total. DRY RUN -- no fetches.")
        return 0

    used = 0
    healed = 0
    t0 = time.time()
    for t in queue:
        if used >= a.budget - a.reserve:
            print(f"\nSTOP: {used:,} est. calls used, within {a.reserve:,} of budget.")
            break
        r = subprocess.run(
            [sys.executable, str(ROOT / "scripts" / "repair_darkpool_days.py"),
             "--ticker", t, "--days", str(a.days)],
            capture_output=True, text=True)
        m = re.search(r"TOTAL upserted:\s*([\d,]+)", r.stdout or "")
        rows_n = int(m.group(1).replace(",", "")) if m else 0
        # MEASURED, not estimated. rows/500 undercounts ~2x because the walk is
        # PER DAY: every ticker pays >=1 call per trading day regardless of
        # volume (DNA: 48 rows = 1 estimated call, ~32 actual). The repair
        # script prints "N page(s)" per day -- pages ARE the API calls.
        _pages = [int(x) for x in re.findall(r"(\d+)\s+page\(s\)", r.stdout or "")]
        calls = sum(_pages) if _pages else max(1, round(rows_n / ROWS_PER_CALL))
        used += calls
        healed += 1
        j["done"][t] = {"rows": rows_n, "est_calls": calls,
                        "at": datetime.now().isoformat(timespec="seconds")}
        print(f"  {t:<6} rows {rows_n:>9,}  ~{calls:>5,} calls   "
              f"(cumulative ~{used:,})")
        if r.returncode != 0:
            print(f"  [warn] {t} exit {r.returncode}: {(r.stderr or '')[:150]}")
        JOURNAL.parent.mkdir(parents=True, exist_ok=True)
        JOURNAL.write_text(json.dumps(j, indent=1))

    j["runs"].append({"at": datetime.now().isoformat(timespec="seconds"),
                      "tickers": healed, "est_calls": used,
                      "minutes": round((time.time() - t0) / 60, 1)})
    JOURNAL.write_text(json.dumps(j, indent=1))
    print(f"\n{healed} ticker(s), ~{used:,} est. calls, "
          f"{round((time.time()-t0)/60,1)} min. Journal: {JOURNAL}")
    print(f"Remaining in queue: {len(queue) - healed}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
