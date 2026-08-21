#!/usr/bin/env python3
"""
dma_levels.py -- per-ticker moving averages computed from prices.db.

WHY THIS EXISTS (2026-08-21)
  NOTHING in the codebase computes a per-ticker moving average. The only 200-DMA
  anywhere is models/regime_classifier.py, and that is SPY-only for market regime.
  So every 50/200-DMA in every research report has been HAND-CARRIED forward.

  NVDA 2026-08-21 carried a 200-DMA of $196.01 against a live ~$206.23 -- a 5%
  error on a level the report used as "the next support below the base." The
  report itself flagged it: "Recompute + pin 50/200-DMA from prices.db each run."

  This is not a control that failed. It is the ABSENCE of a control: nothing was
  wrong, and nothing was checking either. A carried number ages silently while
  looking authoritative, and levels feed straight into the levels map and the
  scenario bands.

WHAT IT DOES
  Computes 20/50/100/200-day simple MAs from raw_bars for one ticker or the whole
  universe, reports distance from the last close, flags golden/death crosses, and
  optionally writes to accuracy.db.dma_levels so reports READ a pinned value
  instead of carrying one.

  Insufficient history is reported as "n/a (N bars)" -- never silently omitted,
  and never back-filled with a shorter window pretending to be a 200-DMA.

USAGE
  python scripts/dma_levels.py NVDA
  python scripts/dma_levels.py NVDA MU MRVL --json
  python scripts/dma_levels.py --all --write
  python scripts/dma_levels.py --all --near 2.0     # only names within 2% of a MA
"""
import argparse
import json
import os
import sqlite3
import sys

ROOT = os.path.expanduser(os.environ.get("ML_QUANT_ROOT", "~/ML_Quant_Fund"))
PRICES = os.path.join(ROOT, "prices.db")
ACC = os.path.join(ROOT, "accuracy.db")
WINDOWS = (20, 50, 100, 200)


def load_closes(con, tickers=None):
    """{ticker: [(d, close)] ascending}. Uses raw_bars: split-adjusted on write,
    same series outcomes are computed from (verified via parity 2026-08-15)."""
    q = ("SELECT ticker, d, close FROM raw_bars "
         "WHERE close IS NOT NULL AND close > 0")
    args = []
    if tickers:
        q += f" AND ticker IN ({','.join('?' * len(tickers))})"
        args = list(tickers)
    q += " ORDER BY ticker, d"
    out = {}
    for tk, d, c in con.execute(q, args):
        out.setdefault(tk.upper(), []).append((d, float(c)))
    return out


def compute(rows):
    """-> dict of last close/date + each MA, distance, and cross state."""
    if not rows:
        return None
    d, last = rows[-1]
    closes = [c for _, c in rows]
    res = {"date": d, "close": last, "bars": len(closes)}
    for w in WINDOWS:
        if len(closes) >= w:
            ma = sum(closes[-w:]) / w
            res[f"ma{w}"] = ma
            res[f"d{w}"] = (last / ma - 1.0) * 100.0
        else:
            res[f"ma{w}"] = None
            res[f"d{w}"] = None

    # Golden / death cross on the 50 vs 200, using the prior session for the
    # crossing check so a same-day flip is reported the day it happens.
    res["cross"] = ""
    if res["ma50"] is not None and res["ma200"] is not None and len(closes) >= 201:
        p50 = sum(closes[-51:-1]) / 50
        p200 = sum(closes[-201:-1]) / 200
        now_above = res["ma50"] > res["ma200"]
        was_above = p50 > p200
        if now_above and not was_above:
            res["cross"] = "GOLDEN"
        elif was_above and not now_above:
            res["cross"] = "DEATH"
        else:
            res["cross"] = "50>200" if now_above else "50<200"
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tickers", nargs="*")
    ap.add_argument("--all", action="store_true", help="every ticker in raw_bars")
    ap.add_argument("--write", action="store_true",
                    help="write to accuracy.db.dma_levels so reports read a PINNED "
                         "value instead of carrying one forward by hand")
    ap.add_argument("--near", type=float, default=None,
                    help="only show names within this %% of any MA (levels that matter)")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--root")
    args = ap.parse_args()

    global ROOT, PRICES, ACC
    if args.root:
        ROOT = os.path.expanduser(args.root)
        PRICES = os.path.join(ROOT, "prices.db")
        ACC = os.path.join(ROOT, "accuracy.db")
    if not os.path.isfile(PRICES):
        sys.exit(f"FATAL: {PRICES} not found")
    if not args.tickers and not args.all:
        ap.error("give tickers or --all")

    con = sqlite3.connect(PRICES, timeout=30)
    tks = [t.upper() for t in args.tickers] or None
    data = load_closes(con, tks)
    con.close()
    if not data:
        sys.exit("FATAL: no price rows found for the requested tickers")

    results = {}
    for tk in sorted(data):
        r = compute(data[tk])
        if r:
            results[tk] = r

    if args.near is not None:
        keep = {}
        for tk, r in results.items():
            for w in WINDOWS:
                dv = r.get(f"d{w}")
                if dv is not None and abs(dv) <= args.near:
                    keep[tk] = r
                    break
        results = keep

    if args.json:
        print(json.dumps(results, indent=2))
    else:
        hdr = (f"{'ticker':<7}{'date':<12}{'close':>9}{'20DMA':>9}{'50DMA':>9}"
               f"{'200DMA':>9}{'vs50':>8}{'vs200':>8}  cross")
        print(hdr)
        print("-" * len(hdr))
        for tk, r in results.items():
            def f(k, w=9, p=2):
                v = r.get(k)
                return f"{v:>{w}.{p}f}" if v is not None else f"{'n/a':>{w}}"
            note = r["cross"]
            if r.get("ma200") is None:
                note = f"n/a ({r['bars']} bars)"
            print(f"{tk:<7}{r['date']:<12}{r['close']:>9.2f}{f('ma20')}{f('ma50')}"
                  f"{f('ma200')}{f('d50',8,1)}{f('d200',8,1)}  {note}")

    if args.write:
        acon = sqlite3.connect(ACC, timeout=30)
        try:
            acon.execute("""CREATE TABLE IF NOT EXISTS dma_levels(
                ticker TEXT NOT NULL, bar_date TEXT NOT NULL, close REAL,
                ma20 REAL, ma50 REAL, ma100 REAL, ma200 REAL,
                d20 REAL, d50 REAL, d100 REAL, d200 REAL,
                cross_state TEXT, bars INTEGER,
                computed_at TEXT DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (ticker, bar_date))""")
            n = 0
            for tk, r in results.items():
                acon.execute(
                    "INSERT OR REPLACE INTO dma_levels(ticker,bar_date,close,"
                    "ma20,ma50,ma100,ma200,d20,d50,d100,d200,cross_state,bars) "
                    "VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    (tk, r["date"], r["close"], r["ma20"], r["ma50"], r["ma100"],
                     r["ma200"], r["d20"], r["d50"], r["d100"], r["d200"],
                     r["cross"], r["bars"]))
                n += 1
            acon.commit()
            print(f"\n# wrote {n} row(s) to accuracy.db.dma_levels")
        finally:
            acon.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
