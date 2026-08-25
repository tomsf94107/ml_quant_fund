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
START = "2016-07-18"   # vendor history floor


def load_closes(con, tickers=None):
    """{ticker: [(d, close)] ascending}, SPLIT-ADJUSTED.

    CORRECTION 2026-08-25. The previous body read raw_bars, on a docstring claim
    that raw_bars was split-adjusted on write and was the same series outcomes is
    computed from. BOTH CLAUSES WERE FALSE:
      - raw_bars is UNADJUSTED. AAPL 2020-08-31 stores 499.23 -> 129.04 across a
        4:1; the adjusted series runs 124.81 -> 129.04.
      - outcomes comes from accuracy/sink.py:666 via mc.download(auto_adjust=True),
        a DIFFERENT series.
    Consequence: 18 of 439 tickers had a split inside the trailing 200 sessions,
    so their MAs averaged pre- and post-split prices. BKNG SMA200 read 2531.80
    against a 213.36 close (11.9x); KLAC 1195.83 vs 181.57; CRWD 445.35 vs 190.68
    AND its SMA50 330.16 -- the 50-day is poisoned too where the split is recent,
    which also fabricates cross_state (BKNG printed "50<200" off a bogus 200).

    Now uses mc.download(auto_adjust=True): the same series outcomes uses,
    verified split-correct on AAPL's 4:1 and validated at 99.32% agreement to
    1e-9 against stored outcomes (2024-25, h=5). Serves from price_cache; 410
    tickers in ~10s. `con` is used only to enumerate tickers and pin the end
    date, so a stale bar_date is REPORTED rather than hidden."""
    import sys as _sys, os as _os
    _R = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
    if _R not in _sys.path:
        _sys.path.insert(0, _R)
    from features import massive_client as _mc
    import pandas as _pd

    tks = ([t.upper() for t in tickers] if tickers else
           [r[0].upper() for r in con.execute(
               "SELECT DISTINCT ticker FROM raw_bars ORDER BY ticker")])
    end = con.execute("SELECT MAX(d) FROM raw_bars").fetchone()[0]
    out, failed = {}, []
    for t in tks:
        try:
            df = _mc.download(t, start=START, end=end, auto_adjust=True, progress=False)
            if df is None or df.empty:
                failed.append((t, "empty")); continue
            if isinstance(df.columns, _pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            cs = df["Close"]
            if isinstance(cs, _pd.DataFrame):
                cs = cs.iloc[:, 0]
            cs.index = _pd.to_datetime(cs.index).tz_localize(None)
            cs = cs[~cs.index.duplicated(keep="last")].sort_index().dropna()
            if cs.empty:
                failed.append((t, "all-nan")); continue
            out[t] = [(str(d.date()), float(v)) for d, v in cs.items()]
        except Exception as e:
            failed.append((t, repr(e)[:60]))
    if failed:
        print(f"# WARNING: {len(failed)} ticker(s) had no usable adjusted series: "
              f"{failed[:10]}", file=_sys.stderr)
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
        # ma100 was COMPUTED and STORED but never displayed -- which is how a
        # 100-day average reached a report labelled "the 200-DMA" (NVDA, Aug 25:
        # "live 200-DMA ~$206.23" vs the true SMA200 of 195.34; SMA100 was 206.99).
        hdr = (f"{'ticker':<7}{'date':<12}{'close':>9}{'20DMA':>9}{'50DMA':>9}"
               f"{'100DMA':>9}{'200DMA':>9}{'vs50':>8}{'vs200':>8}  cross")
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
                  f"{f('ma100')}{f('ma200')}{f('d50',8,1)}{f('d200',8,1)}  {note}")

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
