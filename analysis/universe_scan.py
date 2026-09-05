#!/usr/bin/env python3
"""
universe_scan.py — pass 1 of 2. Rank ~9,900 US symbols by liquidity.

WRITES ONLY to universe_scan.db, a scratch file. Never touches prices.db.

WHY TWO PASSES
    A first attempt fetched candidates in ALPHABETICAL order and spent its
    budget on AACG, AACI, AACIU, AACIW, AACO, AACOU -- SPAC units, warrants and
    shells. A 2,000-name cap ordered that way would never reach a liquid name.

    Liquidity cannot be known before the bars are in hand, so:
      pass 1 (this)  fetch a SHORT recent window for every candidate, compute
                     dollar ADV, write it to a scratch database
      pass 2         fetch full history only for the top N by ADV

    With unlimited Massive calls the first pass is cheap: 9,869 symbols x ~90
    days is a small fraction of the data volume of 2,000 x 10 years.

WHY A SEPARATE DATABASE
    features/price_cache.py writes to prices.db raw_bars on every download. On
    2026-09-05 a fetch run and an analysis run held prices.db at the same time
    and the analysis job's panel came back incomplete -- GEMI, AME and ZM failed
    mid-run and one seed's results had to be discarded.

    So this writes its scan results to universe_scan.db and must be run when
    nothing else is using prices.db, because the price cache underneath still
    writes there. The scratch file keeps the SCAN data out of the price history;
    it does not make the run concurrent-safe.

WHY THIS AT ALL
    analysis/universe_expand.py found 427 of 452 ingested tickers pass a
    liquidity screen against a target of ~2,000. The screen can only choose
    among what has been fetched, so breadth is the binding constraint.

    Breadth matters because the h=40 signal shows wide dispersion across ticker
    draws -- prob>=0.70 ranged +2.14pp to +5.46pp over three samples of 80. A
    cross-sectional rank signal sharpens with the number of names ranked: a top
    decile of 420 is 42 stocks, of 2,000 it is 200. Gu, Kelly & Xiu run ~30,000.

    Twenty model configurations were tested on 2026-09-05 and all landed in the
    same place. None changed how many names were being ranked.

SURVIVORSHIP
    company_tickers.json lists registrants that exist TODAY. Everything that
    delisted 2016-2026 is absent, and a wider universe makes the tilt worse in
    absolute terms: more names that will eventually delist, with the history
    still missing those that already did. Correct for a live universe, wrong for
    a backtest, and fixable only with delisted price history.

    python analysis/universe_scan.py --limit 100      # trial
    python analysis/universe_scan.py                  # full, hours
    python analysis/universe_scan.py --report         # rank what has been scanned
"""
import argparse
import json
import os
import sqlite3
import sys
import time
import urllib.request
from datetime import date, timedelta

SEC_TICKERS = "https://www.sec.gov/files/company_tickers.json"
SCAN_DB = "universe_scan.db"
DDL = """
CREATE TABLE IF NOT EXISTS scan (
    ticker      TEXT PRIMARY KEY,
    n_bars      INTEGER,
    last_bar    TEXT,
    last_close  REAL,
    dollar_adv  REAL,
    status      TEXT NOT NULL,
    scanned_at  TEXT NOT NULL
)
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=90,
                    help="recent window used only to measure liquidity")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--ua", default="atomnguyen research atom@ifrenzy.co")
    ap.add_argument("--sleep", type=float, default=0.03)
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--min-adv", type=float, default=5e6)
    ap.add_argument("--min-price", type=float, default=5.0)
    ap.add_argument("--target", type=int, default=2000)
    args = ap.parse_args()

    con = sqlite3.connect(SCAN_DB, timeout=60)
    con.execute(DDL)
    con.commit()

    if args.report:
        rows = con.execute(
            "SELECT ticker, n_bars, last_close, dollar_adv FROM scan "
            "WHERE status='ok' AND dollar_adv IS NOT NULL "
            "ORDER BY dollar_adv DESC").fetchall()
        tot = con.execute("SELECT COUNT(*), status FROM scan "
                          "GROUP BY status").fetchall()
        con.close()
        print(f"scanned so far: " + ", ".join(f"{n} {s}" for n, s in tot))
        ok = [r for r in rows
              if r[3] >= args.min_adv and (r[2] or 0) >= args.min_price]
        print(f"{len(rows)} with ADV, {len(ok)} clear "
              f"ADV >= ${args.min_adv/1e6:.0f}M and price >= "
              f"${args.min_price:.0f}")
        sel = ok[:args.target]
        print(f"\ntop {len(sel)} by dollar ADV would be the universe")
        if sel:
            print(f"  ADV range ${sel[-1][3]/1e6:.1f}M .. "
                  f"${sel[0][3]/1e9:.1f}B")
            print(f"  first 12: " + ", ".join(r[0] for r in sel[:12]))
            print(f"  last 12 : " + ", ".join(r[0] for r in sel[-12:]))
        with open("universe_candidates.txt", "w") as f:
            for r in sel:
                f.write(r[0] + "\n")
        print(f"\nwrote universe_candidates.txt ({len(sel)} names)")
        print("This is pass 1 output. Pass 2 fetches full history for these.")
        return

    done = {r[0] for r in con.execute("SELECT ticker FROM scan")}
    try:
        req = urllib.request.Request(SEC_TICKERS,
                                     headers={"User-Agent": args.ua})
        with urllib.request.urlopen(req, timeout=60) as r:
            data = json.loads(r.read().decode())
        cand = sorted({str(v.get("ticker", "")).upper().strip()
                       for v in data.values()
                       if str(v.get("ticker", "")).strip().isalpha()
                       and len(str(v.get("ticker", "")).strip()) <= 5})
    except Exception as e:
        con.close()
        raise SystemExit(f"could not fetch the SEC ticker list: {e}")

    todo = [t for t in cand if t not in done]
    print(f"{len(cand):,} candidates, {len(done):,} already scanned, "
          f"{len(todo):,} to go")
    if not todo:
        print("nothing to do -- run with --report")
        con.close()
        return

    sys.path.insert(0, ".")
    from features import massive_client as mc
    end_dt = mc._last_completed_session()
    end = end_dt.strftime("%Y-%m-%d")
    start = (end_dt - timedelta(days=args.days)).strftime("%Y-%m-%d")
    print(f"measuring liquidity over {start} .. {end}\n")

    now = str(date.today())
    n_ok = n_none = n_fail = 0
    first_err = None
    t0 = time.time()
    for i, tk in enumerate(todo, 1):
        if args.limit and i > args.limit:
            break
        try:
            df = mc.download(tk, start=start, end=end,
                             auto_adjust=True, progress=False)
            if df is None or len(df) < 20:
                con.execute("INSERT OR REPLACE INTO scan VALUES "
                            "(?,?,?,?,?,?,?)",
                            (tk, 0 if df is None else len(df), None, None,
                             None, "thin", now))
                n_none += 1
            else:
                closes = [float(v) for v in df["Close"] if v == v]
                vols = [float(v) for v in df["Volume"] if v == v]
                n = min(len(closes), len(vols))
                adv = (sum(closes[k] * vols[k] for k in range(n)) / n
                       if n else None)
                con.execute("INSERT OR REPLACE INTO scan VALUES "
                            "(?,?,?,?,?,?,?)",
                            (tk, len(df), str(df.index[-1])[:10],
                             closes[-1] if closes else None, adv, "ok", now))
                n_ok += 1
        except Exception as e:
            con.execute("INSERT OR REPLACE INTO scan VALUES "
                        "(?,?,?,?,?,?,?)", (tk, None, None, None, None,
                                            "fail", now))
            n_fail += 1
            if first_err is None:
                first_err = f"{tk}: {type(e).__name__}: {e}"
        if (n_ok + n_none + n_fail) % 200 == 0:
            con.commit()
            el = time.time() - t0
            rate = (n_ok + n_none + n_fail) / max(el, 1)
            left = (len(todo) - i) / max(rate, 0.01) / 60
            print(f"  {i:>5}/{len(todo)}  ok {n_ok:>5}  thin {n_none:>5}  "
                  f"fail {n_fail:>4}  {el/60:.0f} min elapsed, "
                  f"~{left:.0f} min left")
        time.sleep(args.sleep)

    con.commit()
    tot = con.execute("SELECT COUNT(*) FROM scan").fetchone()[0]
    con.close()
    print(f"\n  {n_ok} ok, {n_none} thin, {n_fail} failed")
    if first_err:
        print(f"  first failure: {first_err}")
    print(f"  {SCAN_DB} now holds {tot:,} rows")
    print(f"\n  NEXT: python analysis/universe_scan.py --report")
    print(f"  That ranks by ADV and writes universe_candidates.txt for pass 2.")


if __name__ == "__main__":
    main()
