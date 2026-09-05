#!/usr/bin/env python3
"""
universe_fetch.py — ingest daily bars for a wider US universe.

WRITES to prices.db raw_bars. Resumable. INSERT OR IGNORE, so it cannot
overwrite an existing bar; a re-run only fills gaps.

WHY
    analysis/universe_expand.py screened the current database and found 427 of
    452 ingested tickers pass a liquidity screen -- but the target is ~2,000.
    The screen can only choose among what has been fetched, so breadth is the
    binding constraint, not the screen.

    Breadth matters because the h=40 signal found on 2026-09-05 shows wide
    dispersion across ticker samples: prob>=0.70 ranged +2.14pp to +5.46pp over
    three draws of 80 names. A cross-sectional rank signal sharpens with the
    number of names being ranked -- a top decile of 420 is 42 stocks, of 2,000
    it is 200, and the head of that ranking is far more selective. Gu, Kelly &
    Xiu run ~30,000.

    Twenty model configurations were tested on 2026-09-05 and all landed in the
    same place. None of them changed how many names were being ranked.

CANDIDATE SOURCE
    SEC company_tickers.json -- 10,412 US filers, the same file used on
    2026-09-05 to resolve CIKs for the Form 4 backfill. Free, no auth, and it is
    the registrant list rather than a vendor's coverage view.

    That list includes ETFs, trusts, and many names too illiquid to trade. This
    script fetches broadly and lets universe_expand.py screen afterwards, rather
    than pre-guessing liquidity -- a name's ADV is not knowable until its bars
    are in hand.

SURVIVORSHIP, AGAIN AND UNCHANGED
    company_tickers.json lists registrants that exist TODAY. Every company that
    delisted between 2016 and now is absent. Expanding the universe makes the
    survivor tilt WORSE in absolute terms -- more names that will eventually
    delist, and the history still lacks the ones that already did.

    This is correct for a live universe and wrong for a backtest. It cannot be
    fixed here; it can only be sized, the way si_leg_decomp.py sized it for the
    SI brick (79% of that edge sits in low days-to-cover names, which rarely
    delist, so the tilt was near-moot there). Nobody has sized it for the h=40
    signal.

SAFETY
    - INSERT OR IGNORE keyed on (ticker, d); an existing bar is never modified.
    - --dry-run reports what WOULD be fetched and writes nothing.
    - --limit caps the run so the first pass can be checked before committing.
    - Existing tickers are skipped unless --refresh is passed.
    - A per-ticker failure is logged and does not stop the run, but the failure
      COUNT and the first error are printed -- a silent-failure loop is how a
      backfill can appear to succeed while writing nothing.

    python analysis/universe_fetch.py --dry-run
    python analysis/universe_fetch.py --limit 50
    python analysis/universe_fetch.py --target 2000
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="prices.db")
    ap.add_argument("--start", default="2016-07-18",
                    help="Massive's earliest available date")
    ap.add_argument("--target", type=int, default=2000,
                    help="how many NEW tickers to attempt")
    ap.add_argument("--limit", type=int, default=0,
                    help="stop after N tickers; 0 = no cap")
    ap.add_argument("--ua", default="atomnguyen research atom@ifrenzy.co")
    ap.add_argument("--sleep", type=float, default=0.05)
    ap.add_argument("--min-bars", type=int, default=250)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--refresh", action="store_true",
                    help="also re-fetch tickers already present")
    args = ap.parse_args()

    if not os.path.exists(args.db):
        raise SystemExit(f"{args.db} not found -- run from the repo root")

    con = sqlite3.connect(args.db, timeout=60)
    have = {r[0] for r in con.execute("SELECT DISTINCT ticker FROM raw_bars")}
    n0 = con.execute("SELECT COUNT(*) FROM raw_bars").fetchone()[0]
    print(f"raw_bars: {n0:,} rows, {len(have)} tickers already present")

    cols = [r[1] for r in con.execute("PRAGMA table_info(raw_bars)")]
    print(f"raw_bars columns: {', '.join(cols)}")
    need = {"ticker", "d", "close"}
    if not need.issubset(set(cols)):
        con.close()
        raise SystemExit(f"raw_bars is missing one of {need}")

    try:
        req = urllib.request.Request(SEC_TICKERS,
                                     headers={"User-Agent": args.ua})
        with urllib.request.urlopen(req, timeout=60) as r:
            data = json.loads(r.read().decode())
        allt = []
        for v in data.values():
            t = str(v.get("ticker", "")).upper().strip()
            if t and t.isalpha() and len(t) <= 5:
                allt.append(t)
        allt = sorted(set(allt))
        print(f"SEC company_tickers.json: {len(allt):,} candidate symbols "
              f"(alpha, <=5 chars)")
    except Exception as e:
        con.close()
        raise SystemExit(f"could not fetch the SEC ticker list: {e}")

    todo = [t for t in allt if args.refresh or t not in have][:args.target]
    print(f"{len(todo):,} to attempt "
          f"({'including' if args.refresh else 'excluding'} the "
          f"{len(have)} already present)")

    if args.dry_run:
        print(f"\nDRY RUN -- nothing written.")
        print(f"  would attempt: {', '.join(todo[:15])} ...")
        print(f"  from {args.start}, keeping tickers with "
              f">= {args.min_bars} bars")
        con.close()
        return

    sys.path.insert(0, ".")
    from features import massive_client as mc

    end = mc._last_completed_session().strftime("%Y-%m-%d")
    print(f"fetching {args.start} .. {end}\n")

    ins = ("INSERT OR IGNORE INTO raw_bars "
           "(" + ",".join(cols[1:] if cols[0] == "id" else cols) + ") "
           "VALUES (" + ",".join("?" * len(cols[1:] if cols[0] == "id"
                                           else cols)) + ")")
    use_cols = cols[1:] if cols[0] == "id" else cols

    n_ok = n_thin = n_fail = 0
    rows_written = 0
    first_err = None
    t0 = time.time()

    for i, tk in enumerate(todo, 1):
        if args.limit and i > args.limit:
            break
        try:
            df = mc.download(tk, start=args.start, end=end,
                             auto_adjust=True, progress=False)
            if df is None or len(df) < args.min_bars:
                n_thin += 1
                continue
            batch = []
            for idx, row in df.iterrows():
                rec = {"ticker": tk, "d": str(idx)[:10]}
                for c in use_cols:
                    if c in ("ticker", "d"):
                        continue
                    key = {"close": "Close", "open": "Open", "high": "High",
                           "low": "Low", "volume": "Volume",
                           "adj_close": "Close"}.get(c, c.capitalize())
                    v = row.get(key)
                    rec[c] = float(v) if v is not None and v == v else None
                batch.append(tuple(rec.get(c) for c in use_cols))
            con.executemany(ins, batch)
            rows_written += len(batch)
            n_ok += 1
            if n_ok % 50 == 0:
                con.commit()
                el = time.time() - t0
                print(f"  {n_ok:>5} ok  {n_thin:>4} thin  {n_fail:>3} failed  "
                      f"{rows_written:>9,} rows  {el/60:.1f} min")
        except Exception as e:
            n_fail += 1
            if first_err is None:
                first_err = f"{tk}: {type(e).__name__}: {e}"
        time.sleep(args.sleep)

    con.commit()
    n1 = con.execute("SELECT COUNT(*) FROM raw_bars").fetchone()[0]
    tk1 = con.execute("SELECT COUNT(DISTINCT ticker) FROM raw_bars").fetchone()[0]
    con.close()

    print(f"\n  {n_ok} fetched, {n_thin} too thin (< {args.min_bars} bars), "
          f"{n_fail} failed")
    if first_err:
        print(f"  first failure: {first_err}")
    print(f"  raw_bars: {n0:,} -> {n1:,} rows (+{n1-n0:,}), "
          f"{len(have)} -> {tk1} tickers")
    print("\n  NEXT: re-run analysis/universe_expand.py to screen the wider set,")
    print("  then decide whether to adopt tickers_expanded.txt. Nothing")
    print("  downstream changes until tickers.txt does.")
    print("\n  Feature building is roughly 3s/ticker, so a 2,000-name universe")
    print("  is ~2 hours per full pass. The nightly retrain scales with it.")


if __name__ == "__main__":
    main()
