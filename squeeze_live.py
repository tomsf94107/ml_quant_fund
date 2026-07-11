#!/usr/bin/env python3
"""
squeeze_live.py -- Live cost-to-borrow (HTB) scanner across your universe.

WHAT IT DOES
  Pulls the LIVE borrow fee + shares-available (UW /api/shorts/{t}/data, intraday)
  for every ticker in your universe (or a selected subset), ranks them by fee
  (most expensive to borrow first), and tags each by tier. This is the morning
  "who is hard to borrow today" sweep -- the squeeze-CANDIDATE radar.

  NOT a buy signal. High borrow fee / hard-to-borrow means "short pressure is
  building"; your own validated SI brick shows high short interest -> LOWER
  forward returns on average. Crowded shorts underperform on average; the squeeze
  is the rare tail. This scan finds candidates, it does not predict direction.

  This is the LIVE scanner (intraday /data). It is DISTINCT from squeeze_scan.py
  (the multi-day PRICE ramp/ignition detector) and from borrow_fetch.py (the
  bi-monthly SETTLEMENT-history backfill into borrow.db).

REUSES
  - load_universe()  from si_fetch_v2.py   (your validated universe loader)
  - uw_get()         from scripts/monitor_ticker.py  (UW auth + 403 handling)

TIERS (by annualized borrow fee %)
  EXTREME  >= 20%      HIGH  >= 5%      MODERATE  >= 1%      (else easy)

RUN
  squeeze                       # all tickers, full ranked   (alias: no --only)
  squeezeselect BYND GME AMC    # just these                 (alias: --only ...)
  squeeze --top 40              # only the 40 most expensive
  squeeze --min-fee 1           # only names with fee >= 1%  (hide easy-to-borrow)
  squeeze --save                # also write logs/htb_scan_YYYYMMDD.txt
"""

import argparse, os, sys, time
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    from si_fetch_v2 import load_universe
except Exception as e:
    sys.exit(f"Cannot import load_universe from si_fetch_v2 (run from repo root): {e}")

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "scripts"))
try:
    from monitor_ticker import uw_get
except Exception as e:
    sys.exit(f"Cannot import uw_get from scripts/monitor_ticker.py: {e}")

UW_PAUSE = 0.12


def tier(fee_pct):
    if fee_pct is None:
        return "?"
    if fee_pct >= 20:
        return "EXTREME"
    if fee_pct >= 5:
        return "HIGH"
    if fee_pct >= 1:
        return "MODERATE"
    return "easy"


def get_live_fee(ticker):
    """Latest live fee + shares-available from UW /data. fee_rate is a STRING %."""
    data = uw_get(f"/api/shorts/{ticker}/data")
    rows = (data or {}).get("data") or []
    if not rows:
        return None
    r0 = rows[0]  # newest-first
    try:
        fee = float(r0.get("fee_rate")) if r0.get("fee_rate") is not None else None
    except (TypeError, ValueError):
        fee = None
    av = r0.get("short_shares_available")
    try:
        av = int(av) if av is not None else None
    except (TypeError, ValueError):
        av = None
    return {"ticker": ticker.upper(), "fee": fee, "avail": av, "ts": r0.get("timestamp")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--prices-db", default=None)
    ap.add_argument("--si-db", default=None)
    ap.add_argument("--only", nargs="+", default=None, help="specific tickers (overrides universe)")
    ap.add_argument("--top", type=int, default=None, help="show only the top N by fee")
    ap.add_argument("--min-fee", type=float, default=None, help="only show fee >= this %% (hide easy)")
    ap.add_argument("--save", action="store_true", help="also write logs/htb_scan_YYYYMMDD.txt")
    a = ap.parse_args()

    prices_db = a.prices_db or os.path.join(a.root, "prices.db")
    si_db     = a.si_db or os.path.join(a.root, "short_interest.db")

    if a.only:
        universe = [t.upper() for t in a.only]
        scope = f"{len(universe)} selected"
    else:
        uni = load_universe(prices_db, si_db)
        if not uni:
            sys.exit("  universe NOT FOUND -- set --prices-db / --si-db")
        universe = sorted(uni)
        scope = f"{len(universe)} (full universe)"

    print(f"  Scanning {scope} for live borrow fee ...")
    results = []
    empties = errs = 0
    t0 = time.time()
    for i, tk in enumerate(universe, 1):
        try:
            r = get_live_fee(tk)
        except Exception:
            errs += 1
            continue
        if r is None or r["fee"] is None:
            empties += 1
            continue
        results.append(r)
        # light progress ping every 50 on a full run
        if not a.only and i % 50 == 0:
            print(f"    ... {i}/{len(universe)}")
        time.sleep(UW_PAUSE)

    # rank by fee desc
    results.sort(key=lambda r: (r["fee"] if r["fee"] is not None else -1), reverse=True)

    # optional filters
    shown = results
    if a.min_fee is not None:
        shown = [r for r in shown if r["fee"] is not None and r["fee"] >= a.min_fee]
    if a.top is not None:
        shown = shown[:a.top]

    # counts by tier (over ALL scanned, not just shown)
    from collections import Counter
    tc = Counter(tier(r["fee"]) for r in results)

    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    lines = []
    lines.append("=" * 70)
    lines.append(f"  LIVE BORROW-FEE SCAN (HTB radar)   {ts}")
    lines.append(f"  scanned {len(results)} tickers  |  EXTREME {tc.get('EXTREME',0)}  "
                 f"HIGH {tc.get('HIGH',0)}  MODERATE {tc.get('MODERATE',0)}  easy {tc.get('easy',0)}")
    lines.append("  (fuel = candidate strength, NOT a buy signal; high SI -> underperforms on avg)")
    lines.append("  TIERS:  EXTREME >=20%   HIGH >=5%   MODERATE >=1%   easy <1%")
    lines.append("=" * 70)
    lines.append(f"  {'#':>3}  {'TICKER':<7} {'FEE%':>8}  {'TIER':<9} {'SHARES AVAIL':>14}")
    lines.append("  " + "-" * 62)
    for i, r in enumerate(shown, 1):
        fee = f"{r['fee']:.2f}" if r["fee"] is not None else "n/a"
        av = f"{r['avail']:,}" if r["avail"] is not None else "n/a"
        t = tier(r["fee"])
        mark = "  <<<" if t in ("EXTREME", "HIGH") else ""
        lines.append(f"  {i:>3}  {r['ticker']:<7} {fee:>8}  {t:<9} {av:>14}{mark}")
    lines.append("  " + "-" * 62)
    if a.min_fee is not None or a.top is not None:
        lines.append(f"  (showing {len(shown)} of {len(results)} scanned; filters active)")
    if empties or errs:
        lines.append(f"  (no data: {empties}  errors: {errs})")
    lines.append(f"  scan took {time.time()-t0:.0f}s")
    lines.append("=" * 70)

    out = "\n".join(lines)
    print(out)

    if a.save:
        os.makedirs(os.path.join(a.root, "logs"), exist_ok=True)
        fn = os.path.join(a.root, "logs", f"htb_scan_{datetime.now().strftime('%Y%m%d')}.txt")
        with open(fn, "w") as f:
            f.write(out + "\n")
        print(f"  saved -> {fn}")


if __name__ == "__main__":
    main()
