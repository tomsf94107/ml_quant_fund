#!/usr/bin/env python3
"""
borrow_fetch.py -- Backfill UW cost-to-borrow history into borrow.db (borrow_fees).

WHAT IT DOES (Phase 0 of the CTB model-feature track):
  For each ticker in your universe, calls UW /api/shorts/{ticker}/interest-float/v2
  (one call returns the FULL ~5yr bi-monthly settlement history), maps the fields
  to the EXISTING borrow_fees schema, and idempotently upserts.

REUSES (does not reinvent):
  - load_universe()/ro()/Q()  from si_fetch_v2.py  (DB-side, API-agnostic)
  - uw_get()                  from scripts/monitor_ticker.py  (UW auth + 403 handling)

OPTION A (chosen): stores BORROW-specific fields only. SI%/DTC/float are NOT stored
  here -- they live in the validated FINRA short_interest.db and are joined at
  feature-build time. This keeps a single source of truth per field (no UW-vs-FINRA
  SI drift). UW's si_float/dtc are ignored on purpose.

SCHEMA MAPPING (borrow_fees):
  ticker         <- symbol
  asof_date      <- market_date
  borrow_fee_bps <- fee_rate (PERCENT) * 100   [21.83% -> 2183 bps]
  utilization    <- NULL (UW doesn't provide it on this endpoint)
  shares_avail   <- short_shares_available
  is_htb         <- 1 if borrow_fee_bps >= HTB_BPS else 0   [aligned to assess_squeeze 1% tier]
  source         <- "UW:interest-float/v2"
  fetched_at     <- now (UTC ISO)

RUN
  # 10-ticker test first:
  python borrow_fetch.py --limit 10
  # then full universe:
  python borrow_fetch.py
  # inspect without writing:
  python borrow_fetch.py --limit 10 --dry-run
"""

import argparse, os, sqlite3, sys, time
from datetime import datetime, timezone

# ---- reuse si_fetch_v2's DB helpers + universe loader ----
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    from si_fetch_v2 import ro, Q, load_universe
except Exception as e:
    sys.exit(f"Cannot import si_fetch_v2 helpers (run from repo root): {e}")

# ---- reuse monitor_ticker's UW fetcher (auth + 403 handling) ----
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "scripts"))
try:
    from monitor_ticker import uw_get
except Exception as e:
    sys.exit(f"Cannot import uw_get from scripts/monitor_ticker.py: {e}")

BORROW_DB_DEFAULT = "borrow.db"
HTB_BPS = 100.0            # >=1% borrow fee => hard-to-borrow flag (matches assess_squeeze MODERATE tier)
UW_PAUSE = 0.15           # polite pause between tickers (well under 40k/day limit)


def now_iso():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def fetch_ticker(ticker):
    """One UW call -> full settlement history rows for this ticker (or [])."""
    data = uw_get(f"/api/shorts/{ticker}/interest-float/v2")
    return (data or {}).get("data") or []


def map_row(ticker, r):
    """UW row -> borrow_fees tuple, or None if unusable. Option A: borrow fields only."""
    d = r.get("market_date")
    if not d:
        return None
    # fee_rate is a PERCENT string; convert to basis points.
    fee_raw = r.get("fee_rate")
    try:
        fee_bps = float(fee_raw) * 100.0 if fee_raw is not None else None
    except (TypeError, ValueError):
        fee_bps = None
    # shares available (int)
    av = r.get("short_shares_available")
    try:
        av = int(av) if av is not None else None
    except (TypeError, ValueError):
        av = None
    is_htb = 1 if (fee_bps is not None and fee_bps >= HTB_BPS) else 0
    return (
        ticker.upper(),
        d,
        fee_bps,
        None,              # utilization -- not provided by this endpoint
        av,
        is_htb,
        "UW:interest-float/v2",
        now_iso(),
    )


def ensure_schema(con):
    """borrow_fees already exists (verified); create if somehow missing so the
    script is self-standing. Matches the existing schema exactly."""
    con.execute("""
        CREATE TABLE IF NOT EXISTS borrow_fees (
            ticker         TEXT NOT NULL,
            asof_date      TEXT NOT NULL,
            borrow_fee_bps REAL,
            utilization    REAL,
            shares_avail   INTEGER,
            is_htb         INTEGER,
            source         TEXT,
            fetched_at     TEXT,
            PRIMARY KEY (ticker, asof_date)
        )
    """)
    con.execute("CREATE INDEX IF NOT EXISTS idx_borrow_ticker_date ON borrow_fees(ticker, asof_date)")
    con.commit()


def upsert(con, rows):
    """Idempotent upsert on (ticker, asof_date) PK -- mirrors si_fetch_v2 pattern."""
    con.executemany("""
        INSERT OR REPLACE INTO borrow_fees
        (ticker, asof_date, borrow_fee_bps, utilization, shares_avail, is_htb, source, fetched_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, rows)
    con.commit()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--borrow-db", default=None)
    ap.add_argument("--prices-db", default=None)
    ap.add_argument("--si-db", default=None)
    ap.add_argument("--limit", type=int, default=None, help="only first N tickers (test)")
    ap.add_argument("--only", nargs="+", default=None, help="specific tickers, overrides universe")
    ap.add_argument("--dry-run", action="store_true", help="fetch + map + print, no DB write")
    a = ap.parse_args()

    borrow_db = a.borrow_db or os.path.join(a.root, BORROW_DB_DEFAULT)
    prices_db = a.prices_db or os.path.join(a.root, "prices.db")
    si_db     = a.si_db or os.path.join(a.root, "short_interest.db")

    # --- universe (reuse the validated loader) ---
    if a.only:
        universe = [t.upper() for t in a.only]
        print(f"  universe: {len(universe)} tickers (explicit --only)")
    else:
        uni = load_universe(prices_db, si_db)
        if not uni:
            sys.exit("  universe NOT FOUND -- set --prices-db / --si-db")
        universe = sorted(uni)
        print(f"  universe: {len(universe)} tickers (from load_universe)")

    if a.limit:
        universe = universe[:a.limit]
        print(f"  limited to first {len(universe)}: {', '.join(universe)}")

    con = None
    if not a.dry_run:
        con = sqlite3.connect(borrow_db, timeout=30)
        ensure_schema(con)

    tot_rows = tot_ok = tot_empty = tot_err = 0
    t0 = time.time()
    for i, tk in enumerate(universe, 1):
        try:
            uw_rows = fetch_ticker(tk)
        except Exception as e:
            tot_err += 1
            print(f"  [{i}/{len(universe)}] {tk:6s} ERROR: {e}")
            continue
        if not uw_rows:
            tot_empty += 1
            print(f"  [{i}/{len(universe)}] {tk:6s} (no borrow data)")
            time.sleep(UW_PAUSE)
            continue
        mapped = [m for m in (map_row(tk, r) for r in uw_rows) if m]
        tot_rows += len(mapped)
        tot_ok += 1
        # date range + latest fee for a quick sanity line
        dates = sorted(m[1] for m in mapped)
        latest = mapped[0]  # UW returns newest-first; [0] is latest
        # find the newest by date to be safe
        newest = max(mapped, key=lambda m: m[1])
        fee_disp = f"{newest[2]/100:.2f}%" if newest[2] is not None else "n/a"
        print(f"  [{i}/{len(universe)}] {tk:6s} {len(mapped):3d} rows  "
              f"{dates[0]}..{dates[-1]}  latest fee {fee_disp}"
              + ("  [HTB]" if newest[5] == 1 else ""))
        if not a.dry_run:
            upsert(con, mapped)
        time.sleep(UW_PAUSE)

    if con:
        # report final DB state
        after = con.execute("SELECT COUNT(*), COUNT(DISTINCT ticker), MIN(asof_date), MAX(asof_date) FROM borrow_fees").fetchone()
        con.close()
        print("\n  === borrow.db borrow_fees NOW ===")
        print(f"  {after[0]:,} rows | {after[1]} tickers | {after[2]} .. {after[3]}")

    dt = time.time() - t0
    print(f"\n  DONE in {dt:.0f}s | ok:{tot_ok} empty:{tot_empty} err:{tot_err} | rows mapped:{tot_rows:,}"
          + ("  (DRY RUN -- nothing written)" if a.dry_run else ""))


if __name__ == "__main__":
    main()
