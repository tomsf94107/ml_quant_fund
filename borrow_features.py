#!/usr/bin/env python3
"""
borrow_features.py -- Phase 1: build TRANSFORMED borrow-fee features from borrow.db.

WHY TRANSFORMED (not raw fee):
  Raw borrow fee is ~constant (~0.25-0.5%) for ~90% of the universe -> near-zero
  variance -> noise as a universe-wide model feature (the "Prob Global" mistake).
  The signal, if any, is in the MOVEMENT and CROSS-SECTIONAL UNUSUALNESS, not the
  level. These transforms capture that:

    fee_change_1s     : delta fee vs prior settlement (~2wk)  -> rising borrow cost
    fee_change_3s     : delta fee vs 3 settlements ago (~6wk) -> slower build
    avail_change_1s   : delta shares-available vs prior settlement -> supply drying up
    fee_zscore_xsec   : cross-sectional z-score of fee on each date -> how unusual today
    fee_gt_5pct       : binary, fee >= 5% (genuinely hard-to-borrow)

  SETTLEMENT-SPACED, not daily. Names use _1s/_3s (settlements), NOT "_5d" -- the
  bi-monthly data cannot support daily deltas. (Daily deltas would need the live
  /data feed, a separate finer-grained source.)

  Offline. Reads borrow.db.borrow_fees, writes borrow.db.borrow_features. Idempotent.

RUN
  python borrow_features.py                 # build all features
  python borrow_features.py --dry-run       # compute + preview, no write
"""

import argparse, os, sqlite3, sys
from collections import defaultdict
from datetime import datetime, timezone
import statistics

BORROW_DB_DEFAULT = "borrow.db"


def now_iso():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def build_features(borrow_db, dry_run=False):
    con = sqlite3.connect(borrow_db, timeout=30)
    cur = con.cursor()

    # Pull all borrow rows: ticker, date, fee (bps), shares_avail
    rows = cur.execute("""
        SELECT ticker, asof_date, borrow_fee_bps, shares_avail
        FROM borrow_fees
        WHERE asof_date IS NOT NULL
        ORDER BY ticker, asof_date
    """).fetchall()
    if not rows:
        sys.exit("  borrow_fees is empty -- run borrow_fetch.py first")

    # Organize per ticker, chronological
    by_ticker = defaultdict(list)  # ticker -> [(date, fee_bps, avail), ...]
    for tk, d, fee, av in rows:
        by_ticker[tk].append((d, fee, av))

    # Cross-sectional stats per date (for zscore): date -> [fee_bps, ...]
    by_date_fees = defaultdict(list)
    for tk, d, fee, av in rows:
        if fee is not None:
            by_date_fees[d].append(fee)
    date_mean = {}
    date_std = {}
    for d, fees in by_date_fees.items():
        if len(fees) >= 2:
            date_mean[d] = statistics.mean(fees)
            date_std[d] = statistics.pstdev(fees)
        else:
            date_mean[d] = fees[0] if fees else None
            date_std[d] = 0.0

    # Compute features per (ticker, date)
    out = []  # (ticker, asof_date, fee_change_1s, fee_change_3s, avail_change_1s, fee_zscore_xsec, fee_gt_5pct, fetched_at)
    for tk, series in by_ticker.items():
        # series is chronological (ORDER BY asof_date)
        for i, (d, fee, av) in enumerate(series):
            fee_change_1s = None
            fee_change_3s = None
            avail_change_1s = None
            if fee is not None and i >= 1 and series[i-1][1] is not None:
                fee_change_1s = fee - series[i-1][1]
            if fee is not None and i >= 3 and series[i-3][1] is not None:
                fee_change_3s = fee - series[i-3][1]
            if av is not None and i >= 1 and series[i-1][2] is not None:
                avail_change_1s = av - series[i-1][2]
            # cross-sectional z-score of fee on this date
            fee_z = None
            if fee is not None and date_std.get(d):
                sd = date_std[d]
                if sd and sd > 0:
                    fee_z = (fee - date_mean[d]) / sd
            # binary hard-to-borrow at 5% (= 500 bps)
            fee_gt_5pct = 1 if (fee is not None and fee >= 500.0) else 0

            out.append((tk, d, fee_change_1s, fee_change_3s, avail_change_1s,
                        fee_z, fee_gt_5pct, now_iso()))

    print(f"  computed {len(out):,} feature rows across {len(by_ticker)} tickers")

    # preview a few non-null fee_change rows
    sample = [r for r in out if r[2] is not None][:5]
    print("  --- sample (ticker, date, fee_chg_1s_bps, fee_chg_3s_bps, avail_chg_1s, fee_z, htb5) ---")
    for r in sample:
        fc1 = f"{r[2]:+.1f}" if r[2] is not None else "n/a"
        fc3 = f"{r[3]:+.1f}" if r[3] is not None else "n/a"
        ac1 = f"{r[4]:+,.0f}" if r[4] is not None else "n/a"
        fz = f"{r[5]:+.2f}" if r[5] is not None else "n/a"
        print(f"    {r[0]:<6} {r[1]}  fee_chg1s={fc1:>8}  fee_chg3s={fc3:>8}  avail_chg1s={ac1:>12}  z={fz:>6}  htb5={r[6]}")

    if dry_run:
        con.close()
        print("  DRY RUN -- nothing written")
        return

    # Write borrow_features table
    cur.execute("""
        CREATE TABLE IF NOT EXISTS borrow_features (
            ticker          TEXT NOT NULL,
            asof_date       TEXT NOT NULL,
            fee_change_1s   REAL,
            fee_change_3s   REAL,
            avail_change_1s REAL,
            fee_zscore_xsec REAL,
            fee_gt_5pct     INTEGER,
            fetched_at      TEXT,
            PRIMARY KEY (ticker, asof_date)
        )
    """)
    cur.execute("CREATE INDEX IF NOT EXISTS idx_bf_ticker_date ON borrow_features(ticker, asof_date)")
    cur.executemany("""
        INSERT OR REPLACE INTO borrow_features
        (ticker, asof_date, fee_change_1s, fee_change_3s, avail_change_1s,
         fee_zscore_xsec, fee_gt_5pct, fetched_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, out)
    con.commit()

    after = cur.execute("SELECT COUNT(*), COUNT(DISTINCT ticker), MIN(asof_date), MAX(asof_date) FROM borrow_features").fetchone()
    con.close()
    print(f"\n  === borrow_features NOW ===")
    print(f"  {after[0]:,} rows | {after[1]} tickers | {after[2]} .. {after[3]}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--borrow-db", default=None)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    borrow_db = a.borrow_db or os.path.join(a.root, BORROW_DB_DEFAULT)
    if not os.path.isfile(borrow_db):
        sys.exit(f"  borrow.db not found at {borrow_db}")
    build_features(borrow_db, dry_run=a.dry_run)


if __name__ == "__main__":
    main()
