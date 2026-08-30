#!/usr/bin/env python3
"""
ingest_spx.py — copy SPY daily closes from prices.db into warning.db.

WHY A COPY RATHER THAN A CROSS-DB READ
    Every macro input reaches a builder through pit.series_asof, which enforces
    publication-date semantics. Reading prices.db directly would create a second
    path with no vintage discipline, and the first thing to go wrong would be a
    builder quietly seeing a close before it was knowable. One path, one rule.

SPY IS A PROXY FOR SPX, AND IS LABELLED AS ONE
    The registry says "SPX within 3% of 52w high". SPY is an ETF tracking the
    index; it is close but not identical (tracking error, dividend timing). The
    series is stored as SPY_CLOSE, and S2 reports `equity_source` in every
    reading so no result hides which instrument produced it.

COVERAGE LIMIT, VERIFIED 2026-08-28
    prices.db raw_bars holds SPY from 2016-07-18 (2,543 rows). The equity leg is
    therefore computable from roughly 2017-08 onward, once 252+21 sessions exist.
    It cannot be computed for 2000 or 2007. That is a coverage finding, not a
    defect: S2 arms but does not fire red for those dates, and says why.

pub_date = obs_date + 1 day. A close is knowable that evening; the one-day lag is
conservative and matches how every other non-revisable series is stamped.

USAGE
    python warning/ingest_spx.py --prices prices.db --db warning.db
    python warning/ingest_spx.py --prices prices.db --db warning.db --ticker SPY
"""
import argparse
import sqlite3
from datetime import date, timedelta

SERIES_SUFFIX = "_CLOSE"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prices", default="prices.db")
    ap.add_argument("--db", default="warning.db")
    ap.add_argument("--ticker", default="SPY",
                    help="one ticker, or a comma-separated list "
                         "(e.g. SPY,XLP,XLU,XLV)")
    ap.add_argument("--table", default="raw_bars",
                    help="raw_bars (unadjusted close, preferred: closest to a "
                         "price index) or daily_prices (adj_close, dividend-"
                         "adjusted -> a total-return series, which reaches new "
                         "highs earlier and biases a 52w-high test)")
    args = ap.parse_args()

    tickers = [t.strip().upper() for t in args.ticker.split(",") if t.strip()]
    for tkr in tickers:
        _ingest_one(args, tkr)


def _ingest_one(args, ticker):
    src = sqlite3.connect(f"file:{args.prices}?mode=ro", uri=True)
    if args.table == "raw_bars":
        rows = src.execute("SELECT d, close FROM raw_bars WHERE ticker=? "
                           "AND close IS NOT NULL ORDER BY d", (ticker,)).fetchall()
    else:
        rows = src.execute("SELECT date, adj_close FROM daily_prices WHERE ticker=? "
                           "AND adj_close IS NOT NULL ORDER BY date",
                           (ticker,)).fetchall()
    src.close()

    if not rows:
        print(f"  !! no rows for {ticker} in {args.prices}.{args.table}"); return

    series = ticker + SERIES_SUFFIX
    con = sqlite3.connect(args.db)
    before = con.execute("SELECT COUNT(*) FROM data_vintages WHERE series_id=?",
                         (series,)).fetchone()[0]
    n = 0
    for d, close in rows:
        pub = (date.fromisoformat(d) + timedelta(days=1)).isoformat()
        con.execute("INSERT OR IGNORE INTO data_vintages "
                    "(series_id, obs_date, pub_date, value, source) VALUES (?,?,?,?,?)",
                    (series, d, pub, float(close), f"prices.db/{args.table}"))
        n += 1
    con.commit()
    after = con.execute("SELECT COUNT(*) FROM data_vintages WHERE series_id=?",
                        (series,)).fetchone()[0]
    rng = con.execute("SELECT MIN(obs_date), MAX(obs_date) FROM data_vintages "
                      "WHERE series_id=?", (series,)).fetchone()
    con.close()

    print(f"{series}: read {n} rows from {args.prices}.{args.table}")
    print(f"  data_vintages rows {before} -> {after} (+{after - before}); "
          f"obs range {rng[0]}..{rng[1]}")
    print(f"  pub_date = obs_date + 1 day. Re-runs are idempotent.")
    if ticker == "SPY":
        print(f"  NOTE: SPY is a PROXY for SPX and is labelled as such in "
              f"every reading that uses it.")


if __name__ == "__main__":
    main()
