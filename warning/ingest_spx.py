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
import os
import sqlite3
import sys
from datetime import date, timedelta

SERIES_SUFFIX = "_CLOSE"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prices", default="prices.db")
    ap.add_argument("--db", default="warning.db")
    ap.add_argument("--ticker", default="SPY",
                    help="one ticker, or a comma-separated list "
                         "(e.g. SPY,XLP,XLU,XLV)")
    ap.add_argument("--from-massive", action="store_true",
                    help="fetch from Massive instead of prices.db, for tickers "
                         "the price DB does not carry (e.g. RSP). Provenance is "
                         "recorded in the source column so the two routes are "
                         "never confused.")
    ap.add_argument("--start", default="2016-07-18",
                    help="only used with --from-massive")
    ap.add_argument("--table", default="raw_bars",
                    help="raw_bars (unadjusted close, preferred: closest to a "
                         "price index) or daily_prices (adj_close, dividend-"
                         "adjusted -> a total-return series, which reaches new "
                         "highs earlier and biases a 52w-high test)")
    args = ap.parse_args()

    tickers = [t.strip().upper() for t in args.ticker.split(",") if t.strip()]
    for tkr in tickers:
        _ingest_one(args, tkr)


def _from_massive(ticker, start):
    """Direct Massive fetch for tickers absent from prices.db.

    RSP is the case this exists for: S6 needs the equal-weight S&P against the
    cap-weight one, prices.db carries 443 tickers and RSP is not among them, but
    Massive returns 2,511 bars for it. Fetching directly is preferable to
    quietly dropping the signal.

    Massive returns tz-NAIVE UTC timestamps -- the same property that broke the
    intraday reconciler on 2026-08-30 -- so only the DATE is taken here, which is
    all a daily close needs.
    """
    # This file lives in warning/, so Python puts THAT directory on sys.path,
    # not the repo root -- `from features import ...` fails here even though the
    # same import works from a one-liner run at the root. Add the parent
    # explicitly rather than depending on the caller's working directory.
    _root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if _root not in sys.path:
        sys.path.insert(0, _root)
    from features import massive_client as mc
    from datetime import date as _date
    h = mc.download(ticker, start=start, end=_date.today().isoformat(),
                    auto_adjust=True, progress=False)
    if h is None or h.empty:
        return []
    import pandas as pd
    if isinstance(h.columns, pd.MultiIndex):
        h.columns = h.columns.get_level_values(0)
    out = []
    for ts, row in h.iterrows():
        v = row["Close"]
        if v == v:                     # skip NaN
            out.append((str(ts)[:10], float(v)))
    # one row per date; Massive may return intraday granularity
    seen = {}
    for d, v in out:
        seen[d] = v
    return sorted(seen.items())


def _ingest_one(args, ticker):
    if args.from_massive:
        rows = _from_massive(ticker, args.start)
        if not rows:
            print(f"  !! Massive returned nothing for {ticker}")
            return
        _write(args, ticker, rows, f"massive/{ticker}")
        return

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

    _write(args, ticker, rows, f"prices.db/{args.table}")


def _write(args, ticker, rows, provenance):
    series = ticker + SERIES_SUFFIX
    con = sqlite3.connect(args.db)
    before = con.execute("SELECT COUNT(*) FROM data_vintages WHERE series_id=?",
                         (series,)).fetchone()[0]
    n = 0
    for d, close in rows:
        pub = (date.fromisoformat(d) + timedelta(days=1)).isoformat()
        con.execute("INSERT OR IGNORE INTO data_vintages "
                    "(series_id, obs_date, pub_date, value, source) VALUES (?,?,?,?,?)",
                    (series, d, pub, float(close), provenance))
        n += 1
    con.commit()
    after = con.execute("SELECT COUNT(*) FROM data_vintages WHERE series_id=?",
                        (series,)).fetchone()[0]
    rng = con.execute("SELECT MIN(obs_date), MAX(obs_date) FROM data_vintages "
                      "WHERE series_id=?", (series,)).fetchone()
    con.close()

    print(f"{series}: read {n} rows from {provenance}")
    print(f"  data_vintages rows {before} -> {after} (+{after - before}); "
          f"obs range {rng[0]}..{rng[1]}")
    print(f"  pub_date = obs_date + 1 day. Re-runs are idempotent.")
    if ticker == "SPY":
        print(f"  NOTE: SPY is a PROXY for SPX and is labelled as such in "
              f"every reading that uses it.")


if __name__ == "__main__":
    main()
