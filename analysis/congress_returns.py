#!/usr/bin/env python3
"""
congress_returns.py — what each disclosure was worth, against SPY.

READ-ONLY on prices.db. Writes congress_returns into congress_trades.db.

THE ONLY COLUMN THAT MEANS ANYTHING IS `excess`
    A raw return is uninterpretable. In a rising market almost everything goes
    up, so +9% is excellent against a flat tape and poor against +12%. Every
    row here therefore carries the stock's return AND SPY's return over the
    IDENTICAL bar range, and their difference.

    This is not a general principle applied for tidiness. It is the error that
    ran through an entire session on 2026-09-14/15: an h=5 hit rate falling
    61.9% to 37.4% was read as model decay three times, and benchmarking showed
    the field had fallen faster and the edge had RISEN to +9.3pp. The fund's
    own docs/MASTER_TODO_LIST.md 1.1b records the same trap: "The 58% is the
    market's rising tide, not skill."

TWO DATE CONVENTIONS, AND THE GAP BETWEEN THEM IS THE POINT
    filed  -- entry at the first close AFTER filed_at_date. What YOU could have
              earned copying the disclosure. The tradeable number, and the one
              any ranking on the page must use.
    txn    -- entry at the first close after transaction_date. What the MEMBER
              got. Not tradeable: that price was unknowable to the public for a
              median 38 days.

    excess_filed is the copy-trading edge. excess_txn minus excess_filed is
    what the disclosure lag costs, which is the closest thing in this data to
    an insider-edge measure. The 2025 CEPR finding that leadership alphas
    survive "even when portfolios are constructed using public disclosure dates
    rather than actual execution dates" is a statement about exactly this gap.

BUYS AND SELLS ARE NEVER POOLED
    For a buy, a positive excess is a good call. For a sell, a NEGATIVE excess
    is a good call -- they got out before it lagged. Averaging the two together
    produces a number that means nothing, so `excess` is stored UNSIGNED (the
    stock's excess return, whatever the trade direction) and every consumer
    must group by txn_type. The page reports buy and sell columns separately.

WHAT CANNOT BE COMPUTED, AND WHY THE PAGE MUST SAY SO
    Congress trades bonds, private funds, crypto and foreign listings. 787 rows
    carry no priceable ticker at all -- '$BTC', '3G FUND VI LP', 'ABNFX'. A
    further 444 rows (2%) name a real ticker absent from raw_bars, mostly index
    ETFs, money-market funds and dot-notation symbols like BRK.B.

    raw_bars SPY begins 2016-07-18 while filings begin 2013-04-16, so filings
    before mid-2016 have no benchmark and are excluded. The counts are printed;
    a coverage figure that is quietly a subset is worse than a low one.

MATURITY
    A 60-trading-day window cannot resolve until 60 bars have passed. Rows
    inside that window get NULL returns and matured=0 rather than a partial
    figure. The newest three months of filings are therefore absent from every
    return statistic, which is a property of the horizon, not a gap.

USAGE
    python analysis/congress_returns.py --horizon 60
    python analysis/congress_returns.py --status
"""
import argparse
import sqlite3
from bisect import bisect_right
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CDB = ROOT / "congress_trades.db"
PDB = ROOT / "prices.db"

DDL = """
CREATE TABLE IF NOT EXISTS congress_returns (
    ticker           TEXT NOT NULL,
    transaction_date TEXT NOT NULL,
    filed_at_date    TEXT NOT NULL,
    name             TEXT NOT NULL,
    txn_type         TEXT,
    horizon          INTEGER NOT NULL,
    entry_filed      REAL,   -- close after filed_at_date
    exit_filed       REAL,
    ret_filed        REAL,   -- the stock, unsigned
    spy_filed        REAL,   -- SPY over the SAME bars
    excess_filed     REAL,   -- ret_filed - spy_filed
    entry_txn        REAL,
    ret_txn          REAL,
    spy_txn          REAL,
    excess_txn       REAL,
    matured          INTEGER NOT NULL,
    PRIMARY KEY (ticker, transaction_date, filed_at_date, name, txn_type, horizon)
);
CREATE INDEX IF NOT EXISTS idx_cr_filed ON congress_returns(filed_at_date);
CREATE INDEX IF NOT EXISTS idx_cr_name  ON congress_returns(name);
"""


def load_bars(p, tickers):
    """{ticker: (sorted dates, closes)} from raw_bars. The column is `d`."""
    out = {}
    q = ",".join("?" * len(tickers))
    cur = p.execute(
        f"SELECT ticker, d, close FROM raw_bars WHERE ticker IN ({q}) "
        f"AND close IS NOT NULL AND close > 0 ORDER BY ticker, d", tickers)
    acc = defaultdict(lambda: ([], []))
    for tk, d, c in cur:
        acc[tk][0].append(d)
        acc[tk][1].append(float(c))
    return dict(acc)


def window(bars, anchor, H):
    """Entry at the first bar STRICTLY AFTER anchor; exit H bars later.

    Strictly after, not on: a disclosure filed on day D is not actionable at
    that day's close in any realistic sense, and using it would hand the book a
    price it could not have traded. Returns (entry, exit) or None when the
    window has not matured.
    """
    ds, cs = bars
    i = bisect_right(ds, anchor)
    if i >= len(ds) or i + H >= len(ds):
        return None
    return cs[i], cs[i + H]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizon", type=int, default=60,
                    help="trading days, not calendar days")
    ap.add_argument("--status", action="store_true")
    a = ap.parse_args()
    H = a.horizon

    con = sqlite3.connect(CDB, timeout=30)
    con.executescript(DDL)

    if a.status:
        for row in con.execute(
                "SELECT horizon, COUNT(*), SUM(matured), "
                "ROUND(AVG(CASE WHEN matured=1 THEN excess_filed END)*100,2) "
                "FROM congress_returns GROUP BY 1"):
            print(f"  h={row[0]}: {row[1]:,} rows, {row[2]:,} matured, "
                  f"mean excess {row[3]}pp")
        con.close()
        return

    trades = list(con.execute("""
        SELECT ticker, transaction_date, filed_at_date, name, txn_type
        FROM congress_trades
        WHERE ticker IS NOT NULL AND ticker <> '' AND filed_at_date <> ''
    """))
    print(f"{len(trades):,} disclosures with a ticker")

    p = sqlite3.connect(f"file:{PDB}?mode=ro", uri=True, timeout=30)
    want = sorted({t[0] for t in trades} | {"SPY"})
    print(f"loading bars for {len(want)} tickers ...")
    bars = load_bars(p, want)
    p.close()
    spy = bars.get("SPY")
    if not spy:
        print("SPY not in raw_bars -- cannot benchmark, aborting")
        return
    print(f"  {len(bars)} priced, SPY {spy[0][0]} .. {spy[0][-1]} "
          f"({len(spy[0]):,} bars)")

    rows, no_price, pre_spy, unmatured = [], 0, 0, 0
    spy_start = spy[0][0]
    for tk, td, fd, nm, tt in trades:
        b = bars.get(tk)
        if not b:
            no_price += 1
            continue
        if fd < spy_start:
            pre_spy += 1
            continue
        wf = window(b, fd, H)
        sf = window(spy, fd, H)
        if wf is None or sf is None:
            unmatured += 1
            rows.append((tk, td, fd, nm, tt, H, None, None, None, None, None,
                         None, None, None, None, 0))
            continue
        e, x = wf
        se, sx = sf
        rf = x / e - 1.0
        sr = sx / se - 1.0
        wt = window(b, td, H)
        st = window(spy, td, H)
        if wt and st:
            rt = wt[1] / wt[0] - 1.0
            srt = st[1] / st[0] - 1.0
        else:
            rt = srt = None
        rows.append((tk, td, fd, nm, tt, H, e, x, rf, sr, rf - sr,
                     wt[0] if wt else None, rt, srt,
                     (rt - srt) if rt is not None else None, 1))

    con.executemany(
        "INSERT OR REPLACE INTO congress_returns VALUES "
        "(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", rows)
    con.commit()

    print(f"\nstored {len(rows):,} rows at h={H}")
    print(f"  {no_price:,} skipped -- ticker not in raw_bars")
    print(f"  {pre_spy:,} skipped -- filed before SPY history begins "
          f"({spy_start})")
    print(f"  {unmatured:,} stored unmatured -- inside the {H}-bar window")

    print("\nEXCESS VS SPY, matured rows only, buys and sells SEPARATE")
    print(f"  {'':10}{'n':>7}{'raw':>9}{'SPY':>9}{'excess':>9}")
    for tt in ("Buy", "Sell"):
        r = con.execute("""
            SELECT COUNT(*), AVG(ret_filed)*100, AVG(spy_filed)*100,
                   AVG(excess_filed)*100
            FROM congress_returns WHERE matured=1 AND horizon=? AND txn_type=?
        """, (H, tt)).fetchone()
        if r[0]:
            print(f"  {tt:10}{r[0]:>7,}{r[1]:>+8.2f}%{r[2]:>+8.2f}%"
                  f"{r[3]:>+8.2f}pp")
    print("\n  For a BUY a positive excess is a good call. For a SELL a")
    print("  NEGATIVE excess is a good call -- they got out before it lagged.")
    print("  Pooling the two produces a number that means nothing.")

    g = con.execute("""
        SELECT AVG(excess_txn)*100, AVG(excess_filed)*100, COUNT(*)
        FROM congress_returns
        WHERE matured=1 AND horizon=? AND txn_type='Buy'
          AND excess_txn IS NOT NULL
    """, (H,)).fetchone()
    if g[2]:
        print(f"\nDISCLOSURE LAG COST, buys, {g[2]:,} rows")
        print(f"  from transaction date  {g[0]:+.2f}pp   what the member got")
        print(f"  from filing date       {g[1]:+.2f}pp   what you could get")
        print(f"  the lag costs          {g[0]-g[1]:+.2f}pp")
    con.close()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\ninterrupted.")
