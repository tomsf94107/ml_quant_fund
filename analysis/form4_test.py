#!/usr/bin/env python3
"""
form4_test.py — do open-market insider purchases predict anything?

READ-ONLY. Benchmarked against SPY over identical bars.

WHY THIS RUNS BEFORE THE 13F TESTS
    Three sources, three clocks:
        Form 4     ~1 day lag, 383,355 transaction-level rows, NAMED person
        dark pool   same day,  8.75M prints, ANONYMOUS
        13F        45 days,    quarterly snapshot, no transactions at all

    Form 4 is the only one that is both fast AND attributed. It is also the
    only one where the literature runs in favour rather than against: the
    13F prior is Fama-French, who find aggregate active managers have zero
    gross alpha and that even the top 3% can expect zero. A fund manager
    allocating clients' money at a 45-day lag and an officer buying their own
    company with their own after-tax money are not the same claim.

THE 2% THAT MATTERS
    transaction_code splits 383,355 rows as:
        S  124,678   sales -- mostly scheduled 10b5-1, not a view
        A  106,390   grants -- compensation, not a purchase
        M   60,700   option exercises
        F   59,412   shares withheld for tax
        G    8,060   gifts
        P    8,009   OPEN-MARKET PURCHASES  <-- the only discretionary buy
        C    6,331   conversions
        J    5,860   other

    Treating these alike would read tax withholding as insider selling and
    option exercises as accumulation. Only P is someone choosing to spend
    their own money. 1,606 of the 8,009 are C-suite.

SAMPLE WINDOW
    P buys by year: 9 (2016), 16, 38, then 575 (2019), 991, 965, 1313, 1437,
    926, 1146, 572 (2026 partial). The pre-2019 numbers are coverage starting,
    not insiders not buying, so the window is 2019-01-01 onward: ~7,900 events
    across 323 tickers.

ENTRY AT THE FILING DATE, NOT THE TRADE DATE
    The trade is private until the Form 4 posts. Measured lag on P rows is
    about one day -- TPL filed 2026-07-17 for a 07-16 trade -- but entry is
    taken at the close AFTER filing_date regardless, because that is the first
    price a reader of the filing could transact at. Using trade_date would
    hand the test a price nobody could have paid.

USAGE
    python analysis/form4_test.py
    python analysis/form4_test.py --horizon 60
"""
import argparse
import sqlite3
import statistics as st
from bisect import bisect_right
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
INS, PX = ROOT / "insider_trades.db", ROOT / "prices.db"
HORIZONS = (20, 60, 120)


def bars(con, tickers):
    q = ",".join("?" * len(tickers))
    acc = defaultdict(lambda: ([], []))
    for tk, d, c in con.execute(
            f"SELECT ticker, d, close FROM raw_bars WHERE ticker IN ({q}) "
            f"AND close > 0 ORDER BY ticker, d", tickers):
        acc[tk][0].append(d)
        acc[tk][1].append(float(c))
    return dict(acc)


def fwd(b, anchor, h):
    """Return over h TRADING days from the first bar strictly after anchor."""
    ds, cs = b
    i = bisect_right(ds, anchor)
    if i >= len(ds) or i + h >= len(ds):
        return None
    return cs[i + h] / cs[i] - 1.0


def summarise(label, rows, h):
    """rows: list of (excess, year). Prints mean, hit rate and both halves."""
    if len(rows) < 30:
        print(f"  {label:26}{len(rows):>6}   too few")
        return
    ex = [r[0] for r in rows]
    n = len(ex)
    m = st.mean(ex) * 100
    pos = sum(1 for x in ex if x > 0) / n
    sd = st.stdev(ex) * 100 if n > 1 else 0.0
    t = m / (sd / n ** 0.5) if sd else 0.0
    early = [r[0] for r in rows if r[1] <= 2022]
    late = [r[0] for r in rows if r[1] > 2022]
    e = st.mean(early) * 100 if len(early) > 20 else float("nan")
    l = st.mean(late) * 100 if len(late) > 20 else float("nan")
    # BAR, fixed before the run: excess above 1pp, positive in at least 55% of
    # events, and the SIGN holding in both halves. A result living in one half
    # is noise found by looking -- the earnings-season hypothesis died exactly
    # that way, +1.60pp in 2016-2020 against +0.63pp after.
    ok = m > 1.0 and pos >= 0.55 and (e > 0) == (l > 0) and e == e and l == l
    print(f"  {label:26}{n:>6}{m:>+9.2f}{100*pos:>7.0f}%{t:>7.2f}"
          f"{e:>+9.2f}{l:>+9.2f}  {'PASS' if ok else ''}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2019-01-01")
    a = ap.parse_args()

    ic = sqlite3.connect(f"file:{INS}?mode=ro", uri=True)
    rows = ic.execute("""
        SELECT ticker, filing_date, trade_date, shares, price_per_share,
               is_csuite, insider_name, notional_usd, insider_title
        FROM insider_filings_raw
        WHERE transaction_code = 'P'
          AND trade_date >= ? AND filing_date >= trade_date
          AND ticker IS NOT NULL AND shares > 0
          -- PRICE SANITY. 103 of 8,009 P rows carry a price_per_share above
          -- 10,000, and on 4 of them price_per_share EQUALS shares -- the
          -- parser wrote the share count into both columns. AIG shows
          -- 50,000,000 shares at "50,000,000 per share" for a notional of
          -- 2.5e15. Every one of those sorts to the top of a notional
          -- ranking, so the >= $250k cut was built on them. Only BRK.A
          -- trades above $10,000, and it is not in this data.
          AND price_per_share > 0 AND price_per_share < 10000
          AND price_per_share <> shares
        ORDER BY filing_date
    """, (a.start,)).fetchall()
    ic.close()
    print(f"{len(rows):,} open-market purchases (code P) since {a.start}\n")

    pc = sqlite3.connect(f"file:{PX}?mode=ro", uri=True)
    want = sorted({r[0] for r in rows} | {"SPY"})
    B = bars(pc, want)
    pc.close()
    spy = B.get("SPY")
    if not spy:
        print("SPY missing from raw_bars"); return
    priced = sum(1 for t in want if t in B)
    print(f"{priced} of {len(want)} tickers priced\n")

    # CLUSTERS. Two or more DIFFERENT insiders buying the same name inside 30
    # days is a different object from one person topping up -- it is the only
    # cut where independent people reach the same conclusion.
    byname = defaultdict(list)
    ten = {}
    for tk, fd, td, sh, px, cs, who, notional, title in rows:
        byname[tk].append((fd, who))
        ten[(tk, fd, who)] = "10%" in (title or "") or "10 %" in (title or "")
    cluster = set()
    for tk, evs in byname.items():
        evs.sort()
        for i, (fd, who) in enumerate(evs):
            others = {w for f, w in evs
                      if w != who and abs((int(f[:4]) * 372 + int(f[5:7]) * 31
                                           + int(f[8:10]))
                                          - (int(fd[:4]) * 372 + int(fd[5:7]) * 31
                                             + int(fd[8:10]))) <= 30}
            if others:
                cluster.add((tk, fd, who))

    for h in HORIZONS:
        print(f"h={h} trading days from the close AFTER filing_date")
        print(f"  {'cut':26}{'n':>6}{'excess%':>9}{'pos':>8}{'t':>7}"
              f"{'≤2022':>9}{'>2022':>9}")
        # 10% OWNERS ARE A DIFFERENT ACTOR. 1,615 of 8,009 P rows come from
        # holders of more than 10% of a class -- institutions, not company
        # officers. Berkshire files here: 153 P buys across 2 tickers,
        # including Occidental at $55.78 and $57.38, disclosed at a TWO-DAY
        # lag instead of the 45 days a 13F would take. A fund adding to a
        # >10% stake and a director buying a few thousand shares are not the
        # same claim, and the first test pooled them. Registered as a new cut
        # with the same bar, not a retune of a failed one.
        allr, csr, clr, bigr, tenr, offr = [], [], [], [], [], []
        for tk, fd, td, sh, px, cs, who, notional, title in rows:
            b = B.get(tk)
            if not b:
                continue
            r = fwd(b, fd, h)
            s = fwd(spy, fd, h)
            if r is None or s is None:
                continue
            ex = r - s
            yr = int(fd[:4])
            allr.append((ex, yr))
            if cs:
                csr.append((ex, yr))
            if (tk, fd, who) in cluster:
                clr.append((ex, yr))
            if (notional or 0) >= 250_000:
                bigr.append((ex, yr))
            if ten.get((tk, fd, who)):
                tenr.append((ex, yr))
            else:
                offr.append((ex, yr))
        summarise("all P buys", allr, h)
        summarise("C-suite only", csr, h)
        summarise("cluster, 2+ insiders 30d", clr, h)
        summarise("notional >= $250k", bigr, h)
        summarise("10% owners only", tenr, h)
        summarise("officers/directors only", offr, h)
        print()

    print("  BAR, fixed before the run: excess > 1pp, positive in >= 55% of")
    print("  events, and the SIGN holding in both halves. A result living in")
    print("  one half is noise found by looking.")
    print()
    print("  Entry is the close AFTER filing_date, never trade_date -- the")
    print("  trade is private until the form posts, and using the trade price")
    print("  would hand the test a fill nobody could have got.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\ninterrupted.")
