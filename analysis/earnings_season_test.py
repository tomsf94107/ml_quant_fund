#!/usr/bin/env python3
"""
earnings_season_test.py — does the market fall as earnings season approaches?

READ-ONLY. Tests a market-level seasonality claim against 40 quarters.

THE HYPOTHESIS, AS STATED
    "The market goes down as earnings days approach, especially for banks and
    institutions. They sell to take profit before reporting."

    Three parts, and they are not equally testable:
      1. The MARKET falls before earnings season      -> testable, this script
      2. BANKS specifically fall                      -> testable, this script
      3. Institutions are SELLING to cause it         -> partly, needs flow data
      4. They are CONTROLLING the market              -> not a hypothesis

WHAT THE LITERATURE SAYS, WHICH IS MOSTLY THE OPPOSITE
    Lamont & Frazzini document an EARNINGS ANNOUNCEMENT PREMIUM: stock prices
    RISE around scheduled announcement dates, 7-18% annualised excess, present
    since 1927, strongest in large caps, and explicitly "equity prices
    predictably rise in the week prior to announcements and gradually decline
    following."

    Linnainmaa & Zhang dissent, finding low returns BEFORE firms announce,
    attributed to managers guiding expectations down.

    Savor & Wilson find the premium is LOWER when more firms announce at once --
    which is exactly what earnings season is.

    All of that is STOCK-level. This claim is MARKET-level: the index falls as
    the season approaches. That is a different question and the cross-sectional
    literature does not answer it.

THE DISCIPLINE THIS SCRIPT ENFORCES
    The observation that prompted it was "the past 2-3 quarters". That is 2-3
    data points. There are ~40 quarters of SPY in raw_bars since 2016, and a
    calendar effect spotted in 3 and confirmed in 3 is not a finding.

    The bar is registered here in code, before the run:
        mean window return below the unconditional mean for the same window
        length, AND negative in at least 70% of quarters.
    Anything less is noise, and 40 observations is thin even when it passes.

EVENT DEFINITION
    Earnings season start = the earliest announce_date among the big six banks
    (JPM, GS, BAC, C, WFC, MS) for each fiscal quarter. Banks report first and
    their dates are the conventional start of the season.

    earnings.db.announce_date is genuine -- NVDA's quarter ending 07-31
    announced 08-26, AAPL's 06-30 announced 07-30, mean lag 29.4 days over
    19,573 rows. But some rows are wrong: the minimum lag is -5 days, which is
    impossible, and JPM shows a 2-day lag where mid-April is correct. Rows with
    a lag outside 5-75 days are dropped and the count is reported.

USAGE
    python analysis/earnings_season_test.py
    python analysis/earnings_season_test.py --proxy SPY --sector XLF
"""
import argparse
import sqlite3
import sys
from bisect import bisect_left
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BANKS = ("JPM", "GS", "BAC", "C", "WFC", "MS")
WINDOWS = [(-20, -1), (-10, -1), (-5, -1), (0, 5), (0, 10)]


def bars(con, ticker):
    rows = con.execute(
        "SELECT d, close FROM raw_bars WHERE ticker=? AND close>0 ORDER BY d",
        (ticker,)).fetchall()
    return [r[0] for r in rows], [float(r[1]) for r in rows]


def win_ret(ds, cs, anchor, a, b):
    """Return over [anchor+a, anchor+b] in TRADING days, or None."""
    i = bisect_left(ds, anchor)
    if i >= len(ds):
        return None
    s, e = i + a, i + b
    if s < 0 or e >= len(ds) or s >= e:
        return None
    return cs[e] / cs[s] - 1.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--proxy", default="SPY")
    ap.add_argument("--sector", default="XLF")
    a = ap.parse_args()

    ec = sqlite3.connect(f"file:{ROOT/'earnings.db'}?mode=ro", uri=True)
    q = ",".join("?" * len(BANKS))
    rows = ec.execute(f"""
        SELECT substr(report_date,1,10) rd, substr(announce_date,1,10) ad, ticker
        FROM earnings_surprises
        WHERE ticker IN ({q}) AND announce_date IS NOT NULL
          AND report_date >= '2016-01-01'
    """, BANKS).fetchall()
    ec.close()

    # Drop impossible lags. A bank announcing 2 days after quarter end, or
    # BEFORE it, is a data error rather than a fast filer.
    good, bad = {}, 0
    for rd, ad, tk in rows:
        try:
            lag = (sqlite3.connect(":memory:").execute(
                "SELECT julianday(?)-julianday(?)", (ad, rd)).fetchone()[0])
        except Exception:
            bad += 1
            continue
        if lag is None or not (5 <= lag <= 75):
            bad += 1
            continue
        good.setdefault(rd, []).append(ad)

    seasons = sorted((rd, min(ads)) for rd, ads in good.items())
    print(f"{len(seasons)} quarters with a usable bank announce date "
          f"({bad} rows dropped for an impossible lag)\n")
    if len(seasons) < 12:
        print("too few quarters to test")
        return

    pc = sqlite3.connect(f"file:{ROOT/'prices.db'}?mode=ro", uri=True)
    series = {t: bars(pc, t) for t in (a.proxy, a.sector)}
    pc.close()

    print(f"{'window':>12}  {'ticker':6}{'n':>5}{'mean %':>9}{'median':>9}"
          f"{'neg':>7}{'uncond %':>10}{'diff':>8}  bar")
    for lo, hi in WINDOWS:
        for t in (a.proxy, a.sector):
            ds, cs = series[t]
            if not ds:
                continue
            rets = [r for _, ad in seasons
                    if (r := win_ret(ds, cs, ad, lo, hi)) is not None]
            if len(rets) < 12:
                continue
            n = len(rets)
            mean = sum(rets) / n * 100
            med = sorted(rets)[n // 2] * 100
            neg = sum(1 for r in rets if r < 0) / n

            # UNCONDITIONAL baseline: the same window length measured from
            # every bar, so "down 1%" is judged against what that many days
            # normally does rather than against zero. Without this, any
            # negative number looks like a finding in a market that rises.
            k = hi - lo
            allr = [cs[i + k] / cs[i] - 1.0
                    for i in range(0, len(cs) - k, 3)]
            unc = sum(allr) / len(allr) * 100

            # BAR REGISTERED BEFORE THE RUN: below the unconditional mean AND
            # negative in at least 70% of quarters. 40 observations is thin
            # even when it clears; a pass here is a reason to look further,
            # not a signal.
            ok = (mean < unc) and (neg >= 0.70)
            print(f"  [{lo:>3},{hi:>3}]  {t:6}{n:>5}{mean:>+9.2f}{med:>+9.2f}"
                  f"{100*neg:>6.0f}%{unc:>+10.2f}{mean-unc:>+8.2f}  "
                  f"{'PASS' if ok else ''}")

    print("\n  'neg' is the share of quarters with a negative window return.")
    print("  'uncond' is the same window length measured from every third bar")
    print("  of the whole history -- the return that window normally earns.")
    print("  A negative mean in a rising market is not a finding on its own;")
    print("  the diff column is what matters.")
    print("\n  The literature runs the other way at STOCK level: Lamont &")
    print("  Frazzini find prices RISE in the week before an announcement,")
    print("  7-18% annualised, since 1927. This is a MARKET-level test of a")
    print("  different claim, on 40 observations.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\ninterrupted.")
