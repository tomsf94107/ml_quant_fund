#!/usr/bin/env python3
"""
earnings_season_confound.py — is the pre-season drift about EARNINGS at all?

READ-ONLY. Follow-up to analysis/earnings_season_test.py.

WHAT THAT TEST FOUND
    The stated hypothesis was that the market falls as earnings season
    approaches, driven by banks selling to take profit. Measured on 40 quarters
    of SPY and XLF, anchored to the first big-six bank announcement each
    quarter, it came out the OTHER WAY:

        window      SPY mean   uncond   diff    negative
        [-10,-1]     +1.56%    +0.49%  +1.07pp    28%
        [ -5,-1]     +0.86%    +0.17%  +0.69pp    28%
        [-20,-1]     +0.82%    +0.99%  -0.17pp    42%

    XLF was stronger still at +1.74% against +0.39%. Nine of ten windows went
    against the hypothesis and none cleared the registered bar.

    That reproduces Lamont & Frazzini's earnings announcement premium -- prices
    rise in the week before announcements, 7-18% annualised, documented since
    1927 -- but at market level rather than stock level.

THE CONFOUND THIS SCRIPT EXISTS TO KILL
    Big banks report in the SECOND WEEK of January, April, July and October,
    every quarter, almost to the day. So "the 10 days before bank earnings" is
    very nearly "the first week and a half of the first month of a quarter".

    If the market simply drifts up in the first half of those months for
    unrelated reasons -- flows, rebalancing, turn-of-quarter effects -- then the
    +1.07pp has nothing to do with earnings and the anchor is decoration.

    THE TEST: re-run the identical window against a PLACEBO anchor -- the same
    calendar position each quarter, with no earnings content. If the placebo
    produces the same excess, the effect is seasonal, not informational.

    Three placebos, because one could coincide by accident:
      cal_10th   the 10th of Jan/Apr/Jul/Oct, near the real bank dates
      mid_q2     the middle month of each quarter, away from any season start
      shift_30   the real anchor pushed 30 calendar days later

ALSO TESTED
    Sample split. 2016-2020 against 2021-2026. A calendar effect that exists in
    one half and not the other is noise found by looking.

    Costs. The excess is measured gross. At 10bps round trip on SPY, a +1.07pp
    edge four times a year is +4.3pp annual gross and +3.9pp net -- but that
    assumes the whole window is capturable, which it is not.

USAGE
    python analysis/earnings_season_confound.py
"""
import sqlite3
import statistics as st
from bisect import bisect_left
from datetime import date, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BANKS = ("JPM", "GS", "BAC", "C", "WFC", "MS")
LO, HI = -10, -1          # the window that produced the effect


def bars(con, t):
    r = con.execute("SELECT d, close FROM raw_bars WHERE ticker=? AND close>0 "
                    "ORDER BY d", (t,)).fetchall()
    return [x[0] for x in r], [float(x[1]) for x in r]


def win(ds, cs, anchor, lo=LO, hi=HI):
    i = bisect_left(ds, anchor)
    if i >= len(ds):
        return None
    s, e = i + lo, i + hi
    if s < 0 or e >= len(ds) or s >= e:
        return None
    return cs[e] / cs[s] - 1.0


def summarise(label, rets, unc):
    if len(rets) < 8:
        print(f"  {label:16}{len(rets):>4}   too few")
        return None
    n = len(rets)
    m = st.mean(rets) * 100
    neg = sum(1 for r in rets if r < 0) / n
    # t against the unconditional mean, not against zero. "Beats zero" in a
    # market that rises is not a claim.
    sd = st.stdev(rets) * 100 if n > 1 else 0.0
    t = (m - unc) / (sd / n ** 0.5) if sd else 0.0
    print(f"  {label:16}{n:>4}{m:>+9.2f}{unc:>+9.2f}{m-unc:>+8.2f}"
          f"{100*neg:>6.0f}%{t:>7.2f}")
    return m - unc


def main():
    ec = sqlite3.connect(f"file:{ROOT/'earnings.db'}?mode=ro", uri=True)
    q = ",".join("?" * len(BANKS))
    rows = ec.execute(f"""
        SELECT substr(report_date,1,10), substr(announce_date,1,10)
        FROM earnings_surprises WHERE ticker IN ({q})
          AND announce_date IS NOT NULL AND report_date >= '2016-01-01'
    """, BANKS).fetchall()
    ec.close()
    byq = {}
    for rd, ad in rows:
        byq.setdefault(rd, []).append(ad)
    seasons = sorted((rd, min(a)) for rd, a in byq.items())

    pc = sqlite3.connect(f"file:{ROOT/'prices.db'}?mode=ro", uri=True)
    ds, cs = bars(pc, "SPY")
    pc.close()

    k = HI - LO
    allr = [cs[i + k] / cs[i] - 1.0 for i in range(0, len(cs) - k, 3)]
    unc = st.mean(allr) * 100
    print(f"SPY, window [{LO},{HI}] = {k} trading days. "
          f"Unconditional mean {unc:+.2f}%\n")
    print(f"  {'anchor':16}{'n':>4}{'mean %':>9}{'uncond':>9}{'diff':>8}"
          f"{'neg':>7}{'t':>7}")

    # ── the real anchor ──────────────────────────────────────────────────
    real = [r for _, ad in seasons if (r := win(ds, cs, ad)) is not None]
    real_diff = summarise("bank announce", real, unc)

    # ── placebo 1: the 10th of the season-start month ────────────────────
    cal = []
    for rd, _ in seasons:
        y, m, _ = (int(x) for x in rd.split("-"))
        m2 = m + 1 if m < 12 else 1
        y2 = y if m < 12 else y + 1
        r = win(ds, cs, f"{y2:04d}-{m2:02d}-10")
        if r is not None:
            cal.append(r)
    summarise("cal 10th", cal, unc)

    # ── placebo 2: middle month of the quarter, far from any season ──────
    mid = []
    for rd, _ in seasons:
        y, m, _ = (int(x) for x in rd.split("-"))
        m2 = (m + 2 - 1) % 12 + 1
        y2 = y + (1 if m + 2 > 12 else 0)
        r = win(ds, cs, f"{y2:04d}-{m2:02d}-15")
        if r is not None:
            mid.append(r)
    summarise("mid-quarter", mid, unc)

    # ── placebo 3: the real anchor, 30 days later ────────────────────────
    sh = []
    for _, ad in seasons:
        d0 = date.fromisoformat(ad) + timedelta(days=30)
        r = win(ds, cs, d0.isoformat())
        if r is not None:
            sh.append(r)
    summarise("anchor +30d", sh, unc)

    # ── sample split on the real anchor ──────────────────────────────────
    print()
    early = [r for rd, ad in seasons if rd < "2021-01-01"
             and (r := win(ds, cs, ad)) is not None]
    late = [r for rd, ad in seasons if rd >= "2021-01-01"
            and (r := win(ds, cs, ad)) is not None]
    summarise("real 2016-2020", early, unc)
    summarise("real 2021-2026", late, unc)

    print("\n  READING IT")
    print("  If the placebos show the same excess as the bank anchor, the")
    print("  effect is CALENDAR, not earnings -- big banks report in the second")
    print("  week of Jan/Apr/Jul/Oct, so the window is nearly 'the first week")
    print("  and a half of a quarter' and the anchor is decoration.")
    print("  If both halves of the sample disagree, it is noise found by")
    print("  looking. 40 observations is thin either way.")
    if real_diff:
        gross = real_diff * 4
        print(f"\n  If real: {real_diff:+.2f}pp per quarter = {gross:+.2f}pp/yr "
              f"gross, {gross - 0.4:+.2f}pp net at 10bps round trip -- and that")
        print("  assumes the whole window is capturable, which it is not.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\ninterrupted.")
