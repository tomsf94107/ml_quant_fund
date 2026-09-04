#!/usr/bin/env python3
"""
insider_timeseries_test.py — does the TRAJECTORY of insider selling predict?

READ-ONLY. Writes nothing.

WHAT THE PREVIOUS TEST GOT WRONG
    analysis/insider_construction_test.py tested five constructions and found
    nothing: every IC under 0.003, every |t| under 0.5, nulls the same size as
    the signals. That test closed the axis. Two objections to it are correct:

    1. EVERY CONSTRUCTION WAS A WINDOW, NOT A TRAJECTORY. Rolling 7/21/60/90-day
       sums, an acceleration ratio, a per-ADV normalisation -- all of them
       collapse the history into one number about the recent past. None can
       express WHERE IN A DISTRIBUTION a name currently sits. A fund in month 2
       of a fourteen-month unwind and a fund that finished last week can produce
       the same 90-day sum.

    2. ONLY h=5 WAS TESTED. The mechanism -- a large holder distributing a
       position -- operates over months. Cohen, Malloy & Pomorski (JF 2012)
       measured 82bps at a MONTHLY horizon. Testing a months-long mechanism at
       one week and concluding it does not exist is a horizon error.

    This script fixes both.

WHAT OVERHANG ACTUALLY IS, AND WHY IT IS NOT TESTED HERE
    The literature definition is a STOCK variable: "a large block of securities
    that the market knows or suspects will be sold in the near future". What is
    LEFT to sell, not what was already sold. Those are opposite in the case that
    matters -- a seller who just finished looks identical, on any flow measure,
    to one who is 10% through.

    Computing it requires sharesOwnedFollowingTransaction from each Form 4.
    insider_filings_raw does NOT store that field (verified 2026-09-03 against
    the schema), so true overhang needs an EDGAR backfill and is out of scope
    here. This script tests the trajectory features that ARE computable now.

CONSTRUCTIONS
    cum_sold_adv       cumulative net shares disposed since 2021, divided by
                       20-day ADV. A trajectory: it only grows, so it encodes
                       how far through a distribution a name is rather than
                       what happened lately.
    sell_persistence   consecutive 21-day blocks with net selling. This is the
                       "continuous selling" idea directly: not how much, but
                       for how long without interruption.
    sell_slope         change in the 90-day sell rate over the last 90 days,
                       i.e. is the pace building or fading.
    recent_share       fraction of the trailing YEAR's selling that happened in
                       the last 90 days. Concentration -- a distribution that
                       is front-loaded versus one winding down.
    sell_breadth       count of distinct insider names selling in 90 days.
                       Many sellers at once is a different event from one
                       insider's scheduled plan.

HORIZONS
    h = 5, 20, 60 trading days. If the mechanism is months-long, h=60 is where
    it should appear and h=5 is where it should not.

METHOD -- unchanged from the previous test, and non-negotiable
    Keyed on filing_date, not trade_date: the market cannot know a transaction
    until it is filed, and Form 4 allows two business days.
    Per-date Spearman IC, then Newey-West at the horizon lag on the IC series.
    Pooled stock-date rows are not independent and have inflated t-statistics
    10-20x in this project before.
    A shuffle null on every construction at every horizon, reported alongside.

    python analysis/insider_timeseries_test.py
"""
import argparse
import math
import random
import sqlite3
import statistics as st
from collections import defaultdict
from datetime import date, timedelta


def spearman(pairs):
    n = len(pairs)
    if n < 8:
        return None
    def rank(v):
        o = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(v):
            j = i
            while j + 1 < len(v) and v[o[j + 1]] == v[o[i]]:
                j += 1
            a = (i + j) / 2.0 + 1
            for m in range(i, j + 1):
                r[o[m]] = a
            i = j + 1
        return r
    rx = rank([p[0] for p in pairs]); ry = rank([p[1] for p in pairs])
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((rx[i] - mx) * (ry[i] - my) for i in range(n))
    dx = math.sqrt(sum((r - mx) ** 2 for r in rx))
    dy = math.sqrt(sum((r - my) ** 2 for r in ry))
    return num / (dx * dy) if dx and dy else None


def nw_t(series, lag):
    n = len(series)
    if n < 10:
        return None
    m = sum(series) / n
    d = [x - m for x in series]
    var = sum(x * x for x in d) / n
    for k in range(1, min(lag, n - 1) + 1):
        gk = sum(d[i] * d[i - k] for i in range(k, n)) / n
        var += 2 * (1 - k / (lag + 1.0)) * gk
    return m / math.sqrt(var / n) if var > 0 else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--insider-db", default="insider_trades.db")
    ap.add_argument("--prices-db", default="prices.db")
    ap.add_argument("--start", default="2021-01-01")
    ap.add_argument("--min-names", type=int, default=15)
    args = ap.parse_args()

    px = sqlite3.connect(f"file:{args.prices_db}?mode=ro", uri=True)
    close = defaultdict(dict)
    volume = defaultdict(dict)
    for t, d, c, v in px.execute(
            "SELECT ticker, d, close, volume FROM raw_bars WHERE d >= ?",
            (args.start,)):
        if c:
            close[t][d] = c
        if v:
            volume[t][d] = v
    px.close()

    HORIZONS = (5, 20, 60)
    fwd = {h: {} for h in HORIZONS}
    adv = {}
    for t, s in close.items():
        ds = sorted(s)
        for h in HORIZONS:
            for i in range(len(ds) - h):
                a, b = s[ds[i]], s[ds[i + h]]
                if a and b and abs((b - a) / a) < 0.8:
                    fwd[h][(t, ds[i])] = (b - a) / a
        vs = volume.get(t, {})
        vd = sorted(vs)
        for i in range(20, len(vd)):
            w = [vs[vd[j]] for j in range(i - 20, i)]
            m = sum(w) / len(w)
            if m > 0:
                adv[(t, vd[i])] = m

    ic_db = sqlite3.connect(f"file:{args.insider_db}?mode=ro", uri=True)
    rows = ic_db.execute("""
        SELECT ticker, filing_date, transaction_code, shares, insider_name,
               acquired_disposed
        FROM insider_filings_raw WHERE filing_date >= ? AND shares IS NOT NULL
    """, (args.start,)).fetchall()
    ic_db.close()
    print(f"{len(rows):,} filings, {len({r[0] for r in rows})} tickers, "
          f"from {args.start}")

    # per (ticker, filing_date): net disposed shares, and the set of sellers
    day_net = defaultdict(float)
    day_sellers = defaultdict(set)
    for t, fd, code, sh, name, ad in rows:
        fd = str(fd)[:10]
        s = abs(sh or 0)
        if code == "S" or (ad == "D" and code in ("S", "F")):
            day_net[(t, fd)] += s
            if code == "S":
                day_sellers[(t, fd)].add(name or "?")
        elif code == "P":
            day_net[(t, fd)] -= s

    def wsum(t, d, days):
        d0 = date.fromisoformat(d)
        return sum(day_net.get((t, (d0 - timedelta(days=k)).isoformat()), 0.0)
                   for k in range(days))

    def wsellers(t, d, days):
        d0 = date.fromisoformat(d)
        s = set()
        for k in range(days):
            s |= day_sellers.get((t, (d0 - timedelta(days=k)).isoformat()),
                                 set())
        return len(s)

    # cumulative disposal per ticker, as a running trajectory
    cum = defaultdict(dict)
    for t in close:
        run = 0.0
        for d in sorted(close[t]):
            run += day_net.get((t, d), 0.0)
            cum[t][d] = run

    dates = sorted({d for t in close for d in close[t] if d >= args.start})[::5]
    names = ["cum_sold_adv", "sell_persistence", "sell_slope",
             "recent_share", "sell_breadth"]
    ics = {(k, h): [] for k in names for h in HORIZONS}
    nulls = {(k, h): [] for k in names for h in HORIZONS}
    rnd = random.Random(11)

    for d in dates:
        vals = {}
        for t in close:
            if d not in close[t]:
                continue
            a = adv.get((t, d))
            s90 = wsum(t, d, 90)
            s365 = wsum(t, d, 365)
            s180_90 = wsum(t, d, 180) - s90
            if s365 <= 0 and s90 <= 0:
                continue
            v = {}
            if a:
                v["cum_sold_adv"] = -cum[t].get(d, 0.0) / a
            # consecutive 21-day blocks with net selling, up to 12
            p = 0
            for blk in range(12):
                d0 = date.fromisoformat(d) - timedelta(days=21 * blk)
                if wsum(t, d0.isoformat(), 21) > 0:
                    p += 1
                else:
                    break
            v["sell_persistence"] = -float(p)
            v["sell_slope"] = -(s90 - s180_90)
            if s365 > 0:
                v["recent_share"] = -(s90 / s365)
            v["sell_breadth"] = -float(wsellers(t, d, 90))
            vals[t] = v

        for k in names:
            for h in HORIZONS:
                obs = [(vals[t][k], fwd[h][(t, d)]) for t in vals
                       if k in vals[t] and (t, d) in fwd[h]]
                if len(obs) < args.min_names:
                    continue
                r = spearman(obs)
                if r is not None:
                    ics[(k, h)].append(r)
                ys = [o[1] for o in obs]
                rnd.shuffle(ys)
                rn = spearman([(obs[i][0], ys[i]) for i in range(len(ys))])
                if rn is not None:
                    nulls[(k, h)].append(rn)

    print(f"\n  {'construction':<18}{'h':>4}{'dates':>7}{'mean IC':>10}"
          f"{'NW t':>8}{'null t':>9}")
    for k in names:
        for h in HORIZONS:
            v = ics[(k, h)]
            if len(v) < 20:
                print(f"  {k:<18}{h:>4}{len(v):>7}   too few dates")
                continue
            t_ = nw_t(v, h) or 0.0
            nt = nw_t(nulls[(k, h)], h) or 0.0
            flag = "  <<<" if abs(t_) >= 2.5 and abs(nt) < 1.5 else ""
            print(f"  {k:<18}{h:>4}{len(v):>7}{st.mean(v):>+10.4f}"
                  f"{t_:>+8.2f}{nt:>+9.2f}{flag}")
        print()

    print("  Sign: negated so MORE/LONGER selling is MORE NEGATIVE. A POSITIVE "
          "IC means\n  heavier or longer distribution predicts LOWER forward "
          "returns.\n")
    print("  Null t must be near zero. A construction whose null is also large "
          "is void.\n")
    print("  If h=60 shows a signal where h=5 does not, the mechanism is real "
          "but too\n  slow for this model's horizon -- which is a finding "
          "about the MODEL, not\n  about insider selling. It would argue for a "
          "longer-horizon product, or\n  for using it as a position veto "
          "rather than a return predictor.")


if __name__ == "__main__":
    main()
