#!/usr/bin/env python3
"""
insider_breadth_test.py — is the NUMBER of insiders a signal, not the dollars?

READ-ONLY. Writes nothing. No model.

WHY THIS IS NOT A SEVENTH REPEAT
    Six insider constructions were tested on 2026-09-03 and all came back null:
    flow across five Form-4 types, trajectory (persistence, slope, breadth of
    dollars, cumulative), remaining overhang from a sharesOwnedFollowingTransaction
    backfill, buying conditioned on short-interest terciles, the Cohen-Malloy-
    Pomorski routine/opportunistic split, and a model-accuracy gate. The axis was
    closed.

    Two things reopen it.

    FIRST, EVERY ONE WAS RUN AT h=5. The horizon sweep on 2026-09-05 found this
    feature set carries 40-day information and essentially none at 5 -- XGBoost
    is BELOW random at h=3/5 (AUC 0.478) and 0.577 at h=40. And tonight's
    leave-one-out put the [insider] group at -0.433pp with both seeds agreeing,
    measured at h=40. So the family contributes something at the horizon it was
    never tested on.

    SECOND, ALL SIX MEASURED DOLLARS. Two equity-research reports written on the
    same data treat BREADTH as the actual signal:
      "~$417M, unmarked 10b5-1, no new filers -- yellow flag, not red: two
       insiders, below spot (not top-ticking), no escalation"
      "Director-sale follow-through -- a spread to MORE INSIDERS or continuation
       escalates the yellow flag; a first BUY would be the real (bullish) tell"
    Three filings from one director is a liquidity event. Three filings from
    three directors is a view. Dollar flow cannot tell them apart.

WHAT IS MEASURED
    Per ticker per date, over trailing windows:

      n_sellers      distinct insider_name with acquired_disposed='D'
      n_buyers       distinct insider_name with 'A' -- rarer and, per the
                     literature, more informative
      seller_breadth n_sellers scaled by that ticker's own trailing-year max,
                     so a 3-insider month means something different at a company
                     with 5 officers than at one with 40
      new_filers     sellers in this window who did NOT appear in the prior
                     window -- the escalation measure the reports use
      csuite_frac    share of sellers flagged is_csuite
      px_vs_spot     volume-weighted sale price over the close on the filing
                     date. Selling BELOW spot is not top-ticking; selling above
                     it is. The reports treat this as the tell.

    Each tested as a per-date cross-sectional IC against forward returns at
    h=5, h=20 and h=40, with Newey-West and a within-date shuffle null.

PIT DISCIPLINE
    Joined on FILING_DATE, never trade_date. The market learns when the Form 4
    is filed, not when the trade happened, and the gap is real -- an NVDA row
    shows trade 2026-03-20, filed 2026-03-24. Using trade_date would be a
    look-ahead of exactly the kind that voided the PEAD work (report_date was
    the fiscal period end, admitting the figure 14-30 days early, measured at
    IC +0.2612 t=+30).

THE BAR
    NW t > 3.0 per Harvey, Liu & Zhu, not 2.0 -- and this is a seventh look at
    an axis already closed six times, which is precisely the multiple-testing
    situation their hurdle exists for. A result between 2 and 3 here should be
    read as nothing.

    python analysis/insider_breadth_test.py
"""
import argparse
import datetime
import math
import os
import sqlite3
import statistics as st
import sys
import warnings
from collections import defaultdict

warnings.filterwarnings("ignore")

HORIZONS = (5, 20, 40)
T_HURDLE = 3.0


def nw_t(v, lag):
    n = len(v)
    if n < 10:
        return None
    m = sum(v) / n
    d = [x - m for x in v]
    var = sum(x * x for x in d) / n
    for k in range(1, min(lag, n - 1) + 1):
        gk = sum(d[i] * d[i - k] for i in range(k, n)) / n
        var += 2 * (1 - k / (lag + 1.0)) * gk
    return m / math.sqrt(var / n) if var > 0 else None


def spearman(x, y):
    n = len(x)
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
            for k in range(i, j + 1):
                r[o[k]] = a
            i = j + 1
        return r
    rx, ry = rank(x), rank(y)
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((rx[i] - mx) * (ry[i] - my) for i in range(n))
    dx = math.sqrt(sum((v - mx) ** 2 for v in rx))
    dy = math.sqrt(sum((v - my) ** 2 for v in ry))
    return num / (dx * dy) if dx and dy else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--window", type=int, default=90,
                    help="trailing calendar days for the breadth count")
    ap.add_argument("--min-names", type=int, default=25)
    ap.add_argument("--start", default="2019-06-01")
    ap.add_argument("--step", type=int, default=5,
                    help="sample every Nth session; breadth moves slowly")
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()

    for f in ("insider_trades.db", "prices.db"):
        if not os.path.exists(f):
            raise SystemExit(f"{f} not found -- run from the repo root")

    ic = sqlite3.connect("file:insider_trades.db?mode=ro", uri=True)
    rows = ic.execute(
        "SELECT ticker, filing_date, insider_name, acquired_disposed, "
        "price_per_share, notional_usd, is_csuite FROM insider_filings_raw "
        "WHERE filing_date >= ? AND insider_name IS NOT NULL",
        (args.start,)).fetchall()
    ic.close()
    byt = defaultdict(list)
    for t, fd, nm, ad, px, no, cs in rows:
        byt[str(t).upper()].append(
            (str(fd)[:10], nm.strip().upper(), ad, px, no, cs))
    for t in byt:
        # sort on filing_date ONLY. price_per_share and notional_usd are NULL
        # on some rows (the schema documents notional_usd as NULL when price is
        # unknown), and a plain tuple sort compares None against float.
        byt[t].sort(key=lambda x: x[0])
    print(f"insider filings: {len(rows):,} rows, {len(byt)} tickers, "
          f"from {args.start}")

    pc = sqlite3.connect("file:prices.db?mode=ro", uri=True)
    bars = defaultdict(list)
    for t, d, c in pc.execute(
            "SELECT ticker, d, close FROM raw_bars WHERE close > 0 AND d >= ?",
            (args.start,)):
        if t in byt:
            bars[t].append((str(d)[:10], float(c)))
    pc.close()
    for t in bars:
        bars[t].sort()
    have = [t for t in byt if len(bars.get(t, [])) > 300]
    print(f"prices: {len(have)} tickers with both sources\n")
    if len(have) < 50:
        raise SystemExit("too few tickers carry both sources")

    W = datetime.timedelta(days=args.window)
    panel = defaultdict(dict)     # metric -> date -> ticker -> value
    fwd = {h: defaultdict(dict) for h in HORIZONS}

    for t in have:
        b = bars[t]
        fil = byt[t]
        closes = {d: c for d, c in b}
        for i in range(60, len(b) - max(HORIZONS), args.step):
            d, c0 = b[i]
            dd = datetime.date.fromisoformat(d)
            lo = (dd - W).isoformat()
            plo = (dd - W - W).isoformat()
            cur = [x for x in fil if lo <= x[0] <= d]
            prv = [x for x in fil if plo <= x[0] < lo]
            if not cur:
                continue
            sell = {x[1] for x in cur if x[2] == "D"}
            buy = {x[1] for x in cur if x[2] == "A"}
            prev_sell = {x[1] for x in prv if x[2] == "D"}
            yr = [x for x in fil
                  if (dd - datetime.timedelta(days=365)).isoformat()
                  <= x[0] <= d]
            yr_max = max(
                (len({y[1] for y in yr if y[0] <= z[0] and y[2] == "D"})
                 for z in yr), default=0) or 1

            m = {
                "n_sellers": float(len(sell)),
                "n_buyers": float(len(buy)),
                "seller_breadth": len(sell) / yr_max,
                "new_filers": float(len(sell - prev_sell)),
                "csuite_frac": (sum(1 for x in cur if x[2] == "D" and x[5])
                                / max(sum(1 for x in cur if x[2] == "D"), 1)),
            }
            sv = [(x[3], x[4]) for x in cur
                  if x[2] == "D" and x[3] and x[4]]
            if sv and c0:
                wp = sum(p * n for p, n in sv) / sum(n for _, n in sv)
                m["px_vs_spot"] = wp / c0 - 1.0
            for k, v in m.items():
                panel[k].setdefault(d, {})[t] = v
            for h in HORIZONS:
                if i + h < len(b) and c0:
                    r = (b[i + h][1] - c0) / c0
                    if abs(r) < 1.5:
                        fwd[h].setdefault(d, {})[t] = r

    import random
    rnd = random.Random(args.seed)
    print(f"  {'metric':<16}{'h':>4}{'dates':>7}{'IC':>9}{'NW t':>8}"
          f"{'null':>9}")
    for metric in ("n_sellers", "n_buyers", "seller_breadth", "new_filers",
                   "csuite_frac", "px_vs_spot"):
        if metric not in panel:
            continue
        for h in HORIZONS:
            ics, nulls = [], []
            for d, mm in panel[metric].items():
                fr = fwd[h].get(d, {})
                names = [t for t in mm if t in fr]
                if len(names) < args.min_names:
                    continue
                xs = [mm[t] for t in names]
                ys = [fr[t] for t in names]
                if len(set(xs)) < 3:
                    continue
                r = spearman(xs, ys)
                if r is not None:
                    ics.append(r)
                sh = ys[:]
                rnd.shuffle(sh)
                rn = spearman(xs, sh)
                if rn is not None:
                    nulls.append(rn)
            if len(ics) < 20:
                print(f"  {metric:<16}{h:>4}{len(ics):>7}   too few dates")
                continue
            t_ = nw_t(ics, max(1, h // 5)) or 0.0
            print(f"  {metric:<16}{h:>4}{len(ics):>7}{st.mean(ics):>+9.4f}"
                  f"{t_:>+8.2f}{st.mean(nulls):>+9.4f}"
                  + ("   PASSES t>3" if abs(t_) > T_HURDLE else ""))
        print()

    print(f"  Bar is NW t > {T_HURDLE}, not 2.0. This is a SEVENTH look at an")
    print("  axis closed six times on 2026-09-03, which is exactly the")
    print("  multiple-testing situation Harvey, Liu & Zhu's hurdle exists for.")
    print("  Between 2 and 3 here means nothing.\n")
    print("  Joined on FILING_DATE, never trade_date -- the market learns when")
    print("  the Form 4 is filed. An NVDA row shows trade 2026-03-20, filed")
    print("  2026-03-24; using trade_date would repeat the PEAD look-ahead.\n")
    print("  What would be NEW: the six prior tests all measured DOLLARS. Three")
    print("  filings from one director is a liquidity event; three from three")
    print("  directors is a view. Only breadth separates them.")


if __name__ == "__main__":
    main()
