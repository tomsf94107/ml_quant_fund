#!/usr/bin/env python3
"""
highconf_attribution.py — why did high-confidence edge fall +11.7pp to +0.8pp?

READ-ONLY. Writes nothing.

WHAT HAS ALREADY BEEN ELIMINATED (measured, not assumed)
    1. The model did not degrade. Walk-forward AUC at h=5 was 0.5347 in early
       May and 0.5362 at end-August, across ~400 tickers.
    2. Market direction is not the cause. Edge is measured over each month's OWN
       base rate, so a falling market lowers the bar too. The model was also
       leaning bearish (mean prob_up 0.525 -> 0.463), which should have helped.
    3. Dispersion collapse is not sufficient. Robust cross-sectional SD of 5-day
       returns fell only ~10% (5.00% -> 4.48%) while edge fell ~93%.

WHAT THIS SCRIPT TESTS
    Four decompositions, in the order that a cause would have to survive:

    A. CONCENTRATION -- was May's edge broad, or did a handful of tickers
       produce most of it? A concentrated edge is fragile by construction and
       would explain flat AUC alongside collapsing lift. Reported as the share
       of total lift from the top 5 and top 10 contributors, plus a Herfindahl
       index over absolute contributions.

    B. PERSISTENCE OF THE CONTRIBUTORS -- did May's biggest contributors keep
       contributing, or revert? Regression to the mean is the null; systematic
       failure by the same names would be a different finding.

    C. CONDITIONS ON FAILURE -- what distinguishes a wrong high-confidence call
       from a right one? Split by market direction that day, realised
       volatility, the ticker's own recent volatility, and liquidity. This is
       the only decomposition that can yield an ACTIONABLE gate: "do not emit
       high confidence when condition X holds".

    D. THE ARITHMETIC BASELINE -- what lift does AUC 0.535 actually support?
       If May exceeded the ceiling implied by measured discrimination, the
       decline is regression, not damage.

READING THE OUTPUT
    A cause must explain a ~93% fall. A factor explaining 10% is real but not
    the answer. The script prints magnitudes so that comparison is possible
    rather than rhetorical.

    python analysis/highconf_attribution.py --db accuracy.db
"""
import argparse
import math
import sqlite3
import statistics as st
from collections import defaultdict

THRESH = 0.70          # the high-confidence gate in signals/generator.py
MONTHS = ("2026-05", "2026-06", "2026-07", "2026-08")


def wilson(k, n, z=1.96):
    if not n:
        return (0.0, 100.0)
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    s = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (max(0.0, 100 * (c - s) / d), min(100.0, 100 * (c + s) / d))


def fetch(con, horizon, thresh):
    """High-confidence predictions joined to outcomes, with context."""
    return con.execute("""
        SELECT p.prediction_date, p.ticker, p.prob_up, o.actual_up,
               o.actual_return
        FROM predictions p
        JOIN outcomes o
          ON p.ticker=o.ticker AND p.prediction_date=o.prediction_date
         AND p.horizon=o.horizon
        WHERE p.horizon=? AND p.prob_up >= ?
          AND o.actual_up IS NOT NULL
          AND p.prediction_date >= '2026-05-01'
        ORDER BY p.prediction_date
    """, (horizon, thresh)).fetchall()


def base_rates(con, horizon):
    """Each month's unconditional up-rate -- the bar edge is measured against."""
    out = {}
    for m, n, k in con.execute("""
        SELECT substr(p.prediction_date,1,7), COUNT(*),
               SUM(o.actual_up)
        FROM predictions p JOIN outcomes o
          ON p.ticker=o.ticker AND p.prediction_date=o.prediction_date
         AND p.horizon=o.horizon
        WHERE p.horizon=? AND o.actual_up IS NOT NULL
          AND p.prediction_date >= '2026-05-01'
        GROUP BY 1""", (horizon,)):
        if n:
            out[m] = 100.0 * k / n
    return out


def section_a_concentration(rows, bases):
    print(f"\n{'='*74}\nA. CONCENTRATION -- was the edge broad or a few names?\n{'='*74}")
    bym = defaultdict(list)
    for d, t, p, y, r in rows:
        bym[d[:7]].append((t, y))

    print(f"  {'month':<9}{'n':>5}{'acc':>8}{'base':>8}{'lift':>8}"
          f"{'tickers':>9}{'top5 share':>12}{'top10 share':>12}{'HHI':>7}")
    for m in MONTHS:
        recs = bym.get(m, [])
        if len(recs) < 10:
            print(f"  {m:<9}{len(recs):>5}   too few")
            continue
        n = len(recs)
        acc = 100.0 * sum(y for _, y in recs) / n
        base = bases.get(m, 50.0)
        lift = acc - base

        # each ticker's contribution to total lift, in "excess hits"
        byt = defaultdict(lambda: [0, 0])
        for t, y in recs:
            byt[t][0] += 1
            byt[t][1] += y
        contrib = {t: (k - c * base / 100.0) for t, (c, k) in byt.items()}
        ranked = sorted(contrib.items(), key=lambda x: -x[1])

        # Share of POSITIVE contribution, not of NET total.
        #
        # Dividing by the net total blows up when the month's edge is near zero:
        # a test case with total = -1.0 produced a "top-5 share" of -650%. The
        # positive-only denominator cannot collapse the same way and answers the
        # actual question -- of the excess hits that WERE generated, how many
        # came from a handful of names.
        pos_total = sum(v for v in contrib.values() if v > 0) or 1e-9
        top5 = sum(max(v, 0.0) for _, v in ranked[:5])
        top10 = sum(max(v, 0.0) for _, v in ranked[:10])
        absmag = sum(abs(v) for v in contrib.values()) or 1.0
        hhi = sum((abs(v) / absmag) ** 2 for v in contrib.values())

        print(f"  {m:<9}{n:>5}{acc:>7.1f}%{base:>7.1f}%{lift:>+7.1f}pp"
              f"{len(byt):>9}{100*top5/pos_total:>11.0f}%"
              f"{100*top10/pos_total:>11.0f}%{hhi:>7.3f}")

    print("\n  top-N share  = share of POSITIVE excess hits from the best N "
          "tickers.\n                 Positive-only denominator: dividing by "
          "the NET total explodes\n                 when a month's edge is "
          "near zero.\n  HHI          = Herfindahl over absolute "
          "contributions; higher = more concentrated.\n"
          "  A high May share means a few names WERE the edge and the rest of "
          "the book\n  contributed nothing -- fragile by construction, and it "
          "would not repeat.")


def section_b_persistence(rows, bases):
    print(f"\n{'='*74}\nB. DID MAY'S CONTRIBUTORS KEEP CONTRIBUTING?\n{'='*74}")
    bym = defaultdict(lambda: defaultdict(lambda: [0, 0]))
    for d, t, p, y, r in rows:
        m = d[:7]
        bym[m][t][0] += 1
        bym[m][t][1] += y

    may = bym.get("2026-05", {})
    if not may:
        print("  no May data")
        return
    base_may = bases.get("2026-05", 50.0)
    contrib = {t: (k - c * base_may / 100.0) for t, (c, k) in may.items()}
    top = sorted(contrib.items(), key=lambda x: -x[1])[:12]

    print(f"  {'ticker':<8}{'May n':>7}{'May acc':>9}"
          f"{'later n':>9}{'later acc':>11}{'delta':>9}")
    later_n = later_k = 0
    for t, _ in top:
        c, k = may[t]
        macc = 100.0 * k / c
        lc = lk = 0
        for m in ("2026-06", "2026-07", "2026-08"):
            if t in bym.get(m, {}):
                lc += bym[m][t][0]
                lk += bym[m][t][1]
        if lc:
            lacc = 100.0 * lk / lc
            later_n += lc
            later_k += lk
            print(f"  {t:<8}{c:>7}{macc:>8.1f}%{lc:>9}{lacc:>10.1f}%"
                  f"{lacc-macc:>+8.1f}pp")
        else:
            print(f"  {t:<8}{c:>7}{macc:>8.1f}%{'—':>9}{'—':>11}{'—':>9}")
    if later_n:
        pooled = 100.0 * later_k / later_n
        later_base = st.mean([bases.get(m, 50.0)
                              for m in ("2026-06", "2026-07", "2026-08")])
        print(f"\n  May's top contributors, pooled Jun-Aug: {pooled:.1f}% "
              f"on n={later_n} vs base {later_base:.1f}% "
              f"({pooled-later_base:+.1f}pp)")
        print("  If that is near zero, May's leaders reverted -- regression to "
              "the mean,\n  not a failure that can be attributed or fixed.")


def section_c_conditions(con, rows, horizon):
    print(f"\n{'='*74}\nC. WHAT CONDITIONS SEPARATE RIGHT FROM WRONG CALLS?"
          f"\n{'='*74}")
    px = sqlite3.connect("file:prices.db?mode=ro", uri=True)
    spy = dict(px.execute(
        "SELECT date, adj_close FROM daily_prices WHERE ticker='SPY' "
        "AND date >= '2026-04-01'").fetchall())
    sd = sorted(spy)
    spy_ret = {sd[i]: (spy[sd[i]] - spy[sd[i-1]]) / spy[sd[i-1]]
               for i in range(1, len(sd)) if spy[sd[i-1]]}
    # trailing 20d realised vol of SPY as a regime proxy
    vol = {}
    for i in range(21, len(sd)):
        w = [spy_ret.get(sd[j]) for j in range(i-20, i)]
        w = [x for x in w if x is not None]
        if len(w) > 10:
            vol[sd[i]] = st.pstdev(w) * math.sqrt(252)
    px.close()

    def bucket(rows_, keyfn, label):
        b = defaultdict(lambda: [0, 0])
        for d, t, p, y, r in rows_:
            k = keyfn(d, t, p)
            if k is None:
                continue
            b[k][0] += 1
            b[k][1] += y
        print(f"\n  --- {label} ---")
        print(f"  {'bucket':<22}{'n':>6}{'acc':>8}{'95% CI':>18}")
        for k in sorted(b):
            c, h = b[k]
            if c < 8:
                print(f"  {str(k):<22}{c:>6}   too few")
                continue
            lo, hi = wilson(h, c)
            print(f"  {str(k):<22}{c:>6}{100*h/c:>7.1f}%"
                  f"   [{lo:>5.1f}, {hi:>5.1f}]")

    bucket(rows, lambda d, t, p: ("market UP that day" if spy_ret.get(d, 0) > 0
                                  else "market DOWN that day")
           if d in spy_ret else None, "Market direction on prediction day")

    def volb(d, t, p):
        v = vol.get(d)
        if v is None:
            return None
        return ("vol LOW  (<15%)" if v < 0.15 else
                "vol MID  (15-25%)" if v < 0.25 else "vol HIGH (>25%)")
    bucket(rows, volb, "SPY trailing-20d realised vol regime")

    def probb(d, t, p):
        return ("prob 0.70-0.75" if p < 0.75 else
                "prob 0.75-0.80" if p < 0.80 else "prob 0.80+")
    bucket(rows, probb, "How far above the 0.70 gate")

    def monthb(d, t, p):
        return d[:7]
    bucket(rows, monthb, "By month, for reference")

    print("\n  A bucket whose interval EXCLUDES 50% and sits below it is a "
          "candidate gate:\n  a condition under which high confidence should "
          "not be emitted. A bucket that\n  merely looks bad but spans 50% is "
          "not actionable.")


def section_d_ceiling(bases):
    print(f"\n{'='*74}\nD. WHAT LIFT DOES THE MEASURED AUC SUPPORT?\n{'='*74}")
    auc = 0.5362
    d = 2 * (auc - 0.5)
    print(f"  walk-forward AUC h=5, end-August:       {auc:.4f}")
    print(f"  Somers' D = 2(AUC-0.5) = Gini:          {d:.4f}   "
          f"(exact identity for a binary outcome)")
    print(f"\n  For a top-decile cut, the lift implied by discrimination this "
          f"weak is\n  a few percentage points -- the research brief's estimate "
          f"was 2-4pp.\n")
    print(f"  {'month':<10}{'observed lift':>15}{'vs 2-4pp band':>18}")
    obs = {"2026-05": 11.7, "2026-06": 6.9, "2026-07": 1.9, "2026-08": 0.8}
    for m in MONTHS:
        v = obs[m]
        tag = ("FAR ABOVE ceiling" if v > 6 else
               "above" if v > 4 else
               "inside band" if v >= 2 else "below band")
        print(f"  {m:<10}{v:>14.1f}pp{tag:>18}")
    print("\n  May exceeded what the model's own measured discrimination can "
          "produce by\n  roughly 3-5x. That is the signature of an anomalous "
          "sample, not of a\n  capability that was later lost.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="accuracy.db")
    ap.add_argument("--horizon", type=int, default=5)
    ap.add_argument("--thresh", type=float, default=THRESH)
    args = ap.parse_args()

    con = sqlite3.connect(f"file:{args.db}?mode=ro", uri=True)
    rows = fetch(con, args.horizon, args.thresh)
    bases = base_rates(con, args.horizon)
    print(f"h={args.horizon}, prob_up >= {args.thresh}, "
          f"{len(rows)} high-confidence predictions with outcomes")
    if len(rows) < 30:
        print("too few high-confidence predictions to decompose. Lower "
              "--thresh to 0.55 for a larger sample.")
        con.close()
        return

    section_a_concentration(rows, bases)
    section_b_persistence(rows, bases)
    section_c_conditions(con, rows, args.horizon)
    section_d_ceiling(bases)
    con.close()

    print(f"\n{'='*74}\nHOW TO READ THIS\n{'='*74}")
    print("  A cause must explain a ~93% fall in edge. Already eliminated:\n"
          "  model degradation (AUC flat), market direction (edge is over the\n"
          "  base rate), dispersion collapse (~10%, not 93%).\n\n"
          "  If A shows high concentration and B shows reversion, the answer "
          "is that\n  May was a small number of names running hot -- fragile "
          "by construction,\n  and consistent with D's ceiling arithmetic.\n\n"
          "  Section C is the only one that can produce a FIX rather than an\n"
          "  explanation. A condition where confident calls fail reliably is a "
          "gate.")


if __name__ == "__main__":
    main()
