#!/usr/bin/env python3
"""
pct7_deployment_check.py — is PCT7 tradeable at $105k, or only true?

READ-ONLY. Writes nothing.

WHY
    PCT7 passed the gauntlet on 2026-09-05: at threshold 0.30, selected names
    returned +1.46% against a universe mean of +0.34%, per-date NW t = +3.34
    with a clean null, positive every month, and it beat its own session's base
    rate on 48 of 65 days. That is a validated SIGNAL.

    Validated is not deployable. The gauntlet says nothing about whether the
    names can be bought at this size, how many positions run at once, what the
    worst stretch looked like, or whether PCT7 picks the same stocks the SI
    brick already holds -- in which case it is not a second brick at all.

    Everything below is computed from the 65 dates of shadow data already
    logged. No new collection.

WHAT IS CHECKED

  1. CONCENTRATION
     How many distinct tickers and sectors carry the selections, and what share
     the top names take. A signal whose result rests on five tickers is a bet on
     those five.

  2. LIQUIDITY AT $105k
     Dollar ADV of each selected name against a realistic position. With ~4
     names a day at 0.30 and 5-day holds, roughly 20 positions overlap, so a
     position is about $5,250. The test is what fraction of a name's daily
     dollar volume that represents -- above roughly 1% of ADV, market impact
     starts to matter and the backtest's returns are optimistic.

  3. CONCURRENT POSITIONS
     5-day holds mean today's selections overlap the previous four days'. The
     maximum concurrent count sets the real capital requirement, and it is
     larger than the daily count.

  4. DRAWDOWN PATH
     +1.46% mean per selection says nothing about the path. Equal-weighted
     daily cohort returns, cumulated, with the worst peak-to-trough. A signal
     that earns its average through one month and bleeds the rest is
     unholdable at this size.

  5. OVERLAP WITH THE SI BRICK
     The SI brick is long low days-to-cover. If PCT7 selects the same names,
     the two are one position, not two bricks, and the diversification assumed
     in sizing does not exist.

  6. TURNOVER
     How often the selection set changes. High turnover at 5-day holds means
     costs compound faster than the 20bps ladder in the gauntlet assumed.

    python analysis/pct7_deployment_check.py
    python analysis/pct7_deployment_check.py --thresh 0.25 --capital 105000
"""
import argparse
import sqlite3
import statistics as st
from collections import Counter, defaultdict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="accuracy.db")
    ap.add_argument("--prices-db", default="prices.db")
    ap.add_argument("--si-db", default="short_interest.db")
    ap.add_argument("--thresh", type=float, default=0.30)
    ap.add_argument("--capital", type=float, default=105000)
    ap.add_argument("--hold", type=int, default=5)
    args = ap.parse_args()

    con = sqlite3.connect(f"file:{args.db}?mode=ro", uri=True)
    rows = con.execute("""
        SELECT p.prediction_date, p.ticker, p.prob_pct7, o.actual_return
        FROM predictions p JOIN outcomes o ON p.ticker=o.ticker
          AND p.prediction_date=o.prediction_date AND p.horizon=o.horizon
        WHERE p.horizon=5 AND p.prob_pct7 >= ? AND o.actual_return IS NOT NULL
        ORDER BY p.prediction_date
    """, (args.thresh,)).fetchall()
    con.close()
    if len(rows) < 50:
        print(f"only {len(rows)} selections at threshold {args.thresh}")
        return

    dates = sorted({r[0] for r in rows})
    byd = defaultdict(list)
    for d, t, p, r in rows:
        byd[d].append((t, r))
    print(f"PCT7 deployment check — threshold {args.thresh}, "
          f"capital ${args.capital:,.0f}, {args.hold}-day hold")
    print(f"{len(rows)} selections over {len(dates)} dates, "
          f"{len({r[1] for r in rows})} tickers")
    print(f"  {st.mean(len(v) for v in byd.values()):.1f} names/day "
          f"(min {min(len(v) for v in byd.values())}, "
          f"max {max(len(v) for v in byd.values())})\n")

    # ---- 1. concentration ----
    print("1. CONCENTRATION")
    c = Counter(r[1] for r in rows)
    top = c.most_common(10)
    share5 = 100.0 * sum(n for _, n in c.most_common(5)) / len(rows)
    share10 = 100.0 * sum(n for _, n in top) / len(rows)
    print(f"   top 5 tickers = {share5:.0f}% of selections, "
          f"top 10 = {share10:.0f}%")
    print("   " + ", ".join(f"{t}({n})" for t, n in top))
    if share5 > 40:
        print("   !! heavily concentrated -- the result rests on a few names")
    print()

    # ---- 3. concurrent positions ----
    print(f"3. CONCURRENT POSITIONS ({args.hold}-day holds overlap)")
    conc = []
    for i, d in enumerate(dates):
        window = dates[max(0, i - args.hold + 1):i + 1]
        live = set()
        for w in window:
            live |= {t for t, _ in byd[w]}
        conc.append(len(live))
    print(f"   mean {st.mean(conc):.1f}   median {st.median(conc)}   "
          f"max {max(conc)}")
    pos = args.capital / max(st.mean(conc), 1)
    print(f"   -> ~${pos:,.0f} per position at mean concurrency")
    print(f"      ${args.capital / max(conc):,.0f} at peak concurrency\n")

    # ---- 2. liquidity ----
    px = sqlite3.connect(f"file:{args.prices_db}?mode=ro", uri=True)
    dadv = {}
    cl, vo = defaultdict(dict), defaultdict(dict)
    for t, d, c_, v in px.execute(
            "SELECT ticker, d, close, volume FROM raw_bars "
            "WHERE d >= '2026-04-01'"):
        if c_:
            cl[t][d] = c_
        if v:
            vo[t][d] = v
    px.close()
    for t in cl:
        vd = sorted(vo.get(t, {}))
        for i in range(20, len(vd)):
            w = [vo[t][vd[j]] * cl[t].get(vd[j], 0) for j in range(i - 20, i)]
            m = sum(w) / len(w)
            if m > 0:
                dadv[(t, vd[i])] = m

    print(f"2. LIQUIDITY — position as a share of 20-day dollar ADV")
    fracs = []
    thin = []
    for d, t, p, r in rows:
        a = dadv.get((t, d))
        if a:
            f = 100.0 * pos / a
            fracs.append(f)
            if f > 1.0:
                thin.append((t, d, f, a))
    if fracs:
        s = sorted(fracs)
        print(f"   median {s[len(s)//2]:.3f}%   p90 {s[9*len(s)//10]:.3f}%   "
              f"max {max(s):.2f}%")
        print(f"   above 1% of ADV: {len(thin)} of {len(fracs)} selections")
        if thin:
            worst = sorted(thin, key=lambda x: -x[2])[:5]
            for t, d, f, a in worst:
                print(f"     {t:<6} {d}  {f:.2f}% of ${a/1e6:.1f}M ADV")
        print("   Above ~1% of ADV market impact starts to bite and the")
        print("   backtest's returns are optimistic.")
    else:
        print("   no ADV data matched")
    print()

    # ---- 4. drawdown ----
    print("4. DRAWDOWN PATH — equal-weighted daily cohort, cumulated")
    eq = 1.0
    peak = 1.0
    mdd = 0.0
    curve = []
    for d in dates:
        r = st.mean(x[1] for x in byd[d])
        eq *= (1 + r / args.hold)      # a cohort is held `hold` days; scale daily
        peak = max(peak, eq)
        mdd = min(mdd, eq / peak - 1)
        curve.append((d, eq))
    print(f"   cumulative {100*(eq-1):+.1f}% over {len(dates)} cohorts")
    print(f"   max drawdown {100*mdd:.1f}%")
    neg = sum(1 for d in dates if st.mean(x[1] for x in byd[d]) < 0)
    print(f"   negative cohorts: {neg}/{len(dates)} "
          f"({100*neg/len(dates):.0f}%)")
    print("   Return is scaled by 1/hold since a cohort is held several days;")
    print("   this is an approximation, not a backtest with real fills.\n")

    # ---- 5. overlap with the SI brick ----
    print("5. OVERLAP WITH THE SI BRICK (long low days-to-cover)")
    try:
        sc = sqlite3.connect(f"file:{args.si_db}?mode=ro", uri=True)
        latest = sc.execute("SELECT MAX(settlement_date) FROM short_interest"
                            ).fetchone()[0]
        si = dict(sc.execute(
            "SELECT ticker, days_to_cover FROM short_interest "
            "WHERE settlement_date=? AND days_to_cover IS NOT NULL",
            (latest,)))
        sc.close()
        vals = sorted(si.values())
        q = vals[len(vals) // 5] if vals else None
        long_leg = {t for t, v in si.items() if q is not None and v <= q}
        sel = {r[1] for r in rows}
        ov = sel & long_leg
        print(f"   SI settlement {latest}: {len(long_leg)} names in the "
              f"lowest-DTC quintile")
        print(f"   PCT7 selects {len(sel)} distinct tickers; "
              f"{len(ov)} overlap ({100*len(ov)/max(len(sel),1):.0f}%)")
        if ov:
            print(f"     {', '.join(sorted(ov)[:12])}")
        if len(ov) / max(len(sel), 1) > 0.4:
            print("   !! heavy overlap -- these are not two independent bricks")
        else:
            print("   Low overlap: the two select largely different names.")
    except Exception as e:
        print(f"   short interest unavailable: {e}")
    print()

    # ---- 6. turnover ----
    print("6. TURNOVER")
    ch = []
    for i in range(1, len(dates)):
        a = {t for t, _ in byd[dates[i - 1]]}
        b = {t for t, _ in byd[dates[i]]}
        if a or b:
            ch.append(100.0 * len(b - a) / max(len(b), 1))
    if ch:
        print(f"   mean {st.mean(ch):.0f}% of each day's names are new")
        print("   At 5-day holds, high turnover means costs compound faster")
        print("   than the flat 20bps ladder in the gauntlet assumed.")


if __name__ == "__main__":
    main()
