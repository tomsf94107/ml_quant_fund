#!/usr/bin/env python3
"""
ticker_persistence.py — does per-ticker accuracy in one window predict the next?

READ-ONLY. Writes nothing.

WHY THIS EXISTS
    Ranking tickers by high-confidence accuracy produces a list that looks
    actionable: PSX 93%, S 94%, AFL 87%. But those rankings are computed on the
    same window they are read from, so they describe the past. The question that
    matters is whether a ticker that scored well in one period scores well in
    the NEXT one.

    Measured 2026-09-02 on the top 20 only:
        h=3  top-20 held 13/20 (65%)  vs all 77 tickers 58%   -> +7pp
        h=5  top-20 held 11/20 (55%)  vs all 87 tickers 59%   -> -4pp
    i.e. the highest-ranked names performed like the field. This prints the FULL
    list so that is verifiable rather than taken on trust, and adds a rank
    correlation over every ticker rather than a cut at 20.

READING IT
    prior / later   accuracy at prob_up >= 0.55 in each window
    delta           later minus prior, in percentage points
    Spearman        rank correlation between the two windows across ALL tickers.
                    ~0 means a ticker's rank in one window says nothing about
                    its rank in the next. This is the number that settles it,
                    not any individual row.
    decile table    mean later-accuracy by prior-accuracy decile. If skill
                    persists, later accuracy rises with prior decile. If it is
                    noise, the column is flat.

CAVEAT
    Both windows use overlapping outcome horizons (h=3 and h=5 predictions on
    consecutive days share outcome days), so effective sample is smaller than n
    suggests. That widens the true uncertainty on every row; it does not change
    the rank correlation's interpretation.

    python analysis/ticker_persistence.py --db accuracy.db
"""
import argparse
import sqlite3

ETF = {"XLE","XLF","XLK","XLV","XLP","XLU","XLI","XLY","XLB","XLRE","XLC",
       "SPY","QQQ","IWM","RSP","SMH","IGV","ARKK","TLT","GLD","USO","VXX",
       "DIA","EEM","EFA","HYG","LQD","XBI","XOP","XRT","KRE","SOXX","VNQ","SLV"}

Q = """SELECT p.ticker, COUNT(*) n,
         SUM(CASE WHEN (p.prob_up>=0.5)=(o.actual_up=1) THEN 1 ELSE 0 END) k
       FROM predictions p JOIN outcomes o
         ON p.ticker=o.ticker AND p.prediction_date=o.prediction_date
        AND p.horizon=o.horizon
       WHERE p.horizon=? AND p.prob_up>=? AND o.actual_up IS NOT NULL
         AND p.prediction_date>=? AND p.prediction_date<?
       GROUP BY p.ticker HAVING n>=?"""


def spearman(xs, ys):
    def rank(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(v):
            j = i
            while j + 1 < len(v) and v[order[j + 1]] == v[order[i]]:
                j += 1
            avg = (i + j) / 2.0 + 1
            for m in range(i, j + 1):
                r[order[m]] = avg
            i = j + 1
        return r
    rx, ry = rank(xs), rank(ys)
    n = len(xs)
    if n < 3:
        return None
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((rx[i] - mx) * (ry[i] - my) for i in range(n))
    dx = sum((r - mx) ** 2 for r in rx) ** 0.5
    dy = sum((r - my) ** 2 for r in ry) ** 0.5
    return num / (dx * dy) if dx and dy else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="accuracy.db")
    ap.add_argument("--thresh", type=float, default=0.55)
    ap.add_argument("--split", default="2026-07-01")
    ap.add_argument("--start", default="2026-01-01")
    ap.add_argument("--end", default="2026-09-03")
    ap.add_argument("--min-prior", type=int, default=10)
    ap.add_argument("--min-later", type=int, default=8)
    ap.add_argument("--keep-etf", action="store_true")
    args = ap.parse_args()

    con = sqlite3.connect(f"file:{args.db}?mode=ro", uri=True)
    for h in (3, 5):
        a = {t: (n, k) for t, n, k in con.execute(
            Q, (h, args.thresh, args.start, args.split, args.min_prior))
            if args.keep_etf or t not in ETF}
        b = {t: (n, k) for t, n, k in con.execute(
            Q, (h, args.thresh, args.split, args.end, args.min_later))
            if args.keep_etf or t not in ETF}
        both = sorted(set(a) & set(b), key=lambda t: -a[t][1] / a[t][0])
        if len(both) < 3:
            print(f"h={h}: only {len(both)} tickers in both windows")
            continue

        print(f"\n{'='*72}")
        print(f"HORIZON {h}d   prob_up >= {args.thresh}   "
              f"{len(both)} tickers in BOTH windows")
        print(f"  prior  {args.start}..{args.split}  (n>={args.min_prior})")
        print(f"  later  {args.split}..{args.end}  (n>={args.min_later})")
        print("=" * 72)
        print(f"  {'#':>3} {'ticker':<7}{'prior':>8}{'n':>5}"
              f"{'later':>8}{'n':>5}{'delta':>9}")
        for i, t in enumerate(both, 1):
            pa = 100 * a[t][1] / a[t][0]
            la = 100 * b[t][1] / b[t][0]
            print(f"  {i:>3} {t:<7}{pa:>7.1f}%{a[t][0]:>5}"
                  f"{la:>7.1f}%{b[t][0]:>5}{la-pa:>+8.1f}pp")

        pri = [100 * a[t][1] / a[t][0] for t in both]
        lat = [100 * b[t][1] / b[t][0] for t in both]
        rho = spearman(pri, lat)
        held_all = sum(1 for x in lat if x >= 50)
        print(f"\n  Spearman rank correlation prior vs later: "
              f"{rho:+.3f}" if rho is not None else "")
        print(f"  tickers above 50% in the later window: "
              f"{held_all}/{len(both)} ({100*held_all/len(both):.0f}%)")

        print(f"\n  mean LATER accuracy by PRIOR decile "
              f"(flat = prior rank carries no information):")
        d = sorted(zip(pri, lat))
        size = max(1, len(d) // 10)
        for q in range(10):
            chunk = d[q*size:(q+1)*size] if q < 9 else d[q*size:]
            if not chunk:
                continue
            print(f"    decile {q+1:>2}  prior {sum(c[0] for c in chunk)/len(chunk):5.1f}%"
                  f"   -> later {sum(c[1] for c in chunk)/len(chunk):5.1f}%"
                  f"   (n={len(chunk)})")
    con.close()
    print("\nA Spearman near zero and a flat decile column together mean the "
          "ranking\ndescribes the past and does not identify skill. Read those "
          "two before\nany individual row.")


if __name__ == "__main__":
    main()
