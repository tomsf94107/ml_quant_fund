#!/usr/bin/env python3
"""
rate_beta_test.py — does per-ticker RATE SENSITIVITY explain what insider data could not?

READ-ONLY. Writes nothing.

WHY THIS, AFTER SIX NULLS ON INSIDER DATA
    Twelve CRWV research reports written 2026-08-11 through 09-03 name the
    driver on nearly every page, and it is not insider selling:

      8/18  30-year yields hit ~20-year highs; CRWV -12.1%, closed on the low.
            "the debt/rates bear case made manifest -- not a demand problem"
      8/20  a new multibillion-dollar Hudson River Trading deal lands and the
            stock slips 1.4% anyway. "demand keeps arriving, rates keep pricing it"
      8/27  NVDA blows out -- revenue $96.2B, +106%, supply commitments to $279B
            -- CRWV gaps to $91.81 and REVERSES to close at 5% of range.
            "A blowout from its own supplier could not hold a CRWV bounce.
             It is a rate trade, not a demand problem."

    The reports also record beta ~3.2 and ~$35B of debt where interest expense
    swamps operating income. Meanwhile the 30-year went to 5.2-5.3%, a two-decade
    high.

THE GAP IN THE MODEL
    yield_10y is already a feature and carries importance 7.894 -- third-highest
    of ~100. But it is a MARKET-WIDE LEVEL. It cannot express that one name is a
    leveraged rate instrument while a cash-rich megacap is nearly immune. Every
    ticker on a given date sees the same yield_10y.

    What is missing is per-ticker SENSITIVITY: the rolling beta of a stock's
    returns to changes in the yield. That is computable from raw_bars and DGS10,
    both already local.

CONSTRUCTIONS
    rate_beta        rolling 60-day OLS beta of daily stock returns on daily
                     changes in the 10-year yield. Negative beta = falls when
                     yields rise.
    rate_beta_x_dy   that beta times the trailing 5-day yield change. This is
                     the interaction the reports describe: a high-sensitivity
                     name is only punished WHEN yields actually move. Beta alone
                     is a standing characteristic; the product is a signal.
    rate_r2          the R-squared of that regression -- how much of the name's
                     variance rates explain. A name at 3.2 beta with low R2 is
                     noisy; high R2 means rates really are the driver.

    Each is tested two ways:
      PREDICTIVE   per-date IC against forward returns, Newey-West, shuffle null
      CONDITIONING whether the MODEL's own high-confidence accuracy is worse in
                   the high-sensitivity tercile

    The second matters more. A predictor must beat the market. A conditioner only
    needs this model -- which sees price, volume and volatility but no balance
    sheet -- to be systematically fooled in an identifiable state.

HONEST PRIOR
    This is roughly the 65th cell tested in this sequence. The bar is |t| >= 3.0
    with a clean null (Harvey, Liu & Zhu, RFS 2016). A marginal result is more
    likely selection than signal. What distinguishes this test from the previous
    six is that the hypothesis came from independent contemporaneous documents
    rather than from searching the data for something that works.

    python analysis/rate_beta_test.py
"""
import argparse
import math
import random
import sqlite3
import statistics as st
from collections import defaultdict


def spearman(pairs):
    n = len(pairs)
    if n < 10:
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


def nw_t(s, lag):
    n = len(s)
    if n < 10:
        return None
    m = sum(s) / n
    d = [x - m for x in s]
    var = sum(x * x for x in d) / n
    for k in range(1, min(lag, n - 1) + 1):
        gk = sum(d[i] * d[i - k] for i in range(k, n)) / n
        var += 2 * (1 - k / (lag + 1.0)) * gk
    return m / math.sqrt(var / n) if var > 0 else None


def wilson(k, n, z=1.96):
    if not n:
        return (0.0, 100.0)
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    s = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (max(0.0, 100 * (c - s) / d), min(100.0, 100 * (c + s) / d))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prices-db", default="prices.db")
    ap.add_argument("--warning-db", default="warning.db")
    ap.add_argument("--accuracy-db", default="accuracy.db")
    ap.add_argument("--start", default="2021-01-01")
    ap.add_argument("--window", type=int, default=60)
    ap.add_argument("--min-names", type=int, default=25)
    args = ap.parse_args()
    HOR = (5, 20)

    # ---- 10-year yield ----
    y = {}
    try:
        wc = sqlite3.connect(f"file:{args.warning_db}?mode=ro", uri=True)
        for d, v in wc.execute(
                "SELECT obs_date, value FROM data_vintages "
                "WHERE series_id='DGS10' AND obs_date >= ? ORDER BY obs_date",
                (args.start,)):
            y[str(d)[:10]] = v
        wc.close()
    except Exception as e:
        raise SystemExit(f"could not read DGS10 from {args.warning_db}: {e}")
    yd = sorted(y)
    dy = {yd[i]: y[yd[i]] - y[yd[i - 1]] for i in range(1, len(yd))}
    print(f"DGS10: {len(y):,} observations {yd[0]}..{yd[-1]}")

    px = sqlite3.connect(f"file:{args.prices_db}?mode=ro", uri=True)
    close = defaultdict(dict)
    for t, d, c in px.execute(
            "SELECT ticker, d, close FROM raw_bars WHERE d >= ? AND close>0",
            (args.start,)):
        close[t][d] = c
    px.close()

    fwd = {h: {} for h in HOR}
    beta = {}
    r2 = {}
    dy5 = {}
    for t, s in close.items():
        ds = sorted(s)
        rets = {}
        for i in range(1, len(ds)):
            a, b = s[ds[i - 1]], s[ds[i]]
            if a and abs((b - a) / a) < 0.5:
                rets[ds[i]] = (b - a) / a
        for h in HOR:
            for i in range(len(ds) - h):
                a, b = s[ds[i]], s[ds[i + h]]
                if a and b and abs((b - a) / a) < 0.8:
                    fwd[h][(t, ds[i])] = (b - a) / a
        rd = sorted(rets)
        for i in range(args.window, len(rd)):
            w = rd[i - args.window:i]
            xs = [dy.get(d) for d in w]
            ys = [rets[d] for d in w]
            pair = [(a, b) for a, b in zip(xs, ys) if a is not None]
            if len(pair) < args.window * 0.6:
                continue
            mx = sum(p[0] for p in pair) / len(pair)
            my = sum(p[1] for p in pair) / len(pair)
            sxx = sum((p[0] - mx) ** 2 for p in pair)
            sxy = sum((p[0] - mx) * (p[1] - my) for p in pair)
            if sxx <= 0:
                continue
            bta = sxy / sxx
            syy = sum((p[1] - my) ** 2 for p in pair)
            beta[(t, rd[i])] = bta
            r2[(t, rd[i])] = (sxy * sxy / (sxx * syy)) if syy > 0 else 0.0

    # trailing 5-day yield change, per date
    for i in range(5, len(yd)):
        dy5[yd[i]] = y[yd[i]] - y[yd[i - 5]]

    dates = sorted({d for t in close for d in close[t] if d >= args.start})[::5]
    names = ["rate_beta", "rate_beta_x_dy", "rate_r2"]
    ics = {(k, h): [] for k in names for h in HOR}
    nulls = {(k, h): [] for k in names for h in HOR}
    rnd = random.Random(17)

    for d in dates:
        chg = dy5.get(d)
        vals = {}
        for t in close:
            b = beta.get((t, d))
            if b is None:
                continue
            v = {"rate_beta": b, "rate_r2": r2.get((t, d), 0.0)}
            if chg is not None:
                v["rate_beta_x_dy"] = b * chg
            vals[t] = v
        for k in names:
            for h in HOR:
                obs = [(vals[t][k], fwd[h][(t, d)]) for t in vals
                       if k in vals[t] and (t, d) in fwd[h]]
                if len(obs) < args.min_names:
                    continue
                r = spearman(obs)
                if r is not None:
                    ics[(k, h)].append(r)
                ys2 = [o[1] for o in obs]
                rnd.shuffle(ys2)
                rn = spearman([(obs[i][0], ys2[i]) for i in range(len(ys2))])
                if rn is not None:
                    nulls[(k, h)].append(rn)

    print(f"\nPREDICTIVE TEST — {len(dates)} evaluation dates\n")
    print(f"  {'construction':<18}{'h':>4}{'dates':>7}{'mean IC':>10}"
          f"{'NW t':>8}{'null t':>9}")
    for k in names:
        for h in HOR:
            v = ics[(k, h)]
            if len(v) < 20:
                print(f"  {k:<18}{h:>4}{len(v):>7}   too few dates")
                continue
            t_ = nw_t(v, h) or 0.0
            nt = nw_t(nulls[(k, h)], h) or 0.0
            flag = "  <<<" if abs(t_) >= 3.0 and abs(nt) < 1.5 else ""
            print(f"  {k:<18}{h:>4}{len(v):>7}{st.mean(v):>+10.4f}"
                  f"{t_:>+8.2f}{nt:>+9.2f}{flag}")
        print()

    # ---- CONDITIONING TEST: is the model worse on rate-sensitive names? ----
    ac = sqlite3.connect(f"file:{args.accuracy_db}?mode=ro", uri=True)
    preds = ac.execute("""
        SELECT p.ticker, p.prediction_date, p.prob_up, o.actual_up
        FROM predictions p JOIN outcomes o ON p.ticker=o.ticker
          AND p.prediction_date=o.prediction_date AND p.horizon=o.horizon
        WHERE p.horizon=5 AND o.actual_up IS NOT NULL AND p.prob_up IS NOT NULL
    """).fetchall()
    ac.close()

    print(f"CONDITIONING TEST — {len(preds):,} scored h=5 predictions")
    for state, getter in (("rate_beta", lambda t, d: beta.get((t, d))),
                          ("rate_r2", lambda t, d: r2.get((t, d)))):
        rows = [(t, d, p, a, getter(t, d)) for t, d, p, a in preds]
        rows = [r for r in rows if r[4] is not None]
        if len(rows) < 500:
            print(f"\n=== {state}: only {len(rows)} matched, skipped ===")
            continue
        vals = sorted(r[4] for r in rows)
        lo_cut = vals[len(vals) // 3]
        hi_cut = vals[2 * len(vals) // 3]
        print(f"\n=== {state} — terciles at {lo_cut:.4g} / {hi_cut:.4g}, "
              f"{len(rows):,} matched ===")
        print(f"  {'cohort':<14}{'tercile':<8}{'n':>7}{'acc':>8}{'base':>8}"
              f"{'lift':>9}{'95% CI':>18}")
        for cohort, filt in (("all", lambda p: True),
                             ("prob>=0.70", lambda p: p >= 0.70)):
            for label, lo, hi in (("LOW", -1e18, lo_cut),
                                  ("MID", lo_cut, hi_cut),
                                  ("HIGH", hi_cut, 1e18)):
                allsub = [r for r in rows if lo <= r[4] < hi]
                sub = [r for r in allsub if filt(r[2])]
                if len(sub) < 30 or not allsub:
                    print(f"  {cohort:<14}{label:<8}{len(sub):>7}   too few")
                    continue
                n = len(sub)
                k = sum(1 for r in sub if r[3] == 1)
                base = 100.0 * sum(1 for r in allsub if r[3] == 1) / len(allsub)
                acc = 100.0 * k / n
                cl, ch = wilson(k, n)
                print(f"  {cohort:<14}{label:<8}{n:>7}{acc:>7.1f}%{base:>7.1f}%"
                      f"{acc-base:>+8.1f}pp   [{cl:>5.1f},{ch:>5.1f}]")
            print()

    print("  rate_beta is NEGATIVE for a name that falls when yields rise, so "
          "the LOW\n  tercile is the MOST rate-sensitive. The reports' claim "
          "predicts the model\n  is worse there.\n")
    print("  A conditioning result needs high-confidence lift materially worse "
          "in the\n  rate-sensitive tercile with non-overlapping intervals. "
          "This is roughly the\n  65th cell in this sequence; the bar is |t| >= "
          "3.0 or clearly separated CIs.")


if __name__ == "__main__":
    main()
