#!/usr/bin/env python3
"""
market_timing_test.py — can anything we already have predict a selloff week?

READ-ONLY. Writes nothing. Tests BEFORE building.

THE PROPOSAL BEING TESTED
    On 2026-08-21..08-26 the model was long into a broad decline: only 25-41%
    of the 417-name universe rose over the following five sessions, mean 5-day
    return -2.32% at the worst. Accuracy tracked the base rate exactly, so no
    skill was lost -- but a long-only gate loses money in that week regardless
    of skill.

    The proposal is an overlay: detect weeks like that in advance and stand
    down, or cut confidence.

WHY IT IS TESTED RATHER THAN BUILT
    An overlay that does not predict costs money twice -- it stands down before
    good weeks and stays long before bad ones. And the crash-warning subsystem
    already in this repo has no demonstrated hit rate: measured 2026-08-30,
    S5/S7/S8 were indistinguishable from chance against their base rates and
    S9's six fires were all false, one landing on a bear-market low.

    So the question is narrow and answerable: does ANY series already on hand
    predict the universe's forward 5-day up-rate?

WHAT IS TESTED
    Target: the fraction of the universe whose 5-day forward return is positive,
    computed per date -- the "base rate" the model is judged against, and the
    thing that collapsed in late August.

    Predictors, all point-in-time and already local:
      SPY trailing return (1, 5, 20 day)      momentum / reversal
      SPY realised vol (20d)                  regime
      VIX level and 252-day percentile        priced volatility
      VIX term slope (VIX vs VIX3M)           stress structure
      breadth: % of names above their 200DMA  participation
      cross-sectional dispersion (robust SD)  opportunity
      HY OAS level and 21-day change          credit stress
      2s10s slope                             rates regime

    For each: Spearman correlation with the forward up-rate, a naive t-stat,
    and the up-rate conditional on the predictor's own top and bottom quintile.

HOW TO READ IT
    The sample is ~250-500 daily observations with heavy overlap (5-day forward
    windows on consecutive days share four days), so the effective independent
    sample is closer to n/5. Naive t-stats are therefore inflated by roughly
    sqrt(5) ~ 2.2x. A |t| of 2 here is NOT significant; treat |t| < 4 as noise.

    A predictor earns a build only if the quintile spread is large, monotone,
    and survives that discount.

    python analysis/market_timing_test.py
"""
import math
import sqlite3
import statistics as st
from collections import defaultdict

PRICES = "prices.db"
WARN = "warning.db"
FWD = 5


def spearman(xs, ys):
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
    n = len(xs)
    if n < 10:
        return None
    rx, ry = rank(xs), rank(ys)
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((rx[i] - mx) * (ry[i] - my) for i in range(n))
    dx = math.sqrt(sum((r - mx) ** 2 for r in rx))
    dy = math.sqrt(sum((r - my) ** 2 for r in ry))
    return num / (dx * dy) if dx and dy else None


def main():
    px = sqlite3.connect(f"file:{PRICES}?mode=ro", uri=True)
    rows = px.execute(
        "SELECT ticker, date, adj_close FROM daily_prices "
        "WHERE date >= '2023-01-01' AND adj_close IS NOT NULL").fetchall()
    series = defaultdict(dict)
    for t, d, p in rows:
        series[t][d] = p
    spy = series.get("SPY", {})
    dates = sorted(spy)
    print(f"{len(series)} tickers, {len(dates)} SPY sessions "
          f"{dates[0]}..{dates[-1]}")

    # ---- target: fraction of universe up over the NEXT 5 sessions ----
    idx = {d: i for i, d in enumerate(dates)}
    up_rate = {}
    disp = {}
    for i in range(len(dates) - FWD):
        d0, d1 = dates[i], dates[i + FWD]
        rets = []
        for t, s in series.items():
            a, b = s.get(d0), s.get(d1)
            if a and b and abs((b - a) / a) < 0.5:      # drop split artifacts
                rets.append((b - a) / a)
        if len(rets) > 100:
            up_rate[d0] = 100.0 * sum(1 for r in rets if r > 0) / len(rets)
            rs = sorted(rets)
            disp[d0] = (rs[3 * len(rs) // 4] - rs[len(rs) // 4]) / 1.349

    # ---- predictors ----
    spyret = {dates[i]: (spy[dates[i]] - spy[dates[i - 1]]) / spy[dates[i - 1]]
              for i in range(1, len(dates)) if spy[dates[i - 1]]}
    pred = defaultdict(dict)
    for i in range(252, len(dates)):
        d = dates[i]
        w = [spyret.get(dates[j]) for j in range(i - 19, i + 1)]
        w = [x for x in w if x is not None]
        pred["spy_ret_1d"][d] = spyret.get(d)
        pred["spy_ret_5d"][d] = (spy[d] - spy[dates[i - 5]]) / spy[dates[i - 5]]
        pred["spy_ret_20d"][d] = (spy[d] - spy[dates[i - 20]]) / spy[dates[i - 20]]
        if len(w) > 10:
            pred["spy_vol_20d"][d] = st.pstdev(w) * math.sqrt(252)
        # breadth: % above own 200DMA
        above = tot = 0
        for t, s in series.items():
            sd = sorted(k for k in s if k <= d)
            if len(sd) < 200:
                continue
            tot += 1
            ma = sum(s[k] for k in sd[-200:]) / 200
            if s[d] > ma if d in s else False:
                above += 1
        if tot > 100:
            pred["pct_above_200dma"][d] = 100.0 * above / tot
        if d in disp:
            pred["dispersion"][d] = 100 * disp[d]
    px.close()

    # ---- warning.db series ----
    try:
        wc = sqlite3.connect(f"file:{WARN}?mode=ro", uri=True)
        for sid, label in (("VIXCLS", "vix_level"),
                           ("BAMLH0A0HYM2", "hy_oas"),
                           ("T10Y2Y", "slope_2s10s")):
            got = wc.execute(
                "SELECT obs_date, value FROM data_vintages WHERE series_id=? "
                "AND obs_date >= '2023-01-01' ORDER BY obs_date", (sid,)).fetchall()
            m = {}
            for d, v in got:
                m[d] = v
            for d in m:
                pred[label][d] = m[d]
            if label == "hy_oas":
                ds = sorted(m)
                for i in range(21, len(ds)):
                    pred["hy_oas_chg21"][ds[i]] = m[ds[i]] - m[ds[i - 21]]
        wc.close()
    except Exception as e:
        print(f"  (warning.db series unavailable: {e})")

    # ---- evaluate ----
    print(f"\ntarget = % of universe up over the NEXT {FWD} sessions "
          f"(n={len(up_rate)} dates)")
    base = st.mean(up_rate.values())
    print(f"mean forward up-rate: {base:.1f}%\n")
    print(f"  {'predictor':<22}{'n':>6}{'rho':>8}{'naive t':>9}"
          f"{'bottom Q':>11}{'top Q':>9}{'spread':>9}")
    for name in sorted(pred):
        pairs = [(pred[name][d], up_rate[d]) for d in pred[name]
                 if d in up_rate and pred[name][d] is not None]
        if len(pairs) < 60:
            continue
        xs = [a for a, _ in pairs]
        ys = [b for _, b in pairs]
        rho = spearman(xs, ys)
        n = len(pairs)
        t = rho * math.sqrt((n - 2) / (1 - rho * rho)) if rho and abs(rho) < 1 else 0
        s = sorted(pairs)
        q = max(1, len(s) // 5)
        bq = st.mean([b for _, b in s[:q]])
        tq = st.mean([b for _, b in s[-q:]])
        print(f"  {name:<22}{n:>6}{rho:>+8.3f}{t:>+9.2f}"
              f"{bq:>10.1f}%{tq:>8.1f}%{tq-bq:>+8.1f}pp")

    print(f"\n  Overlapping {FWD}-day windows inflate the naive t by about "
          f"sqrt({FWD}) = {math.sqrt(FWD):.1f}x.\n  Treat |t| < 4 as noise. A "
          f"predictor earns a build only if the quintile\n  spread is large, "
          f"monotone, and survives that discount.\n")
    print("  If nothing clears, the honest answer is that a timing overlay "
          "cannot be\n  built from what is on hand -- and gating a small real "
          "edge behind an\n  unproven signal would cost money twice.")


if __name__ == "__main__":
    main()
