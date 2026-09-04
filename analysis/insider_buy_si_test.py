#!/usr/bin/env python3
"""
insider_buy_si_test.py — does insider BUYING predict, and more so when short
interest is high?

READ-ONLY. Writes nothing.

WHY THIS IS A DIFFERENT HYPOTHESIS FROM THE THREE ALREADY KILLED
    Three tests of insider SELLING came back null: flow constructions
    (insider_construction_test.py), trajectory constructions across h=5/20/60
    (insider_timeseries_test.py), and remaining overhang from backfilled
    sharesOwnedFollowingTransaction (overhang_test.py, where per-ticker ICs
    contradicted each other -- FIG +0.674 against BETR -0.364).

    The literature explains those nulls rather than contradicting them:

      - Insider selling is LESS predictive in technology firms; industries with
        faster information diffusion show a weaker link to future returns
        (Journal of Risk and Financial Management, 2026). This universe is
        heavily tech.
      - Sell-side informativeness is WEAKER post-2023, after the Rule 10b5-1
        amendments tightened cooling-off periods and made plan sales more
        mechanical (same study). The tests above ran 2021-2026, mostly
        post-amendment.
      - Predictability is "strongest in equal-weighted portfolios, SHORT holding
        periods, and insider-BUYING signals" (Heckmann, Jacobs & Schwarz, 2023).

    So: buying, not selling. And short holding periods, which is this model's
    horizon rather than an argument against it.

THE CONDITIONING HYPOTHESIS
    Insiders "seek to signal firm quality during adverse conditions and
    counteract short sellers when short interest is high" (Journal of Behavioral
    and Experimental Finance, 2023). Insider buying is partly a RESPONSE to
    short interest, which makes short interest a natural conditioner rather than
    a competing signal.

    That matters here because days-to-cover is this project's one validated
    brick -- per-date IC -0.054, NW-t -4.46, negative every year 2021-2026,
    null control 8.3 sigma. Conditioning a candidate on an edge that is already
    trusted is a stronger test than conditioning on something unproven.

    The INFO factor (Ma & Ringgenberg) combines "the positive signals from
    insider trades with the negative signals from short selling activities and
    option trades" -- all three of which exist in this system.

WHAT IS TESTED
    buy_dollars     insider open-market purchases (code P) over 90 days, in
                    dollars, per ticker-date
    buy_dov         those purchases divided by 20-day dollar ADV
    buy_breadth     count of distinct insiders buying in 90 days -- clustering,
                    which the literature repeatedly finds matters more than size
    net_buy_dov     (purchases - sales) / dollar ADV

    Each is evaluated three ways:
      ALL           the full cross-section
      HIGH-SI       only names in the top tercile of days-to-cover that date
      LOW-SI        only names in the bottom tercile

    If the conditioning hypothesis holds, HIGH-SI shows a materially larger IC
    than LOW-SI. If buying works unconditionally, all three are similar. If
    nothing works, all three are zero -- which after three nulls is the
    honest base case.

POINT-IN-TIME
    Insider data keyed on filing_date, not trade_date: Form 4 allows two
    business days and the market cannot know before the filing exists.
    Short interest keyed on the FINRA settlement date already stored, which
    publishes with roughly an 8-business-day lag; the lag is inherent in the
    stored series.

    python analysis/insider_buy_si_test.py
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--insider-db", default="insider_trades.db")
    ap.add_argument("--prices-db", default="prices.db")
    ap.add_argument("--si-db", default="short_interest.db")
    ap.add_argument("--start", default="2021-05-01")
    ap.add_argument("--min-names", type=int, default=12)
    args = ap.parse_args()
    HOR = (5, 20)

    px = sqlite3.connect(f"file:{args.prices_db}?mode=ro", uri=True)
    close, volume = defaultdict(dict), defaultdict(dict)
    for t, d, c, v in px.execute(
            "SELECT ticker, d, close, volume FROM raw_bars WHERE d >= ?",
            (args.start,)):
        if c:
            close[t][d] = c
        if v:
            volume[t][d] = v
    px.close()

    fwd = {h: {} for h in HOR}
    dadv = {}
    for t, s in close.items():
        ds = sorted(s)
        for h in HOR:
            for i in range(len(ds) - h):
                a, b = s[ds[i]], s[ds[i + h]]
                if a and b and abs((b - a) / a) < 0.8:
                    fwd[h][(t, ds[i])] = (b - a) / a
        vs = volume.get(t, {})
        vd = sorted(vs)
        for i in range(20, len(vd)):
            w = [vs[vd[j]] * s.get(vd[j], 0) for j in range(i - 20, i)]
            m = sum(w) / len(w)
            if m > 0:
                dadv[(t, vd[i])] = m

    # ---- short interest: days to cover, per ticker per settlement ----
    si = defaultdict(dict)
    try:
        sc = sqlite3.connect(f"file:{args.si_db}?mode=ro", uri=True)
        cols = [r[1] for r in sc.execute("PRAGMA table_info(short_interest)")]
        dcol = "days_to_cover" if "days_to_cover" in cols else None
        if dcol:
            for t, d, v in sc.execute(
                    f"SELECT ticker, settlement_date, {dcol} "
                    f"FROM short_interest WHERE {dcol} IS NOT NULL "
                    f"AND settlement_date >= ?", (args.start,)):
                si[t][str(d)[:10]] = v
        sc.close()
        print(f"short interest: {len(si)} tickers"
              + ("" if dcol else "  (no days_to_cover column -- "
                                 "conditioning disabled)"))
    except Exception as e:
        print(f"short interest unavailable ({e}) -- conditioning disabled")

    def si_asof(t, d):
        """Latest settlement at or before d. FINRA publishes ~8 business days
        after settlement, so a settlement dated within ~12 days of d may not
        have been public yet; those are excluded rather than assumed known."""
        s = si.get(t)
        if not s:
            return None
        cand = [k for k in s
                if k <= (date.fromisoformat(d) - timedelta(days=12)).isoformat()]
        return s[max(cand)] if cand else None

    ic = sqlite3.connect(f"file:{args.insider_db}?mode=ro", uri=True)
    rows = ic.execute("""
        SELECT ticker, filing_date, transaction_code, shares, price_per_share,
               insider_name
        FROM insider_filings_raw
        WHERE filing_date >= ? AND shares IS NOT NULL
    """, (args.start,)).fetchall()
    ic.close()

    buy_d, sell_d = defaultdict(float), defaultdict(float)
    buyers = defaultdict(set)
    for t, fd, code, sh, pps, name in rows:
        fd = str(fd)[:10]
        usd = abs(sh or 0) * (pps or 0)
        if code == "P":
            buy_d[(t, fd)] += usd
            buyers[(t, fd)].add(name or "?")
        elif code == "S":
            sell_d[(t, fd)] += usd
    print(f"{len(rows):,} filings; "
          f"{sum(1 for k in buy_d if buy_d[k] > 0):,} ticker-days with "
          f"open-market purchases\n")

    def wsum(dct, t, d, days):
        d0 = date.fromisoformat(d)
        return sum(dct.get((t, (d0 - timedelta(days=k)).isoformat()), 0.0)
                   for k in range(days))

    def wset(t, d, days):
        d0 = date.fromisoformat(d)
        s = set()
        for k in range(days):
            s |= buyers.get((t, (d0 - timedelta(days=k)).isoformat()), set())
        return len(s)

    dates = sorted({d for t in close for d in close[t] if d >= args.start})[::5]
    names = ["buy_dollars", "buy_dov", "buy_breadth", "net_buy_dov"]
    buckets = ("ALL", "HIGH-SI", "LOW-SI")
    ics = {(k, h, b): [] for k in names for h in HOR for b in buckets}
    nulls = {(k, h, b): [] for k in names for h in HOR for b in buckets}
    rnd = random.Random(5)

    for d in dates:
        cand = {}
        for t in close:
            if d not in close[t]:
                continue
            b90 = wsum(buy_d, t, d, 90)
            s90 = wsum(sell_d, t, d, 90)
            if b90 <= 0:
                continue                 # no purchases: no buy signal
            a = dadv.get((t, d))
            v = {"buy_dollars": b90, "buy_breadth": float(wset(t, d, 90))}
            if a:
                v["buy_dov"] = b90 / a
                v["net_buy_dov"] = (b90 - s90) / a
            cand[t] = (v, si_asof(t, d))

        if len(cand) < args.min_names:
            continue
        sis = sorted((v[1] for v in cand.values() if v[1] is not None))
        hi_cut = sis[int(len(sis) * 2 / 3)] if len(sis) >= 6 else None
        lo_cut = sis[int(len(sis) / 3)] if len(sis) >= 6 else None

        for k in names:
            for h in HOR:
                for b in buckets:
                    obs = []
                    for t, (v, sv) in cand.items():
                        if k not in v or (t, d) not in fwd[h]:
                            continue
                        if b == "HIGH-SI" and (hi_cut is None or sv is None
                                               or sv < hi_cut):
                            continue
                        if b == "LOW-SI" and (lo_cut is None or sv is None
                                              or sv > lo_cut):
                            continue
                        obs.append((v[k], fwd[h][(t, d)]))
                    if len(obs) < args.min_names:
                        continue
                    r = spearman(obs)
                    if r is not None:
                        ics[(k, h, b)].append(r)
                    ys = [o[1] for o in obs]
                    rnd.shuffle(ys)
                    rn = spearman([(obs[i][0], ys[i])
                                   for i in range(len(ys))])
                    if rn is not None:
                        nulls[(k, h, b)].append(rn)

    print(f"  {'construction':<14}{'h':>3}{'bucket':>9}{'dates':>7}"
          f"{'mean IC':>10}{'NW t':>8}{'null t':>9}")
    for k in names:
        for h in HOR:
            for b in buckets:
                v = ics[(k, h, b)]
                if len(v) < 20:
                    print(f"  {k:<14}{h:>3}{b:>9}{len(v):>7}   too few dates")
                    continue
                t_ = nw_t(v, h) or 0.0
                nt = nw_t(nulls[(k, h, b)], h) or 0.0
                flag = "  <<<" if abs(t_) >= 2.5 and abs(nt) < 1.5 else ""
                print(f"  {k:<14}{h:>3}{b:>9}{len(v):>7}{st.mean(v):>+10.4f}"
                      f"{t_:>+8.2f}{nt:>+9.2f}{flag}")
            print()

    print("  Sign: values are NOT negated here. More buying is a LARGER value, "
          "so a\n  POSITIVE IC means more insider buying predicts HIGHER "
          "forward returns.\n")
    print("  The conditioning hypothesis predicts HIGH-SI materially above "
          "LOW-SI.\n  Similar ICs across buckets would mean buying works "
          "unconditionally.\n  All three near zero is the base case after "
          "three nulls on selling.\n")
    print("  Null t must be near zero. Treat |t| < 2.5 as unproven -- this is "
          "the fourth\n  test of a related idea, so a marginal result is more "
          "likely selection than\n  signal.")


if __name__ == "__main__":
    main()
