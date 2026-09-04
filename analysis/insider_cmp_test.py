#!/usr/bin/env python3
"""
insider_cmp_test.py — routine vs opportunistic, the split every prior test missed.

READ-ONLY. Writes nothing.

WHY THIS IS NOT A FIFTH SLICE OF THE SAME IDEA
    Four tests have found nothing: flow constructions, trajectory constructions,
    remaining overhang, and buying conditioned on short interest. All four used
    the UNDIFFERENTIATED pool of insider trades.

    Cohen, Malloy & Pomorski (Journal of Finance, 2012) found precisely that
    this is why raw insider trading appears uninformative: routine trades swamp
    the informative ones. After separating them, a long-short on OPPORTUNISTIC
    trades earned 82bps/month value-weighted and 180bps equal-weighted (t=6.07),
    while routine trades earned essentially zero -- and BOTH opportunistic buys
    and opportunistic SELLS predicted returns, which is the exception to the
    usual finding that only buying is informative.

    Routine trades are over half of all insider activity. Every prior test here
    measured a signal diluted by the half that carries no information. This is
    the standard methodological fix, not another slice.

THE CLASSIFICATION
    CMP's rule is mechanical: an insider who traded in the SAME CALENDAR MONTH
    in three consecutive years is a routine trader, and their trades in that
    month are routine. Everything else is opportunistic.

    The intuition is that a scheduled annual liquidation, a vesting-driven sale
    every March, or a fixed 10b5-1 tranche is predictable from the insider's own
    history and therefore carries no news. An irregular trade does.

    This requires several years of history per insider. insider_filings_raw
    spans 2019-2026, so classification uses 2019-2022 and testing runs
    2023-onward -- the classification window and the test window do not overlap,
    which matters because classifying on the same data being tested would leak.

ALSO TESTED, both newly possible
    stake_fraction   shares sold divided by the insider's own holding before the
                     sale, from shares_owned_after backfilled today. Selling 5%
                     of a position and selling 80% are different acts that every
                     raw-share measure treats identically. Only available for the
                     15 backfilled tickers.
    csuite_weighted  sales weighted by is_csuite. The literature consistently
                     finds officer trades more informative than director trades.

HONEST NOTE ON MULTIPLE TESTING
    This is the fifth test in this sequence and roughly the 30th cell overall.
    At |t| = 2.0 with 30 cells, about 1.5 false positives are expected by
    chance. The bar here is therefore |t| >= 3.0 with a clean null, following
    Harvey, Liu & Zhu (RFS 2016), who argue a newly discovered factor should
    clear |t| > 3.0 given the factor zoo. A marginal result is more likely
    selection than signal.

    python analysis/insider_cmp_test.py
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
    ap.add_argument("--class-end", default="2023-01-01",
                    help="classification uses filings BEFORE this date")
    ap.add_argument("--min-names", type=int, default=12)
    args = ap.parse_args()
    HOR = (5, 20, 60)

    ic = sqlite3.connect(f"file:{args.insider_db}?mode=ro", uri=True)
    rows = ic.execute("""
        SELECT ticker, filing_date, transaction_code, shares, price_per_share,
               insider_name, is_csuite
        FROM insider_filings_raw
        WHERE shares IS NOT NULL AND insider_name IS NOT NULL
        ORDER BY filing_date
    """).fetchall()

    # ---- CMP classification on the EARLY window only ----
    # (insider, ticker, month) -> set of years traded
    hist = defaultdict(set)
    for t, fd, code, sh, pps, name, cs in rows:
        fd = str(fd)[:10]
        if fd >= args.class_end or code not in ("P", "S"):
            continue
        hist[(name, t, int(fd[5:7]))].add(int(fd[:4]))

    routine = set()
    for (name, t, mon), yrs in hist.items():
        ys = sorted(yrs)
        run = 1
        for i in range(1, len(ys)):
            run = run + 1 if ys[i] == ys[i - 1] + 1 else 1
            if run >= 3:
                routine.add((name, t, mon))
                break
    print(f"classification window: filings before {args.class_end}")
    print(f"  {len(hist):,} (insider, ticker, month) combinations")
    print(f"  {len(routine):,} classified ROUTINE "
          f"(same month, 3 consecutive years)")

    # ---- holdings, for stake_fraction ----
    prior = {}
    try:
        for t, fd, ins, soa, shr in ic.execute(
                "SELECT ticker, filing_date, insider_norm, shares_owned_after,"
                " shares FROM insider_holdings WHERE shares_owned_after "
                "IS NOT NULL ORDER BY filing_date, seq"):
            prior[(t, str(fd)[:10], ins)] = (soa or 0) + (shr or 0)
        print(f"  {len(prior):,} holding records for stake_fraction")
    except sqlite3.OperationalError:
        print("  insider_holdings not present -- stake_fraction skipped")
    ic.close()

    px = sqlite3.connect(f"file:{args.prices_db}?mode=ro", uri=True)
    close, volume = defaultdict(dict), defaultdict(dict)
    for t, d, c, v in px.execute(
            "SELECT ticker, d, close, volume FROM raw_bars WHERE d >= ?",
            (args.class_end,)):
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

    # ---- per ticker-day aggregates on the TEST window ----
    opp_sell, rou_sell = defaultdict(float), defaultdict(float)
    opp_buy = defaultdict(float)
    cs_sell = defaultdict(float)
    stake = defaultdict(list)
    for t, fd, code, sh, pps, name, cs in rows:
        fd = str(fd)[:10]
        if fd < args.class_end or code not in ("P", "S"):
            continue
        usd = abs(sh or 0) * (pps or 0)
        is_rou = (name, t, int(fd[5:7])) in routine
        if code == "S":
            (rou_sell if is_rou else opp_sell)[(t, fd)] += usd
            if cs:
                cs_sell[(t, fd)] += usd
            held = prior.get((t, fd, (name or "").upper()))
            if held and held > 0:
                stake[(t, fd)].append(min(abs(sh or 0) / held, 1.0))
        elif code == "P" and not is_rou:
            opp_buy[(t, fd)] += usd

    def wsum(dct, t, d, days):
        d0 = date.fromisoformat(d)
        return sum(dct.get((t, (d0 - timedelta(days=k)).isoformat()), 0.0)
                   for k in range(days))

    def wstake(t, d, days):
        d0 = date.fromisoformat(d)
        vals = []
        for k in range(days):
            vals += stake.get((t, (d0 - timedelta(days=k)).isoformat()), [])
        return max(vals) if vals else None

    dates = sorted({d for t in close for d in close[t]
                    if d >= args.class_end})[::5]
    names = ["opp_sell_dov", "rou_sell_dov", "opp_buy_dov",
             "csuite_sell_dov", "stake_fraction"]
    ics = {(k, h): [] for k in names for h in HOR}
    nulls = {(k, h): [] for k in names for h in HOR}
    rnd = random.Random(13)

    for d in dates:
        vals = {}
        for t in close:
            if d not in close[t]:
                continue
            a = dadv.get((t, d))
            if not a:
                continue
            o = wsum(opp_sell, t, d, 90)
            r = wsum(rou_sell, t, d, 90)
            b = wsum(opp_buy, t, d, 90)
            c = wsum(cs_sell, t, d, 90)
            if o + r + b == 0:
                continue
            v = {}
            if o:
                v["opp_sell_dov"] = -o / a
            if r:
                v["rou_sell_dov"] = -r / a
            if b:
                v["opp_buy_dov"] = b / a
            if c:
                v["csuite_sell_dov"] = -c / a
            sf = wstake(t, d, 90)
            if sf is not None:
                v["stake_fraction"] = -sf
            vals[t] = v

        for k in names:
            for h in HOR:
                obs = [(vals[t][k], fwd[h][(t, d)]) for t in vals
                       if k in vals[t] and (t, d) in fwd[h]]
                if len(obs) < args.min_names:
                    continue
                rr = spearman(obs)
                if rr is not None:
                    ics[(k, h)].append(rr)
                ys = [o2[1] for o2 in obs]
                rnd.shuffle(ys)
                rn = spearman([(obs[i][0], ys[i]) for i in range(len(ys))])
                if rn is not None:
                    nulls[(k, h)].append(rn)

    print(f"\n  test window: {args.class_end} onward, "
          f"{len(dates)} evaluation dates")
    print(f"\n  {'construction':<18}{'h':>4}{'dates':>7}{'mean IC':>10}"
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

    print("  Sign: SELL measures are negated so more selling is more negative;")
    print("  a POSITIVE IC means heavier selling predicts LOWER returns. BUY")
    print("  measures are not negated.\n")
    print("  THE COMPARISON THAT MATTERS is opp_sell_dov against rou_sell_dov.")
    print("  CMP predicts opportunistic predicts and routine does not. If both")
    print("  are zero, the split does not rescue the signal and four prior")
    print("  nulls stand.\n")
    print("  Bar is |t| >= 3.0 with a clean null (Harvey, Liu & Zhu 2016) --")
    print("  this is roughly the 30th cell tested in this sequence.")


if __name__ == "__main__":
    main()
