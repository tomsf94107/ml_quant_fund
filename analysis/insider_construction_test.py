#!/usr/bin/env python3
"""
insider_construction_test.py — does ANY insider construction predict at h=5?

READ-ONLY. Writes nothing. Tests before building.

WHY THIS EXISTS
    CRWV scored 9.1% accuracy on 11 high-confidence h=5 predictions -- the model
    was confidently long while a 30% holder distributed 70M shares and officers
    sold ~$730M in a quarter. The obvious inference is "feed insider data to the
    model". But the model ALREADY has five insider features, and insider_90d
    carries importance 4.53 (mid-pack of ~100). CRWV has 170 rows in
    insider_flows, second-highest of any ticker. The data was there.

    So the question is not whether to add insider data. It is whether any
    CONSTRUCTION of it predicts at a 5-day horizon.

WHAT THE CURRENT FEATURE DOES, AND WHY IT MAY NOT WORK
    features/builder.py line 1086-1089: net.rolling(N).sum() -- raw net share
    counts over 7/21/60/90 days. Two problems:

    1. NOT NORMALISED. 200,000 shares is enormous for a small float and
       irrelevant for Apple. Raw counts make those identical.
    2. NO TIMING CONTENT. A rolling 10b5-1 distribution produces a steadily
       negative number every day for months. It says "insiders are selling" on
       the day the stock rises and the day it falls. A level cannot pick weeks.

CONSTRUCTIONS TESTED
    net_shares_90d      the current feature, as a baseline to beat
    days_of_volume      shares sold / 20-day ADV -- how many sessions of normal
                        volume the supply represents. Field & Hanka (JF 2001)
                        found lockup-expiry abnormal returns MORE negative when
                        volume is abnormally high; this is the absorption-
                        capacity version of that, and needs no float figure.
    accel               7-day sell rate / 90-day sell rate -- a tranche above
                        the run-rate is a CHANGE, which a level cannot express.
    sell_only_dov       days_of_volume using code 'S' alone, excluding 'F'
                        (tax withholding) and 'A' (grants), which are mechanical
                        and carry no intent.
    net_dov             (sells - buys) / ADV, since Lakonishok & Lee and others
                        find buys more informative than sells.

POINT-IN-TIME
    Keyed on filing_date, NOT trade_date. The market cannot know about a
    transaction until it is filed, and Form 4 allows two business days. Using
    trade_date would leak up to two days of hindsight into every observation --
    the same class of error that voided this project's earlier PEAD work, where
    report_date turned out to be fiscal-period-end rather than announcement.

METHOD
    Per-DATE Spearman IC against forward 5-day return, then Newey-West on the
    IC series. Pooled stock-date rows are NOT independent -- a market-wide move
    correlates every stock on a date -- and treating them as independent has
    inflated t-statistics by 10-20x in this project before. The unit of
    observation is the date.

    A shuffle null is mandatory: permute the outcome within each date and the IC
    must vanish. Reported alongside, not as an appendix.

    python analysis/insider_construction_test.py
"""
import argparse
import math
import sqlite3
import statistics as st
from collections import defaultdict


def spearman(pairs):
    n = len(pairs)
    if n < 8:
        return None
    xs = [p[0] for p in pairs]
    ys = [p[1] for p in pairs]

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
    rx, ry = rank(xs), rank(ys)
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((rx[i] - mx) * (ry[i] - my) for i in range(n))
    dx = math.sqrt(sum((r - mx) ** 2 for r in rx))
    dy = math.sqrt(sum((r - my) ** 2 for r in ry))
    return num / (dx * dy) if dx and dy else None


def newey_west_t(series, lag=None):
    """t-stat on the mean of an IC series, autocorrelation-corrected.

    lag defaults to the forecast horizon, since overlapping forward windows
    induce autocorrelation of exactly that order.
    """
    n = len(series)
    if n < 10:
        return None
    lag = lag if lag is not None else 5
    m = sum(series) / n
    d = [x - m for x in series]
    g0 = sum(x * x for x in d) / n
    var = g0
    for k in range(1, min(lag, n - 1) + 1):
        gk = sum(d[i] * d[i - k] for i in range(k, n)) / n
        var += 2 * (1 - k / (lag + 1.0)) * gk
    if var <= 0:
        return None
    return m / math.sqrt(var / n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--insider-db", default="insider_trades.db")
    ap.add_argument("--prices-db", default="prices.db")
    ap.add_argument("--horizon", type=int, default=5)
    ap.add_argument("--start", default="2021-01-01")
    ap.add_argument("--min-names", type=int, default=15,
                    help="dates with fewer cross-sectional observations are "
                         "skipped; a handful of names cannot support an IC")
    args = ap.parse_args()
    H = args.horizon

    ic_db = sqlite3.connect(f"file:{args.insider_db}?mode=ro", uri=True)
    px = sqlite3.connect(f"file:{args.prices_db}?mode=ro", uri=True)

    # ---- prices, forward returns, ADV ----
    series = defaultdict(dict)
    for t, d, p in px.execute(
            "SELECT ticker, d, close FROM raw_bars WHERE d >= ? "
            "AND close IS NOT NULL", (args.start,)):
        series[t][d] = p
    vol = defaultdict(dict)
    try:
        for t, d, v in px.execute(
                "SELECT ticker, d, volume FROM raw_bars WHERE d >= ? "
                "AND volume IS NOT NULL", (args.start,)):
            vol[t][d] = v
    except sqlite3.OperationalError:
        print("  raw_bars has no volume column -- days_of_volume cannot be "
              "computed")
    px.close()

    fwd = {}
    adv = {}
    for t, s in series.items():
        ds = sorted(s)
        for i in range(len(ds) - H):
            a, b = s[ds[i]], s[ds[i + H]]
            if a and b and abs((b - a) / a) < 0.5:      # drop split artifacts
                fwd[(t, ds[i])] = (b - a) / a
        vs = vol.get(t, {})
        if vs:
            vd = sorted(vs)
            for i in range(20, len(vd)):
                w = [vs[vd[j]] for j in range(i - 20, i)]
                m = sum(w) / len(w)
                if m > 0:
                    adv[(t, vd[i])] = m

    # ---- insider filings, keyed on FILING date ----
    rows = ic_db.execute("""
        SELECT ticker, filing_date, transaction_code, shares
        FROM insider_filings_raw
        WHERE filing_date >= ? AND shares IS NOT NULL
    """, (args.start,)).fetchall()
    ic_db.close()
    print(f"{len(rows):,} insider filings from {args.start}, "
          f"{len({r[0] for r in rows})} tickers")

    sells = defaultdict(float)     # (ticker, filing_date) -> shares sold
    buys = defaultdict(float)
    allsh = defaultdict(float)
    for t, fd, code, sh in rows:
        fd = str(fd)[:10]
        if code == "S":
            sells[(t, fd)] += abs(sh or 0)
        elif code == "P":
            buys[(t, fd)] += abs(sh or 0)
        allsh[(t, fd)] += (abs(sh or 0) if code in ("S", "F", "A", "M")
                           else 0.0)

    def window(dct, t, d, days):
        """Sum a per-day dict over the trailing `days` calendar days."""
        from datetime import date, timedelta
        d0 = date.fromisoformat(d)
        tot = 0.0
        for k in range(days):
            tot += dct.get((t, (d0 - timedelta(days=k)).isoformat()), 0.0)
        return tot

    # ---- build the constructions, per date ----
    dates = sorted({d for t in series for d in series[t] if d >= args.start})
    dates = dates[::5]        # every 5th date: overlapping windows anyway
    print(f"{len(dates)} evaluation dates\n")

    names = ["net_shares_90d", "days_of_volume", "accel",
             "sell_only_dov", "net_dov"]
    ics = {k: [] for k in names}
    nulls = {k: [] for k in names}
    import random
    rnd = random.Random(7)

    for d in dates:
        obs = {k: [] for k in names}
        for t in series:
            r = fwd.get((t, d))
            if r is None:
                continue
            s90 = window(sells, t, d, 90)
            s7 = window(sells, t, d, 7)
            b90 = window(buys, t, d, 90)
            a = adv.get((t, d))
            if s90 == 0 and b90 == 0:
                continue          # no insider activity: no signal either way
            obs["net_shares_90d"].append((-(s90 - b90), r))
            if a:
                obs["days_of_volume"].append((-s90 / a, r))
                obs["sell_only_dov"].append((-s90 / a, r))
                obs["net_dov"].append((-(s90 - b90) / a, r))
            rate90 = s90 / 90.0
            if rate90 > 0:
                obs["accel"].append((-((s7 / 7.0) / rate90), r))
        for k in names:
            if len(obs[k]) >= args.min_names:
                v = spearman(obs[k])
                if v is not None:
                    ics[k].append(v)
                sh = [x[1] for x in obs[k]]
                rnd.shuffle(sh)
                vn = spearman([(obs[k][i][0], sh[i]) for i in range(len(sh))])
                if vn is not None:
                    nulls[k].append(vn)

    print(f"  {'construction':<18}{'dates':>7}{'mean IC':>10}{'NW t':>9}"
          f"{'null IC':>10}{'null t':>9}")
    for k in names:
        v = ics[k]
        if len(v) < 20:
            print(f"  {k:<18}{len(v):>7}   too few dates")
            continue
        t_ = newey_west_t(v, lag=H)
        nv = nulls[k]
        nt = newey_west_t(nv, lag=H) if len(nv) >= 20 else None
        print(f"  {k:<18}{len(v):>7}{st.mean(v):>+10.4f}"
              f"{(t_ if t_ else 0):>+9.2f}{st.mean(nv):>+10.4f}"
              f"{(nt if nt else 0):>+9.2f}")

    print(f"\n  Sign convention: all constructions are negated so that MORE "
          f"SELLING\n  gives a MORE NEGATIVE value. A POSITIVE IC therefore "
          f"means heavy selling\n  predicts LOWER forward returns -- the "
          f"hypothesis.\n")
    print("  The null column must be ~0. If a construction and its null are "
          "both\n  non-zero, the pipeline leaks and the result is void.\n")
    print("  Newey-West at the horizon lag on a per-date IC series is the "
          "correct test.\n  Pooled stock-date rows are not independent and "
          "have inflated t-stats by\n  10-20x in this project before. Treat "
          "|t| < 2.5 as unproven.")


if __name__ == "__main__":
    main()
