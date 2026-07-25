#!/usr/bin/env python3
"""Dark-pool signed-skew validation gate (v1.0, spec 2026-07-26).

QUESTION: does RTH NBBO-signed dark-pool skew on day t predict forward
returns t+1 (and t+1..t+5) cross-sectionally?

DECISION RULE (pre-registered, HLZ bar):
  - |null-z| >= 3 AND |NW-t| >= 3 on BOTH horizons -> "SIGNAL CANDIDATE"
    (still only: keep accumulating, re-test; NOT production)
  - n_dates < 20                                   -> "UNDERPOWERED"
  - else                                           -> "DEAD"

Offline: reads earnings_monitor.db + prices.db only. No network.
Audit-style: prints its arithmetic, runs the shuffle null (mandatory).
Signal is pluggable via --signal for future streams (aggressor tilt etc.).

Method notes:
  - skew(t,ticker) = (B-S)/(B+S), B/S from NBBO-midpoint signing of RTH
    prints only (ext_hours empty), rows with valid nbbo_bid<=nbbo_ask only.
    NO VWAP fallback here (validation purity; monitor display differs).
  - usability gates mirror the monitor: classified >= $5M, >= 20 classified
    prints, coverage >= 50% of NBBO-priced RTH value.
  - per-date cross-sectional Spearman IC, >= MIN_NAMES names per date.
  - Newey-West t on the IC series, lag = h-1 (overlap correction).
  - null: shuffle forward returns ACROSS names WITHIN each date, N_SHUFFLE
    times -> distribution of mean IC; report real mean IC in sigma units.
  - pooled Spearman also printed but labeled INFLATED (the t=-20 lesson:
    ~persistent signals across overlapping dates are not independent rows).
"""
import argparse
import random
import sqlite3
import statistics
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

DEFAULT_TICKERS = ["MSFT", "GOOG", "AMZN", "BYND", "NVDA", "PLTR", "SMCI",
                   "DDOG", "OKLO", "QUBT", "CRWD", "SNOW", "NVMI"]
SIGNING_FIX_DATE = "2026-07-14"   # NBBO-midpoint signing ships; earlier skew unreliable
MIN_CLASSIFIED_USD = 5_000_000
MIN_CLASSIFIED_PRINTS = 20
MIN_COVERAGE_PCT = 50.0
MIN_NAMES_PER_DATE = 6
N_SHUFFLE = 1000
HLZ_T = 3.0


def rankdata(xs):
    """Average-rank ties, 1-based. Manual to avoid scipy dependency."""
    idx = sorted(range(len(xs)), key=lambda i: xs[i])
    ranks = [0.0] * len(xs)
    i = 0
    while i < len(idx):
        j = i
        while j + 1 < len(idx) and xs[idx[j + 1]] == xs[idx[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[idx[k]] = avg
        i = j + 1
    return ranks


def spearman(x, y):
    if len(x) < 3:
        return None
    rx, ry = rankdata(x), rankdata(y)
    mx, my = statistics.fmean(rx), statistics.fmean(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    dx = sum((a - mx) ** 2 for a in rx) ** 0.5
    dy = sum((b - my) ** 2 for b in ry) ** 0.5
    if dx == 0 or dy == 0:
        return None
    return num / (dx * dy)


def newey_west_t(series, lag):
    """NW t-stat of the mean of `series` with `lag` autocovariance terms."""
    n = len(series)
    if n < 3:
        return None, None
    mu = statistics.fmean(series)
    e = [s - mu for s in series]
    g0 = sum(v * v for v in e) / n
    var = g0
    for l in range(1, min(lag, n - 1) + 1):
        gl = sum(e[i] * e[i - l] for i in range(l, n)) / n
        var += 2.0 * (1.0 - l / (lag + 1.0)) * gl
    if var <= 0:
        return mu, None
    se = (var / n) ** 0.5
    return mu, mu / se


def load_skew(con, tickers, since):
    """(date -> ticker -> skew), with gate bookkeeping printed."""
    out = defaultdict(dict)
    dropped = defaultdict(int)
    for tkr in tickers:
        rows = con.execute(
            "SELECT et_date, price, value_usd, nbbo_bid, nbbo_ask "
            "FROM darkpool_prints WHERE ticker=? AND et_date>=? "
            "AND (ext_hours IS NULL OR ext_hours='') AND canceled=0",
            (tkr, since)).fetchall()
        byday = defaultdict(lambda: {"b": 0.0, "s": 0.0, "n": 0, "neu": 0.0})
        for et, px, val, bid, ask in rows:
            if not et or px is None or not val:
                continue
            if bid is None or ask is None or ask < bid or bid <= 0:
                continue  # NBBO-signable only; no VWAP fallback in validation
            mid = (bid + ask) / 2.0
            d = byday[et]
            if px > mid:
                d["b"] += val; d["n"] += 1
            elif px < mid:
                d["s"] += val; d["n"] += 1
            else:
                d["neu"] += val
        for et, d in byday.items():
            cls = d["b"] + d["s"]
            tot = cls + d["neu"]
            cov = (cls / tot * 100) if tot else 0.0
            if (cls >= MIN_CLASSIFIED_USD and d["n"] >= MIN_CLASSIFIED_PRINTS
                    and cov >= MIN_COVERAGE_PCT):
                out[et][tkr] = (d["b"] - d["s"]) / cls
            else:
                dropped[tkr] += 1
    return out, dropped


def load_prices(tickers):
    con = sqlite3.connect(f"file:{ROOT / 'prices.db'}?mode=ro", uri=True)
    px, missing = {}, []
    for tkr in tickers:
        rows = con.execute(
            "SELECT date, adj_close FROM daily_prices WHERE ticker=? "
            "AND adj_close IS NOT NULL ORDER BY date", (tkr,)).fetchall()
        if not rows:
            missing.append(tkr)
            continue
        px[tkr] = {str(d)[:10]: float(c) for d, c in rows}
    return px, missing


def fwd_return(px_t, dates_t, d, h):
    if d not in px_t:
        return None
    import bisect
    i = bisect.bisect_left(dates_t, d)
    if i >= len(dates_t) or dates_t[i] != d or i + h >= len(dates_t):
        return None
    return px_t[dates_t[i + h]] / px_t[d] - 1.0


def run(signal_name, skew_by_date, px, horizon, rng):
    ics, per_date_rows, pooled_x, pooled_y = [], [], [], []
    date_pairs = []
    for d in sorted(skew_by_date):
        xs, ys = [], []
        for tkr, sk in skew_by_date[d].items():
            if tkr not in px:
                continue
            dates_t = sorted(px[tkr])
            r = fwd_return(px[tkr], dates_t, d, horizon)
            if r is None:
                continue
            xs.append(sk); ys.append(r)
        if len(xs) >= MIN_NAMES_PER_DATE:
            ic = spearman(xs, ys)
            if ic is not None:
                ics.append(ic)
                per_date_rows.append((d, len(xs), ic))
                pooled_x += xs; pooled_y += ys
                date_pairs.append((xs, ys))
    if not ics:
        return None
    mu, nwt = newey_west_t(ics, lag=max(horizon - 1, 0))
    # ---- mandatory null: shuffle fwd returns within each date ----
    null_means = []
    for _ in range(N_SHUFFLE):
        sh = []
        for xs, ys in date_pairs:
            yy = ys[:]
            rng.shuffle(yy)
            ic = spearman(xs, yy)
            if ic is not None:
                sh.append(ic)
        if sh:
            null_means.append(statistics.fmean(sh))
    nmu = statistics.fmean(null_means)
    nsd = statistics.stdev(null_means) if len(null_means) > 2 else float("nan")
    z = (mu - nmu) / nsd if nsd and nsd == nsd and nsd > 0 else None
    pooled = spearman(pooled_x, pooled_y)
    return {"h": horizon, "n_dates": len(ics), "per_date": per_date_rows,
            "mean_ic": mu, "nw_t": nwt, "null_mu": nmu, "null_sd": nsd,
            "z": z, "pooled": pooled, "pooled_n": len(pooled_x)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--signal", default="skew", choices=["skew"],
                    help="signal stream (registry; aggressor tilt needs persistence first)")
    ap.add_argument("--since", default=SIGNING_FIX_DATE)
    ap.add_argument("--db", default=str(ROOT / "earnings_monitor.db"))
    ap.add_argument("--tickers", nargs="*", default=DEFAULT_TICKERS)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    rng = random.Random(a.seed)

    print("=" * 74)
    print(f"  DARK-POOL SIGNED-SKEW VALIDATION GATE  (signal={a.signal}, "
          f"since {a.since}, seed {a.seed})")
    print("=" * 74)
    con = sqlite3.connect(f"file:{a.db}?mode=ro", uri=True)
    skew_by_date, dropped = load_skew(con, a.tickers, a.since)
    px, missing = load_prices(a.tickers)
    if missing:
        print(f"  [gap] no prices.db history for: {', '.join(missing)} "
              f"-- excluded (finding, not silently filled)")
    n_days = len(skew_by_date)
    print(f"  Usable skew days: {n_days}; per-ticker gate-dropped days: "
          f"{dict(sorted(dropped.items())) or 'none'}")
    for h in (1, 5):
        res = run(a.signal, skew_by_date, px, h, rng)
        print("-" * 74)
        if not res:
            print(f"  h={h}: no dates with >= {MIN_NAMES_PER_DATE} names -- nothing to test")
            continue
        _nwt = f"{res['nw_t']:+.2f}" if res["nw_t"] is not None else "n/a"
        print(f"  h={h}  dates={res['n_dates']}  "
              f"mean per-date IC = {res['mean_ic']:+.4f}  NW-t = {_nwt}")
        print(f"       null: mean {res['null_mu']:+.4f} sd {res['null_sd']:.4f} "
              f"({N_SHUFFLE} shuffles)  ->  real IC z = "
              + (f"{res['z']:+.2f} sigma" if res['z'] is not None else "n/a"))
        print(f"       pooled Spearman {res['pooled']:+.4f} on n={res['pooled_n']} "
              f"[INFLATED -- non-independent, shown for audit only]")
        print(f"       per-date: " + "  ".join(
            f"{d}(n{n}){ic:+.2f}" for d, n, ic in res["per_date"]))
    print("=" * 74)
    print("  VERDICT RULE (pre-registered):")
    print(f"    SIGNAL CANDIDATE iff |z|>={HLZ_T} AND |NW-t|>={HLZ_T} on BOTH horizons")
    print(f"    UNDERPOWERED    iff n_dates < 20 (expected outcome at this window)")
    print(f"    DEAD            otherwise")
    print("  Candidate != validated: it buys a monthly re-run, nothing else.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
