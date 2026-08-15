#!/usr/bin/env python3
"""
horizon_decay.py -- map the IC decay curve across holding horizons.

WHY (2026-08-15)
  Production only measures h=1/3/5, and rank-IC rises monotonically across them:
      h=1 +0.020   h=3 +0.040   h=5 +0.051   (80d window, prob_up, bucket-neutral
                                              retention 55% / 53% / 85%)
  That may be the LEFT EDGE of a curve, not a peak. Documented pattern: for many
  cross-sectional signals, t-stats increase monotonically toward the ~1-month
  mark as idiosyncratic daily noise diversifies away. This fund's one validated
  brick (short interest) peaks at h=40, not h=5.

  If IC peaks past h=5, the finding is not "h=5 works" -- it is that the book is
  rebalancing faster than the signal's information horizon. Rebalancing more
  frequently than the horizon adds noise without adding signal.

TWO MEASURES (Qian, Hua & Tilney -- they are different questions)
  HORIZON IC : corr(signal_t, cumulative return t -> t+h).  "How long to hold?"
  LAGGED IC  : corr(signal_t, return over period t+h-1 -> t+h).  "WHERE does the
               return accrue?"  A fast signal can show rising horizon-IC purely
               by dragging along its day-1 edge; lagged IC exposes that. If
               lagged IC is ~0 beyond day 2, the signal is fast and the rising
               horizon-IC is an accumulation artifact.

*** MANDATORY PARITY GATE ***
  raw_bars is UNADJUSTED (price_cache applies backward split adjustment on read;
  prices.db carries separate splits/splits_cache tables). Computing returns
  straight off raw_bars close would manufacture fake -50% moves on every 2:1
  split. So this script REFUSES to report anything until its own h=1/3/5 returns
  reproduce outcomes.actual_return within tolerance. If parity fails, the price
  source or adjustment is wrong and every downstream number would be garbage.

USAGE
  python scripts/horizon_decay.py                     # parity + full curve
  python scripts/horizon_decay.py --horizons 1,3,5,10,20,40,60
  python scripts/horizon_decay.py --sector-neutral --group-col bucket
  python scripts/horizon_decay.py --parity-only       # just run the gate
  python scripts/horizon_decay.py --csv decay.csv
"""
import argparse
import csv as _csv
import math
import os
import random
import sqlite3
import sys
from collections import defaultdict

ROOT = os.path.expanduser(os.environ.get("ML_QUANT_ROOT", "~/ML_Quant_Fund"))
ACC = os.path.join(ROOT, "accuracy.db")
PRICES = os.path.join(ROOT, "prices.db")
META = os.path.join(ROOT, "tickers_metadata.csv")
GROUP_CANDIDATES = ["sector", "bucket", "industry", "group", "tier"]
HLZ_BAR = 3.0


# ------------------------------------------------------------------ stats
def ranks(xs):
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    r = [0.0] * len(xs)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            r[order[k]] = avg
        i = j + 1
    return r


def pearson(a, b):
    n = len(a)
    if n < 3:
        return None
    ma, mb = sum(a) / n, sum(b) / n
    va = sum((x - ma) ** 2 for x in a)
    vb = sum((y - mb) ** 2 for y in b)
    if va <= 0 or vb <= 0:
        return None
    return sum((a[i] - ma) * (b[i] - mb) for i in range(n)) / math.sqrt(va * vb)


def spearman(a, b):
    return pearson(ranks(a), ranks(b))


def nw_tstat(xs, lag):
    n = len(xs)
    if n < 5:
        return None, None, None
    mu = sum(xs) / n
    e = [x - mu for x in xs]
    s = sum(v * v for v in e) / n
    for l in range(1, min(lag, n - 1) + 1):
        g = sum(e[t] * e[t + l] for t in range(n - l)) / n
        s += 2.0 * (1.0 - l / (lag + 1.0)) * g
    if s <= 0:
        return mu, None, None
    se = math.sqrt(s / n)
    return mu, se, mu / se


def block_bootstrap_ci(xs, block, B=1000, alpha=0.05, seed=17):
    """Moving-block bootstrap CI for the mean.

    BUG FIXED 2026-08-15: block came straight from the horizon, so h=60 on
    fewer than 60 dates gave starts = n-block+1 <= 0 and randrange() raised.
    Block is now capped at n//2 -- longer than half the series cannot
    resample meaningfully anyway."""
    n = len(xs)
    if n < 10 or block < 1:
        return None, None
    block = max(1, min(block, n // 2))
    rnd = random.Random(seed)
    nb = math.ceil(n / block)
    starts = n - block + 1
    means = []
    for _ in range(B):
        samp = []
        for _ in range(nb):
            s0 = rnd.randrange(starts)
            samp.extend(xs[s0:s0 + block])
        samp = samp[:n]
        means.append(sum(samp) / len(samp))
    means.sort()
    return means[int(alpha / 2 * B)], means[min(B - 1, int((1 - alpha / 2) * B))]


# ------------------------------------------------------------------ data
def cols(con, t):
    return [r[1] for r in con.execute(f'PRAGMA table_info("{t}")')]


def load_prices(pcon, table, dcol, ccol):
    """{ticker: ([dates asc], [closes])} -- adjusted series."""
    out = defaultdict(lambda: ([], []))
    for tk, d, c in pcon.execute(
            f'SELECT ticker, "{dcol}", "{ccol}" FROM "{table}" '
            f'WHERE "{ccol}" IS NOT NULL ORDER BY ticker, "{dcol}"'):
        ds, cs = out[tk.upper()]
        ds.append(d)
        cs.append(float(c))
    return out


def fwd_return(series, d, h):
    """Cumulative close-to-close return from date d to d+h trading days."""
    ds, cs = series
    lo, hi = 0, len(ds) - 1
    idx = -1
    while lo <= hi:
        mid = (lo + hi) // 2
        if ds[mid] == d:
            idx = mid
            break
        if ds[mid] < d:
            lo = mid + 1
        else:
            hi = mid - 1
    if idx < 0 or idx + h >= len(ds):
        return None
    p0, p1 = cs[idx], cs[idx + h]
    if p0 <= 0:
        return None
    return p1 / p0 - 1.0


def period_return(series, d, h):
    """Return over the SINGLE period t+h-1 -> t+h (for lagged IC)."""
    ds, cs = series
    lo, hi = 0, len(ds) - 1
    idx = -1
    while lo <= hi:
        mid = (lo + hi) // 2
        if ds[mid] == d:
            idx = mid
            break
        if ds[mid] < d:
            lo = mid + 1
        else:
            hi = mid - 1
    if idx < 0 or idx + h >= len(ds) or h < 1:
        return None
    p0, p1 = cs[idx + h - 1], cs[idx + h]
    if p0 <= 0:
        return None
    return p1 / p0 - 1.0


def load_groups(want=None):
    if not os.path.isfile(META):
        sys.exit(f"FATAL: --sector-neutral requested but {META} not found.")
    rows = list(_csv.reader(open(META, newline="")))
    hdr = [h.strip().lower() for h in rows[0]]
    if want:
        if want.lower() not in hdr:
            sys.exit(f"FATAL: --group-col '{want}' not in {META}. Columns: {hdr}")
        g = hdr.index(want.lower())
    else:
        g = next((hdr.index(c) for c in GROUP_CANDIDATES if c in hdr), None)
        if g is None:
            sys.exit(f"FATAL: no grouping column in {META}. Looked for "
                     f"{GROUP_CANDIDATES}; present: {hdr}")
    t = next((i for i, h in enumerate(hdr) if h in ("ticker", "symbol")), 0)
    out = {r[t].strip().upper(): (r[g].strip() or "UNKNOWN")
           for r in rows[1:] if r and len(r) > max(t, g) and r[t].strip()}
    print(f"# neutralising on '{hdr[g]}': {len(out)} tickers, "
          f"{len(set(out.values()))} groups")
    return out


def demean(recs, groups):
    b = defaultdict(list)
    for t, p, r in recs:
        b[groups.get(t, "UNKNOWN")].append((t, p, r))
    out = []
    for rows in b.values():
        if len(rows) < 2:
            continue
        mp = sum(x[1] for x in rows) / len(rows)
        mr = sum(x[2] for x in rows) / len(rows)
        out.extend((t, p - mp, r - mr) for t, p, r in rows)
    return out


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizons", default="1,2,3,5,10,15,20,40,60")
    ap.add_argument("--prob-col", default="prob_up")
    ap.add_argument("--days", type=int, default=365)
    ap.add_argument("--min-names", type=int, default=20)
    ap.add_argument("--no-common-dates", action="store_true",
                    help="allow each horizon its own date window. OFF by "
                         "default: different windows are different TIME "
                         "PERIODS, not different horizons.")
    ap.add_argument("--cert-tol-abs", type=float, default=0.0002,
                    help="absolute slack added to --cert-tol, scaled by the "
                         "return size. Stops penny stocks being excluded for "
                         "sub-cent rounding rather than real error.")
    ap.add_argument("--cert-tol", type=float, default=0.01,
                    help="max |diff| vs stored h=5 for a ticker-date to certify")
    ap.add_argument("--max-ret", type=float, default=2.0,
                    help="sanity cap: drop any computed return beyond +/-this")
    ap.add_argument("--max-exclusion", type=float, default=5.0,
                    help="refuse if more than this %% of rows fail certification")
    ap.add_argument("--parity-tol", type=float, default=0.002,
                    help="max MEAN |diff| vs outcomes.actual_return to pass")
    ap.add_argument("--parity-max", type=float, default=0.01,
                    help="max SINGLE |diff| allowed. A correct price source "
                         "reproduces outcomes to ~1e-6; a single 50%% error is a "
                         "split. Mean alone cannot detect rare catastrophic errors "
                         "-- one split in 400 tickers barely moves it.")
    ap.add_argument("--parity-only", action="store_true")
    ap.add_argument("--sector-neutral", action="store_true")
    ap.add_argument("--group-col")
    ap.add_argument("--include-watchlist", action="store_true")
    ap.add_argument("--bootstrap", type=int, default=1000)
    ap.add_argument("--shuffles", type=int, default=100)
    ap.add_argument("--csv")
    ap.add_argument("--root")
    args = ap.parse_args()

    global ROOT, ACC, PRICES, META
    if args.root:
        ROOT = os.path.expanduser(args.root)
        ACC = os.path.join(ROOT, "accuracy.db")
        PRICES = os.path.join(ROOT, "prices.db")
        META = os.path.join(ROOT, "tickers_metadata.csv")
    for p in (ACC, PRICES):
        if not os.path.isfile(p):
            sys.exit(f"FATAL: {p} not found")

    acon = sqlite3.connect(ACC, timeout=30)
    pcon = sqlite3.connect(PRICES, timeout=30)

    pred_cols = cols(acon, "predictions")
    low = {c.lower(): c for c in pred_cols}
    if args.prob_col.lower() not in low:
        sys.exit(f"FATAL: --prob-col '{args.prob_col}' not in predictions. "
                 f"Available: {', '.join(pred_cols)}")
    pcol = low[args.prob_col.lower()]
    wl = "" if args.include_watchlist or "is_watchlist" not in low \
        else " AND COALESCE(p.is_watchlist,0)=0"

    # ---- choose a price source, then PROVE it
    ptables = [r[0] for r in pcon.execute(
        "SELECT name FROM sqlite_master WHERE type='table'")]
    cand = []
    for t in ("daily_prices", "raw_bars"):
        if t in ptables:
            c = [x.lower() for x in cols(pcon, t)]
            dcol = next((x for x in ("d", "date", "dt", "bar_date") if x in c), None)
            ccol = next((x for x in ("adj_close", "close_adj", "close") if x in c), None)
            if dcol and ccol:
                cand.append((t, dcol, ccol))
    if not cand:
        sys.exit("FATAL: no usable price table found in prices.db")

    horizons = [int(x) for x in args.horizons.split(",")]
    groups = load_groups(args.group_col) if args.sector_neutral else {}

    print(f"# horizon_decay  prob='{pcol}'  lookback={args.days}d")
    print(f"# watchlist: {'INCLUDED' if args.include_watchlist else 'EXCLUDED'}")
    print()

    # ---- PER-ROW CERTIFICATION (replaces the global pass/fail gate) -------
    # v1.0 used ONE global gate: if any price source failed overall, nothing ran.
    # That blocked the whole study on ~1% of rows. Worse, it required diagnosing
    # every cause first (splits -> the CRWD seam -> dividends -> bad bars).
    #
    # v1.1 is CAUSE-AGNOSTIC. Every (ticker, date) must prove itself: recompute
    # the h=5 return from the price series and require it to reproduce the STORED
    # outcomes.actual_return. A row that certifies is trusted at longer horizons
    # too; a row that fails is dropped at ALL horizons -- no matter why it failed.
    #
    # h=1/3/5 are then taken STRAIGHT FROM outcomes (zero parity error by
    # construction, and directly comparable to production accuracy numbers).
    # Only h>5 is computed from prices.
    #
    # Known residual: adj_close and outcomes may sit on opposite sides of the
    # price-return / total-return distinction (WRDS supplies separate split and
    # dividend-reinvestment factors). Certification tolerance absorbs that; the
    # excluded set is REPORTED, never silent.
    print("=" * 74)
    print("PER-ROW CERTIFICATION -- each (ticker,date) must reproduce stored h=5")
    print(f"  tolerance = {args.cert_tol:.4f}   sanity cap = |ret| <= {args.max_ret:.1f}")
    print("=" * 74)

    truth5 = {}
    for tk, d, r in acon.execute(
            "SELECT ticker, prediction_date, actual_return FROM outcomes "
            "WHERE horizon=5 AND actual_return IS NOT NULL "
            "AND prediction_date >= date('now', ?)", (f"-{args.days} days",)):
        truth5[(tk.upper(), d)] = float(r)

    best = None
    for table, dcol, ccol in cand:
        px = load_prices(pcon, table, dcol, ccol)
        cert, fail = set(), defaultdict(int)
        for (tk, d), tv in truth5.items():
            s5 = px.get(tk)
            if not s5:
                continue
            mine = fwd_return(s5, d, 5)
            if mine is None:
                continue
            # Combined absolute + relative tolerance. A purely relative bound is
            # far stricter on penny stocks: VXRT trades $0.45-0.60, where a
            # fraction-of-a-cent price difference exceeds a 1% return tolerance
            # and the name is excluded 46 times for rounding, not corruption.
            tol = max(args.cert_tol, args.cert_tol_abs / max(abs(tv), 0.02))
            if abs(mine - tv) <= tol:
                cert.add((tk, d))
            else:
                fail[tk] += 1
        tot = len(cert) + sum(fail.values())
        rate = (sum(fail.values()) / tot * 100) if tot else 100.0
        print(f"  {table}.{ccol:<12} certified={len(cert):<6} excluded={sum(fail.values()):<5} "
              f"({rate:.2f}%)")
        if fail:
            top = sorted(fail.items(), key=lambda x: -x[1])[:6]
            print(f"      worst: {', '.join(f'{t}({n})' for t, n in top)}")
        if best is None or len(cert) > len(best[3]):
            best = (table, dcol, ccol, cert, px, rate)

    table, dcol, ccol, certified, px, exc_rate = best
    print()
    if exc_rate > args.max_exclusion:
        print(f"REFUSING: exclusion rate {exc_rate:.2f}% exceeds "
              f"--max-exclusion {args.max_exclusion:.1f}%.")
        print("That is not spotty data -- the price source is systematically wrong.")
        return 1
    print(f"Using {table}.{ccol}: {len(certified)} certified ticker-dates "
          f"({exc_rate:.2f}% excluded).")
    print("h=1/3/5 come from outcomes directly; h>5 computed from certified rows.\n")
    if args.parity_only:
        return 0

    # stored outcomes for the horizons production already scores
    stored = defaultdict(dict)
    for h in (1, 3, 5):
        for tk, d, r in acon.execute(
                "SELECT ticker, prediction_date, actual_return FROM outcomes "
                "WHERE horizon=? AND actual_return IS NOT NULL "
                "AND prediction_date >= date('now', ?)", (h, f"-{args.days} days")):
            stored[h][(tk.upper(), d)] = float(r)

    # ---- load signals -----------------------------------------------------
    sig = defaultdict(list)
    for d, tk, p in acon.execute(
            f'SELECT p.prediction_date, p.ticker, p."{pcol}" FROM predictions p '
            f'WHERE p."{pcol}" IS NOT NULL' + wl +
            f" AND p.horizon=1 AND p.prediction_date >= date('now', ?)",
            (f"-{args.days} days",)):
        sig[d].append((tk.upper(), float(p)))
    dates = sorted(d for d, v in sig.items() if len(v) >= args.min_names)
    acon.close()
    pcon.close()
    print(f"# {len(dates)} signal dates, {sum(len(sig[d]) for d in dates)} signals\n")

    # COMMON-DATE RESTRICTION. Without it each horizon uses a DIFFERENT
    # window: h=1 had 106 dates, h=40 only 72 -- and h=40's dates all end ~40
    # sessions EARLIER. That makes long-h cells a different TIME PERIOD, not a
    # longer horizon. Matters given the regime split found earlier. Same defect
    # fixed in validate_confidence_filter.py and not carried over here at first.
    if not args.no_common_dates and len(horizons) > 1:
        hmax = max(horizons)
        ok = set()
        for d in dates:
            n_ok = 0
            for tk, p in sig[d]:
                if (tk, d) not in certified:
                    continue
                s_ = px.get(tk)
                if s_ and fwd_return(s_, d, hmax) is not None:
                    n_ok += 1
                    if n_ok >= args.min_names:
                        break
            if n_ok >= args.min_names:
                ok.add(d)
        print(f'# common-date restriction: {len(dates)} -> {len(ok)} dates '
              f'(all horizons up to h={hmax} on the SAME dates)')
        if len(ok) < 10:
            print(f'#   WARNING: only {len(ok)} common dates -- drop the longest '
                  f'horizon, or pass --no-common-dates (cells NOT comparable).')
        dates = sorted(ok)
        print()

    hdr = (f"{'h':>4}{'dates':>7}{'horizonIC':>11}{'NW-t':>7}{'ICIR':>7}{'hit%':>6}"
           f"{'boot95':>20}{'null-σ':>8}{'laggedIC':>10}{'lag-t':>7}")
    print(hdr)
    print("-" * len(hdr))

    rows_out = []
    for h in horizons:
        hics, lics = [], []
        for d in dates:
            recs, lrecs = [], []
            for tk, p in sig[d]:
                if (tk, d) not in certified:
                    continue                      # failed certification -> drop
                if h in (1, 3, 5):
                    fr = stored[h].get((tk, d))   # production ground truth
                else:
                    s = px.get(tk)
                    fr = fwd_return(s, d, h) if s else None
                if fr is not None and abs(fr) <= args.max_ret:
                    recs.append((tk, p, fr))
                s = px.get(tk)
                pr = period_return(s, d, h) if s else None
                if pr is not None and abs(pr) <= args.max_ret:
                    lrecs.append((tk, p, pr))
            if groups:
                recs = demean(recs, groups)
                lrecs = demean(lrecs, groups)
            if len(recs) >= args.min_names:
                ic = spearman([x[1] for x in recs], [x[2] for x in recs])
                if ic is not None:
                    hics.append(ic)
            if len(lrecs) >= args.min_names:
                lic = spearman([x[1] for x in lrecs], [x[2] for x in lrecs])
                if lic is not None:
                    lics.append(lic)
        if len(hics) < 10:
            print(f"{h:>4}{len(hics):>7}   too few dates")
            continue
        mu, se, t = nw_tstat(hics, h)
        sd = math.sqrt(sum((x - mu) ** 2 for x in hics) / (len(hics) - 1)) if len(hics) > 1 else 0
        icir = mu / sd if sd > 0 else float("nan")
        blo, bhi = block_bootstrap_ci(hics, max(1, h), B=args.bootstrap) if args.bootstrap else (None, None)
        lmu, lse, lt = nw_tstat(lics, 1) if len(lics) >= 10 else (None, None, None)

        nz = None
        if args.shuffles and len(hics) >= 10:
            rnd = random.Random(23)
            nulls = []
            for _ in range(args.shuffles):
                acc = []
                for d in dates:
                    recs = []
                    for tk, p in sig[d]:
                        if (tk, d) not in certified:
                            continue
                        if h in (1, 3, 5):
                            fr = stored[h].get((tk, d))
                        else:
                            s = px.get(tk)
                            fr = fwd_return(s, d, h) if s else None
                        if fr is not None and abs(fr) <= args.max_ret:
                            recs.append((tk, p, fr))
                    if len(recs) < args.min_names:
                        continue
                    rr = [x[2] for x in recs]
                    rnd.shuffle(rr)
                    ic = spearman([x[1] for x in recs], rr)
                    if ic is not None:
                        acc.append(ic)
                if acc:
                    nulls.append(sum(acc) / len(acc))
            if len(nulls) > 2:
                nm = sum(nulls) / len(nulls)
                nsd = math.sqrt(sum((x - nm) ** 2 for x in nulls) / (len(nulls) - 1))
                nz = (mu - nm) / nsd if nsd > 0 else None

        bs = f"[{blo:+.4f},{bhi:+.4f}]" if blo is not None else "n/a"
        ts = f"{t:+.2f}" if t is not None else "n/a"
        ls = f"{lmu:+.5f}" if lmu is not None else "n/a"
        lts = f"{lt:+.2f}" if lt is not None else "n/a"
        nzs = f"{nz:+.2f}" if nz is not None else "n/a"
        hit = 100.0 * sum(1 for x in hics if x > 0) / len(hics)
        print(f"{h:>4}{len(hics):>7}{mu:>+11.5f}{ts:>7}{icir:>7.3f}{hit:>6.1f}"
              f"{bs:>20}{nzs:>8}{ls:>10}{lts:>7}")
        rows_out.append([h, len(hics), mu, t, icir, hit, blo, bhi, nz, lmu, lt])

    print()
    print("HOW TO READ")
    print("  horizonIC rising then flattening -> hold to the flattening point;")
    print("    rebalancing faster than that adds noise without adding signal.")
    print("  laggedIC ~0 beyond day 1-2 while horizonIC keeps rising -> the")
    print("    signal is FAST and the horizon curve is an accumulation artifact,")
    print("    NOT evidence for a longer hold.")
    print("  hit% = share of dates with positive IC (standard IC reporting:")
    print("    mean IC, t, sd, IR, hit rate, n). ~50% with a positive mean means")
    print("    the average is carried by a few dates, not consistent skill.")
    print("  ICIR (mean IC / sd IC) is the stability diagnostic -- a high IC with")
    print("    low ICIR is a signal that works occasionally, not consistently.")
    print(f"  NW lag = h (overlapping returns are MA(h-1)); NW OVER-REJECTS, so t")
    print(f"    is an UPPER BOUND. Bar remains |t|>{HLZ_BAR:.0f} + bootstrap excluding 0.")
    print("  Longer h = heavier overlap = fewer independent blocks. The SHAPE of")
    print("    the curve is the deliverable; individual long-h cells are weak.")

    if args.csv:
        with open(args.csv, "w", newline="") as f:
            w = _csv.writer(f)
            w.writerow(["horizon", "n_dates", "horizon_ic", "nw_t", "icir", "hit_pct",
                        "boot_lo", "boot_hi", "null_sigma", "lagged_ic", "lagged_t"])
            w.writerows(rows_out)
        print(f"\n# wrote {args.csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
