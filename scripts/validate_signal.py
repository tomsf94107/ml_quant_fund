#!/usr/bin/env python3
"""
validate_signal.py -- run ANY (ticker, date, value) signal through the full gate.

WHY THIS EXISTS (2026-08-21)
  Four validators already exist and each is welded to ONE source:
    validate_confidence_filter.py  -> predictions.prob_up
    horizon_decay.py               -> predictions.prob_up
    decile_monotonicity.py         -> predictions.prob_up
    alpha_gate*.py                 -> the exploded alpha panel
  So testing a new information axis means writing a new script, and the gate
  drifts between copies. The options-Greeks table (110,321 rows, loaded
  2026-08-20) could not be tested by any of them.

  This takes the signal from ANY table and applies the whole stack once. Every
  future axis -- vanna, charm, VRP, FTDs, 13F flow, insider -- becomes a
  one-line test rather than a new script.

THE STACK (each piece exists because something broke without it)
  per-date rank-IC       a DATE is the unit, never a stock-date row. Pooling
                         rows produced a fake t=-20 on the SI brick
                         (Two_Brick_Findings 5.1) and a fake +0.96% decile
                         spread on 2026-08-21. Equal-weight per date, always.
  Newey-West, lag = h    overlapping h-day returns are MA(h-1). NW OVER-REJECTS
                         under strong serial correlation (den Haan & Levin 1997),
                         so the reported t is an UPPER BOUND.
  block bootstrap        block = h, mirroring the NW lag; less size-distorted
                         than NW at small n.
  shuffle null           LEAKAGE CHECK ONLY. Anti-conservative, ignores regime
                         variance, NOT part of the bar.
  decile monotonicity    Spearman(decile index, decile mean return). Separates
                         tradeable IC from mid-book IC. alpha_fitness has 1,222
                         alphas at |t|>3 whose top 20 show IC +0.04 with Sharpe
                         -0.51 -- none were ever tested for this.
  net of cost            measured turnover on NON-OVERLAPPING rebalances. A
                         gross edge of t=+4.87 became t=-3.28 net of 10bps in
                         commit 75ae24a9. Cost is frequently the answer.
  group-neutral          demean within bucket from tickers_metadata.csv. The SI
                         brick retained 80%; the confidence filter 12%.
  BH-FDR                 every horizon x neutralisation is a trial.

  BAR: NW |t| > 3 AND bootstrap excludes 0 AND net-of-cost positive
       AND monotonicity >= +0.3 AND BH-FDR survival.

USAGE
  # options Greeks, trailing 60d z-score per ticker
  python scripts/validate_signal.py --db accuracy.db --table options_greeks \\
      --value-col net_gamma --zscore 60 --horizons 1,3,5,10,20

  # any other table
  python scripts/validate_signal.py --db earnings_monitor.db --table darkpool_prints \\
      --value-col signed_notional --date-col executed_at --agg sum

  # raw values, no transform, bucket-neutral
  python scripts/validate_signal.py --db accuracy.db --table options_greeks \\
      --value-col net_vanna --sector-neutral
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
PRICES = os.path.join(ROOT, "prices.db")
META = os.path.join(ROOT, "tickers_metadata.csv")
GROUP_COLS = ("sector", "bucket", "industry", "group", "tier")
HLZ = 3.0


# ---------------------------------------------------------------- stats
def ranks(xs):
    o = sorted(range(len(xs)), key=lambda i: xs[i])
    r = [0.0] * len(xs)
    i = 0
    while i < len(o):
        j = i
        while j + 1 < len(o) and xs[o[j + 1]] == xs[o[i]]:
            j += 1
        av = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            r[o[k]] = av
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


def nw(xs, lag):
    n = len(xs)
    if n < 5:
        return None, None
    mu = sum(xs) / n
    e = [x - mu for x in xs]
    s = sum(v * v for v in e) / n
    for l in range(1, min(lag, n - 1) + 1):
        s += 2.0 * (1.0 - l / (lag + 1.0)) * sum(e[t] * e[t + l] for t in range(n - l)) / n
    if s <= 0:
        return mu, None
    return mu, mu / math.sqrt(s / n)


def boot(xs, block, B=1000, seed=17):
    n = len(xs)
    if n < 10:
        return None, None
    block = max(1, min(block, n // 2))
    rnd = random.Random(seed)
    nb = math.ceil(n / block)
    starts = n - block + 1
    ms = []
    for _ in range(B):
        s = []
        for _ in range(nb):
            i = rnd.randrange(starts)
            s.extend(xs[i:i + block])
        s = s[:n]
        ms.append(sum(s) / len(s))
    ms.sort()
    return ms[int(0.025 * B)], ms[min(B - 1, int(0.975 * B))]


def norm_p(z):
    return 2.0 * (1.0 - 0.5 * (1.0 + math.erf(abs(z) / math.sqrt(2.0))))


def bh(ps, q=0.05):
    idx = sorted(range(len(ps)), key=lambda i: ps[i])
    m = len(ps)
    kmax = -1
    for rank, i in enumerate(idx, 1):
        if ps[i] <= q * rank / m:
            kmax = rank
    return {i for rank, i in enumerate(idx, 1) if rank <= kmax}


# ---------------------------------------------------------------- data
def load_groups():
    if not os.path.isfile(META):
        sys.exit(f"FATAL: --sector-neutral needs {META}")
    rows = list(_csv.reader(open(META, newline="")))
    hdr = [h.strip().lower() for h in rows[0]]
    g = next((hdr.index(c) for c in GROUP_COLS if c in hdr), None)
    if g is None:
        sys.exit(f"FATAL: no grouping column in {META}; looked for {GROUP_COLS}")
    t = next((i for i, h in enumerate(hdr) if h in ("ticker", "symbol")), 0)
    out = {r[t].strip().upper(): (r[g].strip() or "UNKNOWN")
           for r in rows[1:] if r and len(r) > max(t, g) and r[t].strip()}
    print(f"# neutralising on '{hdr[g]}': {len(out)} tickers, "
          f"{len(set(out.values()))} groups")
    return out


def demean(recs, groups):
    b = defaultdict(list)
    for t, s, r in recs:
        b[groups.get(t, "UNKNOWN")].append((t, s, r))
    out = []
    for rows in b.values():
        if len(rows) < 2:
            continue
        ms = sum(x[1] for x in rows) / len(rows)
        mr = sum(x[2] for x in rows) / len(rows)
        out.extend((t, s - ms, r - mr) for t, s, r in rows)
    return out


def load_prices():
    con = sqlite3.connect(PRICES, timeout=30)
    px = defaultdict(lambda: ([], []))
    for t, d, c in con.execute(
            "SELECT ticker,d,close FROM raw_bars WHERE close>0 ORDER BY ticker,d"):
        ds, cs = px[t.upper()]
        ds.append(d)
        cs.append(float(c))
    con.close()
    return px


def fwd(series, d, h):
    ds, cs = series
    lo, hi = 0, len(ds) - 1
    i = -1
    while lo <= hi:
        m = (lo + hi) // 2
        if ds[m] == d:
            i = m
            break
        if ds[m] < d:
            lo = m + 1
        else:
            hi = m - 1
    if i < 0 or i + h >= len(cs) or cs[i] <= 0:
        return None
    return cs[i + h] / cs[i] - 1.0


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True)
    ap.add_argument("--table", required=True)
    ap.add_argument("--value-col", required=True)
    ap.add_argument("--ticker-col", default="ticker")
    ap.add_argument("--date-col", default="date")
    ap.add_argument("--agg", default="last", choices=["last", "sum", "mean"],
                    help="collapse multiple rows per ticker-date")
    ap.add_argument("--zscore", type=int, default=0,
                    help="trailing N-day z-score per ticker (0 = raw). TRAILING "
                         "ONLY -- a full-sample z-score leaks the future.")
    ap.add_argument("--horizons", default="1,3,5,10,20")
    ap.add_argument("--days", type=int, default=400)
    ap.add_argument("--min-names", type=int, default=30)
    ap.add_argument("--deciles", type=int, default=10)
    ap.add_argument("--cost-bps", type=float, default=10.0)
    ap.add_argument("--sector-neutral", action="store_true")
    ap.add_argument("--shuffles", type=int, default=100)
    ap.add_argument("--bootstrap", type=int, default=1000)
    ap.add_argument("--csv")
    ap.add_argument("--root")
    args = ap.parse_args()

    global ROOT, PRICES, META
    if args.root:
        ROOT = os.path.expanduser(args.root)
        PRICES = os.path.join(ROOT, "prices.db")
        META = os.path.join(ROOT, "tickers_metadata.csv")
    dbp = args.db if os.path.isabs(args.db) else os.path.join(ROOT, args.db)
    if not os.path.isfile(dbp):
        sys.exit(f"FATAL: {dbp} not found")

    con = sqlite3.connect(dbp, timeout=30)
    cols = [r[1] for r in con.execute(f'PRAGMA table_info("{args.table}")')]
    if not cols:
        sys.exit(f"FATAL: table '{args.table}' not found in {dbp}")
    for c in (args.ticker_col, args.date_col, args.value_col):
        if c not in cols:
            sys.exit(f"FATAL: column '{c}' not in {args.table}. Have: {', '.join(cols)}")

    raw = con.execute(
        f'SELECT UPPER("{args.ticker_col}"), substr("{args.date_col}",1,10), '
        f'"{args.value_col}" FROM "{args.table}" '
        f'WHERE "{args.value_col}" IS NOT NULL '
        f'  AND substr("{args.date_col}",1,10) >= date(\'now\', ?) '
        f'ORDER BY 1,2', (f"-{args.days} days",)).fetchall()
    con.close()
    if not raw:
        sys.exit("FATAL: no rows in range")

    cell = defaultdict(list)
    for t, d, v in raw:
        try:
            cell[(t, d)].append(float(v))
        except (TypeError, ValueError):
            pass
    sig = {}
    for k, vs in cell.items():
        sig[k] = vs[-1] if args.agg == "last" else (
            sum(vs) if args.agg == "sum" else sum(vs) / len(vs))

    print(f"# validate_signal  {args.table}.{args.value_col}  db={os.path.basename(dbp)}")
    print(f"# {len(raw)} rows -> {len(sig)} ticker-dates, "
          f"{len({k[0] for k in sig})} tickers, {len({k[1] for k in sig})} dates")
    print(f"# transform: {'trailing %dd z-score' % args.zscore if args.zscore else 'raw'}"
          f"   agg={args.agg}")

    if args.zscore:
        per = defaultdict(list)
        for (t, d), v in sig.items():
            per[t].append((d, v))
        z = {}
        for t, rows in per.items():
            rows.sort()
            vals = [v for _, v in rows]
            w = args.zscore
            for i in range(w, len(rows)):
                win = vals[i - w:i]
                m = sum(win) / w
                s = math.sqrt(sum((x - m) ** 2 for x in win) / (w - 1))
                if s > 0:
                    z[(t, rows[i][0])] = (vals[i] - m) / s
        print(f"# z-score cost {len(sig) - len(z)} ticker-dates of warmup")
        sig = z
    if not sig:
        sys.exit("FATAL: no signal left after transform")

    groups = load_groups() if args.sector_neutral else {}
    px = load_prices()
    horizons = [int(x) for x in args.horizons.split(",")]
    print()

    hdr = (f"{'h':>3}{'dates':>7}{'rank-IC':>10}{'IC-t':>6}{'mono':>6}"
           f"{'spread%':>9}{'sp-t':>6}{'spBoot':>17}{'topD%':>8}"
           f"{'net%':>8}{'nreb':>6}")
    print(hdr)
    print("-" * len(hdr))
    res = []
    for h in horizons:
        by_date = defaultdict(list)
        for (t, d), v in sig.items():
            s = px.get(t)
            if not s:
                continue
            r = fwd(s, d, h)
            if r is not None and abs(r) <= 2.0:
                by_date[d].append((t, v, r))
        dates = sorted(d for d, v in by_date.items() if len(v) >= args.min_names)
        if len(dates) < 10:
            print(f"{h:>3}{len(dates):>7}   too few dates")
            continue

        ics, spreads, hi_sets, decs = [], [], [], defaultdict(list)
        for d in dates:
            recs = by_date[d]
            if groups:
                recs = demean(recs, groups)
                if len(recs) < args.min_names:
                    continue
            ic = spearman([x[1] for x in recs], [x[2] for x in recs])
            if ic is not None:
                ics.append(ic)
            sr = sorted(recs, key=lambda x: x[1])
            n = len(sr)
            D = args.deciles
            per_d = [[] for _ in range(D)]
            for i, (_t, _s, r) in enumerate(sr):
                per_d[min(D - 1, i * D // n)].append(r)
            if per_d[0] and per_d[-1]:
                for k, b in enumerate(per_d):
                    if b:
                        decs[k].append(sum(b) / len(b))
                spreads.append(sum(per_d[-1]) / len(per_d[-1])
                               - sum(per_d[0]) / len(per_d[0]))
                k9 = n * (D - 1) // D
                hi_sets.append(frozenset(x[0] for x in sr[k9:]))
        if len(ics) < 10:
            print(f"{h:>3}{len(ics):>7}   too few IC dates")
            continue

        mu, t = nw(ics, h)
        hit = 100.0 * sum(1 for x in ics if x > 0) / len(ics)
        dm = [sum(decs[k]) / len(decs[k]) for k in sorted(decs) if decs[k]]
        mono = spearman(list(range(1, len(dm) + 1)), dm) if len(dm) >= 5 else None
        blo, bhi = boot(ics, h, B=args.bootstrap) if args.bootstrap else (None, None)

        nz = None
        if args.shuffles:
            rnd = random.Random(23)
            nulls = []
            for _ in range(args.shuffles):
                acc = []
                for d in dates:
                    recs = by_date[d]
                    rr = [x[2] for x in recs]
                    rnd.shuffle(rr)
                    ic = spearman([x[1] for x in recs], rr)
                    if ic is not None:
                        acc.append(ic)
                if acc:
                    nulls.append(sum(acc) / len(acc))
            if len(nulls) > 2:
                nm = sum(nulls) / len(nulls)
                ns = math.sqrt(sum((x - nm) ** 2 for x in nulls) / (len(nulls) - 1))
                nz = (mu - nm) / ns if ns > 0 else None

        # SPREAD statistics -- the high-confidence book. Already collected; this
        # only reports what the stack was computing. Adds NO new cell to the
        # multiple-testing count: it is the same pre-registered decile test.
        # rank-IC uses the FULL cross-section; the spread uses only D10-D1, i.e.
        # the concentrated book you would actually hold. When monotonicity is
        # high the two nearly coincide; when it is low they diverge, and the
        # divergence is the point.
        sp_mu, sp_t = nw(spreads, h) if len(spreads) >= 10 else (None, None)
        sp_lo, sp_hi = boot(spreads, h, B=args.bootstrap) if (args.bootstrap and len(spreads) >= 10) else (None, None)
        top_mu = (sum(decs[args.deciles - 1]) / len(decs[args.deciles - 1])
                  if decs.get(args.deciles - 1) else None)

        # net of cost on NON-OVERLAPPING rebalances
        idx = list(range(0, len(spreads), max(1, h)))
        net = nrb = None
        if len(idx) >= 5:
            prev = None
            nets = []
            for i in idx:
                cur = hi_sets[i]
                to = 1.0 if prev is None or not cur else len(cur ^ prev) / max(len(cur | prev), 1)
                prev = cur
                nets.append(spreads[i] - to * args.cost_bps / 10000.0)
            net = sum(nets) / len(nets)
            nrb = len(nets)

        ts = f"{t:+.2f}" if t is not None else "n/a"
        ms = f"{mono:+.2f}" if mono is not None else "n/a"
        sps = f"{sp_mu*100:+.4f}" if sp_mu is not None else "n/a"
        spt = f"{sp_t:+.2f}" if sp_t is not None else "n/a"
        spb = f"[{sp_lo*100:+.3f},{sp_hi*100:+.3f}]" if sp_lo is not None else "n/a"
        tds = f"{top_mu*100:+.4f}" if top_mu is not None else "n/a"
        ns_ = f"{net*100:+.4f}" if net is not None else "n/a"
        print(f"{h:>3}{len(ics):>7}{mu:>+10.5f}{ts:>6}{ms:>6}{sps:>9}{spt:>6}"
              f"{spb:>17}{tds:>8}{ns_:>8}{str(nrb or '-'):>6}")
        res.append(dict(h=h, n=len(ics), ic=mu, t=t, hit=hit, mono=mono,
                        blo=blo, bhi=bhi, nullz=nz, net=net, nrebal=nrb,
                        sp_mu=sp_mu, sp_t=sp_t, sp_lo=sp_lo, sp_hi=sp_hi,
                        top_mu=top_mu,
                        p=norm_p(t) if t is not None else 1.0))

    if not res:
        print("\n# nothing measurable")
        return 0

    print()
    print("  rank-IC/IC-t : FULL cross-section.  spread/sp-t : D10-D1, the")
    print("  concentrated book you would hold.  topD% : top decile alone (raw,")
    print("  includes beta).  net% : spread net of measured turnover cost.")
    print("  hit% and null-sigma are in the CSV; null-sigma is a LEAKAGE check.")
    keep = bh([r["p"] for r in res])
    print()
    print("=" * 74)
    print(f"BAR: NW |t|>{HLZ:.0f} AND bootstrap excludes 0 AND net>0 AND mono>=+0.30 AND BH-FDR")
    print(f"  cells={len(res)}  Bonferroni p<{0.05/len(res):.4f}  BH survivors={len(keep)}")
    ok = [r for i, r in enumerate(res)
          if r["t"] is not None and abs(r["t"]) > HLZ
          and r["blo"] is not None and (r["blo"] > 0 or r["bhi"] < 0)
          and r["net"] is not None and r["net"] > 0
          and r["mono"] is not None and r["mono"] >= 0.30
          and i in keep]
    print()
    if ok:
        print(f"!! {len(ok)} cell(s) CLEAR THE BAR:")
        for r in ok:
            print(f"   h={r['h']} IC={r['ic']:+.5f} t={r['t']:+.2f} mono={r['mono']:+.2f} "
                  f"net={r['net']*100:+.4f}%/{r['h']}d on {r['nrebal']} rebalances")
        print("   Before believing: check episode clustering, regime dependence,")
        print("   and re-run on a cold holdout. n_rebalances is the honest sample.")
    else:
        print("VERDICT: no cell clears the bar. NOT demonstrated to have edge.")
        print("  NW over-rejects, so the reported t is an UPPER BOUND.")
        print("  null-sigma is a LEAKAGE check only -- never read it as a bar.")

    if args.csv:
        with open(args.csv, "w", newline="") as f:
            w = _csv.writer(f)
            w.writerow(["horizon", "n_dates", "rank_ic", "nw_t", "hit_pct", "mono",
                        "boot_lo", "boot_hi", "null_sigma", "net", "n_rebal",
                        "spread", "spread_t", "spread_lo", "spread_hi",
                        "top_decile", "p"])
            for r in res:
                w.writerow([r["h"], r["n"], r["ic"], r["t"], r["hit"], r["mono"],
                            r["blo"], r["bhi"], r["nullz"], r["net"], r["nrebal"],
                            r.get("sp_mu"), r.get("sp_t"), r.get("sp_lo"),
                            r.get("sp_hi"), r.get("top_mu"), r["p"]])
        print(f"\n# wrote {args.csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
