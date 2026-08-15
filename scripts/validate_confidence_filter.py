#!/usr/bin/env python3
"""
validate_confidence_filter.py -- does a high prob_up threshold identify signals
with real edge, ACROSS THE UNIVERSE? Measured the way the validated bricks were.

WHY THIS EXISTS (2026-08-15)
  A per-ticker check on CRWV showed 100% accuracy (11/11) at prob_up>=0.60,
  h=3. Escalating the methodology killed it at every stage:
      per-ticker           -> 3 clustered episodes, ONE V-bottom, p~1 after Bonferroni
      pooled hit rate      -> z=4.97 ... which is 0.36 once dependence is handled
      per-date beta-neutral-> t=1.32, NW ~1.0
  Three separate inflation mechanisms, each already documented in this fund's
  own prior work: market beta, overlapping windows, cross-sectional dependence.

METHOD (mirrors validate_si_v2.py, the method that produced the honest SI t)
  1. PER-DATE cross-sectional metrics. A date is the unit of observation, NOT a
     stock-date row. ~400 tickers on one date share market beta; pooling them as
     independent inflated the SI t-stat ~7x (20,000 rows -> effective ~60).
  2. RANK-IC as the primary metric, not hit rate. Directional accuracy in a
     rising tape is mostly drift: over the sample checked, the LOW-confidence
     bucket still averaged +0.67% per 3 days. Rank-IC is beta-neutral by
     construction. (Research_Report.md: AUC ~ 0.5 + IC/2; accuracy is the wrong
     objective for cross-sectional work.)
  3. NEWEY-WEST with lag = horizon. Overlapping h-day returns sampled daily are
     MA(h-1); NW with ~h lags is standard practice. NOTE: NW is itself biased
     DOWNWARD under strong autocorrelation, so the reported t is an UPPER BOUND.
  4. SHUFFLE NULL. Permute actual_return within each date and recompute. A real
     signal must vanish. Reports how many sigma the observed value sits from the
     null distribution -- the SI brick cleared this at 8.3 sigma.
  5. MULTIPLE TESTING. Every cell scanned is a trial. Reports Bonferroni and
     Benjamini-Hochberg FDR alongside the raw p, and grades against the fund's
     standing Harvey-Liu-Zhu bar of t > 3.
  6. OPTIONAL SECTOR-NEUTRALISATION (--sector-neutral): demeans signal and
     outcome within sector per date, testing whether the effect is
     stock-specific or a sector tilt. The SI brick retained 80%.

USAGE
  python scripts/validate_confidence_filter.py
  python scripts/validate_confidence_filter.py --horizons 1,3,5 --thresholds 0.55,0.60,0.65,0.70
  python scripts/validate_confidence_filter.py --days 365 --shuffles 500
  python scripts/validate_confidence_filter.py --sector-neutral
  python scripts/validate_confidence_filter.py --csv confidence_validation.csv

EXIT CODE
  0 = no cell clears the bar (the expected, honest outcome)
  1 = at least one cell clears t>3 AND the shuffle null -- inspect before believing
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
DB = os.path.join(ROOT, "accuracy.db")
META = os.path.join(ROOT, "tickers_metadata.csv")

PROB_CANDIDATES = ["prob_up", "prob", "probability", "p_up"]
RET_CANDIDATES = ["actual_return", "realized_return", "ret", "fwd_return"]
HLZ_BAR = 3.0


# ---------------------------------------------------------------- statistics
def ranks(xs):
    """Average ranks, ties handled (required for a correct Spearman)."""
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
    cov = sum((a[i] - ma) * (b[i] - mb) for i in range(n))
    return cov / math.sqrt(va * vb)


def spearman(a, b):
    return pearson(ranks(a), ranks(b))


def nw_tstat(xs, lag):
    """Newey-West (Bartlett kernel) t-stat for the mean of a series.
    lag = horizon: overlapping h-day returns are MA(h-1).
    NW is biased downward under strong autocorrelation -> treat as upper bound."""
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
    return mu, se, (mu / se if se > 0 else None)


def net_of_cost(hi_sets, spreads, h, cost_bps):
    """Gross spread -> NET of transaction costs, on NON-OVERLAPPING rebalances.

    WHY. Commit 75ae24a9 (Jul 2026) settled the direction model:
        gross hit edge +1.160pp t=+4.87
        NET of 10bps  -0.122%/trade t=-3.28   6/30 folds, 2/9 years
        "NO REGIME WHERE IT WORKS"
    A signal can be significantly positive GROSS and reliably negative NET. A bar
    that stops at IC or gross spread cannot tell those apart -- so the monthly IC
    log could "pass" a signal already proven dead.

    METHOD. Rebalancing every h days = sampling every h-th date, which also makes
    the series NON-OVERLAPPING: the t-stat needs no NW correction and n is the
    honest number of independent bets. Turnover is MEASURED (share of the
    high-conf set replaced between rebalances); cost = turnover x cost_bps."""
    idx = list(range(0, len(spreads), max(1, h)))
    if len(idx) < 5:
        return None
    gross, nets, turns = [], [], []
    prev = None
    for i in idx:
        cur = hi_sets[i]
        to = 1.0 if prev is None or not cur else len(cur ^ prev) / max(len(cur | prev), 1)
        prev = cur
        gross.append(spreads[i])
        turns.append(to)
        nets.append(spreads[i] - to * cost_bps / 10000.0)
    n = len(nets)
    if n < 2:
        return None
    mu = sum(nets) / n
    sd = math.sqrt(sum((x - mu) ** 2 for x in nets) / (n - 1))
    return (n, sum(turns) / n, sum(gross) / n, mu,
            (mu / (sd / math.sqrt(n))) if sd > 0 else None)


def norm_p_two(z):
    return 2.0 * (1.0 - 0.5 * (1.0 + math.erf(abs(z) / math.sqrt(2.0))))


def bh_fdr(pvals, q=0.05):
    """Benjamini-Hochberg: returns the set of indices that survive at level q."""
    idx = sorted(range(len(pvals)), key=lambda i: pvals[i])
    m = len(pvals)
    keep, kmax = set(), -1
    for rank, i in enumerate(idx, start=1):
        if pvals[i] <= q * rank / m:
            kmax = rank
    for rank, i in enumerate(idx, start=1):
        if rank <= kmax:
            keep.add(i)
    return keep


# ---------------------------------------------------------------- data
def cols(con, t):
    return [r[1] for r in con.execute(f'PRAGMA table_info("{t}")')]


def pick(avail, cands, label, table):
    low = {c.lower(): c for c in avail}
    for c in cands:
        if c in low:
            return low[c]
    sys.exit(f"FATAL: no {label} column in {table}. Saw: {', '.join(avail)}")


GROUP_CANDIDATES = ["sector", "bucket", "industry", "group", "tier"]


def load_groups(want_col=None):
    """Load ticker -> group for neutralisation.

    FAILS LOUDLY. v1.0 returned {} when no 'sector' column existed and carried
    on printing sector_neutral=False -- the same silent-fallback bug class that
    let a dead price feed go unnoticed for 6 months. A requested control that
    silently does not run is worse than no control."""
    if not os.path.isfile(META):
        sys.exit(f"FATAL: --sector-neutral requested but {META} not found.")
    with open(META, newline="") as f:
        rows = list(_csv.reader(f))
    if not rows:
        sys.exit(f"FATAL: {META} is empty.")
    hdr = [h.strip().lower() for h in rows[0]]
    if want_col:
        if want_col.lower() not in hdr:
            sys.exit(f"FATAL: --group-col '{want_col}' not in {META}. "
                     f"Columns: {', '.join(hdr)}")
        gcol = hdr.index(want_col.lower())
    else:
        gcol = next((hdr.index(c) for c in GROUP_CANDIDATES if c in hdr), None)
        if gcol is None:
            sys.exit(f"FATAL: --sector-neutral requested but no grouping column "
                     f"found in {META}.\n       Looked for: {', '.join(GROUP_CANDIDATES)}"
                     f"\n       Columns present: {', '.join(hdr)}"
                     f"\n       Pass one explicitly with --group-col NAME.")
    tcol = next((i for i, h in enumerate(hdr) if h in ("ticker", "symbol")), 0)
    out = {}
    for r in rows[1:]:
        if r and len(r) > max(tcol, gcol) and r[tcol].strip():
            out[r[tcol].strip().upper()] = (r[gcol].strip() or "UNKNOWN")
    if not out:
        sys.exit(f"FATAL: grouping column '{hdr[gcol]}' present but empty.")
    named = sum(1 for v in out.values() if v != "UNKNOWN")
    if named < 10:
        sys.exit(f"FATAL: only {named} tickers have a non-empty "
                 f"'{hdr[gcol]}' value -- neutralisation would be meaningless.")
    print(f"# neutralising on column '{hdr[gcol]}': {len(out)} tickers, "
          f"{len(set(out.values()))} groups")
    return out


def block_bootstrap_ci(xs, block, B=2000, alpha=0.05, seed=17):
    """Moving-block bootstrap CI for the mean.

    WHY, not just NW: Newey-West OVER-REJECTS at small samples with high serial
    correlation (den Haan & Levin 1997) -- it produces too many false positives,
    so its t is an upper bound. Block length is set to mimic the NW lag, which is
    the researched recommendation for matching the autocorrelation structure.
    This also matches audit_combination.py, which already block-bootstraps."""
    n = len(xs)
    if n < 10 or block < 1:
        return None, None
    rnd = random.Random(seed)
    nblocks = math.ceil(n / block)
    starts = n - block + 1
    means = []
    for _ in range(B):
        samp = []
        for _ in range(nblocks):
            s0 = rnd.randrange(starts)
            samp.extend(xs[s0:s0 + block])
        samp = samp[:n]
        means.append(sum(samp) / len(samp))
    means.sort()
    lo = means[int(alpha / 2 * B)]
    hi = means[min(B - 1, int((1 - alpha / 2) * B))]
    return lo, hi


def demean_by_sector(recs, sectors):
    """recs: [(ticker, prob, ret)] -> same, demeaned within sector."""
    buckets = defaultdict(list)
    for t, p, r in recs:
        buckets[sectors.get(t, "UNKNOWN")].append((t, p, r))
    out = []
    for _, rows in buckets.items():
        if len(rows) < 2:
            continue
        mp = sum(x[1] for x in rows) / len(rows)
        mr = sum(x[2] for x in rows) / len(rows)
        out.extend((t, p - mp, r - mr) for t, p, r in rows)
    return out


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizons", default="1,3,5")
    ap.add_argument("--thresholds", default="0.55,0.60,0.65,0.70")
    ap.add_argument("--days", type=int, default=365, help="lookback in calendar days")
    ap.add_argument("--min-names", type=int, default=20,
                    help="min tickers on a date for it to count")
    ap.add_argument("--min-hi", type=int, default=5,
                    help="min high-conf names on a date for the spread test")
    ap.add_argument("--shuffles", type=int, default=200)
    ap.add_argument("--sector-neutral", action="store_true",
                    help="demean signal+outcome within group per date (fails loudly "
                         "if no grouping column)")
    ap.add_argument("--group-col", default=None,
                    help="column in tickers_metadata.csv to neutralise on "
                         "(default: first of sector/bucket/industry/group/tier)")
    ap.add_argument("--prob-col", default="prob_up",
                    help="comma list of probability columns to test, e.g. "
                         "prob_up,prob_raw,prob_up_global. Each is run separately "
                         "and compared -- prob_up is POST-OVERLAY, prob_raw is the "
                         "model output before the multiplier stack.")
    ap.add_argument("--no-common-dates", action="store_true",
                    help="allow prob columns to be compared on DIFFERENT date sets. "
                         "Off by default: columns populated over different periods "
                         "are not comparable, and comparing them is a silent apples-"
                         "to-oranges error.")
    ap.add_argument("--include-watchlist", action="store_true",
                    help="include is_watchlist=1 rows (they are excluded by default: "
                         "the system defines watchlist names as predictions-only, "
                         "explicitly OUT of accuracy scoring)")
    ap.add_argument("--cost-bps", type=float, default=10.0,
                    help="round-trip cost in bps applied to MEASURED turnover. "
                         "10bps is where commit 75ae24a9 showed the direction "
                         "model go from t=+4.87 gross to t=-3.28 net.")
    ap.add_argument("--bootstrap", type=int, default=2000,
                    help="moving-block bootstrap resamples (0 to disable)")
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--log", action="store_true",
                    help="append results to accuracy.db table ic_history. Use on a "
                         "monthly cron to accumulate the IC series -- the recent-window "
                         "effect needs ~74 more dates to reach t>3, so the answer comes "
                         "from ACCUMULATION, not from re-running on the same data.")
    ap.add_argument("--csv")
    ap.add_argument("--db")
    args = ap.parse_args()

    random.seed(args.seed)
    dbp = args.db or DB
    if not os.path.isfile(dbp):
        sys.exit(f"FATAL: {dbp} not found")

    con = sqlite3.connect(dbp, timeout=30)
    pred_cols = cols(con, "predictions")
    low = {c.lower(): c for c in pred_cols}
    pcols = []
    for want in [x.strip() for x in args.prob_col.split(",") if x.strip()]:
        if want.lower() not in low:
            sys.exit(f"FATAL: --prob-col '{want}' not in predictions. "
                     f"Available: {', '.join(pred_cols)}")
        pcols.append(low[want.lower()])
    has_wl = "is_watchlist" in low
    wl_clause = ""
    if has_wl and not args.include_watchlist:
        wl_clause = " AND COALESCE(p.is_watchlist,0)=0"
    rcol = pick(cols(con, "outcomes"), RET_CANDIDATES, "return", "outcomes")
    # --group-col alone used to be a SILENT NO-OP: it was ignored unless
    # --sector-neutral was also passed, and the run printed sector_neutral=False
    # while reporting raw numbers. Same silent-fallback class as the empty-gap
    # bug. --group-col now IMPLIES neutralisation.
    if args.group_col and not args.sector_neutral:
        print(f"# --group-col '{args.group_col}' given -> enabling --sector-neutral")
        args.sector_neutral = True
    sectors = load_groups(args.group_col) if args.sector_neutral else {}

    horizons = [int(x) for x in args.horizons.split(",")]
    thresholds = [float(x) for x in args.thresholds.split(",")]

    print("# validate_confidence_filter")
    print(f"# db={dbp}  prob={pcols}  ret='{rcol}'  lookback={args.days}d")
    if has_wl:
        print(f"# watchlist rows: {'INCLUDED' if args.include_watchlist else 'EXCLUDED'} "
              f"(is_watchlist=1 -- system defines these as out of accuracy scoring)")
    else:
        print("# NOTE: no is_watchlist column found")
    print(f"# unit of observation = DATE (not stock-date row)")
    print(f"# NW lag = horizon (overlapping h-day returns are MA(h-1)); NW is")
    print(f"#   biased DOWN under autocorrelation -> reported t is an UPPER BOUND")
    print(f"# shuffles={args.shuffles}  sector_neutral={bool(sectors)}"
          + (f" ({len(set(sectors.values()))} sectors)" if sectors else ""))
    print()

    # ---- common-date intersection across prob columns (per horizon)
    common = {}
    if len(pcols) > 1 and not args.no_common_dates:
        for h in horizons:
            sets = []
            for pc in pcols:
                ds = con.execute(
                    f'SELECT p.prediction_date FROM predictions p JOIN outcomes o '
                    f'  ON p.ticker=o.ticker AND p.prediction_date=o.prediction_date '
                    f'  AND p.horizon=o.horizon '
                    f'WHERE p.horizon=? AND o."{rcol}" IS NOT NULL '
                    f'  AND p."{pc}" IS NOT NULL' + wl_clause +
                    f"  AND p.prediction_date >= date('now', ?) "
                    f'GROUP BY 1 HAVING COUNT(*) >= ?',
                    (h, f"-{args.days} days", args.min_names)).fetchall()
                sets.append({r[0] for r in ds})
            inter = set.intersection(*sets) if sets else set()
            common[h] = inter
            sizes = " / ".join(f"{pc}={len(sv)}" for pc, sv in zip(pcols, sets))
            print(f"# h={h} date coverage: {sizes}  -> COMMON={len(inter)}")
            if len(inter) < 10:
                print(f"#   WARNING: only {len(inter)} common dates -- comparison "
                      f"is underpowered. Consider dropping the sparsest column.")
        print()

    results = []
    for pcol in pcols:
      if len(pcols) > 1:
        print(f"########## prob column: {pcol} ##########")
      for h in horizons:
          rows = con.execute(
              f'SELECT p.prediction_date, p.ticker, p."{pcol}", o."{rcol}" '
              f'FROM predictions p JOIN outcomes o '
              f'  ON p.ticker=o.ticker AND p.prediction_date=o.prediction_date '
              f'  AND p.horizon=o.horizon '
              f'WHERE p.horizon=? AND o."{rcol}" IS NOT NULL AND p."{pcol}" IS NOT NULL '
              + wl_clause +
              f"  AND p.prediction_date >= date('now', ?) "
              f'ORDER BY p.prediction_date', (h, f"-{args.days} days")).fetchall()

          by_date = defaultdict(list)
          for d, t, p, r in rows:
              by_date[d].append((t.upper(), float(p), float(r)))
          dates = sorted(d for d, v in by_date.items() if len(v) >= args.min_names)
          if common.get(h):
              dates = [d for d in dates if d in common[h]]
          if len(dates) < 10:
              print(f"h={h}: only {len(dates)} usable dates -- skipped")
              continue

          # ---- per-date rank-IC (primary metric)
          ics, panels = [], {}
          for d in dates:
              recs = by_date[d]
              if sectors:
                  recs = demean_by_sector(recs, sectors)
                  if len(recs) < args.min_names:
                      continue
              panels[d] = recs
              ic = spearman([x[1] for x in recs], [x[2] for x in recs])
              if ic is not None:
                  ics.append(ic)
          if len(ics) < 10:
              print(f"h={h}: too few IC dates -- skipped")
              continue

          mu, se, t = nw_tstat(ics, h)
          # shuffle null on the IC
          null_means = []
          for _ in range(args.shuffles):
              acc = []
              for d in panels:
                  recs = panels[d]
                  rr = [x[2] for x in recs]
                  random.shuffle(rr)
                  ic = spearman([x[1] for x in recs], rr)
                  if ic is not None:
                      acc.append(ic)
              if acc:
                  null_means.append(sum(acc) / len(acc))
          nz = None
          if len(null_means) > 2:
              nm = sum(null_means) / len(null_means)
              nsd = math.sqrt(sum((x - nm) ** 2 for x in null_means) / (len(null_means) - 1))
              nz = (mu - nm) / nsd if nsd > 0 else None

          rows_total = sum(len(by_date[d]) for d in dates)
          print(f"h={h}  dates={len(ics)}  stock-date rows={rows_total}  "
                f"(pooling rows would overstate n by ~{rows_total/max(1,len(ics)):.0f}x)")
          tt = f"{t:+.2f}" if t is not None else "n/a"
          nzs = f"{nz:+.2f}" if nz is not None else "n/a"
          pv = norm_p_two(t) if t is not None else 1.0
          blo, bhi = (None, None)
          if args.bootstrap:
              blo, bhi = block_bootstrap_ci(ics, h, B=args.bootstrap)
          bs = f"[{blo:+.5f},{bhi:+.5f}]" if blo is not None else "n/a"
          excl = (blo is not None and (blo > 0 or bhi < 0))
          print(f"  RANK-IC   mean={mu:+.5f}")
          print(f"    SIGNIFICANCE  NW-t={tt} (p={pv:.4f})  [bar |t|>{HLZ_BAR:.0f}] "
                f"-- NW OVER-REJECTS, treat as UPPER BOUND")
          print(f"    BOOTSTRAP     95% CI {bs}  block={h}  "
                f"{'excludes 0' if excl else 'SPANS 0'}")
          print(f"    LEAKAGE CHECK shuffle-sigma={nzs}  -- this is NOT a significance")
          print(f"                  test; it only asks whether the association is a")
          print(f"                  pipeline artifact. It ignores regime variance and")
          print(f"                  runs ANTI-CONSERVATIVE. Never read it as a bar.")
          results.append(dict(h=h, thr=None, metric="rank_ic", n=len(ics), col=pcol,
                              mean=mu, t=t, p=pv, nullz=nz, blo=blo, bhi=bhi))

          # ---- per-date high-conf spread (secondary, beta-neutral)
          for thr in thresholds:
              spreads, accs, bases = [], [], []
              hi_sets = []
              for d in dates:
                  recs = by_date[d]  # raw, not demeaned: spread is already same-date
                  hi = [r for _, p, r in recs if p >= thr]
                  lo = [r for _, p, r in recs if p < thr]
                  if len(hi) < args.min_hi or len(lo) < args.min_hi:
                      continue
                  spreads.append(sum(hi) / len(hi) - sum(lo) / len(lo))
                  hi_sets.append(frozenset(x[0] for x in recs if x[1] >= thr))
                  accs.append(sum(1 for r in hi if r > 0) / len(hi))
                  allr = [r for _, _, r in recs]
                  bases.append(sum(1 for r in allr if r > 0) / len(allr))
              if len(spreads) < 10:
                  print(f"  thr>={thr:.2f}  only {len(spreads)} usable dates -- skipped")
                  continue
              m2, se2, t2 = nw_tstat(spreads, h)
              p2 = norm_p_two(t2) if t2 is not None else 1.0
              acc = sum(accs) / len(accs) * 100
              base = sum(bases) / len(bases) * 100
              t2s = f"{t2:+.2f}" if t2 is not None else "n/a"
              print(f"  thr>={thr:.2f}  dates={len(spreads):>4}  spread={m2*100:+.4f}%/{h}d  "
                    f"NW-t={t2s}  p={p2:.4f}  acc={acc:.1f}% vs base {base:.1f}% "
                    f"({acc-base:+.1f}pp)")
              noc = net_of_cost(hi_sets, spreads, h, args.cost_bps)
              _net = dict(zip(("n_rebal", "turnover", "gross", "net_spread", "net_t"),
                              noc)) if noc else {}
              if noc:
                  nn, to, gr, ne, tn = noc
                  tns = f"{tn:+.2f}" if tn is not None else "n/a"
                  verdict = ("NET NEGATIVE" if ne <= 0
                             else ("net t<2" if (tn or 0) < 2 else "NET SURVIVES"))
                  print(f"      net@{args.cost_bps:.0f}bps: {nn} rebalances  "
                        f"turnover={to*100:.0f}%  gross={gr*100:+.4f}%  "
                        f"NET={ne*100:+.4f}%  t={tns}  -> {verdict}")
              b2lo, b2hi = (None, None)
              if args.bootstrap:
                  b2lo, b2hi = block_bootstrap_ci(spreads, h, B=args.bootstrap)
              results.append(dict(h=h, thr=thr, metric="spread", n=len(spreads), col=pcol,
                                  mean=m2, t=t2, p=p2, nullz=None,
                                  blo=b2lo, bhi=b2hi, **_net))
          print()

    con.close()
    if not results:
        print("# no cells produced -- check horizons/lookback")
        return 0

    # ---- multiple testing across every cell scanned
    pvals = [r["p"] for r in results]
    m = len(pvals)
    bonf = 0.05 / m
    keep = bh_fdr(pvals, 0.05)
    if len(pcols) > 1:
        print("=" * 78)
        print("PROB-COLUMN COMPARISON (rank-IC) -- does the overlay add or destroy?")
        if not args.no_common_dates:
            print("  (restricted to COMMON dates -- columns are directly comparable)")
        print(f"  {'column':<24}{'h':>3}{'dates':>7}{'rank-IC':>11}{'NW-t':>8}")
        for r in results:
            if r["metric"] == "rank_ic":
                tv = f"{r['t']:+.2f}" if r["t"] is not None else "n/a"
                print(f"  {r['col']:<24}{r['h']:>3}{r['n']:>7}{r['mean']:>+11.5f}{tv:>8}")
        print()
    print("=" * 78)
    print(f"MULTIPLE TESTING -- {m} cells scanned")
    print(f"  Bonferroni threshold p < {bonf:.5f}")
    print(f"  Benjamini-Hochberg FDR q=0.05 survivors: "
          f"{len(keep)} of {m}")
    def _boot_ok(r):
        return r.get("blo") is None or r["blo"] > 0 or r["bhi"] < 0
    survivors = [r for i, r in enumerate(results)
                 if r["t"] is not None and abs(r["t"]) > HLZ_BAR
                 and _boot_ok(r) and i in keep]
    print("  bar = NW |t|>3  AND  block-bootstrap CI excludes 0  AND  BH-FDR")
    print("  (leakage check is diagnostic only and is NOT part of the bar)")
    print()
    if survivors:
        print(f"!! {len(survivors)} cell(s) clear t>{HLZ_BAR:.0f} + null + FDR:")
        for r in survivors:
            print(f"   h={r['h']} thr={r['thr']} {r['metric']} "
                  f"mean={r['mean']:+.5f} t={r['t']:+.2f}")
        print("   Inspect before believing: check episode clustering, regime")
        print("   dependence, and re-run on a cold holdout.")
    else:
        print("VERDICT: no cell clears the bar (t>3, shuffle null, FDR).")
        print("  The high-confidence filter is NOT demonstrated to have edge.")
        print("  This is a NEGATIVE RESULT, not a failed measurement -- it closes")
        print("  a direction rather than leaving it to be re-litigated.")

    if args.log:
        lcon = sqlite3.connect(dbp, timeout=30)
        try:
            lcon.execute("""CREATE TABLE IF NOT EXISTS ic_history(
                run_date TEXT, window_days INTEGER, prob_col TEXT, horizon INTEGER,
                metric TEXT, threshold REAL, neutralised INTEGER, group_col TEXT,
                n_dates INTEGER, mean REAL, nw_t REAL, p REAL,
                boot_lo REAL, boot_hi REAL, leak_sigma REAL,
                cost_bps REAL, n_rebal INTEGER, turnover REAL,
                gross_spread REAL, net_spread REAL, net_t REAL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY(run_date, window_days, prob_col, horizon, metric,
                            threshold, neutralised))""")
            # Additive migration: the table predates the net-of-cost columns.
            # Gross history stays valid; net accrues from here. Without this the
            # monthly log would accumulate against a bar (gross) that commit
            # 75ae24a9 already showed cannot distinguish live from dead.
            have = {r[1] for r in lcon.execute("PRAGMA table_info(ic_history)")}
            for col, typ in (("cost_bps", "REAL"), ("n_rebal", "INTEGER"),
                             ("turnover", "REAL"), ("gross_spread", "REAL"),
                             ("net_spread", "REAL"), ("net_t", "REAL")):
                if col not in have:
                    lcon.execute(f"ALTER TABLE ic_history ADD COLUMN {col} {typ}")
            try:
                from zoneinfo import ZoneInfo
                from datetime import datetime as _dtm
                rd = _dtm.now(ZoneInfo("America/New_York")).date().isoformat()
            except Exception:
                from datetime import date as _dte
                rd = _dte.today().isoformat()
            gcol = args.group_col or ("auto" if sectors else None)
            n = 0
            for r in results:
                lcon.execute(
                    "INSERT OR REPLACE INTO ic_history(run_date,window_days,prob_col,"
                    "horizon,metric,threshold,neutralised,group_col,n_dates,mean,nw_t,"
                    "p,boot_lo,boot_hi,leak_sigma,cost_bps,n_rebal,turnover,"
                    "gross_spread,net_spread,net_t) "
                    "VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    (rd, args.days, r.get("col"), r["h"], r["metric"],
                     -1.0 if r["thr"] is None else r["thr"],
                     1 if sectors else 0, gcol, r["n"], r["mean"], r["t"], r["p"],
                     r.get("blo"), r.get("bhi"), r["nullz"],
                     args.cost_bps if r["metric"] == "spread" else None,
                     r.get("n_rebal"), r.get("turnover"),
                     r.get("gross"), r.get("net_spread"), r.get("net_t")))
                n += 1
            lcon.commit()
            hist = lcon.execute(
                "SELECT COUNT(DISTINCT run_date) FROM ic_history").fetchone()[0]
            print(f"\n# logged {n} row(s) to ic_history  ({hist} run date(s) accumulated)")
        finally:
            lcon.close()

    if args.csv:
        with open(args.csv, "w", newline="") as f:
            w = _csv.writer(f)
            w.writerow(["prob_col", "horizon", "threshold", "metric", "n_dates", "mean", "nw_t",
                        "p", "boot_lo", "boot_hi", "leak_sigma"])
            for r in results:
                w.writerow([r.get("col"), r["h"], r["thr"], r["metric"], r["n"],
                            round(r["mean"], 6),
                            None if r["t"] is None else round(r["t"], 3),
                            round(r["p"], 5),
                            None if r.get("blo") is None else round(r["blo"], 6),
                            None if r.get("bhi") is None else round(r["bhi"], 6),
                            None if r["nullz"] is None else round(r["nullz"], 2)])
        print(f"\n# wrote {args.csv}")

    return 1 if survivors else 0


if __name__ == "__main__":
    sys.exit(main())
