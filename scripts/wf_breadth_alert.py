#!/usr/bin/env python3
"""
wf_breadth_alert.py -- alert when walk-forward model breadth decays.

WHY (2026-08-15)
  walk_forward_history runs weekly (Sunday) and writes ~386 tickers x 3 horizons.
  Nothing in production READS it -- the only consumers are parity-check scripts.
  So this went unnoticed:

      h=5, share of tickers with AUC > 0.55
        2026-06-29  38.1%
        2026-07-13  28.3%
        2026-07-27  28.1%
        2026-08-09  27.5%      <- -28% breadth in six weeks

  Average AUC barely moved (0.539 -> 0.535), which is exactly why a mean-based
  check would miss it. BREADTH is the sensitive statistic: the mean can hold
  while the tail of genuinely-predictable names thins out.

  Also alerts on incomplete runs. 2026-06-08 covered only 140 tickers and
  2026-06-22 h=1 only 10 -- silent partial runs that nothing flagged.

WHAT IT CHECKS (per horizon, latest run vs a trailing baseline)
  1. breadth  = share of tickers with AUC > --auc-bar, vs the median of the
                previous --baseline runs. Alerts on a relative drop.
  2. mean AUC = alerts if it falls below --min-auc.
  3. coverage = alerts if the ticker count falls well below the trailing median
                (an incomplete run).

USAGE
  python scripts/wf_breadth_alert.py
  python scripts/wf_breadth_alert.py --auc-bar 0.55 --drop-pct 20 --baseline 6
  python scripts/wf_breadth_alert.py --history          # full trend table

EXIT CODE
  0 = all clear    1 = at least one alert   -> suitable for cron alerting
"""
import argparse
import os
import sqlite3
import statistics
import sys

ROOT = os.path.expanduser(os.environ.get("ML_QUANT_ROOT", "~/ML_Quant_Fund"))
DB = os.path.join(ROOT, "accuracy.db")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--auc-bar", type=float, default=0.55,
                    help="AUC a ticker must exceed to count toward breadth")
    ap.add_argument("--drop-pct", type=float, default=20.0,
                    help="alert if breadth falls this %% below the baseline median")
    ap.add_argument("--min-auc", type=float, default=0.51,
                    help="alert if mean AUC falls below this")
    ap.add_argument("--coverage-pct", type=float, default=15.0,
                    help="alert if ticker count is this %% below baseline median")
    ap.add_argument("--min-breadth", type=float, default=0.05,
                    help="below this breadth a horizon is treated as dead and "
                         "relative changes are reported but NOT alerted")
    ap.add_argument("--baseline", type=int, default=6,
                    help="number of prior runs forming the baseline")
    ap.add_argument("--since", default=None,
                    help="ignore runs before this date (YYYY-MM-DD). Needed because "
                         "pre-2026-07-13 walk-forwards ran on a SHALLOW 2022+ panel "
                         "(free-tier key capped history at 2yr): ~9 folds over ONE "
                         "regime, so AUC breadth was optimistic. After the backfill "
                         "to 2016 it is ~30 folds over five regimes. Anchoring a peak "
                         "on the shallow era makes an alarm that can never clear.")
    ap.add_argument("--history", action="store_true", help="print the full trend")
    ap.add_argument("--db")
    args = ap.parse_args()

    dbp = args.db or DB
    if not os.path.isfile(dbp):
        sys.exit(f"FATAL: {dbp} not found")
    con = sqlite3.connect(dbp, timeout=30)
    try:
        rows = con.execute(
            "SELECT run_date, horizon, COUNT(*) n, AVG(auc) avg_auc, "
            "  SUM(CASE WHEN auc > ? THEN 1 ELSE 0 END) * 1.0 / COUNT(*) breadth "
            "FROM walk_forward_history WHERE auc IS NOT NULL "
            "GROUP BY run_date, horizon ORDER BY run_date, horizon",
            (args.auc_bar,)).fetchall()
    except sqlite3.Error as e:
        sys.exit(f"FATAL: cannot read walk_forward_history: {e}")
    finally:
        con.close()

    if args.since:
        before = len(rows)
        rows = [r for r in rows if r[0] >= args.since]
        print(f"# --since {args.since}: {before} -> {len(rows)} run-horizon rows")
    if not rows:
        sys.exit("FATAL: no walk_forward_history rows in range")

    by_h = {}
    for rd, h, n, a, b in rows:
        by_h.setdefault(h, []).append((rd, n, a, b))

    latest = max(r[0] for r in rows)
    print(f"# wf_breadth_alert  db={dbp}")
    print(f"# latest run={latest}  AUC bar={args.auc_bar}  baseline={args.baseline} runs\n")

    if args.history:
        for h in sorted(by_h):
            print(f"h={h}")
            print(f"  {'run':<12}{'tickers':>8}{'avg AUC':>10}{'breadth':>9}")
            for rd, n, a, b in by_h[h]:
                print(f"  {rd:<12}{n:>8}{a:>10.4f}{b*100:>8.1f}%")
            print()

    alerts = []
    drops = sorted({x[0] for h in by_h for x in by_h[h]
                    if x[1] < statistics.median(y[1] for y in by_h[h]) * 0.5})
    if drops:
        print(f"# incomplete runs excluded from baselines: {', '.join(drops)}\n")
    print(f"{'h':>3}{'tickers':>8}{'avg AUC':>10}{'breadth':>9}{'base':>8}"
          f"{'vs med':>9}{'vs peak':>8}  status")
    print("-" * 72)
    for h in sorted(by_h):
        series = by_h[h]
        if len(series) < 2:
            print(f"{h:>3}   only {len(series)} run(s) -- no baseline")
            continue
        rd, n, a, b = series[-1]
        # EXCLUDE INCOMPLETE RUNS from the baseline. 2026-05-22 ran 4 tickers
        # (h=3 breadth 50%), 2026-05-29 ran 3, 2026-06-22 h=1 ran 10. A 4-ticker
        # run is noise, and max() would happily adopt its breadth as the "peak".
        full = [x for x in series[:-1]]
        if full:
            med_n = statistics.median(x[1] for x in full)
            full = [x for x in full if x[1] >= med_n * 0.5]
        prior = full[-args.baseline:] or series[:-1]
        base_b = statistics.median(x[3] for x in prior)
        base_n = statistics.median(x[1] for x in prior)
        chg = ((b - base_b) / base_b * 100) if base_b > 0 else 0.0

        # Peak drawdown as well as vs-median. A trailing median MOVES WITH a slow
        # decline and absorbs it: the real 38.1% -> 27.5% erosion happened over six
        # weeks, so a 6-run median showed only -6.5% and read "ok". Median catches
        # step changes; peak drawdown catches gradual decay. Both are needed.
        peak_b = max(x[3] for x in prior) if prior else b
        dd = ((b - peak_b) / peak_b * 100) if peak_b > 0 else 0.0

        msgs = []
        # A relative drop on a near-zero base is not information: h=1 went 2.3%
        # -> 1.0%, which is "-55%" but both numbers mean the horizon is dead.
        trivial = max(base_b, peak_b) < args.min_breadth
        if trivial:
            msgs.append(f"breadth < {args.min_breadth*100:.0f}% throughout "
                        f"(horizon effectively dead; relative change not meaningful)")
        if not trivial and base_b > 0 and chg <= -args.drop_pct:
            msgs.append(f"BREADTH -{abs(chg):.0f}% vs median")
        if not trivial and peak_b > 0 and dd <= -args.drop_pct:
            msgs.append(f"BREADTH -{abs(dd):.0f}% from peak {peak_b*100:.1f}%")
        if not trivial and a < args.min_auc:
            msgs.append(f"AUC {a:.4f} < {args.min_auc}")
        if base_n > 0 and n < base_n * (1 - args.coverage_pct / 100):
            msgs.append(f"COVERAGE {n} vs {base_n:.0f}")
        status = "  ".join(msgs) if msgs else "ok"
        if msgs and not trivial:
            alerts.append((h, status))
        print(f"{h:>3}{n:>8}{a:>10.4f}{b*100:>8.1f}%{base_b*100:>7.1f}%"
              f"{chg:>+8.1f}%{dd:>+8.1f}%  {status}")

    print()
    if alerts:
        print(f"!! {len(alerts)} alert(s):")
        for h, s in alerts:
            print(f"   h={h}: {s}")
        print()
        print("   Breadth decay with a flat mean = the tail of predictable names is")
        print("   thinning, not that every name got worse. Check whether the universe")
        print("   changed (new thin-history tickers dilute breadth) before concluding")
        print("   the model decayed.")
        return 1
    print("all clear.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
