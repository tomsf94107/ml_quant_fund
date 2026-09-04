#!/usr/bin/env python3
"""
phase_2h_backtest.py — would the PCT7 overlay improve production BUYs?

READ-ONLY. Writes nothing. This is Phase 2H's own "(f) Test: backtest on past
predictions" step, run against fourteen weeks of already-logged data.

WHY H.1 CAN BE SKIPPED
    docs/phase_2H_overlay_spec.md specifies a shadow week: compute the overlay
    decision, log it, do not act, monitor for one week, then promote. That
    shadow week is unnecessary. Phase epsilon has been logging prob_pct7 since
    2026-05-25 and has 24,741 predictions -- the exact data H.1 would have
    collected, fourteen times over.

THE RULE UNDER TEST, from the spec
    if signal == "BUY" and prob_pct7 < OVERLAY_THRESHOLD:
        signal = "HOLD"

    Spec recommends 0.10 ("lenient first, tune up after seeing data") and lists
    0.05 / 0.10 / 0.13 / 0.15 / 0.20 as candidates. This tests those plus 0.25
    and 0.30, because the 2026-09-05 gauntlet found the economics turn at 0.30:
    names selected at prob_pct7 >= 0.20 returned +0.29% against a universe mean
    of +0.34% -- no value -- while at 0.30 they returned +1.46%, +1.13pp over
    the universe and surviving 40bps of cost.

    So the spec's recommended threshold may be too lenient for this purpose too,
    and the whole ladder is reported rather than a single choice.

THE SPEC'S OWN VALIDATION METRIC
    "Of those downgraded, what fraction actually FAILED (return < 0% in 5d)?
     If downgrade-precision > base-fail-rate, overlay is adding value."

    That is reported directly. Two more are added, because precision alone can
    mislead:

      KEPT-BUY performance. The point is not to identify bad trades -- it is to
      leave a better book. Hit rate and mean return of the BUYs that SURVIVE
      the filter, against all BUYs, is what decides that.

      COST OF THE FILTER. Every downgrade forgoes a trade. If the downgraded
      set contains winners at anything like the kept rate, the filter is just
      shrinking the book for nothing.

SCOPE, per the spec's gap-check (g)
    h=5 only. PCT7 was trained on h=5 and applying it to h=1 or h=3 would be
    using a model outside its target.

    python analysis/phase_2h_backtest.py
    python analysis/phase_2h_backtest.py --buy-col prob_eff --buy-thresh 0.70
"""
import argparse
import math
import sqlite3
import statistics as st


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
    ap.add_argument("--db", default="accuracy.db")
    ap.add_argument("--buy-col", default="signal",
                    help="'signal' uses the logged BUY label; or a probability "
                         "column such as prob_eff with --buy-thresh")
    ap.add_argument("--buy-thresh", type=float, default=None)
    args = ap.parse_args()

    con = sqlite3.connect(f"file:{args.db}?mode=ro", uri=True)
    cols = [r[1] for r in con.execute("PRAGMA table_info(predictions)")]
    if args.buy_col not in cols:
        raise SystemExit(f"{args.buy_col} not in predictions: {cols}")

    if args.buy_col == "signal":
        where = "p.signal = 'BUY'"
        label = "signal = 'BUY'"
    else:
        if args.buy_thresh is None:
            raise SystemExit("--buy-thresh required with a probability column")
        where = f"p.{args.buy_col} >= {args.buy_thresh}"
        label = f"{args.buy_col} >= {args.buy_thresh}"

    rows = con.execute(f"""
        SELECT p.prediction_date, p.ticker, p.prob_pct7, o.actual_return,
               o.actual_up
        FROM predictions p JOIN outcomes o ON p.ticker=o.ticker
          AND p.prediction_date=o.prediction_date AND p.horizon=o.horizon
        WHERE p.horizon=5 AND {where}
          AND p.prob_pct7 IS NOT NULL AND o.actual_return IS NOT NULL
    """).fetchall()
    con.close()

    if len(rows) < 100:
        print(f"only {len(rows)} h=5 BUYs with prob_pct7 -- "
              f"try --buy-col prob_eff --buy-thresh 0.70")
        return

    n = len(rows)
    fail = sum(1 for r in rows if r[3] < 0)
    base_fail = 100.0 * fail / n
    base_hit = 100.0 * sum(1 for r in rows if r[3] >= 0) / n
    base_ret = 100.0 * st.mean(r[3] for r in rows)
    print(f"Phase 2H overlay backtest — h=5, {label}")
    print(f"{n:,} BUYs, {len({r[0] for r in rows})} dates, "
          f"{len({r[1] for r in rows})} tickers")
    print(f"  base: {base_hit:.1f}% positive, {base_fail:.1f}% fail "
          f"(<0%), mean return {base_ret:+.2f}%\n")

    print("SPEC METRIC — 'of those downgraded, what fraction actually FAILED?'")
    print("  overlay adds value if downgrade-fail-rate > base-fail-rate\n")
    print(f"  {'thresh':>7}{'downgraded':>12}{'% of book':>11}"
          f"{'fail rate':>11}{'vs base':>10}")
    for cut in (0.05, 0.10, 0.13, 0.15, 0.20, 0.25, 0.30):
        down = [r for r in rows if r[2] < cut]
        if len(down) < 20:
            print(f"  {cut:>7.2f}{len(down):>12}   too few")
            continue
        f = 100.0 * sum(1 for r in down if r[3] < 0) / len(down)
        print(f"  {cut:>7.2f}{len(down):>12}{100*len(down)/n:>10.0f}%"
              f"{f:>10.1f}%{f - base_fail:>+9.1f}pp")

    print("\nWHAT ACTUALLY MATTERS — the book you are LEFT with")
    print(f"  {'thresh':>7}{'kept':>8}{'hit':>8}{'95% CI':>16}"
          f"{'mean ret':>10}{'vs base':>10}{'forgone':>9}")
    for cut in (0.05, 0.10, 0.13, 0.15, 0.20, 0.25, 0.30):
        kept = [r for r in rows if r[2] >= cut]
        down = [r for r in rows if r[2] < cut]
        if len(kept) < 30:
            print(f"  {cut:>7.2f}{len(kept):>8}   too few kept")
            continue
        k = sum(1 for r in kept if r[3] >= 0)
        lo, hi = wilson(k, len(kept))
        mr = 100.0 * st.mean(r[3] for r in kept)
        fg = (100.0 * st.mean(r[3] for r in down)) if down else 0.0
        print(f"  {cut:>7.2f}{len(kept):>8}{100*k/len(kept):>7.1f}%"
              f"   [{lo:>5.1f},{hi:>5.1f}]{mr:>9.2f}%{mr-base_ret:>+9.2f}pp"
              f"{fg:>8.2f}%")

    print("\n  'forgone' is the mean return of the trades the filter REMOVES.")
    print("  If that is close to the kept mean, the filter is shrinking the")
    print("  book without improving it -- precision on failures is not enough.")
    print("\n  Spec recommends 0.10 as a lenient start. The 2026-09-05 gauntlet")
    print("  found the economics turn at 0.30: PCT7-selected names returned")
    print("  +0.29% at the 0.20 cut against a universe mean of +0.34%, and")
    print("  +1.46% at 0.30. A threshold good for one purpose need not be good")
    print("  for the other, which is why the whole ladder is shown.")


if __name__ == "__main__":
    main()
