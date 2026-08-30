#!/usr/bin/env python3
"""
track_model_eval.py — monthly evaluation of the daily direction program (rule D).

Appends one row per horizon to model_eval_history in accuracy.db and checks the
pre-registered triggers. Never overwrites: the path of the metric over time is
the point, and an overwritten metric is how accuracy_cache became meaningless.

DECISION RULE D, agreed 2026-08-30
    Metric   winsorized long-short decile spread at h=3, Newey-West t, lag 2.
             h=3 is the only horizon showing anything; h=1 and h=5 are flat.

    Baseline locked 2026-08-30 on 103 rebalance dates:
             NW-t 1.12, AUC 0.5261, spread +0.39% gross, +0.19% net of 10bps.

    GRADUATE      NW-t >= 2.5 on >= 250 dates, sign stable, positive net of
                  10bps -> promote to a proper walk-forward test. Promotion
                  means "tested seriously", NOT "live".
    CIRCUIT BREAK sign flips negative with |NW-t| > 1.5 -> flag and require a
                  ruling. This is what killed the SELL signal (n=1,234, 51.7%,
                  sign flipped) and catching it automatically is free.
    REVIEW        at >= 250 dates, lay the numbers against the baseline for a
                  human decision.
    NO AUTO-RETIRE. The program keeps running until it graduates or the operator
                  retires it. Chosen deliberately: its marginal cost is compute
                  already being spent.

    Why 2.5 and not 2.0: this is checked monthly, which is a multiple-
    comparisons problem. The project's momentum gauntlet cleared t = +3.19, so
    2.5 is already a concession.

    Honest arithmetic, recorded so nobody is surprised later: if the effect size
    stays at +0.39%, 250 dates gives t ~ 1.8 and 400 dates gives t ~ 2.2 --
    neither reaches 2.5. Graduation therefore requires the EFFECT to grow, not
    just the sample. That is possible (the split and ticker-reuse fixes of
    2026-08-30 may have been suppressing it) but it is not the default path.

USAGE
    python track_model_eval.py                 # compute, print, append
    python track_model_eval.py --dry-run       # compute and print only
    python track_model_eval.py --history       # show the recorded path
"""
import argparse
import os
import sqlite3
import sys
from collections import defaultdict
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

DB = "accuracy.db"
BASELINE = {"date": "2026-08-30", "h": 3, "nw_t": 1.12, "auc": 0.5261,
            "spread": 0.003879, "n_dates": 103}
GRADUATE_T = 2.5
GRADUATE_DATES = 250
BREAK_T = 1.5
PRIMARY_H = 3
COST_BPS = 10

DDL = """
CREATE TABLE IF NOT EXISTS model_eval_history (
    asof_date     TEXT NOT NULL,
    horizon       INTEGER NOT NULL,
    n_rows        INTEGER,
    n_dates       INTEGER,
    auc           REAL,
    spread_gross  REAL,
    spread_net    REAL,
    nw_t          REAL,
    naive_t       REAL,
    sign_pct      REAL,
    verdict       TEXT,
    created_at    TEXT NOT NULL,
    PRIMARY KEY (asof_date, horizon)
)
"""


def load_modules():
    import importlib.util

    def _load(name, path):
        spec = importlib.util.spec_from_file_location(name, path)
        m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(m)
        return m

    here = os.path.dirname(os.path.abspath(__file__))
    ev = _load("ev", os.path.join(here, "economic_value.py"))
    pa = _load("pa", os.path.join(here, "pooled_accuracy.py"))
    return ev, pa


def evaluate(con, ev, pa):
    rows = con.execute("""
        SELECT p.horizon, p.prediction_date, p.prob_up, o.actual_return, o.actual_up
        FROM predictions p JOIN outcomes o
          ON p.ticker=o.ticker AND p.prediction_date=o.prediction_date
         AND p.horizon=o.horizon
        WHERE p.prob_up IS NOT NULL AND o.actual_return IS NOT NULL
    """).fetchall()

    by_h = defaultdict(list)
    for h, d, prob, ret, up in rows:
        by_h[h].append((d, prob, ret, up))

    out = []
    for h in sorted(by_h):
        recs = by_h[h]
        trio = [(d, p, r) for d, p, r, _ in recs]
        wins, _clipped = ev.winsorize_by_date(trio)
        _dec, daily_spread, _long = ev.deciles_by_date(wins)
        if not daily_spread:
            continue
        sp = [s for _, s in daily_spread]
        nw_t, naive_t = ev.newey_west_t(sp, lag=max(h - 1, 1))
        gross = ev.mean(sp)
        net = gross - 2 * COST_BPS / 10000.0
        a = pa.auc([p for _, p, _, _ in recs], [u for _, _, _, u in recs])
        pos = sum(1 for s in sp if s > 0)
        out.append({
            "horizon": h, "n_rows": len(recs), "n_dates": len(sp),
            "auc": a, "spread_gross": gross, "spread_net": net,
            "nw_t": nw_t, "naive_t": naive_t,
            "sign_pct": 100.0 * pos / len(sp),
        })
    return out


def verdict_for(r):
    """Rule D. Returns (verdict, message)."""
    if r["horizon"] != PRIMARY_H:
        return "SECONDARY", "not the primary horizon; recorded for context only"
    t = r["nw_t"]
    if t is None:
        return "NO_DATA", "t undefined"
    if t < -BREAK_T:
        return "CIRCUIT_BREAK", (
            f"sign has flipped NEGATIVE with |t| = {abs(t):.2f} > {BREAK_T}. "
            "This is the failure that killed the SELL signal. Requires a ruling.")
    if t >= GRADUATE_T and r["n_dates"] >= GRADUATE_DATES and r["spread_net"] > 0:
        return "GRADUATE", (
            f"t = {t:.2f} >= {GRADUATE_T} on {r['n_dates']} dates, positive net "
            f"of {COST_BPS}bps. Promote to a walk-forward test -- not to live.")
    if r["n_dates"] >= GRADUATE_DATES:
        return "REVIEW_DUE", (
            f"{r['n_dates']} dates reached without graduating (t = {t:.2f}). "
            "Scheduled operator review.")
    return "ACCUMULATING", (
        f"t = {t:.2f}, {r['n_dates']}/{GRADUATE_DATES} dates. No action.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=DB)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--history", action="store_true")
    args = ap.parse_args()

    con = sqlite3.connect(args.db)
    con.execute(DDL)

    if args.history:
        print(f"{'asof':>12} {'h':>3} {'dates':>6} {'AUC':>7} {'gross':>9} "
              f"{'net10':>9} {'NW t':>7}  verdict")
        for r in con.execute("SELECT asof_date, horizon, n_dates, auc, "
                             "spread_gross, spread_net, nw_t, verdict "
                             "FROM model_eval_history ORDER BY asof_date, horizon"):
            print(f"{r[0]:>12} {r[1]:>3} {r[2]:>6} {r[3] or 0:>7.4f} "
                  f"{(r[4] or 0)*100:>8.4f}% {(r[5] or 0)*100:>8.4f}% "
                  f"{r[6] or 0:>7.2f}  {r[7]}")
        return

    ev, pa = load_modules()
    results = evaluate(con, ev, pa)
    asof = datetime.now().strftime("%Y-%m-%d")

    print(f"MODEL EVALUATION  {asof}   (rule D, agreed 2026-08-30)")
    print(f"baseline: h={BASELINE['h']} NW-t {BASELINE['nw_t']:.2f}, "
          f"AUC {BASELINE['auc']:.4f}, spread {BASELINE['spread']*100:+.4f}%, "
          f"{BASELINE['n_dates']} dates\n")
    print(f"  {'h':>3} {'rows':>7} {'dates':>6} {'AUC':>7} {'gross':>9} "
          f"{'net10':>9} {'NW t':>7} {'sign%':>7}  verdict")

    for r in results:
        v, msg = verdict_for(r)
        print(f"  {r['horizon']:>3} {r['n_rows']:>7} {r['n_dates']:>6} "
              f"{r['auc'] or 0:>7.4f} {r['spread_gross']*100:>8.4f}% "
              f"{r['spread_net']*100:>8.4f}% {r['nw_t'] or 0:>7.2f} "
              f"{r['sign_pct']:>6.1f}%  {v}")
        if r["horizon"] == PRIMARY_H:
            print(f"       -> {msg}")
            d_t = (r["nw_t"] or 0) - BASELINE["nw_t"]
            d_s = r["spread_gross"] - BASELINE["spread"]
            print(f"       vs baseline: NW-t {d_t:+.2f}, "
                  f"spread {d_s*100:+.4f}pp, "
                  f"dates {r['n_dates'] - BASELINE['n_dates']:+d}")

        if not args.dry_run:
            con.execute(
                "INSERT OR REPLACE INTO model_eval_history "
                "(asof_date, horizon, n_rows, n_dates, auc, spread_gross, "
                " spread_net, nw_t, naive_t, sign_pct, verdict, created_at) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                (asof, r["horizon"], r["n_rows"], r["n_dates"], r["auc"],
                 r["spread_gross"], r["spread_net"], r["nw_t"], r["naive_t"],
                 r["sign_pct"], v, datetime.now().isoformat(timespec="seconds")))

    if args.dry_run:
        print("\nDRY RUN -- nothing appended.")
    else:
        con.commit()
        print(f"\nappended {len(results)} rows to model_eval_history "
              f"(asof {asof})")
    con.close()


if __name__ == "__main__":
    main()
