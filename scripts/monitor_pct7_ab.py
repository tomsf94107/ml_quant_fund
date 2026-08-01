#!/usr/bin/env python3
"""
scripts/monitor_pct7_ab.py — Phase epsilon A/B monitoring

Compares prob_pct7 (shadow-logged) vs prob_up (production) on hit rates.

Key distinction:
  prob_up        predicts "any positive in N days" (binary, ~50% base rate)
  prob_pct7      predicts "+7% move in 5 days" (binary, ~13% base rate)

To fairly evaluate prob_pct7, we use a different ground truth:
  actual_up      → return > 0 (matches prob_up)
  actual_pct7    → return >= 0.07 (matches prob_pct7)

Usage:
  python scripts/monitor_pct7_ab.py
  python scripts/monitor_pct7_ab.py --since 2026-05-25
  python scripts/monitor_pct7_ab.py --threshold 0.20  # prob_pct7 BUY threshold

Reports:
  - n_predictions (with prob_pct7)
  - n_with_outcomes (joined to outcomes table)
  - prob_up hit rate (actual_up=1 given prob_up>0.5)
  - prob_pct7 hit rate (actual_pct7=1 given prob_pct7>threshold)
  - Calibration: bucketed hit rates per prob bin
"""
import argparse
import sqlite3
from pathlib import Path

DB = Path("accuracy.db")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--since", default="2026-05-25",
                        help="Start date (inclusive) for predictions, default Phase eps launch")
    parser.add_argument("--threshold", type=float, default=0.20,
                        help="prob_pct7 BUY threshold (default 0.20, since training base rate was 13%)")
    parser.add_argument("--horizon", type=int, default=5, help="Horizon (PCT7 only trained on h=5)")
    args = parser.parse_args()

    if not DB.exists():
        print(f"DB not found: {DB}")
        return

    conn = sqlite3.connect(str(DB), timeout=30)

    # 1. Count predictions logged with prob_pct7
    n_preds = conn.execute("""
        SELECT COUNT(*) FROM predictions
        WHERE prediction_date >= ? AND horizon = ? AND prob_pct7 IS NOT NULL
    """, (args.since, args.horizon)).fetchone()[0]
    print(f"Predictions since {args.since} (h={args.horizon}) with prob_pct7: {n_preds}")

    # 2. Count those with outcomes joined
    n_joined = conn.execute("""
        SELECT COUNT(*)
        FROM predictions p
        JOIN outcomes o
          ON p.ticker = o.ticker
         AND p.prediction_date = o.prediction_date
         AND p.horizon = o.horizon
        WHERE p.prediction_date >= ? AND p.horizon = ? AND p.prob_pct7 IS NOT NULL
    """, (args.since, args.horizon)).fetchone()[0]
    print(f"With outcomes joined: {n_joined}")

    if n_joined == 0:
        print(f"\nNo outcomes yet. Wait until {args.horizon} business days after first prediction.")
        return

    # 3. prob_up hit rate (predicted up = actual up)
    prob_up_hit = conn.execute("""
        SELECT
            SUM(CASE WHEN p.prob_up > 0.5 AND o.actual_up = 1 THEN 1
                     WHEN p.prob_up <= 0.5 AND o.actual_up = 0 THEN 1
                     ELSE 0 END) * 1.0 / COUNT(*) AS hit_rate,
            COUNT(*) AS n,
            AVG(o.actual_up) * 1.0 AS base_rate
        FROM predictions p
        JOIN outcomes o
          ON p.ticker = o.ticker AND p.prediction_date = o.prediction_date AND p.horizon = o.horizon
        WHERE p.prediction_date >= ? AND p.horizon = ? AND p.prob_pct7 IS NOT NULL
    """, (args.since, args.horizon)).fetchone()
    print(f"\nprob_up (any positive prediction):")
    print(f"  Hit rate: {prob_up_hit[0]:.3f} (n={prob_up_hit[1]}, base rate={prob_up_hit[2]:.3f})")

    # 4. prob_pct7 hit rate (predicted +7% move = actual +7% move)
    prob_pct7_hit = conn.execute("""
        SELECT
            SUM(CASE WHEN p.prob_pct7 > ? AND o.actual_return >= 0.07 THEN 1
                     WHEN p.prob_pct7 <= ? AND o.actual_return < 0.07 THEN 1
                     ELSE 0 END) * 1.0 / COUNT(*) AS hit_rate,
            COUNT(*) AS n,
            AVG(CASE WHEN o.actual_return >= 0.07 THEN 1.0 ELSE 0.0 END) AS base_rate_pct7,
            SUM(CASE WHEN p.prob_pct7 > ? THEN 1 ELSE 0 END) AS n_buy_pct7,
            SUM(CASE WHEN p.prob_pct7 > ? AND o.actual_return >= 0.07 THEN 1 ELSE 0 END) AS n_buy_hit_pct7
        FROM predictions p
        JOIN outcomes o
          ON p.ticker = o.ticker AND p.prediction_date = o.prediction_date AND p.horizon = o.horizon
        WHERE p.prediction_date >= ? AND p.horizon = ? AND p.prob_pct7 IS NOT NULL
    """, (args.threshold, args.threshold, args.threshold, args.threshold, args.since, args.horizon)).fetchone()

    print(f"\nprob_pct7 (>= +7% in {args.horizon}d) at threshold {args.threshold}:")
    print(f"  Hit rate (binary acc): {prob_pct7_hit[0]:.3f} (n={prob_pct7_hit[1]})")
    print(f"  Base rate +7%:         {prob_pct7_hit[2]:.3f}")
    if prob_pct7_hit[3] > 0:
        precision = prob_pct7_hit[4] / prob_pct7_hit[3]
        print(f"  BUY precision: {prob_pct7_hit[4]}/{prob_pct7_hit[3]} = {precision:.3f}")

    # 5. Calibration check — bucket prob_pct7 and look at actual hit rate
    print(f"\nCalibration buckets (prob_pct7 -> actual_pct7 rate):")
    print(f"  {'bucket':<15} {'n':>5} {'avg_pred':>10} {'actual':>10}")
    buckets = conn.execute("""
        SELECT
            CASE
                WHEN p.prob_pct7 < 0.05 THEN '[0.00, 0.05)'
                WHEN p.prob_pct7 < 0.10 THEN '[0.05, 0.10)'
                WHEN p.prob_pct7 < 0.15 THEN '[0.10, 0.15)'
                WHEN p.prob_pct7 < 0.20 THEN '[0.15, 0.20)'
                WHEN p.prob_pct7 < 0.30 THEN '[0.20, 0.30)'
                ELSE '[0.30, 1.00]'
            END AS bucket,
            COUNT(*) AS n,
            AVG(p.prob_pct7) AS avg_pred,
            AVG(CASE WHEN o.actual_return >= 0.07 THEN 1.0 ELSE 0.0 END) AS actual_rate
        FROM predictions p
        JOIN outcomes o ON p.ticker = o.ticker AND p.prediction_date = o.prediction_date AND p.horizon = o.horizon
        WHERE p.prediction_date >= ? AND p.horizon = ? AND p.prob_pct7 IS NOT NULL
        GROUP BY bucket ORDER BY bucket
    """, (args.since, args.horizon)).fetchall()
    for b in buckets:
        print(f"  {b[0]:<15} {b[1]:>5} {b[2]:>10.3f} {b[3]:>10.3f}")

    conn.close()


if __name__ == "__main__":
    main()
