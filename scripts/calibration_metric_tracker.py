"""
Calibration metric tracker.

Computes Expected Calibration Error (ECE) per horizon for closed predictions
(those with outcomes). Outputs both prob_raw and prob_eff calibration.

ECE = sum over buckets of |bucket_avg_prob - bucket_hit_rate| * (bucket_n / total_n)

Lower is better. ECE = 0 means perfect calibration.

Usage:
  python scripts/calibration_metric_tracker.py
  python scripts/calibration_metric_tracker.py --since 2026-05-08
  python scripts/calibration_metric_tracker.py --signal BUY --horizon 5

Output: per-horizon ECE for prob_raw and prob_eff, with calibration table.
"""
import argparse
import sqlite3
import sys
from datetime import datetime


def compute_ece(rows, n_buckets=10):
    """rows: list of (prob, hit) tuples. Returns ECE."""
    if not rows:
        return None
    total = len(rows)
    rows.sort(key=lambda x: x[0])
    bucket_size = total // n_buckets
    if bucket_size == 0:
        return None

    ece = 0.0
    buckets_used = 0
    for i in range(n_buckets):
        start = i * bucket_size
        end = start + bucket_size if i < n_buckets - 1 else total
        bucket = rows[start:end]
        if not bucket:
            continue
        avg_prob = sum(r[0] for r in bucket) / len(bucket)
        hit_rate = sum(r[1] for r in bucket) / len(bucket)
        ece += abs(avg_prob - hit_rate) * len(bucket) / total
        buckets_used += 1
    return ece


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--since", default="2026-05-08", help="Start date YYYY-MM-DD")
    ap.add_argument("--signal", default="BUY", choices=["BUY", "HOLD", "ALL"])
    ap.add_argument("--horizon", type=int, default=None, help="If set, only this horizon")
    ap.add_argument("--db", default="accuracy.db")
    args = ap.parse_args()

    sig_filter = "" if args.signal == "ALL" else f"AND p.signal = '{args.signal}'"
    h_filter = f"AND p.horizon = {args.horizon}" if args.horizon else ""

    sql = f"""
        SELECT p.horizon, p.prob_up, p.prob_raw, o.actual_up
        FROM predictions p
        JOIN outcomes o USING(ticker, prediction_date, horizon)
        WHERE p.prediction_date >= ?
          AND o.actual_up IS NOT NULL
          AND p.prob_up IS NOT NULL
          {sig_filter}
          {h_filter}
    """
    
    conn = sqlite3.connect(args.db, timeout=30)
    rows = conn.execute(sql, (args.since,)).fetchall()
    conn.close()

    if not rows:
        print(f"No data since {args.since} matching filters.")
        return

    print(f"\n{'='*60}")
    print(f"  Calibration Metric Tracker")
    print(f"  Window: {args.since} → now")
    print(f"  Signal: {args.signal}")
    if args.horizon:
        print(f"  Horizon: {args.horizon}")
    print(f"  Total rows: {len(rows)}")
    print(f"{'='*60}\n")

    # Group by horizon
    by_h = {}
    for h, peff, praw, actual in rows:
        by_h.setdefault(h, []).append((peff, praw, actual))

    for h in sorted(by_h.keys()):
        data = by_h[h]
        eff_data = [(r[0], r[2]) for r in data if r[0] is not None]
        raw_data = [(r[1], r[2]) for r in data if r[1] is not None]

        ece_eff = compute_ece(eff_data, n_buckets=10)
        ece_raw = compute_ece(raw_data, n_buckets=10)
        
        avg_eff = sum(r[0] for r in eff_data) / len(eff_data) if eff_data else 0
        avg_raw = sum(r[0] for r in raw_data) / len(raw_data) if raw_data else 0
        hit_rate = sum(r[1] for r in eff_data) / len(eff_data) if eff_data else 0

        print(f"h={h}  n={len(data):4d}")
        print(f"  avg_prob_eff: {avg_eff*100:5.1f}%   avg_prob_raw: {avg_raw*100:5.1f}%   hit_rate: {hit_rate*100:5.1f}%")
        if ece_eff is not None:
            print(f"  ECE(prob_eff): {ece_eff*100:5.2f}pp")
        if ece_raw is not None:
            print(f"  ECE(prob_raw): {ece_raw*100:5.2f}pp")
        if ece_eff is not None and ece_raw is not None:
            improvement = ece_eff - ece_raw
            sign = "+" if improvement > 0 else ""
            print(f"  Δ (eff-raw):   {sign}{improvement*100:.2f}pp  ({'eff worse' if improvement > 0 else 'eff better'})")
        print()


if __name__ == "__main__":
    main()
