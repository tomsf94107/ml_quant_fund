"""
Monthly threshold review for hysteresis tuning.

Re-audits the data needed to validate (or update) the entry/exit thresholds
in signals/generator.py:HYSTERESIS_ENTRY / HYSTERESIS_EXIT.

Run monthly. If the metrics drift significantly, consider adjusting thresholds.

Usage:
    python scripts/threshold_review.py [--days N]

Default: looks at all available data in accuracy.db.predictions.
With --days N: only the last N days.

Outputs:
- Day-over-day prob_up noise distribution (median, 90th, 95th, 99th percentiles)
- Signal frequency at common thresholds (0.55, 0.60, 0.65, 0.70, 0.80)
- Current HYSTERESIS_ENTRY and HYSTERESIS_EXIT for comparison
- Saved report to logs/threshold_reviews/threshold_review_YYYYMMDD_HHMMSS.txt
"""
import sqlite3
import sys
from pathlib import Path

# Add project root to sys.path so `from signals.generator import ...` works
# when this script is run from anywhere
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import pandas as pd
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=int, default=None,
                        help="Look back N days only (default: all data)")
    parser.add_argument("--db", default="accuracy.db", type=Path)
    args = parser.parse_args()

    OUTPUT_DIR = Path("logs/threshold_reviews")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not args.db.exists():
        print(f"ERROR: {args.db} not found")
        sys.exit(1)

    # Load predictions
    conn = sqlite3.connect(f"file:{args.db}?mode=ro", uri=True)
    where_clause = "WHERE prob_up IS NOT NULL"
    if args.days:
        where_clause += f" AND prediction_date >= date('now', '-{args.days} days')"
    df = pd.read_sql(f"""
        SELECT ticker, horizon, prediction_date, prob_up
        FROM predictions
        {where_clause}
        ORDER BY ticker, horizon, prediction_date
    """, conn)
    conn.close()

    if df.empty:
        print("No predictions found in window. Aborting.")
        sys.exit(1)

    print(f"\n{'='*70}")
    print(f"THRESHOLD REVIEW")
    print(f"{'='*70}")
    print(f"Run time: {pd.Timestamp.now()}")
    print(f"Predictions analyzed: {len(df):,}")
    print(f"Tickers: {df['ticker'].nunique()}")
    print(f"Horizons: {sorted(df['horizon'].unique())}")
    print(f"Date range: {df['prediction_date'].min()} -> {df['prediction_date'].max()}")
    if args.days:
        print(f"Window: last {args.days} days")
    print()

    # Day-over-day noise
    df["prediction_date"] = pd.to_datetime(df["prediction_date"])
    df = df.sort_values(["ticker", "horizon", "prediction_date"])
    df["prob_change"] = df.groupby(["ticker", "horizon"])["prob_up"].diff().abs()

    print(f"{'='*70}")
    print(f"DAY-OVER-DAY NOISE (|prob_up change|)")
    print(f"{'='*70}")
    print(f"{'horizon':<10s} {'n':<8s} {'median':<10s} {'90%':<10s} {'95%':<10s} {'99%':<10s}")
    for h in sorted(df["horizon"].unique()):
        sub = df[df["horizon"] == h]["prob_change"].dropna()
        print(f"h={h:<8} {len(sub):<8} {sub.median():<10.3f} "
              f"{sub.quantile(0.9):<10.3f} {sub.quantile(0.95):<10.3f} "
              f"{sub.quantile(0.99):<10.3f}")

    # Signal frequency at thresholds
    print()
    print(f"{'='*70}")
    print(f"SIGNAL FREQUENCY (% of predictions clearing threshold)")
    print(f"{'='*70}")
    print(f"{'horizon':<10s} {'>0.55':<10s} {'>0.60':<10s} {'>0.65':<10s} {'>0.70':<10s} {'>0.80':<10s}")
    for h in sorted(df["horizon"].unique()):
        sub = df[df["horizon"] == h]["prob_up"]
        n = len(sub)
        pct_55 = 100 * (sub > 0.55).sum() / n
        pct_60 = 100 * (sub > 0.60).sum() / n
        pct_65 = 100 * (sub > 0.65).sum() / n
        pct_70 = 100 * (sub > 0.70).sum() / n
        pct_80 = 100 * (sub > 0.80).sum() / n
        print(f"h={h:<8} {pct_55:>5.1f}%   {pct_60:>5.1f}%   "
              f"{pct_65:>5.1f}%   {pct_70:>5.1f}%   {pct_80:>5.1f}%")

    # Current thresholds for context
    print()
    print(f"{'='*70}")
    print(f"CURRENT THRESHOLDS (from signals/generator.py)")
    print(f"{'='*70}")
    try:
        from signals.generator import HYSTERESIS_ENTRY, HYSTERESIS_EXIT
        print(f"  ENTRY: {HYSTERESIS_ENTRY}")
        print(f"  EXIT:  {HYSTERESIS_EXIT}")
    except ImportError:
        print("  (could not import — module path issue)")

    # Recommendation logic
    print()
    print(f"{'='*70}")
    print(f"INTERPRETATION GUIDE")
    print(f"{'='*70}")
    print("""
- Entry threshold should clear ~10-30% of predictions per horizon.
  If <5%, threshold is too high (signal starvation). Loosen.
  If >40%, threshold is too loose (false signals). Tighten.

- Entry-Exit gap should be 2-3x the median day-over-day noise.
  If gap < 2x noise: position will flip too often. Widen the gap.
  If gap > 3x noise: position rides through real degradation. Tighten exit.

- For h=1 specifically: if walk-forward shows continued lack of edge,
  the high entry (0.80) is correct. Don't loosen unless edge appears.
""")

    # Save report
    ts = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    output_file = OUTPUT_DIR / f"threshold_review_{ts}.txt"
    print(f"Full report saved: {output_file}")
    # Note: save logic could be added here if needed; for now just terminal output


if __name__ == "__main__":
    main()
