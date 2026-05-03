"""
One-time bootstrap of SP500 monthly history from Yahoo Finance.

This script runs ONCE. It uses yfinance to pull ^GSPC daily history from
1960 to today, aggregates to monthly end-of-period closes, and writes the
result to recession/data/sp500_history.csv.

The CSV is then committed to git as a permanent artifact. The recession
ingest pipeline (manual_sources.fetch_sp500) reads from this CSV. yfinance
is not used again.

Why this design:
  - FRED only has ~10 years of SP500 history (post-2015)
  - Massive Polygon Indices: launched March 2023 (~3 years history)
  - Unusual Whales: focused on options/flow, not deep index history
  - Macrotrends/Stooq: free archives but no clean API
  - Yahoo via yfinance: 1928+ history, free, but library can break

By running yfinance ONCE and committing the output, we get:
  - Full historical depth (1960-2026)
  - Reproducibility (CSV in git, not subject to upstream changes)
  - Independence from yfinance (it could break tomorrow; doesn't matter)
  - Clear audit trail (the CSV is a stable artifact)

Run:
    python -m recession.data.bootstrap_sp500_history

To force a refresh of the CSV (e.g., to extend through a new year):
    python -m recession.data.bootstrap_sp500_history --force

Idempotent: by default, refuses to overwrite if the CSV already exists.
"""
from __future__ import annotations

import argparse
import csv
import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path

logger = logging.getLogger(__name__)

OUTPUT_CSV = Path(__file__).resolve().parent / "sp500_history.csv"
START_DATE = "1960-01-01"
TICKER = "^GSPC"


def aggregate_to_monthly_eop(daily_rows: list[tuple[str, float]]) -> list[dict]:
    """
    Take a list of (date_iso, close) tuples sorted ascending, return monthly
    end-of-period observations.

    Returns dicts with: month (YYYY-MM), close, vintage_date (last trading
    day + 1).
    """
    monthly: dict[str, tuple[str, float]] = {}      # ym -> (last_day, close)
    for date_iso, close in daily_rows:
        ym = date_iso[:7]
        if ym not in monthly or date_iso > monthly[ym][0]:
            monthly[ym] = (date_iso, close)

    out = []
    for ym in sorted(monthly):
        last_day_iso, close = monthly[ym]
        vintage = (datetime.fromisoformat(last_day_iso)
                    + timedelta(days=1)).date().isoformat()
        out.append({
            "month":        ym,
            "close":        close,
            "vintage_date": vintage,
        })
    return out


def main() -> int:
    parser = argparse.ArgumentParser(
        description="One-time bootstrap of SP500 monthly history from Yahoo"
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Overwrite the existing CSV (default: refuse if present)",
    )
    parser.add_argument(
        "--start", default=START_DATE,
        help=f"Start date (default: {START_DATE})",
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    if OUTPUT_CSV.exists() and not args.force:
        logger.error(
            "Refusing to overwrite existing %s. Use --force to overwrite.",
            OUTPUT_CSV,
        )
        return 1

    try:
        import yfinance as yf
    except ImportError:
        logger.error(
            "yfinance not installed. This is a one-time dependency for the "
            "bootstrap. Install with: pip install yfinance"
        )
        return 1

    end_date = datetime.now().strftime("%Y-%m-%d")
    logger.info("Pulling %s from Yahoo: %s → %s", TICKER, args.start, end_date)

    try:
        df = yf.download(
            TICKER,
            start=args.start,
            end=end_date,
            progress=False,
            auto_adjust=True,
        )
    except Exception as e:
        logger.error("Yahoo fetch failed: %s", e)
        return 1

    if df.empty:
        logger.error("Yahoo returned empty dataframe")
        return 1

    # Normalize multi-level columns (newer yfinance versions)
    if hasattr(df.columns, "get_level_values"):
        df.columns = df.columns.get_level_values(0)

    if "Close" not in df.columns:
        logger.error("Expected 'Close' column not found. Got: %s",
                     list(df.columns))
        return 1

    # Convert to list of (date_iso, close) tuples
    daily = []
    for ts, row in df.iterrows():
        try:
            close = float(row["Close"])
        except (ValueError, TypeError):
            continue
        date_iso = ts.strftime("%Y-%m-%d") if hasattr(ts, "strftime") else str(ts)[:10]
        daily.append((date_iso, close))

    daily.sort()
    logger.info("Got %d daily observations", len(daily))

    if not daily:
        logger.error("No usable rows after parsing")
        return 1

    # Aggregate to monthly EOP
    monthly = aggregate_to_monthly_eop(daily)
    logger.info(
        "Aggregated to %d monthly observations (%s → %s)",
        len(monthly), monthly[0]["month"], monthly[-1]["month"],
    )

    # Write CSV
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["month", "close", "vintage_date"])
        writer.writeheader()
        writer.writerows(monthly)

    size_kb = OUTPUT_CSV.stat().st_size / 1024
    logger.info("Wrote %s (%d rows, %.1f KB)", OUTPUT_CSV, len(monthly), size_kb)

    print(f"\n✓ Bootstrap complete.")
    print(f"  Output: {OUTPUT_CSV}")
    print(f"  Rows:   {len(monthly)} (monthly EOP, {monthly[0]['month']} → {monthly[-1]['month']})")
    print(f"\nNext: commit this CSV to git.")
    print(f"  git add {OUTPUT_CSV.relative_to(Path.cwd()) if OUTPUT_CSV.is_relative_to(Path.cwd()) else OUTPUT_CSV}")
    print(f"  git commit -m \"Bootstrap SP500 history from Yahoo (one-time)\"")
    print(f"\nThen rerun the recession ingest:")
    print(f"  rm -rf recession/cache/manual/sp500_yahoo.csv  # remove old cache if any")
    print(f"  sqlite3 recession.db \"DELETE FROM features_monthly WHERE feature_name='SP500';\"")
    print(f"  python -m recession.data.ingest --backfill --verbose")
    return 0


if __name__ == "__main__":
    sys.exit(main())
