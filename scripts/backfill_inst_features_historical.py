"""
scripts/backfill_inst_features_historical.py
─────────────────────────────────────────────
Backfill the 4 inst_* features into accuracy.db.prediction_features for
all past predictions that have NULL inst values.

Uses features.institutional_features.load_institutional_features_pit_fast
(DuckDB-backed, ~5ms per ticker per date batch).

Session F + E followup, May 22 2026.
"""

from __future__ import annotations

import logging
import os
import sqlite3
import sys
import time
from pathlib import Path

import pandas as pd

os.environ.setdefault("ML_QUANT_INST_FEATURES", "1")

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)

DB_PATH = Path("accuracy.db")

INST_COLS = [
    "inst_signed_flow_5d",
    "inst_signed_flow_30d",
    "inst_block_buy_sell_7d",
    "inst_auction_imbal_5d",
]


def main():
    conn = sqlite3.connect(DB_PATH, timeout=30)
    conn.execute("PRAGMA journal_mode=WAL")

    # Find rows needing backfill
    rows = conn.execute(
        """
        SELECT DISTINCT ticker, prediction_date
        FROM prediction_features
        WHERE inst_signed_flow_5d IS NULL
        ORDER BY ticker, prediction_date
        """
    ).fetchall()

    n = len(rows)
    log.info(f"Found {n} (ticker, date) pairs needing inst backfill")

    if n == 0:
        log.info("Nothing to do")
        conn.close()
        return

    # Group by ticker for efficient bulk lookup
    from collections import defaultdict
    by_ticker = defaultdict(list)
    for t, d in rows:
        by_ticker[t].append(d)

    log.info(f"Spread across {len(by_ticker)} tickers")

    from features.institutional_features import load_institutional_features_pit_fast

    n_updated = 0
    n_failed = 0
    start = time.time()

    for i, (ticker, dates) in enumerate(by_ticker.items(), 1):
        try:
            date_idx = pd.DatetimeIndex(dates)
            df_inst = load_institutional_features_pit_fast(ticker, date_idx)

            # Update each row
            for d in dates:
                d_pd = pd.Timestamp(d)
                if d_pd not in df_inst.index:
                    continue
                vals = df_inst.loc[d_pd]

                # Skip if all NaN (no data available for that date)
                if pd.isna(vals[INST_COLS[0]]):
                    continue

                conn.execute(
                    f"""
                    UPDATE prediction_features
                    SET inst_signed_flow_5d = ?,
                        inst_signed_flow_30d = ?,
                        inst_block_buy_sell_7d = ?,
                        inst_auction_imbal_5d = ?
                    WHERE ticker = ? AND prediction_date = ?
                    """,
                    (
                        float(vals.get("inst_signed_flow_5d", 0) or 0),
                        float(vals.get("inst_signed_flow_30d", 0) or 0),
                        float(vals.get("inst_block_buy_sell_7d", 0) or 0),
                        float(vals.get("inst_auction_imbal_5d", 0) or 0),
                        ticker,
                        d,
                    )
                )
                n_updated += 1

            if i % 20 == 0:
                conn.commit()
                log.info(f"[{i}/{len(by_ticker)}] {ticker}: cumulative {n_updated} updates")

        except Exception as e:
            n_failed += 1
            log.error(f"  {ticker} backfill failed: {type(e).__name__}: {e}")

    conn.commit()
    conn.close()

    elapsed = time.time() - start
    log.info(f"DONE — {n_updated} rows updated, {n_failed} tickers failed, {elapsed:.0f}s")


if __name__ == "__main__":
    main()
