#!/usr/bin/env python3
"""
scripts/ab_daily_workflow.py

True A/B sentiment test, run AFTER Pipeline C completes.

What it does:
  1. Reads existing signals_cache.json (= Run B, sentiment ON)
  2. Saves backup as ab_cache_B_YYYYMMDD.json
  3. Re-generates signals with use_sentiment=False (Run A)
  4. Saves Run A as ab_cache_A_YYYYMMDD.json
  5. Restores signals_cache.json from B (so dashboard/trading uses ON version)

Why this is faster than re-running Pipeline C:
  - Skips Stage 0 (no sentiment fetch needed for OFF run)
  - Skips Stage 1 (UW snapshot reuses cached data from Run B)
  - Only re-runs Stage 2 (signal generation) with use_sentiment=False
  - Uses already-built feature DataFrames (cached by builder)

Runtime: ~5 min (vs 70 min for full re-run)

Usage:
  After Pipeline C completes (~20:09 VN), run:
    python scripts/ab_daily_workflow.py

  Or add to pipeline_C_preopen.sh as Stage 3.
"""
from __future__ import annotations
import sys, os, json, shutil, sqlite3, logging
from pathlib import Path
from datetime import datetime
from typing import Any

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s'
)
log = logging.getLogger(__name__)

ROOT = Path("/Users/atomnguyen/Desktop/ML_Quant_Fund")
CACHE = ROOT / "data" / "signals_cache.json"


def main():
    today = datetime.now().strftime("%Y%m%d")
    cache_a = ROOT / "data" / f"ab_cache_A_{today}.json"
    cache_b = ROOT / "data" / f"ab_cache_B_{today}.json"

    if not CACHE.exists():
        log.error("No signals_cache.json — run Pipeline C first")
        sys.exit(1)

    # Step 1: Save Run B (sentiment ON, current cache)
    shutil.copy(CACHE, cache_b)
    with open(cache_b) as f:
        b_data = json.load(f)
    log.info(f"Saved Run B (sentiment ON): {len(b_data.get('signals', []))} signals")

    # Step 2: Re-generate signals with sentiment OFF
    log.info("Re-generating signals with sentiment OFF...")

    from features.builder import build_feature_dataframe
    from signals.generator import generate_signals
    from scripts.daily_runner import (
        load_tickers, today_et, BUY_THRESHOLD, HORIZONS, TRAIN_START
    )

    tickers = load_tickers()
    run_date = today_et()
    log.info(f"Tickers: {len(tickers)} | Horizons: {HORIZONS}")

    a_results = []
    failed = []

    for i, ticker in enumerate(tickers, 1):
        if i % 25 == 0:
            log.info(f"  [{i}/{len(tickers)}] processed")
        try:
            df = build_feature_dataframe(ticker, start_date=TRAIN_START)

            for horizon in HORIZONS:
                try:
                    # KEY DIFFERENCE: use_sentiment=False
                    sig = generate_signals(
                        ticker, df,
                        horizon=horizon,
                        confidence_threshold=BUY_THRESHOLD,
                        use_sentiment=False,
                    )

                    a_results.append({
                        "ticker":          ticker,
                        "horizon":         horizon,
                        "signal":          sig.today_signal,
                        "prob":            sig.today_prob,
                        "prob_eff":        sig.today_prob_eff,
                        "run_date":        run_date,
                        "current_price":   sig.current_price,
                    })
                except Exception as e:
                    failed.append((ticker, horizon, str(e)))
        except Exception as e:
            log.warning(f"{ticker}: {e}")
            failed.append((ticker, "all", str(e)))

    # Step 3: Save Run A (sentiment OFF)
    a_data = {
        "generated_at": datetime.now().isoformat(),
        "date":         run_date,
        "use_sentiment": False,
        "label":        "A_sentiment_off",
        "signals":      a_results,
    }
    with open(cache_a, 'w') as f:
        json.dump(a_data, f, indent=2)
    log.info(f"Saved Run A (sentiment OFF): {len(a_results)} signals — failed={len(failed)}")

    # Step 4: Restore signals_cache.json from Run B (so live system uses ON version)
    shutil.copy(cache_b, CACHE)
    log.info(f"Restored signals_cache.json from Run B")

    # Step 5: Quick comparison summary
    sigs_a = {(s['ticker'], s['horizon']): s for s in a_results}
    sigs_b = {(s['ticker'], s['horizon']): s for s in b_data.get('signals', [])}

    common = set(sigs_a.keys()) & set(sigs_b.keys())
    diffs = []
    for k in common:
        pa = sigs_a[k].get('prob_eff', sigs_a[k].get('prob', 0.5))
        pb = sigs_b[k].get('prob_eff', sigs_b[k].get('prob', 0.5))
        sa = sigs_a[k].get('signal', '?')
        sb = sigs_b[k].get('signal', '?')
        if abs(pa - pb) > 0.01 or sa != sb:
            diffs.append((k[0], k[1], pa, pb, sa, sb))

    log.info(f"\n{'='*60}")
    log.info(f"  Daily AB — Sentiment ON vs OFF")
    log.info(f"  Date: {run_date}")
    log.info(f"  Common predictions: {len(common)}")
    log.info(f"  Predictions changed: {len(diffs)}")
    log.info(f"{'='*60}")

    # Top 10 biggest changes
    if diffs:
        log.info(f"\n  Top 10 changes (by abs prob_eff diff):")
        for ticker, horizon, pa, pb, sa, sb in sorted(
            diffs, key=lambda x: abs(x[3]-x[2]), reverse=True
        )[:10]:
            arrow = '↑' if pb > pa else '↓'
            sig_changed = '*' if sa != sb else ' '
            log.info(f"    {sig_changed}{ticker:6s} h={horizon}d: "
                     f"A={pa:.3f}({sa}) B={pb:.3f}({sb}) {arrow} ({pb-pa:+.3f})")

    # Count signal flips per horizon
    log.info(f"\n  Signal flips per horizon:")
    for h in [1, 3, 5]:
        flips = sum(1 for d in diffs if d[1] == h and d[4] != d[5])
        log.info(f"    h={h}d: {flips} flips")

    log.info(f"\nFiles saved:")
    log.info(f"  Run A (OFF): {cache_a.name}")
    log.info(f"  Run B (ON):  {cache_b.name}")
    log.info(f"\nTomorrow after market close, run:")
    log.info(f"  python scripts/ab_compare_outcomes.py --date {run_date}")


if __name__ == "__main__":
    main()
