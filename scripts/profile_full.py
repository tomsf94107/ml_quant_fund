"""
Full memory leak profiler — mirrors daily_runner per-ticker logic exactly.

Calls build_feature_dataframe + generate_signals (3 horizons) per ticker,
takes tracemalloc snapshots every 10 tickers, identifies leak sources.

Usage:
    python -m scripts.profile_full
"""
import sys
import gc
import tracemalloc
import time
import os
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import resource
def rss_mb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 / 1024

LOG_DIR = ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)
log_path = LOG_DIR / f"profile_full_{datetime.now().strftime('%Y%m%d_%H%M')}.log"
log_file = open(log_path, "w", buffering=1)

def log(msg):
    line = f"{datetime.now().strftime('%H:%M:%S')}  {msg}"
    print(line)
    log_file.write(line + "\n")

log("="*70)
log(f"FULL MEMORY LEAK PROFILER (mirrors daily_runner)")
log(f"Log: {log_path}")
log(f"Initial RSS: {rss_mb():.0f}MB")
log("="*70)

tracemalloc.start(25)

# Heavy imports — same as daily_runner
log("Importing modules...")
import_start = time.time()
from features.builder import build_feature_dataframe
from features import massive_client as mc
from signals.generator import generate_signals
log(f"Imports done in {time.time()-import_start:.1f}s, RSS={rss_mb():.0f}MB")

# Constants from daily_runner
TRAIN_START = "2024-01-01"
HORIZONS = [1, 3, 5]
BUY_THRESHOLD = 0.55

def load_tickers():
    with open(ROOT / "tickers.txt") as f:
        return [t.strip() for t in f if t.strip()]

tickers = load_tickers()
log(f"Loaded {len(tickers)} tickers")

baseline_snap = None
last_snap = None
results = []

for i, ticker in enumerate(tickers, 1):
    cur_rss = rss_mb()

    if i == 2 or i % 10 == 0:
        snap = tracemalloc.take_snapshot()
        if baseline_snap is None:
            baseline_snap = snap
            log(f"[trace] === BASELINE at ticker {i} (RSS={cur_rss:.0f}MB) ===")
        else:
            log(f"[trace] === Ticker {i} (RSS={cur_rss:.0f}MB) ===")
            log(f"[trace] Top 10 growth files since BASELINE:")
            diff_base = snap.compare_to(baseline_snap, "filename")
            for stat in diff_base[:10]:
                log(f"[trace]   {stat}")
            if last_snap is not None:
                log(f"[trace] Top 5 growth files since LAST 10 tickers:")
                diff_last = snap.compare_to(last_snap, "filename")
                for stat in diff_last[:5]:
                    log(f"[trace]   {stat}")
        last_snap = snap

    log(f"[{i:3d}/{len(tickers)}] {ticker}  RSS={cur_rss:.0f}MB")

    try:
        # Pre-check
        check_df = mc.download(ticker, start="2024-12-01", end="2025-01-01",
                                auto_adjust=True, progress=False)
        if check_df.empty:
            log(f"  ⚠ {ticker} no Massive data — skipping")
            del check_df
            continue
        del check_df

        # Build features
        df = build_feature_dataframe(ticker, start_date=TRAIN_START)
        log(f"  features built: {df.shape}, RSS={rss_mb():.0f}MB")

        # Generate signals for each horizon (THIS is what was missing in profile_memory_leak)
        for horizon in HORIZONS:
            try:
                sig = generate_signals(ticker, df, horizon=horizon,
                                        confidence_threshold=BUY_THRESHOLD)
                results.append({
                    "ticker": ticker,
                    "horizon": horizon,
                    "signal": sig.today_signal,
                    "prob": sig.today_prob,
                })
                # Explicit cleanup of signal result
                del sig
            except Exception as e:
                log(f"  ✗ {ticker} h={horizon} signal failed: {e}")

        # Explicit cleanup
        del df
        gc.collect()

    except Exception as e:
        log(f"  ✗ {ticker} failed: {e}")
        continue

    # Stop early to avoid OOM kill
    if cur_rss > 2500:
        log(f"!!! RSS exceeded 2500MB — stopping early")
        break

# Final snapshot
final_snap = tracemalloc.take_snapshot()
log("="*70)
log(f"FINAL ANALYSIS")
log(f"Final RSS: {rss_mb():.0f}MB")
log(f"Tickers processed: {i}")
log(f"Signals generated: {len(results)}")
log("="*70)

if baseline_snap:
    log("\n=== TOP 25 LEAKING FILES (since baseline) ===")
    diff = final_snap.compare_to(baseline_snap, "filename")
    for stat in diff[:25]:
        log(f"  {stat}")

    log("\n=== TOP 25 LEAKING LINES (since baseline) ===")
    diff_line = final_snap.compare_to(baseline_snap, "lineno")
    for stat in diff_line[:25]:
        log(f"  {stat}")

log_file.close()
print(f"\n✅ Done. Log: {log_path}")
