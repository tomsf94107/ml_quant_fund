"""
Standalone memory leak profiler for daily_runner per-ticker logic.

Calls the same imports/functions as daily_runner's loop, but with
tracemalloc snapshots every 10 tickers. Does NOT touch daily_runner.py.

Usage:
    python -m scripts.profile_memory_leak

Output: writes to logs/profile_memory_leak_YYYYMMDD_HHMM.log
"""
import sys
import gc
import tracemalloc
import time
import os
from pathlib import Path
from datetime import datetime

# Project root setup
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

# Load tickers from same source as daily_runner
def load_tickers():
    with open(ROOT / "tickers.txt") as f:
        return [t.strip() for t in f if t.strip()]

# Mem helpers
import resource
def rss_mb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 / 1024

# Logging
LOG_DIR = ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)
log_path = LOG_DIR / f"profile_memory_leak_{datetime.now().strftime('%Y%m%d_%H%M')}.log"
log_file = open(log_path, "w", buffering=1)

def log(msg):
    line = f"{datetime.now().strftime('%H:%M:%S')}  {msg}"
    print(line)
    log_file.write(line + "\n")

# Start
log("="*70)
log(f"MEMORY LEAK PROFILER")
log(f"Log: {log_path}")
log(f"Python: {sys.version.split()[0]}")
log(f"Initial RSS: {rss_mb():.0f}MB")
log("="*70)

# Start tracemalloc BEFORE any heavy imports
tracemalloc.start(25)

# Heavy imports (mirror daily_runner)
log("Importing modules...")
import_start = time.time()
from features.builder import build_feature_dataframe
from features import massive_client as mc
log(f"Imports done in {time.time()-import_start:.1f}s, RSS={rss_mb():.0f}MB")

# Constants from daily_runner
TRAIN_START = "2024-01-01"
HORIZONS = [1, 3, 5]

# Setup
tickers = load_tickers()
log(f"Loaded {len(tickers)} tickers")

baseline_snap = None
last_snap = None
results = []

# Loop
for i, ticker in enumerate(tickers, 1):
    cur_rss = rss_mb()

    # Snapshot every 10 tickers (and at ticker 2 for baseline)
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
        # Pre-check (mirror daily_runner)
        check_df = mc.download(ticker, start="2024-12-01", end="2025-01-01",
                                auto_adjust=True, progress=False)
        if check_df.empty:
            log(f"  ⚠ {ticker} no Massive data — skipping")
            continue

        # Build features (the heavy operation)
        df = build_feature_dataframe(ticker, start_date=TRAIN_START)
        log(f"  features built: {df.shape}")

        # Force GC like daily_runner
        gc.collect()

    except Exception as e:
        log(f"  ✗ {ticker} failed: {e}")
        continue

    # Stop early if memory is dangerous to avoid OOM kill
    if cur_rss > 2500:
        log(f"!!! RSS exceeded 2500MB — stopping early to avoid OOM kill")
        break

# Final snapshot
final_snap = tracemalloc.take_snapshot()
log("="*70)
log(f"FINAL ANALYSIS")
log(f"Final RSS: {rss_mb():.0f}MB")
log(f"Tickers processed: {i}")
log("="*70)

if baseline_snap:
    log("\n=== TOP 20 LEAKING FILES (since baseline) ===")
    diff = final_snap.compare_to(baseline_snap, "filename")
    for stat in diff[:20]:
        log(f"  {stat}")

    log("\n=== TOP 20 LEAKING LINES (since baseline) ===")
    diff_line = final_snap.compare_to(baseline_snap, "lineno")
    for stat in diff_line[:20]:
        log(f"  {stat}")

log_file.close()
print(f"\n✅ Done. Log: {log_path}")
