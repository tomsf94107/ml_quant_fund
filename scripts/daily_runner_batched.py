"""
Subprocess-batched daily_runner.

Runs daily_runner across all 125 tickers in 3 separate Python subprocesses
to avoid cumulative memory state hitting kernel jetsam threshold around
ticker 99-114 on this Mac.

Each subprocess:
  - Starts fresh (~100MB Python baseline)
  - Processes ~42 tickers
  - Peaks at ~400-500MB RSS
  - Writes to cache (cache merge logic preserves earlier batches)
  - Writes to DB (INSERT OR REPLACE preserves earlier batches)
  - Exits cleanly, releasing all C-extension memory

Cache merge logic in daily_runner.py (commit 3a97ae4) ensures rich data
from all 3 batches is combined into final cache.

Usage:
    python -m scripts.daily_runner_batched

Memory profile:
    Single-process run: 248MB → 372MB+ → kernel kill at ~ticker 99
    Batched run:        Each batch starts at 100MB, peaks ~400MB, exits clean
"""
import sys
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

PYTHON = "/Users/atomnguyen/.pyenv/versions/ml_quant_310/bin/python"
TICKERS_FILE = ROOT / "tickers.txt"


def load_tickers():
    return [t.strip() for t in TICKERS_FILE.read_text().splitlines() if t.strip()]


def run_batch(start_from: str | None, end_at: str | None, batch_label: str):
    """Spawn a fresh Python subprocess for one batch."""
    print(f"\n{'='*60}")
    print(f"BATCH {batch_label}: {start_from or '<start>'} → {end_at or '<end>'}")
    print(f"{'='*60}")

    args = [PYTHON, "-c", f"""
import sys
sys.path.insert(0, '.')
from scripts.daily_runner import run_daily
run_daily(force=True, start_from={repr(start_from)}, end_at={repr(end_at)})
"""]

    env = {"DAILY_RUNNER_FORCE_GC": "1", "PATH": "/usr/bin:/bin:/usr/sbin:/sbin"}
    import os
    env.update({k: v for k, v in os.environ.items() if k not in env})

    start = time.time()
    result = subprocess.run(args, env=env, cwd=str(ROOT))
    elapsed = time.time() - start

    print(f"\nBATCH {batch_label}: exit={result.returncode}, elapsed={elapsed:.1f}s")
    return result.returncode


def main():
    tickers = load_tickers()
    n = len(tickers)
    print(f"Loaded {n} tickers")

    # 3 batches of ~42 tickers each
    batch_size = (n + 2) // 3
    batches = []
    for i in range(0, n, batch_size):
        chunk = tickers[i:i + batch_size]
        batches.append((chunk[0], chunk[-1]))

    print(f"Splitting into {len(batches)} batches:")
    for i, (s, e) in enumerate(batches, 1):
        idx_s = tickers.index(s)
        idx_e = tickers.index(e)
        print(f"  Batch {i}: {s} ({idx_s+1}) → {e} ({idx_e+1})  [{idx_e - idx_s + 1} tickers]")

    print()
    overall_start = time.time()

    for i, (start_ticker, end_ticker) in enumerate(batches, 1):
        # First batch: start_from=None means run from beginning
        sf = None if i == 1 else start_ticker
        rc = run_batch(sf, end_ticker, f"{i}/{len(batches)}")
        if rc != 0:
            print(f"\n⚠ Batch {i} exited with non-zero code {rc}")
            print("Continuing to next batch (cache merge handles partial results)")

    overall_elapsed = time.time() - overall_start
    print(f"\n{'='*60}")
    print(f"ALL BATCHES COMPLETE — total {overall_elapsed:.1f}s ({overall_elapsed/60:.1f}min)")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
