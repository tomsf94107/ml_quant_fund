"""
Subprocess-batched walk_forward.

Per memory #34/35, walk_forward has hung overnight repeatedly. Single
process loading 8 years × 125 tickers of features hits cumulative
memory + network state limits.

Each subprocess processes BATCH_SIZE=10 tickers, peaks ~1-2GB, exits
clean. Output CSVs use append mode so all 125 tickers accumulate.

Combines:
- Phase 1: faulthandler diagnostics (in walk_forward.py)
- Phase 2: Massive bounded retries + session reset (massive_client.py)
- Phase 2: yfinance hardening + PIT gate (yf_resilient.py)
- Phase 2: provider circuit breaker (builder.py)
- Phase 3: subprocess batching (this file)

Usage:
    python -m models.walk_forward_batched --horizon 1
    python -m models.walk_forward_batched --horizon 3
    python -m models.walk_forward_batched --horizon 5
"""
import sys
import subprocess
import time
import os
import argparse
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

PYTHON = "/Users/atomnguyen/.pyenv/versions/ml_quant_310/bin/python"
TICKERS_FILE = ROOT / "tickers.txt"
REPORT_DIR = ROOT / "reports"
BATCH_SIZE = 10  # smaller than daily_runner (42) — walk_forward heavier per ticker


def load_tickers():
    return [t.strip() for t in TICKERS_FILE.read_text().splitlines()
            if t.strip() and not t.startswith("#")]


def run_batch(start_from: str, end_at: str, horizon: int, start_date: str, batch_label: str) -> int:
    print(f"\n{'='*60}")
    print(f"BATCH {batch_label}: {start_from} → {end_at}  h={horizon}d")
    print(f"{'='*60}")

    args = [PYTHON, "-m", "models.walk_forward",
            "--all",
            "--horizon", str(horizon),
            "--start", start_date,
            "--start-from", start_from,
            "--end-at", end_at]

    env = {k: v for k, v in os.environ.items()}
    # PIT mode: disable yfinance fallback for reproducibility
    env["ML_QUANT_ALLOW_YFINANCE_FALLBACK"] = "0"

    t0 = time.time()
    result = subprocess.run(args, env=env, cwd=str(ROOT))
    elapsed = time.time() - t0
    print(f"\nBATCH {batch_label}: exit={result.returncode}, elapsed={elapsed:.1f}s ({elapsed/60:.1f}min)")
    return result.returncode


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizon", type=int, default=1, choices=[1, 3, 5])
    ap.add_argument("--start", type=str, default="2018-01-01")
    args = ap.parse_args()

    tickers = load_tickers()
    n = len(tickers)
    print(f"Loaded {n} tickers, horizon={args.horizon}d, start={args.start}")

    # Clear stale report so batches append fresh
    for name in [f"walkforward_summary_h{args.horizon}.csv",
                 f"walkforward_folds_h{args.horizon}.csv"]:
        p = REPORT_DIR / name
        if p.exists():
            p.unlink()
            print(f"Cleared stale {p.name}")

    # Build batches
    batches = []
    for i in range(0, n, BATCH_SIZE):
        chunk = tickers[i:i + BATCH_SIZE]
        batches.append((chunk[0], chunk[-1]))

    print(f"\nSplitting into {len(batches)} batches (BATCH_SIZE={BATCH_SIZE}):")
    for i, (s, e) in enumerate(batches, 1):
        print(f"  Batch {i:2d}: {s} → {e}")

    print()
    overall_start = time.time()
    failed = []

    for i, (start_ticker, end_ticker) in enumerate(batches, 1):
        rc = run_batch(start_ticker, end_ticker, args.horizon, args.start, f"{i}/{len(batches)}")
        if rc != 0:
            failed.append(i)

    overall_elapsed = time.time() - overall_start
    print(f"\n{'='*60}")
    print(f"ALL BATCHES COMPLETE — total {overall_elapsed:.1f}s ({overall_elapsed/60:.1f}min)")
    if failed:
        print(f"⚠ Failed batches: {failed}")
    print(f"{'='*60}")
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
