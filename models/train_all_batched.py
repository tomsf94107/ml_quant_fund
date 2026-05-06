"""
Subprocess-batched train_all.

Runs train_all in 3 fresh Python subprocesses to avoid cumulative memory
state from build_feature_dataframe + XGBoost training across 125 tickers
hitting kernel jetsam threshold.

Each subprocess processes ~42 tickers, peaks ~1-2GB RSS, exits clean.

Usage:
    python -m models.train_all_batched
"""
import sys
import subprocess
import time
import os
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

PYTHON = "/Users/atomnguyen/.pyenv/versions/ml_quant_310/bin/python"
TICKERS_FILE = ROOT / "tickers.txt"
BATCH_SIZE = 42  # safely below death zone of ~99 tickers


def load_tickers():
    return [t.strip() for t in TICKERS_FILE.read_text().splitlines()
            if t.strip() and not t.startswith("#")]


def run_batch(tickers_subset: list[str], batch_label: str) -> int:
    """Spawn fresh Python subprocess for one batch."""
    print(f"\n{'='*60}")
    print(f"BATCH {batch_label}: {tickers_subset[0]} → {tickers_subset[-1]}  [{len(tickers_subset)} tickers]")
    print(f"{'='*60}")

    tickers_repr = repr(tickers_subset)
    code = f"""
import sys
sys.path.insert(0, '.')
from models.train_all import train_all
train_all(tickers={tickers_repr}, verbose=True)
"""

    args = [PYTHON, "-c", code]
    env = {k: v for k, v in os.environ.items()}

    start = time.time()
    result = subprocess.run(args, env=env, cwd=str(ROOT))
    elapsed = time.time() - start

    print(f"\nBATCH {batch_label}: exit={result.returncode}, elapsed={elapsed:.1f}s ({elapsed/60:.1f}min)")
    return result.returncode


def main():
    tickers = load_tickers()
    n = len(tickers)
    print(f"Loaded {n} tickers")

    # Clear stale training_report.csv so batches append fresh.
    # Without this, batches would append to old report from previous day.
    report_path = ROOT / "models" / "saved" / "training_report.csv"
    if report_path.exists():
        report_path.unlink()
        print(f"Cleared stale {report_path.name}")

    batches = []
    for i in range(0, n, BATCH_SIZE):
        chunk = tickers[i:i + BATCH_SIZE]
        batches.append(chunk)

    print(f"Splitting into {len(batches)} batches (BATCH_SIZE={BATCH_SIZE}):")
    for i, chunk in enumerate(batches, 1):
        idx_s = tickers.index(chunk[0])
        idx_e = tickers.index(chunk[-1])
        print(f"  Batch {i}: {chunk[0]} (#{idx_s+1}) → {chunk[-1]} (#{idx_e+1})  [{len(chunk)} tickers]")

    print()
    overall_start = time.time()
    failed_batches = []

    for i, chunk in enumerate(batches, 1):
        rc = run_batch(chunk, f"{i}/{len(batches)}")
        if rc != 0:
            failed_batches.append(i)
            print(f"⚠ Batch {i} non-zero exit, continuing")

    overall_elapsed = time.time() - overall_start
    print(f"\n{'='*60}")
    print(f"ALL BATCHES COMPLETE — total {overall_elapsed:.1f}s ({overall_elapsed/60:.1f}min)")
    if failed_batches:
        print(f"⚠ Failed batches: {failed_batches}")
    print(f"Each subprocess merged training_report.csv via train_all's existing CSV write.")
    print(f"NOTE: training_report.csv only contains the LAST batch. Use models/saved/*.joblib for per-ticker models.")
    print(f"{'='*60}")
    return 0 if not failed_batches else 1


if __name__ == "__main__":
    sys.exit(main())
