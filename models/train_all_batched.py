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
TICKERS_FILE   = ROOT / "tickers.txt"
WATCHLIST_FILE = ROOT / "tickers_watchlist.txt"
BATCH_SIZE = 42  # safely below death zone of ~99 tickers


def _read_ticker_file(path):
    """Read a one-ticker-per-line file, skip blanks and comments."""
    if not path.exists():
        return []
    return [t.strip() for t in path.read_text().splitlines()
            if t.strip() and not t.startswith("#")]


def load_tickers():
    """Load main universe + watchlist, dedupe via dict.fromkeys.

    May 12 2026 — sibling fix to commit 9ee9d6f in train_all.py.
    Without this, Pipeline B (which uses train_all_batched.py at
    07:00 VN Tue-Sat) silently excludes watchlist tickers like
    BYND/RZLV from retraining. They drift onto stale models when
    the feature set changes (e.g. May 7 76→79 feature ship caused
    feature_names mismatch errors and prob_up=0.0 outputs).
    """
    return list(dict.fromkeys(
        _read_ticker_file(TICKERS_FILE) + _read_ticker_file(WATCHLIST_FILE)
    ))


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
