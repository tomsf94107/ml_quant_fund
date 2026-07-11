"""
analysis/test_incremental_parity.py - prove incremental == full rebuild.
Builds the latest date via incremental, compares to the existing full-rebuild
parquet for that SAME date on disk. SHIP ONLY IF byte-identical.
Run: python -m analysis.test_incremental_parity
"""
import sys
from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from analysis.build_alpha_panel import build_alpha_panel_incremental, DEFAULT_OUTPUT_DIR


def main():
    out = Path(DEFAULT_OUTPUT_DIR)
    existing = sorted(out.glob("*.parquet"))
    if not existing:
        print("FAIL: no existing parquet to compare against"); return
    latest_file = existing[-1]
    date_str = latest_file.stem  # e.g. 2026-06-15
    print(f"comparing incremental vs full-rebuild for {date_str}")

    full = pd.read_parquet(latest_file).sort_index(axis=0).sort_index(axis=1)

    # write incremental to a SEPARATE dir so we don't clobber the real one
    tmp = out.parent / "alpha_panel_inctest"
    tmp.mkdir(exist_ok=True)
    build_alpha_panel_incremental(output_dir=tmp, target_date=date_str,
                                  parallel=True, verbose=False)
    inc_file = tmp / f"{date_str}.parquet"
    if not inc_file.exists():
        print(f"FAIL: incremental wrote no parquet for {date_str}"); return
    inc = pd.read_parquet(inc_file).sort_index(axis=0).sort_index(axis=1)

    print(f"full:        {full.shape[0]} tickers x {full.shape[1]} alphas")
    print(f"incremental: {inc.shape[0]} tickers x {inc.shape[1]} alphas")

    if list(full.columns) != list(inc.columns):
        only_full = set(full.columns) - set(inc.columns)
        only_inc = set(inc.columns) - set(full.columns)
        print(f"FAIL: column mismatch. only_full={list(only_full)[:5]} only_inc={list(only_inc)[:5]}")
        return
    if list(full.index) != list(inc.index):
        print(f"FAIL: ticker-row mismatch ({len(full.index)} vs {len(inc.index)})")
        return

    diff = (full.fillna(-9e9) != inc.fillna(-9e9))
    nbad = int(diff.values.sum())
    if nbad == 0:
        print("\n*** PASS — incremental IDENTICAL to full rebuild, SHIP OK ***")
    else:
        # show worst offenders
        bad_cols = diff.sum(axis=0).sort_values(ascending=False)
        bad_cols = bad_cols[bad_cols > 0]
        print(f"\n*** FAIL — {nbad} cell mismatches across {len(bad_cols)} alphas — DO NOT SHIP ***")
        print("worst alphas:", list(bad_cols.head(8).index))
        print("(likely warmup too short for these ops — widen warmup_days)")


if __name__ == "__main__":
    main()
