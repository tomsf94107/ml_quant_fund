"""
analysis/test_explode_parity.py - REAL-DATA correctness gate for parallel explode.
Builds actual base panels for a ticker subset, runs serial explode_panels AND
explode_panels_parallel with the REAL operators, asserts byte-identical output.
SHIP ONLY IF THIS PASSES. Run: python -m analysis.test_explode_parity
"""
import sys, time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from analysis.build_alpha_panel import (
    build_panels_from_tickers, explode_panels, load_bucket_map, load_tickers,
)
from analysis.explode_parallel import explode_panels_parallel


def main():
    all_t = load_tickers()
    subset = all_t[:15]
    print(f"parity test on {len(subset)} tickers: {subset}")
    try:
        bmap = load_bucket_map()
    except Exception:
        bmap = None

    panels = build_panels_from_tickers(subset, "2024-01-01", None, verbose=False)
    print(f"base panels: {len(panels)} features")

    t0 = time.time()
    serial = explode_panels(panels, bucket_map=bmap, verbose=False)
    t_ser = time.time() - t0

    t0 = time.time()
    par = explode_panels_parallel(panels, bucket_map=bmap, verbose=False)
    t_par = time.time() - t0

    print(f"\nserial: {len(serial)} alphas in {t_ser:.1f}s")
    print(f"parallel: {len(par)} alphas in {t_par:.1f}s  (speedup {t_ser/max(t_par,0.01):.1f}x)")

    if set(serial) != set(par):
        miss = set(serial) ^ set(par)
        print(f"FAIL: key mismatch ({len(miss)}): {list(miss)[:5]}")
        return
    bad = []
    for k in serial:
        a, b = serial[k], par[k]
        if not a.fillna(-9e9).equals(b.fillna(-9e9)):
            bad.append(k)
    print(f"value mismatches: {len(bad)}", bad[:5] if bad else "")
    print("\n*** PASS — parallel IDENTICAL to serial on real data, SHIP OK ***"
          if not bad else "\n*** FAIL — DO NOT SHIP ***")


if __name__ == "__main__":
    main()
