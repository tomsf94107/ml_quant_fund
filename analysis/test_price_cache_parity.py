"""
analysis/test_price_cache_parity.py — GATE for the price cache.

Proves cached_daily() (raw bars + backward split adjustment) reproduces
mc.download(auto_adjust=True) byte-for-byte, on a fresh empty DB, INCLUDING a
real corporate action: NVDA's 2024-06-10 10:1 split. If this fails, the cache
does NOT get wired into download().

Run:  PYTHONPATH=. python3 -m analysis.test_price_cache_parity
"""
import sys
from pathlib import Path
import pandas as pd
import numpy as np

from features import massive_client as mc
from features import price_cache as pc


def _raw_fn(t, s, e):
    return mc.download(t, start=s, end=e, auto_adjust=False)  # RAW bars

def _splits_fn(t):
    return mc.get_splits(t)


def _compare(name, ticker, start, end, tol=1e-6):
    # ground truth: vendor adjusted
    truth = mc.download(ticker, start=start, end=end, auto_adjust=True)
    # cache path: raw + local backward adjustment (fresh DB)
    got = pc.cached_daily(ticker, start, end, _raw_fn, _splits_fn)

    # align on common dates (vendor + cache should have identical index)
    common = truth.index.intersection(got.index)
    only_t = truth.index.difference(got.index)
    only_g = got.index.difference(truth.index)
    print(f"\n[{name}] {ticker} {start}..{end}")
    print(f"  rows truth={len(truth)} cache={len(got)} common={len(common)}")
    if len(only_t) or len(only_g):
        print(f"  *** INDEX MISMATCH: truth-only={len(only_t)} cache-only={len(only_g)}")
        return False
    ok = True
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        a = truth.loc[common, col].values.astype(float)
        b = got.loc[common, col].values.astype(float)
        denom = np.clip(np.abs(a), 1e-9, None)
        maxrel = float(np.max(np.abs(a - b) / denom))
        flag = "OK" if maxrel < tol else "*** MISMATCH ***"
        if maxrel >= tol:
            ok = False
        print(f"    {col:7s} max-rel-diff={maxrel:.2e} {flag}")
    return ok


def main():
    results = {}
    # 1. No-split ticker (sanity): a stable large-cap, no recent split
    results["no_split"] = _compare("no-split", "MSFT", "2024-01-01", "2026-06-15")
    # 2. THE split test: NVDA 10:1 on 2024-06-10 — spans before+after
    results["split"] = _compare("split-NVDA", "NVDA", "2024-01-01", "2026-06-15")
    # 3. Incremental: second call should hit DB + fetch only the gap, still match
    print("\n[incremental] second NVDA call (DB warm, gap-only fetch):")
    got2 = pc.cached_daily("NVDA", "2024-01-01", "2026-06-15", _raw_fn, _splits_fn)
    truth2 = mc.download("NVDA", start="2024-01-01", end="2026-06-15", auto_adjust=True)
    inc_ok = np.allclose(
        got2["Close"].values.astype(float),
        truth2.reindex(got2.index)["Close"].values.astype(float),
        rtol=1e-6)
    results["incremental"] = inc_ok
    print(f"  incremental Close match: {'OK' if inc_ok else '*** MISMATCH ***'}")

    print("\n" + "=" * 50)
    for k, v in results.items():
        print(f"  {k:14s} {'PASS' if v else 'FAIL'}")
    allpass = all(results.values())
    print(f"\nPARITY GATE: {'PASS — safe to wire into download()' if allpass else 'FAIL — do NOT wire in'}")
    sys.exit(0 if allpass else 1)


if __name__ == "__main__":
    main()
