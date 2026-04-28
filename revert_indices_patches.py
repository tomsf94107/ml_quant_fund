#!/usr/bin/env python3
"""
Revert patches 3-5 in features/builder.py.
Indices (^VIX, ^VIX3M, ^TNX) need to stay on yfinance because Massive Stocks
Developer plan does not include indices namespace.

Run from project root:
  python3 revert_indices_patches.py

Idempotent.
"""
from pathlib import Path

PATH = Path("features/builder.py")
text = PATH.read_text()
original_len = len(text)

reverts = [
    # Site 3: ^VIX3M back to yf with caret
    (
        '        _vix3m = mc.download("VIX3M", start=start_str, end=end_str,\n                              auto_adjust=True, progress=False)',
        '        _vix3m = yf.download("^VIX3M", start=start_str, end=end_str,\n                              auto_adjust=True, progress=False)',
    ),
    # Site 4: ^VIX back to yf with caret
    (
        '        _vix_raw = mc.download("VIX", start=start_str, end=end_str,\n                               auto_adjust=True, progress=False)',
        '        _vix_raw = yf.download("^VIX", start=start_str, end=end_str,\n                               auto_adjust=True, progress=False)',
    ),
    # Site 5: ^TNX back to yf with caret
    (
        '        tnx_raw = mc.download("TNX", start=start_str, end=end_str,\n                               auto_adjust=True, progress=False)',
        '        tnx_raw = yf.download("^TNX", start=start_str, end=end_str,\n                               auto_adjust=True, progress=False)',
    ),
]

for i, (old, new) in enumerate(reverts, 3):
    if old in text:
        text = text.replace(old, new, 1)
        print(f"[OK] Reverted patch {i} (indices stay yfinance)")
    else:
        print(f"[SKIP] Patch {i} marker not found (already reverted or different)")

if len(text) != original_len:
    PATH.write_text(text)
    print(f"\n[OK] Wrote {PATH} ({original_len} -> {len(text)} chars)")
else:
    print(f"\n[NOOP] No changes")
