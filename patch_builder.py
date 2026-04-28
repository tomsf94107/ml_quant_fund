#!/usr/bin/env python3
"""
Migration patch for features/builder.py
Replaces yfinance with massive_client.

Run from project root:
  python3 patch_builder.py

Idempotent: safe to run multiple times.
"""
from pathlib import Path

PATH = Path("features/builder.py")

text = PATH.read_text()
original_len = len(text)

# Patch 1: add import at top (after existing imports)
import_marker = "import yfinance as yf"
import_replacement = "import yfinance as yf\nfrom features import massive_client as mc"
if import_marker in text and "from features import massive_client as mc" not in text:
    text = text.replace(import_marker, import_replacement, 1)
    print("[OK] Added massive_client import")
else:
    print("[SKIP] Import already present or marker missing")

# Patches 2-7: replace each yf.download call with mc.download
# Format-preserving: same args, just different module
replacements = [
    # Site 1: line ~128 in _download()
    (
        '    df = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)',
        '    df = mc.download(ticker, start=start, end=end, auto_adjust=True, progress=False)',
    ),
    # Site 2: line ~157 in _market_return()
    (
        '        tmp = yf.download(etf, start=start, end=end,\n                          auto_adjust=True, progress=False)',
        '        tmp = mc.download(etf, start=start, end=end,\n                          auto_adjust=True, progress=False)',
    ),
    # Site 3: line ~425 ^VIX3M
    (
        '        _vix3m = yf.download("^VIX3M", start=start_str, end=end_str,\n                              auto_adjust=True, progress=False)',
        '        _vix3m = mc.download("VIX3M", start=start_str, end=end_str,\n                              auto_adjust=True, progress=False)',
    ),
    # Site 4: line ~427 ^VIX
    (
        '        _vix_raw = yf.download("^VIX", start=start_str, end=end_str,\n                               auto_adjust=True, progress=False)',
        '        _vix_raw = mc.download("VIX", start=start_str, end=end_str,\n                               auto_adjust=True, progress=False)',
    ),
    # Site 5: line ~450 ^TNX
    (
        '        tnx_raw = yf.download("^TNX", start=start_str, end=end_str,\n                               auto_adjust=True, progress=False)',
        '        tnx_raw = mc.download("TNX", start=start_str, end=end_str,\n                               auto_adjust=True, progress=False)',
    ),
    # Site 6: line ~512 yf.Ticker(ticker).info for short interest
    (
        '            _info = yf.Ticker(ticker).info',
        '            _info = mc.get_short_interest(ticker)',
    ),
]

for i, (old, new) in enumerate(replacements, 1):
    if old in text:
        text = text.replace(old, new, 1)
        print(f"[OK] Patch {i}: applied")
    else:
        print(f"[FAIL] Patch {i}: original text not found, skipping")

# Write back
if len(text) != original_len:
    PATH.write_text(text)
    print(f"\n[OK] Wrote {PATH} ({original_len} -> {len(text)} chars)")
else:
    print(f"\n[NOOP] No changes to write (idempotent)")
