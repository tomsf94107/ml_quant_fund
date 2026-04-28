#!/usr/bin/env python3
"""
Patch features/massive_client.py to route Yahoo-style symbols
(DX-Y.NYB, dollar index) to yfinance.

These have ".NYB" suffix or other Yahoo-specific formats that don't
exist in Massive's universe.

Run from project root:
  python3 patch_dxy_routing.py

Idempotent.
"""
from pathlib import Path

PATH = Path("features/massive_client.py")
text = PATH.read_text()
orig_len = len(text)

# Add DX-Y.NYB to INDEX_SYMBOLS set, and update _is_index to also catch .NYB suffix
old_block = '''INDEX_SYMBOLS = {
    "^VIX", "^VIX3M", "^VIX9D",        # CBOE volatility indices
    "^TNX", "^TYX", "^FVX", "^IRX",    # Treasury yield indices
    "^GSPC", "^DJI", "^IXIC", "^RUT",  # Major US equity indices
    "^FTSE", "^N225", "^HSI", "^GDAXI", # International indices
    "ES=F", "NQ=F", "YM=F", "RTY=F",   # Equity index futures
    "CL=F", "GC=F", "SI=F", "HG=F",    # Commodity futures
}


def _is_index(symbol):
    """Return True if symbol should route to yfinance instead of Massive."""
    return symbol in INDEX_SYMBOLS or symbol.startswith("^") or symbol.endswith("=F")'''

new_block = '''INDEX_SYMBOLS = {
    "^VIX", "^VIX3M", "^VIX9D",        # CBOE volatility indices
    "^TNX", "^TYX", "^FVX", "^IRX",    # Treasury yield indices
    "^GSPC", "^DJI", "^IXIC", "^RUT",  # Major US equity indices
    "^FTSE", "^N225", "^HSI", "^GDAXI", # International indices
    "ES=F", "NQ=F", "YM=F", "RTY=F",   # Equity index futures
    "CL=F", "GC=F", "SI=F", "HG=F",    # Commodity futures
    "DX-Y.NYB",                         # US Dollar Index (Yahoo-only format)
}


def _is_index(symbol):
    """Return True if symbol should route to yfinance instead of Massive."""
    if symbol in INDEX_SYMBOLS:
        return True
    if symbol.startswith("^"):
        return True
    if symbol.endswith("=F"):
        return True
    if symbol.endswith(".NYB"):  # Yahoo Board (currencies, commodities)
        return True
    return False'''

if old_block in text:
    text = text.replace(old_block, new_block, 1)
    print("[OK] Added DX-Y.NYB and .NYB suffix to index routing")
else:
    print("[FAIL] INDEX_SYMBOLS block not found")

if len(text) != orig_len:
    PATH.write_text(text)
    print(f"[OK] Wrote {PATH} ({orig_len} -> {len(text)} chars)")
else:
    print("[NOOP] No changes")
