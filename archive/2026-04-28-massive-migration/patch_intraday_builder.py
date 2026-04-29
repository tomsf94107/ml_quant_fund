#!/usr/bin/env python3
"""
Migration patch for features/intraday_builder.py.
Replaces 2 yfinance call sites with massive_client.

Run from project root:
  python3 patch_intraday_builder.py

Idempotent.
"""
from pathlib import Path

PATH = Path("features/intraday_builder.py")
text = PATH.read_text()
original_len = len(text)

# Patch 1: add import (after the top yfinance import)
import_marker = "import yfinance as yf"
import_replacement = "import yfinance as yf\nfrom features import massive_client as mc"
if import_marker in text and "from features import massive_client as mc" not in text:
    text = text.replace(import_marker, import_replacement, 1)
    print("[OK] Added massive_client import")
else:
    print("[SKIP] Import already present or marker missing")

# Patch 2: site 1 line ~147 — single-ticker 5min bars
old1 = '    data = yf.download(\n        ticker, period="2d", interval="5m",\n        progress=False, auto_adjust=True\n    )'
new1 = '    data = mc.download(\n        ticker, period="2d", interval="5m",\n        progress=False, auto_adjust=True\n    )'
if old1 in text:
    text = text.replace(old1, new1, 1)
    print("[OK] Patch 1 (single ticker 5min): applied")
else:
    print("[FAIL] Patch 1 marker not found")

# Patch 3: site 2 line ~331 — batch download in get_all_intraday_signals()
# Note: also remove the local "import yfinance as yf" inside that function
old_local_import = "def get_all_intraday_signals(tickers: list) -> list:\n    \"\"\"\n    Batch download all tickers at once to avoid yfinance per-ticker caching issues.\n    \"\"\"\n    import yfinance as yf\n    import pandas as pd"
new_local_import = "def get_all_intraday_signals(tickers: list) -> list:\n    \"\"\"\n    Batch download all tickers at once via Massive (replaces yfinance).\n    \"\"\"\n    from features import massive_client as mc\n    import pandas as pd"
if old_local_import in text:
    text = text.replace(old_local_import, new_local_import, 1)
    print("[OK] Patch 2a (local import in batch fn): applied")
else:
    print("[FAIL] Patch 2a marker not found")

old2 = '    raw = yf.download(\n        tickers, period="2d", interval="5m",\n        progress=False, auto_adjust=True, group_by="ticker"\n    )'
new2 = '    raw = mc.download(\n        tickers, period="2d", interval="5m",\n        progress=False, auto_adjust=True\n    )'
if old2 in text:
    text = text.replace(old2, new2, 1)
    print("[OK] Patch 2b (batch download): applied")
else:
    print("[FAIL] Patch 2b marker not found")

if len(text) != original_len:
    PATH.write_text(text)
    print(f"\n[OK] Wrote {PATH} ({original_len} -> {len(text)} chars)")
else:
    print(f"\n[NOOP] No changes")
