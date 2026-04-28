#!/usr/bin/env python3
"""
Migration patch for regime files:
- models/regime_classifier.py (2 call sites)
- models/regime_models.py (1 call site)

Indices (^VIX, ^GSPC) automatically route to yfinance via hybrid logic
in massive_client. Stocks/ETFs (SPY, TLT, GLD, USO) go to Massive.

Run from project root:
  python3 patch_regime_files.py

Idempotent.
"""
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
# File 1: models/regime_classifier.py
# ─────────────────────────────────────────────────────────────────────────────
PATH1 = Path("models/regime_classifier.py")
text1 = PATH1.read_text()
orig1 = len(text1)

# Add import (after the function-level "import yfinance as yf")
old_local_import_1a = '''    """Fetch SPY, VIX, TLT, GLD for regime analysis."""
    import yfinance as yf'''
new_local_import_1a = '''    """Fetch SPY, VIX, TLT, GLD for regime analysis."""
    import yfinance as yf
    from features import massive_client as mc'''
if old_local_import_1a in text1:
    text1 = text1.replace(old_local_import_1a, new_local_import_1a, 1)
    print("[OK] regime_classifier.py: added import (site 1)")
else:
    print("[SKIP] regime_classifier.py site 1 import marker not found")

# Site 1 download: SPY/VIX/TLT/GLD/GSPC/USO
old_dl_1a = '''    try:
        raw = yf.download(tickers, start=start, end=end,
                           auto_adjust=True, progress=False)'''
new_dl_1a = '''    try:
        raw = mc.download(tickers, start=start, end=end,
                           auto_adjust=True, progress=False)'''
if old_dl_1a in text1:
    text1 = text1.replace(old_dl_1a, new_dl_1a, 1)
    print("[OK] regime_classifier.py: patched download (site 1)")
else:
    print("[FAIL] regime_classifier.py site 1 download marker not found")

# Site 2: another fetch_regime_data variant (lines ~410-417)
# This block has its own "import yfinance as yf"
old_local_import_1b = '''    import yfinance as yf

    end   = datetime.today().strftime("%Y-%m-%d")
    start = (datetime.today() - timedelta(days=lookback_days + 60)).strftime("%Y-%m-%d")

    try:
        tickers = ["SPY", "^VIX", "TLT"]
        raw  = yf.download(tickers, start=start, end=end,
                            auto_adjust=True, progress=False)'''

new_local_import_1b = '''    import yfinance as yf
    from features import massive_client as mc

    end   = datetime.today().strftime("%Y-%m-%d")
    start = (datetime.today() - timedelta(days=lookback_days + 60)).strftime("%Y-%m-%d")

    try:
        tickers = ["SPY", "^VIX", "TLT"]
        raw  = mc.download(tickers, start=start, end=end,
                            auto_adjust=True, progress=False)'''

if old_local_import_1b in text1:
    text1 = text1.replace(old_local_import_1b, new_local_import_1b, 1)
    print("[OK] regime_classifier.py: patched site 2")
else:
    print("[FAIL] regime_classifier.py site 2 marker not found")

if len(text1) != orig1:
    PATH1.write_text(text1)
    print(f"[OK] Wrote {PATH1} ({orig1} -> {len(text1)} chars)")

# ─────────────────────────────────────────────────────────────────────────────
# File 2: models/regime_models.py
# ─────────────────────────────────────────────────────────────────────────────
PATH2 = Path("models/regime_models.py")
text2 = PATH2.read_text()
orig2 = len(text2)

# Site 1: SPY/VIX/TLT download
old_dl_2 = '''    # Download SPY + VIX
    try:
        raw = yf.download(["SPY", "^VIX", "TLT"], start=start, end=end,
                           auto_adjust=True, progress=False)'''
new_dl_2 = '''    # Download SPY + VIX
    try:
        from features import massive_client as mc
        raw = mc.download(["SPY", "^VIX", "TLT"], start=start, end=end,
                           auto_adjust=True, progress=False)'''
if old_dl_2 in text2:
    text2 = text2.replace(old_dl_2, new_dl_2, 1)
    print("[OK] regime_models.py: patched download")
else:
    print("[FAIL] regime_models.py marker not found")

if len(text2) != orig2:
    PATH2.write_text(text2)
    print(f"[OK] Wrote {PATH2} ({orig2} -> {len(text2)} chars)")
