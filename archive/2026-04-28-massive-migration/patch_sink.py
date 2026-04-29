#!/usr/bin/env python3
"""
Migration patch for accuracy/sink.py.
Replaces 3 yfinance call sites with massive_client.

The existing Polygon fallback chain (_fetch_price_fallback) stays as-is —
it already uses Polygon (= Massive) and Alpha Vantage. With Massive now as
PRIMARY via this patch, the fallback chain becomes a backup only.

Run from project root:
  python3 patch_sink.py

Idempotent.
"""
from pathlib import Path

PATH = Path("accuracy/sink.py")
text = PATH.read_text()
original_len = len(text)

# Patch 1: add massive_client import (after yfinance import in the function scope)
# sink.py uses local imports `import yfinance as yf` inside two functions
# We need to add `from features import massive_client as mc` to those scopes
# Easiest: do it via string replace on the local imports.

# Site 1: line ~407 — reconcile_outcomes loop
old1 = '''        try:
            px = yf.download(
                ticker, start=str(min_date), end=str(max_date),
                auto_adjust=True, progress=False
            )
            if isinstance(px.columns, pd.MultiIndex):
                px.columns = px.columns.get_level_values(0)
            if px.empty:
                raise ValueError("Empty yfinance response")
            close = px["Close"].squeeze()
        except Exception as e:
            print(f"  \u26a0 yfinance failed for {ticker}: {e} \u2014 trying fallbacks")'''

new1 = '''        try:
            from features import massive_client as mc
            px = mc.download(
                ticker, start=str(min_date), end=str(max_date),
                auto_adjust=True, progress=False
            )
            if isinstance(px.columns, pd.MultiIndex):
                px.columns = px.columns.get_level_values(0)
            if px.empty:
                raise ValueError("Empty Massive response")
            close = px["Close"].squeeze()
        except Exception as e:
            print(f"  \u26a0 Massive primary failed for {ticker}: {e} \u2014 trying yfinance/fallbacks")'''

if old1 in text:
    text = text.replace(old1, new1, 1)
    print("[OK] Patch 1 (reconcile_outcomes): applied")
else:
    print("[FAIL] Patch 1 marker not found")

# Site 2: line ~736 — minute-bar fetch for intraday outcome reconciliation
old2 = '''            # Fetch actual price at outcome time
            import pandas as pd
            hist = yf.download(ticker,
                               start=outcome_dt.strftime("%Y-%m-%d"),
                               end=(outcome_dt + timedelta(days=1)).strftime("%Y-%m-%d"),
                               interval="1m", auto_adjust=True, progress=False)'''

new2 = '''            # Fetch actual price at outcome time
            import pandas as pd
            from features import massive_client as mc
            hist = mc.download(ticker,
                               start=outcome_dt.strftime("%Y-%m-%d"),
                               end=(outcome_dt + timedelta(days=1)).strftime("%Y-%m-%d"),
                               interval="1m", auto_adjust=True, progress=False)'''

if old2 in text:
    text = text.replace(old2, new2, 1)
    print("[OK] Patch 2 (intraday outcome fetch): applied")
else:
    print("[FAIL] Patch 2 marker not found")

# Site 3: line ~912 — SPY benchmark fetch
old3 = '''    try:
        spy = yf.download("SPY", start=min_date_ext, end=max_date,
                          auto_adjust=True, progress=False)
        if isinstance(spy.columns, pd.MultiIndex):
            spy.columns = spy.columns.get_level_values(0)'''

new3 = '''    try:
        from features import massive_client as mc
        spy = mc.download("SPY", start=min_date_ext, end=max_date,
                          auto_adjust=True, progress=False)
        if isinstance(spy.columns, pd.MultiIndex):
            spy.columns = spy.columns.get_level_values(0)'''

if old3 in text:
    text = text.replace(old3, new3, 1)
    print("[OK] Patch 3 (SPY benchmark): applied")
else:
    print("[FAIL] Patch 3 marker not found")

if len(text) != original_len:
    PATH.write_text(text)
    print(f"\n[OK] Wrote {PATH} ({original_len} -> {len(text)} chars)")
else:
    print(f"\n[NOOP] No changes")
