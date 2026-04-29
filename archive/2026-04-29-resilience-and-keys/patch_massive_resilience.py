#!/usr/bin/env python3
"""
DNS resilience patch for features/massive_client.py.

Two changes:
1. Bump _request_with_retry from 3 attempts to 6 attempts
   - Old: 1+2+4 = 7s tolerance for network blips
   - New: 1+2+4+8+16+32 = 63s tolerance
   - This catches transient DNS failures lasting up to ~1 min

2. Add per-ticker yfinance fallback in _download_single
   - If Massive fails after all retries for ONE ticker, try yfinance for that ticker
   - Pipeline continues for other 124 tickers instead of aborting

Run from project root:
  python3 patch_massive_resilience.py

Idempotent.
"""
from pathlib import Path
import shutil
from datetime import datetime

PATH = Path("features/massive_client.py")
text = PATH.read_text()
orig_len = len(text)

# Backup
backup = PATH.with_suffix(f".py.bak.resilience.{datetime.now().strftime('%Y%m%d_%H%M%S')}")
shutil.copy(PATH, backup)
print(f"[OK] Backed up to {backup}")

# ─────────────────────────────────────────────────────────────────────────────
# Patch 1: Bump retries from 3 to 6 in _request_with_retry signature
# ─────────────────────────────────────────────────────────────────────────────
old_sig = "def _request_with_retry(url, params, timeout=15, max_retries=3):"
new_sig = "def _request_with_retry(url, params, timeout=15, max_retries=6):"

if old_sig in text:
    text = text.replace(old_sig, new_sig, 1)
    print("[OK] Patch 1: bumped max_retries default 3 -> 6")
elif new_sig in text:
    print("[SKIP] Patch 1: already at 6 retries")
else:
    print("[FAIL] Patch 1: signature not found")

# ─────────────────────────────────────────────────────────────────────────────
# Patch 2: Add yfinance fallback in _download_single
# When Massive's _request_with_retry exhausts retries, try yfinance for THAT ticker
# ─────────────────────────────────────────────────────────────────────────────

old_single = '''def _download_single(ticker, start_str, end_str, mult, span, auto_adjust):
    url = f"{BASE_URL}/v2/aggs/ticker/{ticker}/range/{mult}/{span}/{start_str}/{end_str}"
    params = {
        "adjusted": "true" if auto_adjust else "false",
        "sort": "asc",
        "limit": 50000,
        "apiKey": API_KEY,
    }
    data = _request_with_retry(url, params)'''

new_single = '''def _download_single(ticker, start_str, end_str, mult, span, auto_adjust):
    url = f"{BASE_URL}/v2/aggs/ticker/{ticker}/range/{mult}/{span}/{start_str}/{end_str}"
    params = {
        "adjusted": "true" if auto_adjust else "false",
        "sort": "asc",
        "limit": 50000,
        "apiKey": API_KEY,
    }
    try:
        data = _request_with_retry(url, params)
    except Exception as e:
        # Massive failed all retries — try yfinance fallback for this ticker
        log.warning(f"massive failed for {ticker} after retries, falling back to yfinance: {e}")
        return _download_yfinance_fallback(ticker, start_str, end_str,
                                           f"{mult}{span[0]}" if span != "day" else "1d",
                                           auto_adjust)'''

if old_single in text:
    text = text.replace(old_single, new_single, 1)
    print("[OK] Patch 2: added yfinance fallback in _download_single")
elif "yfinance fallback for this ticker" in text:
    print("[SKIP] Patch 2: fallback already in place")
else:
    print("[FAIL] Patch 2: _download_single signature not found")

# ─────────────────────────────────────────────────────────────────────────────
# Patch 3: Add the _download_yfinance_fallback helper function
# (renamed from _download_yfinance which already exists for indices, but the
# fallback for stocks needs a different interval handling)
# ─────────────────────────────────────────────────────────────────────────────

# Check if helper already exists
if "_download_yfinance_fallback" not in text:
    # Insert helper function at end of file
    helper = '''

def _download_yfinance_fallback(ticker, start_str, end_str, interval, auto_adjust):
    """
    Fallback to yfinance when Massive exhausts retries for a single ticker.
    Used to keep pipeline alive when one ticker has transient network issue.
    Returns same yfinance-shape DataFrame as _download_single would.
    """
    try:
        import yfinance as yf
        df = yf.download(ticker, start=start_str, end=end_str,
                         interval=interval, auto_adjust=auto_adjust, progress=False)
        if df.empty:
            log.warning(f"yfinance fallback also empty for {ticker}")
            return pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])
        # Flatten MultiIndex if yfinance returned one (single-ticker case)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df.index.name = "Date"
        if interval == "1d":
            df.index = df.index.normalize()
        log.info(f"yfinance fallback succeeded for {ticker} ({len(df)} rows)")
        return df
    except Exception as e:
        log.error(f"yfinance fallback also failed for {ticker}: {e}")
        # Return empty so caller can proceed
        return pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])
'''
    text = text + helper
    print("[OK] Patch 3: added _download_yfinance_fallback helper function")
else:
    print("[SKIP] Patch 3: _download_yfinance_fallback already present")

# Write changes
if len(text) != orig_len:
    PATH.write_text(text)
    print(f"\n[OK] Wrote {PATH} ({orig_len} -> {len(text)} chars)")
else:
    print(f"\n[NOOP] No changes")

print()
print("Verify with:")
print('  python -c "from features import massive_client as mc; print(mc._request_with_retry.__defaults__)"')
print()
print("Expected: (15, 6)  -- timeout=15, max_retries=6")
