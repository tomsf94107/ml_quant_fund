#!/usr/bin/env python3
"""
Fix the resilience patch error:
  _download_single fallback calls _download_yfinance_fallback (undefined)
  but the existing function in this file is _download_yfinance

Change the call to use the existing function. Same behavior, no missing helper.

Run from project root:
  python3 fix_yfinance_fallback_name.py

Idempotent.
"""
from pathlib import Path
import shutil
from datetime import datetime

PATH = Path("features/massive_client.py")
text = PATH.read_text()
orig_len = len(text)

backup = PATH.with_suffix(f".py.bak.fallbackfix.{datetime.now().strftime('%Y%m%d_%H%M%S')}")
shutil.copy(PATH, backup)
print(f"[OK] Backed up to {backup}")

# Fix: change _download_yfinance_fallback to _download_yfinance (existing function)
old_call = '''        log.warning(f"massive failed for {ticker} after retries, falling back to yfinance: {e}")
        return _download_yfinance_fallback(ticker, start_str, end_str,
                                           f"{mult}{span[0]}" if span != "day" else "1d",
                                           auto_adjust)'''

new_call = '''        log.warning(f"massive failed for {ticker} after retries, falling back to yfinance: {e}")
        return _download_yfinance(ticker, start_str, end_str,
                                  f"{mult}{span[0]}" if span != "day" else "1d",
                                  auto_adjust)'''

if old_call in text:
    text = text.replace(old_call, new_call, 1)
    print("[OK] Changed call to use existing _download_yfinance")
elif new_call in text:
    print("[SKIP] Already using _download_yfinance")
else:
    print("[FAIL] Old call pattern not found — manual review needed")

if len(text) != orig_len:
    PATH.write_text(text)
    print(f"[OK] Wrote {PATH} ({orig_len} -> {len(text)} chars)")

print()
print("Verify:")
print('  /Users/atomnguyen/.pyenv/versions/ml_quant_310/bin/python -c "from features import massive_client as mc; print(\\"OK\\")"')
