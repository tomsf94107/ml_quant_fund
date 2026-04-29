#!/usr/bin/env python3
"""
Add @lru_cache to model loading in:
  - models/classifier.py
  - models/ensemble.py
  - models/regime_models.py

This time using plain str.replace (no regex) to avoid the control-char
injection issue from the earlier attempt.

Expected speedup: model files only loaded from disk once per Python process.
Across 125 tickers x 3 horizons = ~750 loads currently, reduced to ~375 unique
files cached in memory. 2-3x speedup on runfund/Pipeline C.

Run from project root:
  python3 patch_lru_cache.py

Idempotent — safe to run multiple times.
"""
from pathlib import Path

HELPER = '''
from functools import lru_cache as _lru_cache

@_lru_cache(maxsize=None)
def _cached_joblib_load(path_str):
    """Cache joblib.load results across calls. Models are immutable post-train.
    
    Cache key is the file path string. First call loads from disk; subsequent
    calls return the cached object. Reduces disk I/O during pipeline runs that
    re-load the same models repeatedly.
    """
    return joblib.load(path_str)

'''


def patch_file(path_str, call_old, call_new):
    """
    Patch a model file:
    1. Insert helper after the line containing 'import joblib'
    2. Replace joblib.load(...) call with cached version
    """
    path = Path(path_str)
    text = path.read_text()
    orig_len = len(text)

    # Step 1: Insert helper if not already present
    if "_cached_joblib_load" in text:
        print(f"[SKIP] {path_str}: helper already present")
    else:
        # Find the line with 'import joblib' and insert helper right after
        lines = text.split("\n")
        new_lines = []
        inserted = False
        for line in lines:
            new_lines.append(line)
            if not inserted and line.strip() == "import joblib":
                new_lines.append(HELPER.rstrip("\n"))
                inserted = True
        if inserted:
            text = "\n".join(new_lines)
            print(f"[OK] {path_str}: inserted helper after 'import joblib'")
        else:
            print(f"[FAIL] {path_str}: 'import joblib' line not found, skipping")
            return

    # Step 2: Replace joblib.load(...) call site
    if call_old in text:
        if call_new in text:
            print(f"[SKIP] {path_str}: call site already patched")
        else:
            text = text.replace(call_old, call_new, 1)
            print(f"[OK] {path_str}: patched call site")
    else:
        if call_new in text:
            print(f"[SKIP] {path_str}: call site already shows replacement")
        else:
            print(f"[FAIL] {path_str}: call site marker '{call_old}' not found")

    if len(text) != orig_len:
        path.write_text(text)
        print(f"[OK] {path_str}: wrote ({orig_len} -> {len(text)} chars)")


# ─────────────────────────────────────────────────────────────────────────────
# Patches
# ─────────────────────────────────────────────────────────────────────────────

# classifier.py — line ~193, inside TrainResult.load classmethod
patch_file(
    "models/classifier.py",
    call_old="        return joblib.load(path)",
    call_new="        return _cached_joblib_load(str(path))",
)

print()

# ensemble.py — line ~64, inside EnsembleResult.load classmethod
patch_file(
    "models/ensemble.py",
    call_old="        return joblib.load(path)",
    call_new="        return _cached_joblib_load(str(path))",
)

print()

# regime_models.py — line ~276, inside load_regime_model function
patch_file(
    "models/regime_models.py",
    call_old="    return joblib.load(path) if path.exists() else None",
    call_new="    return _cached_joblib_load(str(path)) if path.exists() else None",
)

print()
print("=" * 60)
print("Done. Verify with:")
print('  python -c "from models.classifier import _cached_joblib_load; print(\\"OK\\")"')
