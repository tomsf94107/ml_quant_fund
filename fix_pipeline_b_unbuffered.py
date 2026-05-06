#!/usr/bin/env python3
"""
Add -u flag to scripts/pipeline_B_train_predict.sh so train_all output
is unbuffered. Without -u, cron's log shows only stderr (HTTP errors)
making it look "stuck" when actually progressing fine.

With -u, you see real-time training progress.

Run from project root:
  python3 fix_pipeline_b_unbuffered.py

Idempotent.
"""
from pathlib import Path
import shutil
from datetime import datetime

PATH = Path("scripts/pipeline_B_train_predict.sh")
text = PATH.read_text()
orig_len = len(text)

backup = PATH.with_suffix(f".sh.bak.unbuffered.{datetime.now().strftime('%Y%m%d_%H%M%S')}")
shutil.copy(PATH, backup)
print(f"[OK] Backed up to {backup}")

old = '$PYTHON -m models.train_all \\'
new = '$PYTHON -u -m models.train_all \\'

if old in text and new not in text:
    text = text.replace(old, new, 1)
    print("[OK] Added -u flag to train_all invocation")
elif new in text:
    print("[SKIP] -u flag already present")
else:
    print("[FAIL] Could not find Stage 2 invocation marker")

if len(text) != orig_len:
    PATH.write_text(text)
    print(f"[OK] Wrote {PATH} ({orig_len} -> {len(text)} chars)")
else:
    print(f"[NOOP] No changes")

print()
print("Verify:")
print("  grep '\\$PYTHON.*train_all' scripts/pipeline_B_train_predict.sh")
