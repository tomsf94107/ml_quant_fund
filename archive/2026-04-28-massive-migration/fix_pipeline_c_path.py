#!/usr/bin/env python3
"""
Fix cron PATH issue in pipeline_C_preopen.sh.

Problem: cron uses minimal PATH that doesn't include /opt/homebrew/bin,
so commands like `timeout` (GNU coreutils) aren't found.

Fix: add explicit PATH export at top of script so all homebrew tools
are available regardless of cron's default environment.

Run from project root:
  python3 fix_pipeline_c_path.py

Idempotent.
"""
from pathlib import Path
import shutil
from datetime import datetime

PATH = Path("scripts/pipeline_C_preopen.sh")
text = PATH.read_text()

# Backup
backup_path = PATH.with_suffix(f".sh.bak.path.{datetime.now().strftime('%Y%m%d_%H%M%S')}")
shutil.copy(PATH, backup_path)
print(f"[OK] Backed up to {backup_path}")

# Detection: check if PATH export already exists
already_patched = "/opt/homebrew/bin:/opt/homebrew/sbin" in text

if already_patched:
    print("[SKIP] PATH export already present")
else:
    # Insert PATH export right after `set -euo pipefail`
    old = "set -euo pipefail\n\nROOT="
    new = """set -euo pipefail

# Cron compatibility: explicit PATH so homebrew tools (timeout, etc.) are found
export PATH="/opt/homebrew/bin:/opt/homebrew/sbin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin:$PATH"

ROOT="""

    if old in text:
        text = text.replace(old, new, 1)
        PATH.write_text(text)
        print(f"[OK] Inserted PATH export after 'set -euo pipefail'")
    else:
        print(f"[FAIL] Could not find insertion marker 'set -euo pipefail'")

print()
print("Verify with:")
print("  head -25 scripts/pipeline_C_preopen.sh")
print()
print("After fix, tomorrow's 7:30 PM cron will find /opt/homebrew/bin/timeout.")
print("Sentiment Stage 0 should run cleanly.")
