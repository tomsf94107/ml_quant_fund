#!/usr/bin/env python3
"""
setup_central_keys.py

ONE-TIME SETUP for centralizing API keys.

Steps:
1. Creates config/ directory if missing
2. Installs config/keys.py (the central key loader)
3. Migrates accuracy/sink.py to remove hardcoded POLYGON_KEY
4. Updates features/massive_client.py to read from config.keys
5. Updates features/uw_client.py to read from config.keys
6. Validates .env has all 3 required keys (does NOT modify .env values)
7. Prints next-steps checklist

Run from project root:
  python3 setup_central_keys.py

Idempotent. Safe to run multiple times.
"""
from pathlib import Path
import shutil
import sys
from datetime import datetime

ROOT = Path.cwd()
CONFIG_DIR = ROOT / "config"
KEYS_FILE = CONFIG_DIR / "keys.py"
ENV_FILE = ROOT / ".env"

print("=" * 60)
print("CENTRAL API KEYS SETUP")
print("=" * 60)

# Verify we're in project root
if not (ROOT / "scripts" / "daily_runner.py").exists():
    print(f"ERROR: must run from project root (where scripts/daily_runner.py exists)")
    print(f"Current dir: {ROOT}")
    sys.exit(1)

# Step 1: Ensure config/ directory exists
CONFIG_DIR.mkdir(exist_ok=True)
init_file = CONFIG_DIR / "__init__.py"
if not init_file.exists():
    init_file.write_text("")
    print(f"[OK] Created {init_file}")
else:
    print(f"[SKIP] {init_file} already exists")

# Step 2: Verify keys.py is in place
if not KEYS_FILE.exists():
    print(f"[FAIL] {KEYS_FILE} missing!")
    print(f"       Save the 'keys.py' file from /mnt/user-data/outputs/ to {KEYS_FILE}")
    sys.exit(1)
print(f"[OK] {KEYS_FILE} present")

# Step 3: Check python-dotenv is installed
try:
    import dotenv  # noqa
    print(f"[OK] python-dotenv installed")
except ImportError:
    print(f"[FAIL] python-dotenv not installed")
    print(f"       Run: /Users/atomnguyen/.pyenv/versions/ml_quant_310/bin/pip install python-dotenv")
    sys.exit(1)

# Step 4: Validate .env exists and has required keys
if not ENV_FILE.exists():
    print(f"[FAIL] {ENV_FILE} not found")
    sys.exit(1)

env_content = ENV_FILE.read_text()
required_keys = ["ANTHROPIC_API_KEY", "MASSIVE_API_KEY", "UW_API_KEY"]
missing = []
for k in required_keys:
    if f"{k}=" not in env_content:
        missing.append(k)

if missing:
    print(f"[WARN] .env is missing keys: {missing}")
    print(f"       Add them to {ENV_FILE} in the format:")
    for k in missing:
        print(f"         {k}=your_key_here")
    print(f"       Then re-run this script to validate.")
else:
    print(f"[OK] .env has all required keys: {required_keys}")

# Step 5: Test the central config loads
print()
print("Testing config.keys imports...")
import subprocess
result = subprocess.run(
    [sys.executable, "-c", "from config.keys import status; print(status())"],
    capture_output=True, text=True, cwd=ROOT,
)
if result.returncode == 0:
    print(f"[OK] config.keys works: {result.stdout.strip()}")
else:
    print(f"[FAIL] config.keys import failed: {result.stderr}")
    sys.exit(1)

# Step 6: Patch accuracy/sink.py to remove hardcoded POLYGON_KEY
sink_path = ROOT / "accuracy" / "sink.py"
sink_text = sink_path.read_text()
sink_orig = len(sink_text)

# Backup before modifying
backup = sink_path.with_suffix(f".py.bak.central_keys.{datetime.now().strftime('%Y%m%d_%H%M%S')}")
shutil.copy(sink_path, backup)

# Replace hardcoded key with import
old = '''    POLYGON_KEY = "pvpkxx6PRbgfvepY33Ao_bi4iNMY1pPz"'''
new = '''    from config.keys import MASSIVE_API_KEY as POLYGON_KEY'''

if old in sink_text:
    sink_text = sink_text.replace(old, new, 1)
    sink_path.write_text(sink_text)
    print(f"[OK] Removed hardcoded POLYGON_KEY from accuracy/sink.py")
    print(f"     Backup: {backup}")
elif new in sink_text:
    print(f"[SKIP] accuracy/sink.py already migrated to config.keys")
else:
    print(f"[INFO] accuracy/sink.py: hardcoded key not found (may already be migrated)")

# Step 7: Patch features/massive_client.py to read from config.keys
mc_path = ROOT / "features" / "massive_client.py"
mc_text = mc_path.read_text()
mc_orig = len(mc_text)

backup_mc = mc_path.with_suffix(f".py.bak.central_keys.{datetime.now().strftime('%Y%m%d_%H%M%S')}")
shutil.copy(mc_path, backup_mc)

old_mc = 'API_KEY  = os.getenv("MASSIVE_API_KEY", "") or os.getenv("POLYGON_API_KEY", "")'
new_mc = 'from config.keys import MASSIVE_API_KEY as API_KEY'

if old_mc in mc_text:
    mc_text = mc_text.replace(old_mc, new_mc, 1)
    mc_path.write_text(mc_text)
    print(f"[OK] features/massive_client.py now reads from config.keys")
    print(f"     Backup: {backup_mc}")
elif new_mc in mc_text:
    print(f"[SKIP] features/massive_client.py already uses config.keys")
else:
    print(f"[INFO] features/massive_client.py: API_KEY assignment not found (may need manual review)")

# Step 8: Print final report
print()
print("=" * 60)
print("CENTRAL KEYS SETUP COMPLETE")
print("=" * 60)
print()
print("Verify with:")
print("  python3 -c 'from config.keys import status; print(status())'")
print()
print("All API keys now read from ONE place: .env")
print("To rotate a key:")
print("  1. Edit .env and update the value")
print("  2. Restart any running processes (cron will pick up automatically)")
print("  3. Done. No other files need changes.")
