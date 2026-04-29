#!/usr/bin/env python3
"""
Patch scripts/pipeline_C_preopen.sh to add sentiment as Stage 0.

Behavior:
- Adds Stage 0 that runs daily_sentiment.py with 15-min timeout
- If sentiment fails or times out, Pipeline C continues anyway (sentiment
  is not critical — predictions can still be generated without fresh sentiment)
- Updates header docstring to reflect new sequence
- Logs sentiment to pipeline_C log dir for unified viewing

Run from project root:
  python3 patch_pipeline_c_chain.py

Idempotent.
"""
from pathlib import Path
import shutil
from datetime import datetime

PATH = Path("scripts/pipeline_C_preopen.sh")
text = PATH.read_text()
original_len = len(text)

# Backup first
backup_path = PATH.with_suffix(f".sh.bak.{datetime.now().strftime('%Y%m%d_%H%M%S')}")
shutil.copy(PATH, backup_path)
print(f"[OK] Backed up to {backup_path}")

# Patch 1: Update header docstring to reflect 3 stages now
old_header = '''# scripts/pipeline_C_preopen.sh
# ─────────────────────────────────────────────────────────────────────────────
# PRE-OPEN FRESH RUNFUND
# Runs 8 PM VN (9 AM ET) Mon-Fri — 30 min before US market open
# Refreshes live UW signals and re-generates today's signal with fresh data
# Uses the model trained by Pipeline B that morning
# ─────────────────────────────────────────────────────────────────────────────
#   Stage 1: UW full snapshot (short interest, analyst, FTDs, seasonality)
#   Stage 2: Run daily predictions again with fresh live features
# ─────────────────────────────────────────────────────────────────────────────'''

new_header = '''# scripts/pipeline_C_preopen.sh
# ─────────────────────────────────────────────────────────────────────────────
# PRE-OPEN FRESH RUNFUND
# Runs 7:30 PM VN (8:30 AM ET) Mon-Fri — ~1 hour before US market open
# Chains sentiment + UW snapshot + fresh daily runner predictions
# Uses the model trained by Pipeline B that morning
# ─────────────────────────────────────────────────────────────────────────────
#   Stage 0: Daily sentiment scoring (~5-8 min, non-critical, continues on fail)
#   Stage 1: UW full snapshot (short interest, analyst, FTDs, seasonality)
#   Stage 2: Run daily predictions again with fresh live features
# ─────────────────────────────────────────────────────────────────────────────'''

if old_header in text:
    text = text.replace(old_header, new_header, 1)
    print("[OK] Updated header docstring")
elif new_header in text:
    print("[SKIP] Header already updated")
else:
    print("[FAIL] Header marker not found")

# Patch 2: Insert Stage 0 right before "# ── Stage 1: UW full snapshot"
old_stage_1_marker = '''log "=== PIPELINE C START ==="

# ── Stage 1: UW full snapshot ────────────────────────────────────────────────'''

new_stage_0_block = '''log "=== PIPELINE C START ==="

# ── Stage 0: Daily sentiment (non-critical, 15-min timeout) ──────────────────
log "Stage 0: Daily sentiment scoring"
SENT_START=$(date +%s)
if timeout 900 $PYTHON scripts/daily_sentiment.py \\
    > "$LOGDIR/00_sentiment.log" 2>&1; then
    SENT_DUR=$(($(date +%s) - SENT_START))
    log "Stage 0 OK (${SENT_DUR}s)"
else
    SENT_RC=$?
    SENT_DUR=$(($(date +%s) - SENT_START))
    if [ $SENT_RC -eq 124 ]; then
        log "Stage 0 TIMEOUT after 15min (continuing anyway)"
    else
        log "Stage 0 FAILED rc=$SENT_RC (continuing anyway, sentiment is non-critical)"
    fi
    osascript -e "display notification \\"Pipeline C: sentiment failed (continuing)\\" with title \\"ML Quant Fund\\"" 2>/dev/null || true
fi

# ── Stage 1: UW full snapshot ────────────────────────────────────────────────'''

if old_stage_1_marker in text and "Stage 0:" not in text:
    text = text.replace(old_stage_1_marker, new_stage_0_block, 1)
    print("[OK] Inserted Stage 0 (sentiment with 15-min timeout)")
elif "Stage 0:" in text:
    print("[SKIP] Stage 0 already present")
else:
    print("[FAIL] Stage 1 insertion marker not found")

if len(text) != original_len:
    PATH.write_text(text)
    print(f"\n[OK] Wrote {PATH} ({original_len} -> {len(text)} chars)")
else:
    print(f"\n[NOOP] No changes")

print()
print("Next steps:")
print("1. Test the script manually:")
print("   bash scripts/pipeline_C_preopen.sh")
print("2. Update crontab to remove standalone sentiment, change Pipeline C time:")
print("   crontab -e")
print("   Remove: 30 19 * * 1-5 ... daily_sentiment.py ...")
print("   Change: 0 20 * * 1-5 ... pipeline_C_preopen.sh")
print("       To: 30 19 * * 1-5 ... pipeline_C_preopen.sh")
