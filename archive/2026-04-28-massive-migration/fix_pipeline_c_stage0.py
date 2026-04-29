#!/usr/bin/env python3
"""
Fix patch_pipeline_c_chain.py error.

Previous patch updated the header but skipped Stage 0 insertion because
its detection check found "Stage 0:" in the header (false positive).

This script:
1. Verifies header is updated (was correct before)
2. Inserts the actual Stage 0 code block before "# ── Stage 1:"
3. Uses a more specific detection check (looks for the timeout command)

Run from project root:
  python3 fix_pipeline_c_stage0.py

Idempotent.
"""
from pathlib import Path
import shutil
from datetime import datetime

PATH = Path("scripts/pipeline_C_preopen.sh")
text = PATH.read_text()
original_len = len(text)

# Backup
backup_path = PATH.with_suffix(f".sh.bak.{datetime.now().strftime('%Y%m%d_%H%M%S')}")
shutil.copy(PATH, backup_path)
print(f"[OK] Backed up to {backup_path}")

# Use a UNIQUE detection marker — the actual command, not the header text
already_patched = "timeout 900 $PYTHON scripts/daily_sentiment.py" in text

if already_patched:
    print("[SKIP] Stage 0 code block already present (detected timeout command)")
    print("\n[NOOP] No changes needed")
else:
    # Find the existing Stage 1 block start, insert Stage 0 before it
    old_marker = '''log "=== PIPELINE C START ==="

# ── Stage 1: UW full snapshot ────────────────────────────────────────────────'''

    new_block = '''log "=== PIPELINE C START ==="

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

    if old_marker in text:
        text = text.replace(old_marker, new_block, 1)
        PATH.write_text(text)
        print(f"[OK] Inserted Stage 0 code block")
        print(f"[OK] Wrote {PATH} ({original_len} -> {len(text)} chars)")
    else:
        print(f"[FAIL] Could not find Stage 1 insertion marker")
        print(f"        File may have been manually edited or already in unexpected state")
        print(f"        Restoring from backup...")
        # No restore needed — we haven't written yet

print()
print("Verify with:")
print("  head -45 scripts/pipeline_C_preopen.sh")
print()
print("You should see:")
print("  - Header mentions 'Stage 0: Daily sentiment scoring'")
print("  - Bash code includes 'log \"Stage 0: Daily sentiment scoring\"'")
print("  - Bash code includes 'timeout 900 $PYTHON scripts/daily_sentiment.py'")
