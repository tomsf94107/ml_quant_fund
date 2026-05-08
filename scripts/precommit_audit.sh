#!/usr/bin/env bash
# scripts/precommit_audit.sh
# ─────────────────────────────────────────────────────────────────
# Pre-commit audit for ML Quant Fund.
# Built May 8 2026 to enforce Rule 32-35 mechanically.
# Catches the same class of bugs we fixed this session:
#   - Validator BUY threshold mismatch (0.70 vs 0.55)
#   - save_feature_importance signature mismatch
#   - Validator destructive auto-fix
#   - Hardcoded thresholds without context
#
# USAGE:
#   scripts/precommit_audit.sh          # run audit on staged files
#   scripts/precommit_audit.sh --skip "reason"  # bypass with logged reason
# ─────────────────────────────────────────────────────────────────

set -uo pipefail

# ── Bypass handling ─────────────────────────────────────────────
if [ "${1:-}" = "--skip" ]; then
    REASON="${2:-no reason}"
    mkdir -p logs
    echo "$(date '+%Y-%m-%d %H:%M:%S')  SKIP  reason=$REASON  user=$(whoami)" \
         >> logs/audit_skips.log
    echo "⚠️  Audit SKIPPED — reason: $REASON"
    echo "   Logged to logs/audit_skips.log"
    exit 0
fi

# ── Color codes ─────────────────────────────────────────────────
RED='\033[0;31m'
YEL='\033[0;33m'
GRN='\033[0;32m'
NC='\033[0m'

# ── Paths considered critical ───────────────────────────────────
CRITICAL_DIRS="signals/ models/ scripts/ accuracy/ features/"

echo "════════════════════════════════════════════════════════════"
echo "  PRECOMMIT AUDIT — ML Quant Fund"
echo "  $(date '+%Y-%m-%d %H:%M:%S')"
echo "════════════════════════════════════════════════════════════"

# ── Get staged files ────────────────────────────────────────────
STAGED=$(git diff --cached --name-only --diff-filter=ACM)

if [ -z "$STAGED" ]; then
    echo -e "${YEL}⚠ No staged files. Run: git add <files> first.${NC}"
    exit 1
fi

# Filter to critical paths only
CRITICAL_STAGED=""
for f in $STAGED; do
    for dir in $CRITICAL_DIRS; do
        if [[ "$f" == ${dir}* ]]; then
            CRITICAL_STAGED="$CRITICAL_STAGED $f"
            break
        fi
    done
done

CRITICAL_STAGED=$(echo $CRITICAL_STAGED | xargs)

if [ -z "$CRITICAL_STAGED" ]; then
    echo -e "${GRN}✅ No staged files in critical paths. Audit not required.${NC}"
    echo "   Staged files: $STAGED"
    exit 0
fi

echo "Critical files staged:"
for f in $CRITICAL_STAGED; do
    echo "  - $f"
done
echo ""

WARN_COUNT=0

# ── Check 1: Function signature changes with callers ────────────
echo "── Check 1: Function signature changes ─────────────────────"
for f in $CRITICAL_STAGED; do
    [[ "$f" == *.py ]] || continue
    
    # Get changed function definitions
    CHANGED_DEFS=$(git diff --cached "$f" 2>/dev/null | \
                   grep -E "^\+def |^-def " | sort -u)
    
    if [ -n "$CHANGED_DEFS" ]; then
        echo "  $f has signature changes — verify all callers compatible:"
        echo "$CHANGED_DEFS" | sed 's/^/    /'
        # Find function names and grep callers
        FNAMES=$(echo "$CHANGED_DEFS" | grep -oE "def [a-zA-Z_]+" | \
                 sed 's/def //' | sort -u)
        for fname in $FNAMES; do
            CALLERS=$(grep -rn "${fname}(" --include="*.py" 2>/dev/null | \
                      grep -v ".bak\|__pycache__\|recession\|def ${fname}" | wc -l | tr -d ' ')
            echo "    └─ ${fname}: ${CALLERS} caller(s) in codebase"
        done
        WARN_COUNT=$((WARN_COUNT + 1))
    fi
done
[ $WARN_COUNT -eq 0 ] && echo -e "  ${GRN}✅ No signature changes detected${NC}"

# ── Check 2: Hardcoded thresholds ───────────────────────────────
echo ""
echo "── Check 2: Hardcoded thresholds (0.55, 0.65, 0.70, 0.75, 0.80) ─"
THRESHOLD_HITS=0
for f in $CRITICAL_STAGED; do
    [[ "$f" == *.py ]] || continue
    NEW_THRESHOLDS=$(git diff --cached "$f" 2>/dev/null | \
                     grep -E "^\+" | grep -vE "^\+\+\+" | \
                     grep -oE "0\.(55|65|70|75|80)" | sort -u)
    if [ -n "$NEW_THRESHOLDS" ]; then
        echo "  $f introduces threshold values:"
        echo "$NEW_THRESHOLDS" | sed 's/^/    /'
        THRESHOLD_HITS=$((THRESHOLD_HITS + 1))
    fi
done
if [ $THRESHOLD_HITS -gt 0 ]; then
    echo -e "  ${YEL}⚠ Threshold values detected — verify intentional and${NC}"
    echo -e "  ${YEL}   match generator/validator/daily_runner expectations.${NC}"
    WARN_COUNT=$((WARN_COUNT + 1))
else
    echo -e "  ${GRN}✅ No new thresholds in staged changes${NC}"
fi

# ── Check 3: Destructive operations ─────────────────────────────
echo ""
echo "── Check 3: Destructive SQL operations in new code ─────────"
DESTRUCTIVE_HITS=0
for f in $CRITICAL_STAGED; do
    [[ "$f" == *.py ]] || continue
    DESTRUCTIVE=$(git diff --cached "$f" 2>/dev/null | \
                  grep -E "^\+" | grep -vE "^\+\+\+" | \
                  grep -iE "UPDATE [a-z_]+ SET|DELETE FROM|DROP TABLE|TRUNCATE")
    if [ -n "$DESTRUCTIVE" ]; then
        echo "  $f introduces destructive SQL:"
        echo "$DESTRUCTIVE" | head -3 | sed 's/^/    /'
        DESTRUCTIVE_HITS=$((DESTRUCTIVE_HITS + 1))
    fi
done
if [ $DESTRUCTIVE_HITS -gt 0 ]; then
    echo -e "  ${YEL}⚠ Destructive ops detected — Rule 35: must require${NC}"
    echo -e "  ${YEL}   explicit flag (--fix, FORCE=1, etc) to execute.${NC}"
    WARN_COUNT=$((WARN_COUNT + 1))
else
    echo -e "  ${GRN}✅ No destructive SQL in staged changes${NC}"
fi

# ── Check 4: Predictions table schema awareness ─────────────────
echo ""
echo "── Check 4: Schema-touching changes ────────────────────────"
SCHEMA_HITS=0
for f in $CRITICAL_STAGED; do
    [[ "$f" == *.py ]] || continue
    HITS=$(git diff --cached "$f" 2>/dev/null | \
           grep -E "^\+" | grep -vE "^\+\+\+" | \
           grep -iE "predictions|outcomes|feature_importance_history|prediction_features" | wc -l | tr -d ' ')
    if [ $HITS -gt 0 ]; then
        echo "  $f touches $HITS line(s) referencing schema tables"
        SCHEMA_HITS=$((SCHEMA_HITS + 1))
    fi
done
if [ $SCHEMA_HITS -gt 0 ]; then
    echo -e "  ${YEL}⚠ Schema-touching changes detected.${NC}"
    echo -e "  ${YEL}   Verify all readers/writers compatible with changes.${NC}"
    echo -e "  ${YEL}   Files reading 'predictions' table:${NC}"
    grep -rln "FROM predictions\|INSERT.*predictions\|UPDATE predictions" \
         --include="*.py" 2>/dev/null | grep -v ".bak\|__pycache__\|recession" | \
         head -10 | sed 's/^/    /'
    WARN_COUNT=$((WARN_COUNT + 1))
else
    echo -e "  ${GRN}✅ No schema-touching changes${NC}"
fi

# ── Check 5: Backup verification ────────────────────────────────
echo ""
echo "── Check 5: Backup verification ────────────────────────────"
NO_BACKUP=0
for f in $CRITICAL_STAGED; do
    [[ "$f" == *.py ]] || continue
    DIRNAME=$(dirname "$f")
    BASENAME=$(basename "$f")
    BACKUP_COUNT=$(ls -1 "$DIRNAME"/"$BASENAME".bak.before_* 2>/dev/null | wc -l | tr -d ' ')
    if [ "$BACKUP_COUNT" -eq 0 ]; then
        echo "  $f has no .bak.before_* backup"
        NO_BACKUP=$((NO_BACKUP + 1))
    fi
done
if [ $NO_BACKUP -gt 0 ]; then
    echo -e "  ${YEL}⚠ $NO_BACKUP file(s) missing backups${NC}"
    echo -e "  ${YEL}   Consider: cp <file> <file>.bak.before_<feature>${NC}"
    WARN_COUNT=$((WARN_COUNT + 1))
else
    echo -e "  ${GRN}✅ All staged files have backups${NC}"
fi

# ── Check 6: Final ACK ──────────────────────────────────────────
echo ""
echo "════════════════════════════════════════════════════════════"
if [ $WARN_COUNT -eq 0 ]; then
    echo -e "${GRN}✅ ALL CHECKS PASSED — safe to commit${NC}"
    exit 0
fi

echo -e "${YEL}⚠ $WARN_COUNT warning(s) detected${NC}"
echo ""
echo "Rule 32-35 acknowledgment required."
echo "Type AUDITED to confirm you've reviewed warnings:"
read -r ACK
if [ "$ACK" != "AUDITED" ]; then
    echo -e "${RED}❌ Audit not acknowledged. Commit blocked.${NC}"
    echo "   Re-run audit, OR bypass with:"
    echo "   $0 --skip \"reason for bypass\""
    exit 1
fi

echo -e "${GRN}✅ Audit acknowledged. Proceeding with commit.${NC}"
exit 0
