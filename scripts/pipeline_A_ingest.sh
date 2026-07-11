#!/bin/bash
export LANG=en_US.UTF-8
export LC_ALL=en_US.UTF-8
# scripts/pipeline_A_ingest.sh
# ─────────────────────────────────────────────────────────────────────────────
# POST-CLOSE INGEST PIPELINE
# Runs 16:00 ET Mon-Fri (was 03:00 VN before ET migration)
# Populates DB tables that Pipeline B's retrain/predict reads from
# ─────────────────────────────────────────────────────────────────────────────
#   Stage 1: Insider ETL (incremental, 7-day window — fast)
#   Stage 2: UW post-market snapshot (dark pool + skew)
#   Stage 3: Feature validator (catches any bad DB rows early)
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail  # exit on any error, undefined var, or pipe failure

ROOT=/Users/atomnguyen/Desktop/ML_Quant_Fund
PYTHON=/Users/atomnguyen/.pyenv/versions/ml_quant_310/bin/python
DATE_TAG=$(date +%Y%m%d)
LOGDIR=$ROOT/logs/pipeline_A_$DATE_TAG
mkdir -p "$LOGDIR"

cd $ROOT
source /Users/atomnguyen/.zshrc 2>/dev/null || true

# Load .env file (feature flags, API keys) - added 2026-05-21
if [ -f "$ROOT/.env" ]; then
    set -a
    source "$ROOT/.env"
    set +a
fi

# Marker file — Pipeline B checks for this before running
MARKER=$ROOT/logs/.pipeline_A_done_$DATE_TAG
rm -f "$MARKER"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOGDIR/pipeline.log"
}

fail() {
    log "FAILED at $1"
    log "See $LOGDIR/ for details"
    # Optional: osascript for macOS desktop notification
    osascript -e "display notification \"Pipeline A failed at $1\" with title \"ML Quant Fund\"" 2>/dev/null || true
    exit 1
}

log "=== PIPELINE A START ==="

# ── Stage 1: Insider ETL (incremental) ───────────────────────────────────────
log "Stage 1: Insider ETL (7-day incremental)"
$PYTHON -c "
from pathlib import Path
from data.etl_insider import run_insider_etl
tickers = [t.strip().upper() for t in Path('tickers.txt').read_text().splitlines()
           if t.strip() and not t.startswith('#')]
r = run_insider_etl(tickers, verbose=False)  # incremental (adaptive days_back)
total = sum(r.values())
print(f'insider ETL: {total} rows across {len(tickers)} tickers')
" > "$LOGDIR/01_insider.log" 2>&1 || fail "Stage 1 (etl_insider)"
log "Stage 1 OK"

# ── Stage 2: UW post-market snapshot ─────────────────────────────────────────
log "Stage 2: UW post-market snapshot"
$PYTHON scripts/daily_uw_snapshot.py --mode post_market \
    > "$LOGDIR/02_uw_snap.log" 2>&1 || fail "Stage 2 (uw_snapshot)"
log "Stage 2 OK"

# ── Stage 2.5: Institutional dark-pool trade ingest ──────────────────────────
# Added 2026-06-27. Feeds inst_signed_flow_5d/30d (silently dead May21-Jun27:
# never cron'd). Per-day window = short run = reliable (5wk backfill failed as
# one multi-hour process; short daily pulls don't). --no-cursor + upsert =
# self-healing (missed day recovered within 3d). NON-FATAL (never blocks B).
# Freshness guard fires a desktop alert if data >4d stale -> cannot rot silently.
log "Stage 2.5: Institutional dark-pool ingest (3-day incremental)"
/opt/homebrew/bin/timeout 2400 $PYTHON features/institutional_ingest.py --tickers-file tickers.txt --days-back 3 --no-cursor \
    > "$LOGDIR/025_institutional.log" 2>&1 || log "WARN Stage 2.5 ingest errors/timeout (non-fatal, see 025_institutional.log)"
# Sync SQLite -> DuckDB. The FEATURE reads DuckDB (institutional_features.py:25),
# NOT SQLite. This sync was built 2026-05-22 and never cron'd -> DuckDB froze at
# May20 while SQLite (when it ran) moved on -> feature saw NULL all June. Without
# this line the ingest above is useless to the model. Idempotent (id-watermark).
$PYTHON -m scripts.migrations.sync_inst_to_duckdb \
    >> "$LOGDIR/025_institutional.log" 2>&1 || log "WARN Stage 2.5 DuckDB sync failed (non-fatal, see 025_institutional.log)"
# Staleness guard on DUCKDB (the store the feature reads), not SQLite. Guarding
# SQLite would report 'fresh' while the feature starves on a stale mirror.
$PYTHON -c "
import duckdb, datetime, sys
c = duckdb.connect('institutional_trades.duckdb', read_only=True)
m = c.execute('SELECT MAX(trade_date) FROM institutional_trades').fetchone()[0]
c.close()
m = str(m)[:10] if m else None
if not m or (datetime.date.today() - datetime.date.fromisoformat(m)).days > 4:
    print(f'STALE duckdb latest={m}'); sys.exit(1)
print(f'fresh duckdb latest={m}')
" >> "$LOGDIR/025_institutional.log" 2>&1 \
    || { osascript -e 'display notification "institutional_trades DuckDB STALE >4d - feature is starving" with title "ML Quant Fund"' 2>/dev/null || true; log "WARN institutional DuckDB STALE >4d"; }
log "Stage 2.5 OK"

# ── Stage 3: Feature validator ───────────────────────────────────────────────
log "Stage 3: Feature validator"
$PYTHON scripts/feature_validator.py --fix \
    > "$LOGDIR/03_feature_validator.log" 2>&1 || fail "Stage 3 (feature_validator)"
log "Stage 3 OK"

# ── Stage 4: Polygon revenue refresh ─────────────────────────────────────────
# Added May 23 2026. Polygon owns rev_actual / rev_estimate / rev_surprise.
# Must run before Pipeline B (which retrains models, calls etl_earnings via
# train_all.py). etl_earnings was changed to UPSERT eps_* only (preserving 
# rev_*) so order doesn't matter for correctness, but we run Polygon here
# so rev_growth features are fresh in the panel that Pipeline B trains on.
# MOVED to weekly cron Sun 05:00 VN (Jun 12: 87min nightly for 0 rows): log "Stage 4: Polygon revenue refresh"
# MOVED to weekly cron Sun 05:00 VN (Jun 12: 87min nightly for 0 rows): $PYTHON -m data.etl_polygon_revenue --all \
# MOVED to weekly cron Sun 05:00 VN (Jun 12: 87min nightly for 0 rows):     > "$LOGDIR/04_polygon_revenue.log" 2>&1 || fail "Stage 4 (polygon_revenue)"
# MOVED to weekly cron Sun 05:00 VN (Jun 12: 87min nightly for 0 rows): log "Stage 4 OK"

# Mark success so Pipeline B can run
touch "$MARKER"
log "=== PIPELINE A COMPLETE ==="
