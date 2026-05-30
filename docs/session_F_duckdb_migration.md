# Session F: Migrate institutional_trades to native DuckDB

Target: Friday, May 22 2026
Estimated time: 8 hours focused work
Rationale: Plan to scale to 300-500 tickers in 6-12 months. SQLite
window functions (Option D) ceiling ~500M rows; DuckDB native handles
10B+ rows.

## Scope (F-heavy)

REWRITE: institutional_trades from SQLite to native DuckDB file.
Pipeline A writes directly to .duckdb; all consumers read .duckdb.

OUT OF SCOPE (defer to later sessions):
- finbert_filings stays in sentiment.db SQLite (small data, no scale need)
- accuracy.db stays SQLite (transactional workload, MVCC desirable)
- insider_trades stays SQLite
- Any other DB stays SQLite

## Files to modify

1. features/institutional_ingest.py
   - Replace sqlite3.connect with duckdb.connect
   - Reimplement cursor table logic (no INSERT OR IGNORE in DuckDB)
   - Re-verify 4 bug fixes from commit cb15b86:
     (a) Bug A: silent except RuntimeError -> raise
     (b) Bug B: cursor only advances on total_inserted>0
     (c) Bug C: pagination day-boundary check ordering + loop detection
     (d) Bug D: skip rows with missing executed_at

2. features/institutional_features.py
   - Replace _signed_flow_pct etc with DuckDB window function queries
   - Keep public API: get_institutional_features(ticker, as_of_date)
   - Keep load_institutional_features_pit signature

3. features/builder.py
   - Update import path if needed
   - Verify NaN handling unchanged

4. Pipeline A cron script
   - May need to update DB path env var

## DuckDB schema

```sql
CREATE TABLE institutional_trades (
    ticker          VARCHAR NOT NULL,
    trade_ts        TIMESTAMP NOT NULL,
    side            VARCHAR(1),
    size            BIGINT,
    price           DOUBLE,
    notional        DOUBLE,
    is_block        BOOLEAN,
    is_sweep        BOOLEAN,
    is_auction      BOOLEAN,
    -- ... preserve all existing columns
    PRIMARY KEY (ticker, trade_ts)  -- composite for fast PIT lookups
);

-- Partition / index strategy
CREATE INDEX idx_ticker_ts ON institutional_trades(ticker, trade_ts);
```

## Migration script outline

```python
# scripts/migrate_inst_to_duckdb.py
import sqlite3, duckdb
src = sqlite3.connect("institutional_trades.db")
dst = duckdb.connect("institutional_trades.duckdb")
# Schema create
dst.execute(CREATE_SCHEMA_SQL)
# Bulk insert (DuckDB can directly read from SQLite)
dst.execute("""
    INSTALL sqlite_scanner; LOAD sqlite_scanner;
    ATTACH 'institutional_trades.db' AS src (TYPE sqlite, READ_ONLY);
    INSERT INTO institutional_trades 
    SELECT * FROM src.institutional_trades;
""")
# Verify row count
src_count = sqlite3.connect("institutional_trades.db").execute("SELECT COUNT(*) FROM institutional_trades").fetchone()[0]
dst_count = dst.execute("SELECT COUNT(*) FROM institutional_trades").fetchone()[0]
assert src_count == dst_count, f"Row count mismatch: src={src_count} dst={dst_count}"
```

## Byte-equivalence verifier

```python
# scripts/verify_inst_migration.py
# For 10 sample tickers across 30 as_of_dates:
# 1. Call old get_institutional_features (SQLite-backed)
# 2. Call new get_institutional_features (DuckDB-backed)
# 3. Assert all 6 features match within float epsilon
```

## Rollback plan

- Keep institutional_trades.db SQLite file (don't delete)
- Add env var ML_QUANT_INST_BACKEND=sqlite|duckdb to toggle
- Pipeline A writes to BOTH for 1 week
- After 1 week of clean DuckDB runs, deprecate SQLite write path

## Rule #1 verification gates

Before deploying:
- (a) Audit: re-read all 4 bug fixes against new DuckDB code
- (b) No silent errors: every except logged or raised
- (c) Flag-flip path: ML_QUANT_INST_BACKEND tested both values
- (d) Verify script: scripts/verify_inst_migration.py shows byte-equiv
- (e) Built-not-known: actually run before claiming it works
- (f) Test with real data: full Pipeline A dry run
- (g) Gap-check: any other code touching institutional_trades.db
- (h) Verify chain: code→data→feature→model→output
- (i) Compiled OK ≠ verified

## Decision points to resolve Saturday morning

1. WAL mode for DuckDB? (it auto-handles ACID, may not need)
2. Daily vacuum/compact? (DuckDB columnar storage doesn't fragment like SQLite)
3. Read-only connection mode for Pipeline B/C? (yes, to avoid lock contention)
4. Keep SQLite source for 1 week or 1 month?
