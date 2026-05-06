#!/usr/bin/env python3
"""
write_skew_from_log.py

Parses Stage 1 log to extract skew data and writes directly to options_skew_history.
Recovers data when DB writes failed during Stage 1 run.

Run from project root:
  python3 write_skew_from_log.py logs/pipeline_C_20260429/01_uw_snap_v3.log
"""
import re
import sys
import sqlite3
from datetime import date, datetime
from pathlib import Path

if len(sys.argv) < 2:
    print("Usage: python3 write_skew_from_log.py <log_file>")
    sys.exit(1)

log_path = Path(sys.argv[1])
if not log_path.exists():
    print(f"Log file not found: {log_path}")
    sys.exit(1)

snapshot_date = str(date.today())
now_iso = datetime.now().isoformat()

# Regex to match lines like:
#   [ 64/127] META   dp=6.0% skew=+0.002 NEUTRAL
# or:
#   [127/127] RZLV   dp=4.7% skew=+0.097 BEARISH
pattern = re.compile(
    r'\[\s*\d+/\d+\]\s+(\S+)\s+.*skew=([+-]?\d+\.\d+)\s+(BEARISH|BULLISH|NEUTRAL)'
)

rows = []
with open(log_path) as f:
    for line in f:
        m = pattern.search(line)
        if m:
            ticker = m.group(1)
            skew = float(m.group(2))
            signal = m.group(3)
            rows.append((ticker, skew, signal))

print(f"Parsed {len(rows)} skew rows from log")

if not rows:
    print("No rows to write")
    sys.exit(0)

# Connect and write
db_path = Path("accuracy.db")
if not db_path.exists():
    print(f"DB not found: {db_path}")
    sys.exit(1)

conn = sqlite3.connect(db_path, timeout=30)
written = 0
errors = 0
for ticker, skew, signal in rows:
    try:
        conn.execute("""
            INSERT OR REPLACE INTO options_skew_history
                (date, ticker, skew_25d, skew_signal, source, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (snapshot_date, ticker, skew, signal, 'massive', now_iso))
        written += 1
    except Exception as e:
        print(f"Error writing {ticker}: {e}")
        errors += 1

conn.commit()
conn.close()

print(f"Wrote {written} rows, {errors} errors")

# Verify
conn = sqlite3.connect(db_path)
total = conn.execute(
    "SELECT COUNT(*) FROM options_skew_history WHERE date = ?",
    (snapshot_date,)
).fetchone()[0]
massive = conn.execute(
    "SELECT COUNT(*) FROM options_skew_history WHERE date = ? AND source = 'massive'",
    (snapshot_date,)
).fetchone()[0]
print(f"DB now has {total} rows for {snapshot_date} ({massive} from massive)")
conn.close()
