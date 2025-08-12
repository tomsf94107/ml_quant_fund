#!/usr/bin/env python3
import pandas as pd
from sqlalchemy import create_engine
from pathlib import Path

# ────────────────────────────────────────────────────────────────────────────
# Script: seed_db.py
# Purpose: Seed forecast_accuracy.db from a CSV of evaluation logs
# ────────────────────────────────────────────────────────────────────────────

# Determine script directory (assumes this file lives in forecast_eval/)
script_dir = Path(__file__).resolve().parent

# 1) Load the existing CSV of evaluation logs
csv_path = script_dir / "forecast_evals.csv"
if not csv_path.exists():
    print(f"❌ CSV not found at {csv_path}")
    exit(1)

df = pd.read_csv(csv_path, parse_dates=["timestamp"])
print(f"Read {len(df)} rows from {csv_path}")

# 2) Create (or open) your SQLite accuracy database at repo root
repo_root = script_dir.parent
db_path   = repo_root / "forecast_accuracy.db"
engine    = create_engine(f"sqlite:///{db_path}")

# 3) Write the DataFrame into a table named 'forecast_accuracy'
df.to_sql("forecast_accuracy", engine, if_exists="replace", index=False)
print(f"✅ Seeded {db_path.name} with {len(df)} rows in table 'forecast_accuracy'")
