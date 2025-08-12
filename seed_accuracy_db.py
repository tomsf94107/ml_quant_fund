#!/usr/bin/env python3
import pandas as pd
from sqlalchemy import create_engine
from pathlib import Path

# ────────────────────────────────────────────────────────────────────────────
# Script: seed_accuracy_db.py
# Purpose: Seed forecast_accuracy.db from your CSV of evaluation logs
# ────────────────────────────────────────────────────────────────────────────

# 1) Load the existing CSV of evaluation logs
csv_path = repo_root / "forecast_eval" / "forecast_evals.csv"
# 2) Load the CSV of evaluation logs
csv_path = repo_root / "forecast_evals.csv"
if not csv_path.exists():
    print(f"❌ CSV not found at {csv_path}")
    exit(1)
df = pd.read_csv(csv_path, parse_dates=["timestamp"])
print(f"Read {len(df)} rows from {csv_path}")

# 3) Open or create the accuracy DB
db_path = repo_root / "forecast_accuracy.db"
engine = create_engine(f"sqlite:///{db_path}")

# 4) Write into forecast_accuracy table (replace any existing)
df.to_sql("forecast_accuracy", engine, if_exists="replace", index=False)
print(f"✅ Seeded {db_path.name} with {len(df)} rows in table 'forecast_accuracy'")

