#!/usr/bin/env python3
# load_sec_insider_sqlite.py

import os
import glob
import sqlite3
import pandas as pd

# Directories containing your SEC CSVs
BASE_DIR = "SEC files"
TX_DIR = os.path.join(BASE_DIR, "Transactions")
HOLD_DIR = os.path.join(BASE_DIR, "Holdings")
DB_PATH = "insider_trades.db"


def load_folder_to_table(conn, folder, table_name):
    """
    Reads all CSVs in `folder` and writes to `table_name` in SQLite, detecting date/shares columns.
    """
    paths = sorted(glob.glob(os.path.join(folder, "*.csv")))
    if not paths:
        raise FileNotFoundError(f"No CSV files found in {folder}")

    dfs = []
    for path in paths:
        df = pd.read_csv(path)
        # detect date column
        low = {c.lower(): c for c in df.columns}
        for key in ('date', 'ds', 'filed_date'):
            if key in low:
                date_col = low[key]
                break
        else:
            raise ValueError(f"No date column in {os.path.basename(path)}")
        # detect share column (including 'shr')
        share_cands = [c for c in df.columns if ('share' in c.lower() or 'shr' in c.lower())]
        if table_name == 'transactions':
            picks = [c for c in share_cands if 'net' in c.lower() or 'trade' in c.lower()]
        else:
            picks = [c for c in share_cands if 'held' in c.lower() or 'own' in c.lower()]
        share_col = picks[0] if picks else (share_cands[0] if share_cands else None)
        if share_col is None:
            raise ValueError(f"No shares column in {os.path.basename(path)}")
        # normalize
        df = df.rename(columns={date_col: 'date', share_col: 'shares'})
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df = df[['ticker', 'date', 'shares']]
        dfs.append(df)

    full = pd.concat(dfs, ignore_index=True)
    full['shares'] = pd.to_numeric(full['shares'], errors='coerce').fillna(0.0)
    full.to_sql(table_name, conn, if_exists='replace', index=False)
    print(f"✅ Loaded {len(full)} rows into '{table_name}' from {len(paths)} files.")


def main():
    conn = sqlite3.connect(DB_PATH)
    load_folder_to_table(conn, TX_DIR, 'transactions')
    load_folder_to_table(conn, HOLD_DIR, 'holdings')
    cur = conn.cursor()
    cur.execute("CREATE INDEX IF NOT EXISTS idx_tx_tkd ON transactions(ticker, date);")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_hold_tkd ON holdings(ticker, date);")
    conn.commit()
    conn.close()
    print(f"✅ SQLite DB ready at '{DB_PATH}' with transactions & holdings tables.")

if __name__ == '__main__':
    main()
