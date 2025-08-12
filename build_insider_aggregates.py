#!/usr/bin/env python3
# build_insider_aggregates.py

import sqlite3
import pandas as pd

DB_PATH = "insider_trades.db"

def main():
    # 1) Load raw transactions and holdings
    conn = sqlite3.connect(DB_PATH)
    tx = pd.read_sql(
        "SELECT ticker, date, shares FROM transactions",
        conn,
        parse_dates=["date"]
    )
    hold = pd.read_sql(
        "SELECT ticker, date, shares FROM holdings",
        conn,
        parse_dates=["date"]
    )
    conn.close()

    # 2) Compute net daily insider trades (buys minus sells)
    #    (if your CSVs already folded buys/sells into one 'shares' sign, this is just sum)
    daily = (
        tx
        .groupby(["ticker", "date"], as_index=False)
        .shares
        .sum()
        .rename(columns={"shares": "net_shares"})
    )

    # 3) Add rolling windows for 7-day & 21-day net flow
    daily = daily.sort_values(["ticker", "date"])
    daily["insider_7d"] = (
        daily
        .groupby("ticker")["net_shares"]
        .transform(lambda x: x.rolling(7, min_periods=1).sum())
    )
    daily["insider_21d"] = (
        daily
        .groupby("ticker")["net_shares"]
        .transform(lambda x: x.rolling(21, min_periods=1).sum())
    )

    # 4) Persist aggregates back into SQLite
    conn = sqlite3.connect(DB_PATH)
    daily.to_sql("insider_flows", conn, if_exists="replace", index=False)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_flows_tkd ON insider_flows(ticker, date);")
    conn.commit()
    conn.close()

    print(f"✅ Written insider_flows ({len(daily)} rows) to {DB_PATH}")

if __name__ == "__main__":
    main()
