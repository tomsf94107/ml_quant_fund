from data.etl_insider import fetch_insider_trades
df = fetch_insider_trades("GGG", mode="excel")
print("Found rows:", len(df))
print(df.head())

