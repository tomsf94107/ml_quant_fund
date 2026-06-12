"""
scripts/universe_integrate.py - Track A stage 3: integrate screened survivors.
- Appends pass_screen tickers to tickers.txt (backup + dedupe, idempotent)
- Appends metadata rows to tickers_metadata.csv (GICS -> bucket, tier=expansion)
Repeats the May 28 (+25, d73d7b5/c809b68) recipe at scale.
Run: python -m scripts.universe_integrate          (dry-run prints plan)
     python -m scripts.universe_integrate --apply  (writes)
"""
import argparse, shutil, time
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent

GICS_TO_BUCKET = {
    "Financials": "Financials", "Health Care": "Healthcare",
    "Industrials": "Industrials", "Consumer Discretionary": "Consumer Disc",
    "Consumer Staples": "Consumer Staples", "Energy": "Energy",
    "Utilities": "Power", "Real Estate": "REITs", "Materials": "Materials",
    "Communication Services": "Ad Tech", "Information Technology": "Consumer Tech",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()

    scr = pd.read_csv(ROOT / "data/universe_candidates_screened.csv")
    adds = scr[scr.pass_screen].copy()
    cur = [l.strip() for l in open(ROOT / "tickers.txt") if l.strip()]
    cur_set = {t.upper() for t in cur if not t.startswith("#")}
    adds = adds[~adds.ticker.isin(cur_set)]
    print(f"current: {len(cur_set)} | survivors to add: {len(adds)} | post: {len(cur_set) + len(adds)}")

    meta_path = ROOT / "tickers_metadata.csv"
    meta = pd.read_csv(meta_path)
    print(f"metadata now: {len(meta)} rows, columns: {list(meta.columns)}")
    have_meta = set(meta[meta.columns[0]].astype(str).str.upper())
    new_meta = adds[~adds.ticker.isin(have_meta)].copy()

    new_meta["bucket"] = new_meta.gics_sector.map(GICS_TO_BUCKET).fillna("Other")
    print("\nbucket assignment of adds:")
    print(new_meta.bucket.value_counts().to_string())

    if not args.apply:
        print("\nDRY RUN. Re-run with --apply to write. Sample adds:")
        print(adds.head(8)[["ticker", "name", "gics_sector"]].to_string(index=False))
        return

    ts = time.strftime("%Y%m%d_%H%M%S")
    shutil.copy(ROOT / "tickers.txt", ROOT / f"tickers.txt.bak.expansion_{ts}")
    shutil.copy(meta_path, ROOT / f"tickers_metadata.csv.bak.expansion_{ts}")

    with open(ROOT / "tickers.txt", "a") as f:
        pass  # NO comment lines in tickers.txt: production readers do not filter hash lines (Jun 12 lesson)
        for t in adds.ticker:
            f.write(t + "\n")

    # build metadata rows matching existing column structure
    tcol = meta.columns[0]
    rows = []
    for _, r in new_meta.iterrows():
        row = {c: "" for c in meta.columns}
        row[tcol] = r.ticker
        if "bucket" in meta.columns:
            row["bucket"] = r.bucket
        if "tier" in meta.columns:
            row["tier"] = "expansion"
        if "name" in meta.columns:
            row["name"] = r["name"]
        if "sector" in meta.columns:
            row["sector"] = r.bucket
        rows.append(row)
    meta_out = pd.concat([meta, pd.DataFrame(rows)], ignore_index=True)
    meta_out.to_csv(meta_path, index=False)

    final = [l.strip() for l in open(ROOT / "tickers.txt")
             if l.strip() and not l.startswith("#")]
    print(f"\nAPPLIED: tickers.txt {len(cur_set)} -> {len(set(final))} | metadata {len(meta)} -> {len(meta_out)}")
    print(f"backups: tickers.txt.bak.expansion_{ts} + tickers_metadata.csv.bak.expansion_{ts}")
    print("\nPOST-INTEGRATION CHECKLIST:")
    print("  1. python -m data.etl_xbrl_fundamentals            (fundamentals for new names, ~5min)")
    print("  2. nohup insider raw ETL --since 2019 for new names (background, hours)")
    print("  3. tonight: watch Pipeline B duration (245 new names = long feature backfill)")
    print("  4. ~Jul: post-expansion re-test battery (TODO checkpoint)")


if __name__ == "__main__":
    main()
