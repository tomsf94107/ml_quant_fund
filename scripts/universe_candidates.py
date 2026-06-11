"""
scripts/universe_candidates.py - Track A step 1: candidate ticker list for 149 -> ~400.
Sources: S&P 500 + S&P 400 (Wikipedia GICS tables). Rules:
  - exclude current universe (tickers.txt)
  - SECTOR DIVERSITY: current book is tech/AI-heavy; adds are quota-capped per GICS
    sector to deliberately overweight what we LACK (financials, health, industrials,
    staples, energy, utilities, REITs, materials)
  - SP500 names rank ahead of SP400 within each sector (liquidity proxy; real
    liquidity screen happens at backfill when Massive serves actual volume)
Output: data/universe_candidates.csv (ticker, name, gics_sector, source, rank)
Run: python -m scripts.universe_candidates --target 260
"""
import argparse
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
WIKI_500 = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
WIKI_400 = "https://en.wikipedia.org/wiki/List_of_S%26P_400_companies"

# adds quota per GICS sector (sums ~260). Tech deliberately LOW: we own it already.
QUOTA = {
    "Financials": 40, "Health Care": 40, "Industrials": 38,
    "Consumer Discretionary": 28, "Consumer Staples": 22, "Energy": 20,
    "Utilities": 16, "Real Estate": 16, "Materials": 16,
    "Communication Services": 12, "Information Technology": 12,
}


def fetch(url, source):
    import urllib.request
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (Macintosh) research script"})
    with urllib.request.urlopen(req, timeout=30) as r:
        html = r.read().decode("utf-8", errors="ignore")
    tables = pd.read_html(html)
    t = None
    for tb in tables:
        cols = [str(c).lower() for c in tb.columns]
        if any("symbol" in c for c in cols) and any("gics sector" in c for c in cols):
            t = tb
            break
    if t is None:
        raise RuntimeError(f"no constituent table found at {url}")
    sym = [c for c in t.columns if "ymbol" in str(c)][0]
    name = [c for c in t.columns if "ecurity" in str(c) or "ompany" in str(c)][0]
    sect = [c for c in t.columns if "GICS Sector" in str(c)][0]
    out = t[[sym, name, sect]].copy()
    out.columns = ["ticker", "name", "gics_sector"]
    out["ticker"] = out["ticker"].astype(str).str.strip().str.upper()
    out["source"] = source
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=int, default=260)
    args = ap.parse_args()

    current = {l.strip().upper() for l in open(ROOT / "tickers.txt")
               if l.strip() and not l.startswith("#")}
    print(f"current universe: {len(current)} tickers")

    sp5 = fetch(WIKI_500, "SP500")
    sp4 = fetch(WIKI_400, "SP400")
    allc = pd.concat([sp5, sp4]).drop_duplicates("ticker")
    allc = allc[~allc.ticker.isin(current)]
    allc = allc[~allc.ticker.str.contains(r"\.")]  # skip dual-class dots (BRK.B etc) for feed simplicity
    print(f"candidates after exclusions: {len(allc)}")

    picks = []
    scale = args.target / sum(QUOTA.values())
    for sector, q in QUOTA.items():
        pool = allc[allc.gics_sector == sector]
        pool = pd.concat([pool[pool.source == "SP500"], pool[pool.source == "SP400"]])
        take = pool.head(max(1, int(round(q * scale))))
        picks.append(take)
        print(f"  {sector:24s} quota {q:3d} -> taking {len(take):3d} (pool {len(pool)})")
    out = pd.concat(picks).reset_index(drop=True)
    out["rank"] = out.index + 1
    dest = ROOT / "data/universe_candidates.csv"
    out.to_csv(dest, index=False)
    print(f"\nwrote {len(out)} candidates -> {dest}")
    print("sector mix of adds:")
    print(out.gics_sector.value_counts().to_string())
    print("\nNEXT: stage-2 liquidity/coverage screen via Massive backfill (drops dead feeds),")
    print("then pipeline extension. Current+adds =", len(current), "+", len(out))


if __name__ == "__main__":
    main()
