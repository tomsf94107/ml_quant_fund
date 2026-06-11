"""
scripts/universe_screen.py - Track A stage 2: Massive coverage probe on candidates.
Reads data/universe_candidates.csv, checks which tickers Massive serves with
1500+ trading days since 2019 (the momentum/qv validation window requirement).
Writes data/universe_candidates_screened.csv with pass_screen column.
Run: nohup python3 -m scripts.universe_screen > logs/universe_screen.log 2>&1 &
"""
import sys
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
MIN_DAYS = 1500


def main():
    from analysis.momentum_purged_wf import build_close_panel
    cand = pd.read_csv(ROOT / "data/universe_candidates.csv")
    print(f"probing {len(cand)} candidates against Massive...")
    panel = build_close_panel(cand.ticker.tolist(), "2019-01-01")
    rows = []
    for t in cand.ticker:
        if t in panel.columns:
            s = panel[t].dropna()
            rows.append({"ticker": t, "days": len(s),
                         "start": str(s.index.min().date()) if len(s) else None})
        else:
            rows.append({"ticker": t, "days": 0, "start": None})
    cov = cand.merge(pd.DataFrame(rows), on="ticker")
    cov["pass_screen"] = cov["days"] >= MIN_DAYS
    dest = ROOT / "data/universe_candidates_screened.csv"
    cov.to_csv(dest, index=False)
    print(f"\nscreened: {int(cov.pass_screen.sum())}/{len(cov)} pass ({MIN_DAYS}+ days)")
    print("passing by sector:")
    print(cov[cov.pass_screen].gics_sector.value_counts().to_string())
    fails = cov[~cov.pass_screen]
    if len(fails):
        print(f"\nfailing ({len(fails)}): " + ", ".join(fails.ticker.head(30)))
    print(f"\nwrote {dest}")


if __name__ == "__main__":
    main()
