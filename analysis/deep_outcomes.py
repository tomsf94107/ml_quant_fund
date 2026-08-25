#!/usr/bin/env python3
"""Forward h-day returns for the deep panel, split-adjusted.

WHY accuracy/sink.py cannot be reused: reconcile_outcomes() iterates
`FROM predictions p LEFT JOIN outcomes o WHERE o.id IS NULL` -- it reconciles
LOGGED PREDICTIONS, not a universe. No predictions exist before 2020 or for the
~250 names outside the core list, so `outcomes` covers 2020-01 onward at 126-159
names (rising to ~401 after the Jun-2026 universe expansion). It cannot label a
2017-start, 410-name panel. Its `tickers` arg only FILTERS that set.

METHOD, deliberately identical to sink.py:583 so the two are comparable:
  - mc.download(auto_adjust=True): VERIFIED split-adjusted (AAPL 2020-08-31 4:1
    prints 124.81 -> 129.04, no 4x jump) and verified NOT to rewrite raw_bars,
    which stays unadjusted (price_cache only writes past max(d)).
  - POSITIONAL forward return on the ticker's OWN session index, not a market
    calendar: a name with missing bars (BNY had 71) gets h of its own sessions.
  - Replicates sink.py's `if actual_ret == 0.0: continue`. An undocumented
    filter that drops exactly-flat windows; kept here ONLY so validation can
    reach a floating-point match. Revisit before using these labels in anger.

Writes `deep_outcomes` in accuracy.db. NEVER `outcomes` -- that is the live
accuracy system's table.

  python analysis/deep_outcomes.py --horizon 5
  python analysis/deep_outcomes.py --validate --horizon 5
"""
import argparse, os, sqlite3, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import pandas as pd

DDL = """CREATE TABLE IF NOT EXISTS deep_outcomes (
  ticker TEXT NOT NULL, date TEXT NOT NULL, horizon INTEGER NOT NULL,
  outcome_date TEXT NOT NULL, actual_return REAL NOT NULL,
  PRIMARY KEY (ticker, date, horizon))"""


def build(h, start, end, drop_zero=True):
    from features import massive_client as mc
    tks = sorted({l.strip().upper() for l in open(ROOT / "tickers.txt") if l.strip()})
    con = sqlite3.connect(ROOT / "accuracy.db", timeout=60)
    con.execute(DDL); con.commit()
    t0, ok, fail, rows = time.time(), 0, [], 0
    for i, t in enumerate(tks, 1):
        try:
            px = mc.download(t, start=start, end=end, auto_adjust=True, progress=False)
            if px is None or px.empty:
                fail.append((t, "empty")); continue
            if isinstance(px.columns, pd.MultiIndex):
                px.columns = px.columns.get_level_values(0)
            cs = px["Close"]
            if isinstance(cs, pd.DataFrame):
                cs = cs.iloc[:, 0]
            cs.index = pd.to_datetime(cs.index).tz_localize(None)
            cs = cs[~cs.index.duplicated(keep="last")].sort_index().dropna()
            v, idx = cs.values, cs.index
            batch = []
            for pos in range(len(v) - h):
                a, b = float(v[pos]), float(v[pos + h])
                if a == 0 or a != a or b != b:
                    continue
                r = (b - a) / a
                if r != r or (drop_zero and r == 0.0):
                    continue
                batch.append((t, str(idx[pos].date()), h,
                              str(idx[pos + h].date()), r))
            con.executemany("INSERT OR REPLACE INTO deep_outcomes VALUES (?,?,?,?,?)", batch)
            con.commit(); rows += len(batch); ok += 1
        except Exception as e:
            fail.append((t, repr(e)[:60]))
        if i % 50 == 0:
            print(f"  ...{i}/{len(tks)}  rows={rows}  {time.time()-t0:.0f}s", flush=True)
    n = con.execute("SELECT COUNT(*) FROM deep_outcomes WHERE horizon=?", (h,)).fetchone()[0]
    print(f"\n# {ok}/{len(tks)} tickers, {rows} rows written, table holds {n} at h={h}")
    assert n >= rows, f"ABORT: table has {n}, wrote {rows}"
    if fail:
        print(f"# FAILED ({len(fail)}): {fail[:10]}")
    con.close()
    return 0


def validate(h):
    """Positive control: recomputed vs stored, where both exist and coverage is
    98-99% dense (2024-2025, ~155 names). Must match to floating point."""
    con = sqlite3.connect(ROOT / "accuracy.db", timeout=60)
    df = pd.read_sql("""
        SELECT o.ticker, o.prediction_date d, o.actual_return old, n.actual_return new,
               o.outcome_date od_old, n.outcome_date od_new
        FROM outcomes o JOIN deep_outcomes n
          ON n.ticker=o.ticker AND n.date=o.prediction_date AND n.horizon=o.horizon
        WHERE o.horizon=? AND o.prediction_date BETWEEN '2024-01-01' AND '2025-12-31'
    """, con, params=(h,))
    stored = pd.read_sql("""SELECT COUNT(*) c FROM outcomes WHERE horizon=?
        AND prediction_date BETWEEN '2024-01-01' AND '2025-12-31'""", con, params=(h,)).c[0]
    con.close()
    if df.empty:
        print("NO OVERLAP -- run the build first"); return 1
    df["diff"] = (df.new - df.old).abs()
    dm = (df.od_new != df.od_old).sum()
    print(f"overlap {len(df)} of {stored} stored rows ({len(df)/stored:.1%})")
    print(f"max abs diff : {df['diff'].max():.3e}")
    print(f"> 1e-9       : {(df['diff'] > 1e-9).sum()}")
    print(f"> 1e-4       : {(df['diff'] > 1e-4).sum()}")
    print(f"outcome_date mismatches: {dm}")
    if (df["diff"] > 1e-4).any():
        print("\nworst 10:")
        print(df.nlargest(10, "diff")[["ticker","d","old","new","diff","od_old","od_new"]].to_string(index=False))
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizon", type=int, default=5)
    ap.add_argument("--start", default="2016-07-18")
    ap.add_argument("--end", default="2026-08-14")
    ap.add_argument("--validate", action="store_true")
    ap.add_argument("--keep-zero", action="store_true",
                    help="do NOT replicate sink.py's zero-return filter")
    a = ap.parse_args()
    sys.exit(validate(a.horizon) if a.validate
             else build(a.horizon, a.start, a.end, drop_zero=not a.keep_zero))
