#!/usr/bin/env python3
"""
READ-ONLY deep probe #2 for the ML Quant Fund.

Answers four specific questions the first scan could not:
  Q1. Which probability column does live scoring actually use?
      -> compares prob_raw vs prob_up vs prob_up_global_ranker in predictions,
         and inspects accuracy_cache to infer what hit-rate is computed on.
  Q2. Are p0-2_A1/A2/batch1 experiment arms, and is accuracy.db the live book?
      -> compares date ranges, row counts, distinct tickers, horizons across
         the parallel `outcomes` tables to fingerprint each arm.
  Q3. How are alpha_fitness vs alpha_fitness_by_ticker keyed?
      -> reports distinct `alpha` counts, horizons, whether market-wide flag,
         date ranges -> tells us the true search width for Deflated Sharpe.
  Q4. What's in institutional_trades.duckdb?
      -> lists tables/cols/rowcounts if duckdb is importable.

SAFETY: read-only. SQLite opened mode=ro&immutable=1. DuckDB read_only=True.
Only SELECT / PRAGMA / information_schema. No writes of any kind.

USAGE (from your project root, env active):
    python deep_probe.py
    python deep_probe.py --root .
"""

import argparse, os, sqlite3, traceback
from pathlib import Path

LINE = "=" * 78
def banner(t): print("\n" + LINE + "\n" + t + "\n" + LINE)
def sub(t):   print("\n" + "-"*78 + "\n" + t + "\n" + "-"*78)

def ro(path):
    uri = f"file:{os.path.abspath(path)}?mode=ro&immutable=1"
    return sqlite3.connect(uri, uri=True, timeout=5)

def q(conn, sql, params=()):
    return conn.execute(sql, params).fetchall()

def has_table(conn, name):
    r = q(conn, "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (name,))
    return bool(r)

def cols_of(conn, t):
    return [r[1] for r in q(conn, f'PRAGMA table_info("{t}")')]

def find(root, names):
    """Find specific db filenames anywhere under root (including root itself)."""
    hits = {}
    # os.walk includes root as the first dirpath, so files directly in root
    # ARE covered — but we make it explicit and robust to symlinks/odd cwd.
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in
                       (".git","__pycache__",".venv","venv","node_modules",".cache")]
        for fn in filenames:
            if fn in names:
                hits.setdefault(fn, os.path.join(dirpath, fn))
    # belt-and-suspenders: also check exact paths in root
    for fn in names:
        cand = os.path.join(root, fn)
        if fn not in hits and os.path.isfile(cand):
            hits[fn] = cand
    return hits

def safe(label, fn):
    try:
        return fn()
    except Exception as e:
        print(f"    [FAIL {label}] {type(e).__name__}: {e}")
        return None

# ---------------------------------------------------------------- Q1
def q1_prob_columns(acc_path):
    banner("Q1) WHICH PROBABILITY COLUMN DOES SCORING USE?")
    if not acc_path:
        print("  accuracy.db not found"); return
    conn = ro(acc_path)
    try:
        if not has_table(conn, "predictions"):
            print("  no predictions table in accuracy.db"); return
        pcols = cols_of(conn, "predictions")
        prob_like = [c for c in pcols if c.lower().startswith("prob") or
                     c.lower() in ("confidence",)]
        print(f"  predictions prob-like columns: {prob_like}")

        # Non-null coverage + spread for each prob-like column
        sub("Coverage + distribution of each prob column (non-null %, min/median/max)")
        n = q(conn, "SELECT COUNT(*) FROM predictions")[0][0]
        for c in prob_like:
            try:
                nn = q(conn, f'SELECT COUNT("{c}") FROM predictions')[0][0]
                mn, mx = q(conn, f'SELECT MIN("{c}"), MAX("{c}") FROM predictions')[0]
                # median via ordered offset
                med = None
                if nn:
                    off = nn//2
                    r = q(conn, f'SELECT "{c}" FROM predictions WHERE "{c}" IS NOT NULL '
                                f'ORDER BY "{c}" LIMIT 1 OFFSET {off}')
                    med = r[0][0] if r else None
                pct = (100.0*nn/n) if n else 0
                print(f"    {c:24s} nonnull={pct:5.1f}%  min={mn}  median={med}  max={mx}")
            except Exception as e:
                print(f"    {c:24s} [FAIL] {e}")

        # How different are raw vs up vs ranker, where all present?
        sub("Divergence: prob_raw vs prob_up vs prob_up_global_ranker (sample 5 rows where they differ)")
        present = [c for c in ("prob_raw","prob_up","prob_up_global_ranker",
                               "prob_eff_uncapped","prob_up_global","prob_pct7")
                   if c in pcols]
        if len(present) >= 2:
            a, b = present[0], present[1]
            rows = safe("divergence", lambda: q(
                conn,
                f'SELECT ticker, prediction_date, horizon, '
                + ", ".join(f'"{c}"' for c in present)
                + f' FROM predictions WHERE "{a}" IS NOT NULL AND "{b}" IS NOT NULL '
                  f'AND ABS("{a}"-"{b}") > 0.01 '
                  f'ORDER BY prediction_date DESC LIMIT 5'))
            if rows:
                print(f"    cols: ['ticker','prediction_date','horizon'] + {present}")
                for r in rows: print(f"    {r}")
            else:
                print(f"    (no rows where {a} and {b} differ by >0.01 — they may be ~identical)")
            # correlation-ish: count how often they differ at all
            for b in present[1:]:
                d = safe(f"diffcount {present[0]} vs {b}", lambda b=b: q(
                    conn, f'SELECT COUNT(*) FROM predictions '
                          f'WHERE "{present[0]}" IS NOT NULL AND "{b}" IS NOT NULL '
                          f'AND ABS("{present[0]}"-"{b}")>0.01')[0][0])
                tot = safe("tot", lambda b=b: q(
                    conn, f'SELECT COUNT(*) FROM predictions '
                          f'WHERE "{present[0]}" IS NOT NULL AND "{b}" IS NOT NULL')[0][0])
                if d is not None and tot:
                    print(f"    {present[0]} vs {b}: differ >0.01 in {d}/{tot} "
                          f"({100.0*d/tot:.1f}%) of co-present rows")

        # signal threshold inference: what prob corresponds to BUY vs HOLD?
        sub("Signal vs prob: does `signal` track prob_up or prob_raw? (avg prob by signal)")
        if "signal" in pcols:
            for c in [x for x in ("prob_up","prob_raw","prob_up_global_ranker") if x in pcols]:
                rows = safe(f"signal-by-{c}", lambda c=c: q(
                    conn, f'SELECT signal, COUNT(*), '
                          f'ROUND(AVG("{c}"),4), ROUND(MIN("{c}"),4), ROUND(MAX("{c}"),4) '
                          f'FROM predictions WHERE "{c}" IS NOT NULL '
                          f'GROUP BY signal ORDER BY 2 DESC LIMIT 8'))
                if rows:
                    print(f"    by signal, column={c}: [signal, n, avg, min, max]")
                    for r in rows: print(f"      {r}")

        # accuracy_cache structure to see what it stores
        sub("accuracy_cache structure (to infer what hit-rate is computed on)")
        if has_table(conn, "accuracy_cache"):
            ac = cols_of(conn, "accuracy_cache")
            print(f"    accuracy_cache cols: {ac}")
            rows = safe("acc sample", lambda: q(conn,
                'SELECT * FROM accuracy_cache ORDER BY rowid DESC LIMIT 3'))
            if rows:
                for r in rows: print(f"      {tuple(str(x)[:30] for x in r)}")
        else:
            print("    (no accuracy_cache table)")
    finally:
        conn.close()

# ---------------------------------------------------------------- Q2
def q2_experiment_arms(paths):
    banner("Q2) PRODUCTION vs EXPERIMENT ARMS (fingerprint the parallel outcomes tables)")
    # paths: dict filename -> fullpath for the candidate dbs
    fingerprints = []
    for fname, p in paths.items():
        if not p: continue
        conn = ro(p)
        try:
            if not has_table(conn, "outcomes"):
                continue
            ocols = cols_of(conn, "outcomes")
            n = q(conn, "SELECT COUNT(*) FROM outcomes")[0][0]
            dr = safe(f"daterange {fname}", lambda: q(conn,
                "SELECT MIN(prediction_date), MAX(prediction_date) FROM outcomes")[0]
                if "prediction_date" in ocols else (None,None))
            nt = safe(f"tickers {fname}", lambda: q(conn,
                "SELECT COUNT(DISTINCT ticker) FROM outcomes")[0][0]
                if "ticker" in ocols else None)
            hz = safe(f"horizons {fname}", lambda: [r[0] for r in q(conn,
                "SELECT DISTINCT horizon FROM outcomes ORDER BY horizon")]
                if "horizon" in ocols else None)
            fingerprints.append((fname, n, dr, nt, hz))
        finally:
            conn.close()
    if not fingerprints:
        print("  no outcomes tables found among candidates"); return
    print(f"  {'db':22s} {'rows':>8s}  {'tickers':>7s}  date_range                 horizons")
    for fname, n, dr, nt, hz in fingerprints:
        drs = f"{dr[0]}..{dr[1]}" if dr and dr[0] else "?"
        print(f"  {fname:22s} {n:8d}  {str(nt):>7s}  {drs:26s} {hz}")
    print("\n  Interpretation hints:")
    print("  - Largest row-count + widest date range + most tickers = likely PRODUCTION.")
    print("  - Identical schemas with smaller/disjoint ranges = experiment arms (A/B).")
    print("  - Same date range but different tickers = universe-split experiments.")

# ---------------------------------------------------------------- Q3
def q3_alpha_fitness(acc_path):
    banner("Q3) ALPHA FITNESS KEYING + TRUE SEARCH WIDTH (for Deflated Sharpe)")
    if not acc_path:
        print("  accuracy.db not found"); return
    conn = ro(acc_path)
    try:
        for t in ("alpha_fitness", "alpha_fitness_by_ticker"):
            if not has_table(conn, t):
                print(f"  {t}: MISSING"); continue
            c = cols_of(conn, t)
            n = q(conn, f'SELECT COUNT(*) FROM "{t}"')[0][0]
            sub(f"{t} (rows={n})")
            print(f"    cols: {c}")
            # distinct alphas = search width
            if "alpha" in c:
                na = q(conn, f'SELECT COUNT(DISTINCT alpha) FROM "{t}"')[0][0]
                print(f"    DISTINCT alpha (search width): {na}")
            if "horizon" in c:
                hz = [r[0] for r in q(conn, f'SELECT DISTINCT horizon FROM "{t}" ORDER BY horizon')]
                print(f"    horizons: {hz}")
            if "is_market_wide" in c:
                mw = q(conn, f'SELECT is_market_wide, COUNT(*) FROM "{t}" GROUP BY is_market_wide')
                print(f"    is_market_wide breakdown: {mw}")
            if "scored_date" in c:
                dr = q(conn, f'SELECT MIN(scored_date), MAX(scored_date) FROM "{t}"')[0]
                print(f"    scored_date range: {dr[0]} .. {dr[1]}")
            # distribution of rank_ic and ic_t to see how many 'pass'
            if "rank_ic" in c and "ic_t" in c:
                stats = q(conn, f'SELECT ROUND(AVG(rank_ic),4), ROUND(MIN(rank_ic),4), '
                                f'ROUND(MAX(rank_ic),4), ROUND(AVG(ic_t),3), ROUND(MAX(ic_t),3) '
                                f'FROM "{t}"')[0]
                print(f"    rank_ic avg/min/max = {stats[0]}/{stats[1]}/{stats[2]} ; "
                      f"ic_t avg/max = {stats[3]}/{stats[4]}")
                # how many clear |t|>3 (Harvey-Liu-Zhu bar) — naive, pre-deflation
                p3 = q(conn, f'SELECT COUNT(*) FROM "{t}" WHERE ABS(ic_t)>3')[0][0]
                p2 = q(conn, f'SELECT COUNT(*) FROM "{t}" WHERE ABS(ic_t)>2')[0][0]
                print(f"    rows with |ic_t|>2: {p2} ; |ic_t|>3: {p3}  "
                      f"(pre-deflation, BEFORE multiple-testing correction)")
            # sample top alphas
            if "alpha" in c and "rank_ic" in c:
                rows = safe("top", lambda: q(conn,
                    f'SELECT alpha, horizon, rank_ic, ic_t '
                    + (', sharpe, turnover, fitness' if 'fitness' in c else '')
                    + f' FROM "{t}" ORDER BY ABS(ic_t) DESC LIMIT 8'))
                if rows:
                    print(f"    top by |ic_t|:")
                    for r in rows: print(f"      {r}")
    finally:
        conn.close()

# ---------------------------------------------------------------- Q4
def q4_duckdb(root):
    banner("Q4) institutional_trades.duckdb CONTENTS")
    # locate it
    target = None
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in
                       (".git","__pycache__",".venv","venv","node_modules")]
        for fn in filenames:
            if fn == "institutional_trades.duckdb":
                target = os.path.join(dirpath, fn); break
        if target: break
    if not target:
        print("  institutional_trades.duckdb not found"); return
    print(f"  found: {target}  ({os.path.getsize(target)/1e6:.2f} MB)")
    try:
        import duckdb
    except Exception as e:
        print(f"  duckdb not importable: {e}")
        print("  -> run: pip install duckdb   (then re-run this script)")
        return
    try:
        con = duckdb.connect(target, read_only=True)
    except Exception as e:
        print(f"  open (read_only) FAILED: {e}"); return
    try:
        tbls = con.execute("SELECT table_name FROM information_schema.tables "
                           "ORDER BY table_name").fetchall()
        print(f"  {len(tbls)} tables: {[t[0] for t in tbls]}")
        for (t,) in tbls:
            try:
                cols = con.execute("SELECT column_name, data_type FROM information_schema.columns "
                                   f"WHERE table_name='{t}' ORDER BY ordinal_position").fetchall()
                rc = con.execute(f'SELECT COUNT(*) FROM "{t}"').fetchone()[0]
                print(f"\n  == {t} (rows={rc}) ==")
                print(f"     cols: {[(c[0],c[1]) for c in cols]}")
                # date range if a date-ish col exists
                datecol = next((c[0] for c in cols if c[0].lower() in
                                ("trade_date","date","trade_ts","fetched_at")), None)
                if datecol:
                    dr = con.execute(f'SELECT MIN("{datecol}"), MAX("{datecol}") FROM "{t}"').fetchone()
                    print(f"     {datecol} range: {dr[0]} .. {dr[1]}")
            except Exception as e:
                print(f"  [FAIL {t}] {e}")
    finally:
        try: con.close()
        except Exception: pass

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    args = ap.parse_args()
    root = os.path.expanduser(args.root)

    banner("ML QUANT FUND — READ-ONLY DEEP PROBE #2")
    print("Read-only. No writes to any database. SELECT/PRAGMA/information_schema only.")
    print(f"Root: {os.path.abspath(root)}")

    wanted = {"accuracy.db","p0-2_A1.db","p0-2_A2.db","p0-2_batch1.db"}
    found = find(root, wanted)
    print("\nLocated dbs:")
    for k in sorted(wanted):
        print(f"  {k:18s} -> {found.get(k,'NOT FOUND')}")

    acc = found.get("accuracy.db")
    q1_prob_columns(acc)
    q2_experiment_arms({k: found.get(k) for k in wanted})
    q3_alpha_fitness(acc)
    q4_duckdb(root)

    banner("DONE — paste this ENTIRE output back")

if __name__ == "__main__":
    try:
        main()
    except Exception:
        print("\n[UNEXPECTED ERROR] paste this traceback back:")
        traceback.print_exc()
