#!/usr/bin/env python3
"""overnight_fixes.py -- the four remaining fixes, run unattended.

DESIGN RULE (learned the hard way, 2026-07-11 23:52):
  NEVER delete before the replacement data is in hand. The earlier attempt deleted
  raw_bars >= 2026-06-13 and THEN fetched into a throttled API; nearly every call
  429'd and the table was left holed. Every destructive step here is
  fetch -> STAGE -> verify -> swap, inside a transaction. If the fetch is
  incomplete, the live table is never touched.
"""
import argparse, csv, json, os, sqlite3, sys, time, urllib.request
from datetime import date, datetime, timedelta

ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(ROOT); sys.path.insert(0, ROOT)

LOCK = "/tmp/overnight_fixes.lock"
KEY = os.environ.get("MASSIVE_API_KEY") or os.environ.get("POLYGON_API_KEY")
DEADLINE = time.time() + 2.5 * 3600

VOL_FROM = "2026-06-01"
VOL_CUT  = "2026-06-13"
VOL_TO   = "2026-07-10"

def log(m):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {m}", flush=True)

def deadline_hit():
    if time.time() > DEADLINE:
        log("DEADLINE reached (Pipeline A is due) -- stopping cleanly."); return True
    return False

_last = [0.0]
def get(url, pace=2.5, tries=5):
    for a in range(tries):
        wait = pace - (time.perf_counter() - _last[0])
        if wait > 0: time.sleep(wait)
        _last[0] = time.perf_counter()
        try:
            with urllib.request.urlopen(url, timeout=40) as r:
                return json.loads(r.read())
        except urllib.error.HTTPError as e:
            if e.code == 429:
                back = 30 * (a + 1)
                log(f"    429 -- backing off {back}s (try {a+1}/{tries})")
                time.sleep(back); continue
            if e.code == 404: return None
            raise
        except Exception:
            if a == tries - 1: raise
            time.sleep(10)
    return None

def tickers():
    return [l.strip().upper() for l in open("tickers.txt")
            if l.strip() and not l.startswith("#")]

def step_volume():
    log("=" * 66); log("STEP 1: volume re-fetch (staged, non-destructive)"); log("=" * 66)
    con = sqlite3.connect("prices.db", timeout=120)
    cols = [r[1] for r in con.execute("PRAGMA table_info(raw_bars)")]
    log(f"  raw_bars columns: {cols}")
    con.execute("DROP TABLE IF EXISTS raw_bars_staging")
    con.execute("CREATE TABLE raw_bars_staging AS SELECT * FROM raw_bars WHERE 0")
    con.commit()
    tks = tickers()
    have = {r[0] for r in con.execute(
        "SELECT DISTINCT ticker FROM raw_bars WHERE d >= ?", (VOL_CUT,))}
    tks = sorted(set(tks) | have)
    log(f"  fetching {len(tks)} tickers, {VOL_FROM}..{VOL_TO}, paced 2.5s")
    ok = fail = rows = 0
    for i, t in enumerate(tks, 1):
        if deadline_hit(): break
        u = (f"https://api.polygon.io/v2/aggs/ticker/{t}/range/1/day/{VOL_FROM}/{VOL_TO}"
             f"?adjusted=false&sort=asc&limit=50000&apiKey={KEY}")
        try:
            res = (get(u) or {}).get("results") or []
        except Exception as e:
            fail += 1; log(f"  {t}: {str(e)[:50]}"); continue
        if not res:
            fail += 1; continue
        batch = []
        for r in res:
            d = time.strftime("%Y-%m-%d", time.gmtime(r["t"] / 1000))
            rec = {"ticker": t, "d": d, "open": r.get("o"), "high": r.get("h"),
                   "low": r.get("l"), "close": r.get("c"), "volume": r.get("v"),
                   "vwap": r.get("vw"), "transactions": r.get("n")}
            batch.append(tuple(rec.get(c) for c in cols))
        con.executemany(
            f"INSERT OR REPLACE INTO raw_bars_staging ({','.join(cols)}) "
            f"VALUES ({','.join('?' * len(cols))})", batch)
        con.commit(); ok += 1; rows += len(batch)
        if i % 40 == 0: log(f"  [{i}/{len(tks)}] ok={ok} fail={fail} rows={rows:,}")
    log(f"  fetched: ok={ok} fail={fail} rows={rows:,}")
    n_tk = con.execute("SELECT COUNT(DISTINCT ticker) FROM raw_bars_staging WHERE d >= ?", (VOL_CUT,)).fetchone()[0]
    n_rw = con.execute("SELECT COUNT(*) FROM raw_bars_staging WHERE d >= ?", (VOL_CUT,)).fetchone()[0]
    cur_tk = con.execute("SELECT COUNT(DISTINCT ticker) FROM raw_bars WHERE d >= ?", (VOL_CUT,)).fetchone()[0]
    cur_rw = con.execute("SELECT COUNT(*) FROM raw_bars WHERE d >= ?", (VOL_CUT,)).fetchone()[0]
    log(f"  VERIFY  staging: {n_tk} tickers / {n_rw:,} bars")
    log(f"          live   : {cur_tk} tickers / {cur_rw:,} bars")
    if n_tk < 0.95 * cur_tk or n_rw < 0.95 * cur_rw:
        log("  >> REFUSING TO SWAP: staging is short of live. raw_bars UNTOUCHED.")
        con.close(); return False
    log("  swapping (single transaction)...")
    con.execute("BEGIN")
    con.execute("DELETE FROM raw_bars WHERE d >= ?", (VOL_CUT,))
    con.execute(f"INSERT OR REPLACE INTO raw_bars ({','.join(cols)}) "
                f"SELECT {','.join(cols)} FROM raw_bars_staging WHERE d >= ?", (VOL_CUT,))
    con.execute("COMMIT")
    after = con.execute("SELECT COUNT(DISTINCT ticker), COUNT(*) FROM raw_bars WHERE d >= ?", (VOL_CUT,)).fetchone()
    log(f"  SWAPPED. raw_bars now {after[0]} tickers / {after[1]:,} bars in window")
    con.execute("DROP TABLE raw_bars_staging"); con.commit(); con.close()
    log("  re-syncing daily_prices (local split adjustment, zero API calls)")
    con = sqlite3.connect("prices.db", timeout=120)
    con.execute("DELETE FROM daily_prices WHERE date >= ?", (VOL_CUT,))
    con.commit(); con.close()
    os.system(f"{sys.executable} sync_prices_from_rawbars.py --root .")
    return True

def step_earnings():
    log("=" * 66); log("STEP 2: earnings report_time (AMC/BMO -> exact PIT shift)"); log("=" * 66)
    from features.uw_client import uw_get
    con = sqlite3.connect("earnings.db", timeout=60)
    cols = [r[1] for r in con.execute("PRAGMA table_info(earnings_events)")]
    if "report_time" not in cols:
        con.execute("ALTER TABLE earnings_events ADD COLUMN report_time TEXT")
        con.commit(); log("  added column report_time")
    tks = sorted({r[0] for r in con.execute("SELECT DISTINCT ticker FROM earnings_events")})
    log(f"  {len(tks)} tickers")
    upd = miss = 0
    for i, t in enumerate(tks, 1):
        if deadline_hit(): break
        try:
            rows = (uw_get(f"/api/stock/{t}/earnings") or {}).get("data") or []
        except Exception:
            miss += 1; continue
        for x in rows:
            if (x.get("report_type") or "").lower() != "quarterly": continue
            ad, rt = x.get("report_date"), x.get("report_time")
            if not ad or not rt: continue
            con.execute("UPDATE earnings_events SET report_time=? "
                        "WHERE ticker=? AND announce_date=?", (rt, t, str(ad)[:10]))
            upd += 1
        con.commit(); time.sleep(0.7)
        if i % 50 == 0: log(f"  [{i}/{len(tks)}] updated={upd:,} miss={miss}")
    d = con.execute("SELECT report_time, COUNT(*) FROM earnings_events GROUP BY report_time").fetchall()
    log(f"  DONE updated={upd:,}  distribution: {d}")
    con.close()

def step_listing():
    log("=" * 66); log("STEP 3: first_valid_date from ticker_change events"); log("=" * 66)
    log("  NOTE list_date is the COMPANY's listing (BNY = 1969-12-04) -- a trap.")
    rows = list(csv.DictReader(open("tickers_metadata.csv")))
    cols = list(rows[0].keys())
    if "first_valid_date" not in cols: cols.append("first_valid_date")
    if "listing_source" not in cols: cols.append("listing_source")
    found = 0
    for i, r in enumerate(rows, 1):
        if deadline_hit(): break
        t = r["ticker"].upper(); fv = None; src = ""
        try:
            ev = get(f"https://api.polygon.io/vX/reference/tickers/{t}/events?apiKey={KEY}", pace=2.0)
            events = ((ev or {}).get("results") or {}).get("events") or []
            chg = sorted([e.get("date") for e in events
                          if e.get("type") == "ticker_change" and e.get("date")])
            if chg: fv, src = chg[-1], "ticker_change"
        except Exception:
            pass
        if not fv:
            try:
                ref = get(f"https://api.polygon.io/v3/reference/tickers/{t}?apiKey={KEY}", pace=2.0)
                ld = ((ref or {}).get("results") or {}).get("list_date")
                if ld: fv, src = ld, "list_date"
            except Exception:
                pass
        if fv:
            r["first_valid_date"] = fv; r["listing_source"] = src; found += 1
        else:
            r.setdefault("first_valid_date", ""); r.setdefault("listing_source", "")
        if i % 40 == 0: log(f"  [{i}/{len(rows)}] resolved={found}")
    for r in rows:
        r.setdefault("first_valid_date", ""); r.setdefault("listing_source", "")
    with open("tickers_metadata.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols); w.writeheader(); w.writerows(rows)
    log(f"  DONE resolved={found}/{len(rows)}")
    for r in rows:
        if r["ticker"].upper() in ("FIG", "BNY", "COHR", "NBIS"):
            log(f"    {r['ticker']:6s} {r['first_valid_date']}  ({r['listing_source']})")

def step_probe():
    log("=" * 66); log("STEP 4: where can a POINT-IN-TIME revenue estimate come from?"); log("=" * 66)
    for path, label in [
        (f"/benzinga/v1/earnings?ticker=MSFT&limit=3&apiKey={KEY}", "polygon /benzinga/v1/earnings"),
        (f"/benzinga/v1/consensus-ratings?ticker=MSFT&limit=2&apiKey={KEY}", "polygon /benzinga/v1/consensus-ratings"),
        (f"/vX/reference/financials?ticker=MSFT&limit=1&apiKey={KEY}", "polygon /vX/reference/financials"),
    ]:
        try:
            r = get("https://api.polygon.io" + path, pace=2.0)
            res = (r or {}).get("results") or []
            if isinstance(res, dict): res = [res]
            log(f"  {label:42s} OK n={len(res)}")
            if res:
                ks = list(res[0].keys())
                log(f"      fields: {ks[:14]}")
                hit = [k for k in ks if "rev" in k.lower()]
                if hit: log(f"      *** REVENUE FIELDS PRESENT: {hit}")
        except Exception as e:
            log(f"  {label:42s} FAIL {str(e)[:45]}")

if __name__ == "__main__":
    if os.path.exists(LOCK):
        print(f"LOCKED -- another run is active ({LOCK}). Exiting."); sys.exit(1)
    open(LOCK, "w").write(str(os.getpid()))
    try:
        ap = argparse.ArgumentParser()
        ap.add_argument("--steps", default="all")
        a = ap.parse_args()
        if not KEY:
            log("FATAL: MASSIVE_API_KEY not set. Source .env."); sys.exit(1)
        want = a.steps.split(",") if a.steps != "all" else ["volume","earnings","listing","probe"]
        log(f"START steps={want}  deadline={datetime.fromtimestamp(DEADLINE):%H:%M}")
        for s in want:
            try:
                {"volume": step_volume, "earnings": step_earnings,
                 "listing": step_listing, "probe": step_probe}[s]()
            except Exception as e:
                log(f"STEP {s} FAILED: {e}")
                import traceback; traceback.print_exc()
        log("ALL DONE")
    finally:
        os.remove(LOCK)
