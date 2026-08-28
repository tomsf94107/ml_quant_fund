#!/usr/bin/env python3
"""
fetch_free_history.py — one-shot + weekly refresh of every FREE dataset the
warning system uses (Part 0 of the report). Run on YOUR machine (written in
a no-network sandbox; URLs verified accessible 2026-08-23 in the study —
re-check any that 404 and see the report's Source Register).

    python fetch_free_history.py --db warning.db --out data/raw

What it pulls (and why):
  FRED (daily/weekly/monthly/quarterly)   -> data_vintages (pub_date = pull date)
      DGS10 DTB3 BAA AAA BAA10YM ABCOMP DRTSCILM SOFR CSUSHPINSA HOUST
      BAMLH0A0HYM2 BAMLC0A0CM   <- ROLLING 3-YEAR WINDOW ONLY: archive forever,
                                   never overwrite older pulls (that's your history)
  ALFRED vintages                          -> PAYEMS GDPC1 INDPRO UNRATE (all vintages)
  Cboe free CSVs                           -> VIX, VXO archives, VIX3M, VIX6M, VIX9D,
                                              SKEW, VVIX, COR1M/3M, P/C ratio archives
  Cboe per-day archive pages (2000-2019)   -> VXO/VIX OHLC + total/index/equity/OEX/SPX
                                              put-call volumes (THE 2000-era unlock)
  CFE per-contract VIX futures settles     -> term structure 2004+
  Shiller ie_data                          -> CAPE etc., 1871+
  Ken French daily factors                 -> survivorship-safe backbone, 1926+
  Ritter IPO statistics PDFs               -> issuance-quality leg (manual parse)
  FINRA margin statistics                  -> leverage leg, 1997+

Idempotent: every artifact lands in --out with a dated filename; FRED/ALFRED
rows upsert into data_vintages keyed by (series, obs_date, pub_date).
"""

import argparse, csv, io, os, sqlite3, sys, time, urllib.request
from datetime import date, timedelta

FRED = ["DGS10", "DTB3", "BAA", "AAA", "BAA10YM", "ABCOMP", "DRTSCILM",
        "SOFR", "CSUSHPINSA", "HOUST", "BAMLH0A0HYM2", "BAMLC0A0CM",
        # Cboe-originated volatility indices, redistributed by FRED. Added
        # 2026-08-28 because cdn.cboe.com is DNS-blackholed to 127.0.0.1 from
        # this host; FRED is the same Cboe data via a reachable route, NOT a
        # substitute dataset. History matches the registry exactly:
        #   VIXCLS 1990-01-02+ (F2)   VXVCLS 2007-12-04+ (F3 VIX3M)
        #   VXOCLS 1986-01-02..2021-09-23 (VXO, discontinued; covers the 2000 era)
        "VIXCLS", "VXVCLS", "VXOCLS"]
ALFRED = ["PAYEMS", "GDPC1", "INDPRO", "UNRATE"]

CBOE_CSVS = {
    "VIX_History.csv":    "https://cdn.cboe.com/api/global/us_indices/daily_prices/VIX_History.csv",
    "VXO_History.csv":    "https://cdn.cboe.com/api/global/us_indices/daily_prices/VXO_History.csv",
    "vxoarchive.xls":     "https://cdn.cboe.com/resources/us/indices/vxoarchive.xls",       # 1986-2003
    "vxocurrent.csv":     "https://cdn.cboe.com/resources/us/indices/vxocurrent.csv",       # 2004-2021
    "VIX3M_History.csv":  "https://cdn.cboe.com/api/global/us_indices/daily_prices/VIX3M_History.csv",
    "VIX6M_History.csv":  "https://cdn.cboe.com/api/global/us_indices/daily_prices/VIX6M_History.csv",
    "VIX9D_History.csv":  "https://cdn.cboe.com/api/global/us_indices/daily_prices/VIX9D_History.csv",
    "SKEW_History.csv":   "https://cdn.cboe.com/api/global/us_indices/daily_prices/SKEW_History.csv",
    "VVIX_History.csv":   "https://cdn.cboe.com/api/global/us_indices/daily_prices/VVIX_History.csv",
    "COR1M_History.csv":  "https://cdn.cboe.com/api/global/us_indices/daily_prices/COR1M_History.csv",
    "COR3M_History.csv":  "https://cdn.cboe.com/api/global/us_indices/daily_prices/COR3M_History.csv",
    "pcratioarchive.csv": "https://cdn.cboe.com/resources/options/volume_and_call_put_ratios/pcratioarchive.csv",  # 1995-2003
    "totalpc.csv":        "https://cdn.cboe.com/resources/options/volume_and_call_put_ratios/totalpc.csv",         # 2006-2019
    "equitypc.csv":       "https://cdn.cboe.com/resources/options/volume_and_call_put_ratios/equitypc.csv",
    "indexpc.csv":        "https://cdn.cboe.com/resources/options/volume_and_call_put_ratios/indexpc.csv",
    "vixpc.csv":          "https://cdn.cboe.com/resources/options/volume_and_call_put_ratios/vixpc.csv",
}

CBOE_DAILY_PAGE = ("https://cdn.cboe.com/resources/us/options/market_statistics/"
                   "daily/cone/archive/html/{d}.html")           # d = YYYY-MM-DD
CFE_SETTLE = ("https://cdn.cboe.com/resources/futures/archive/volume-and-price/"
              "CFE_{m}{yy}_VX.csv")                              # m = F..Z month code
SHILLER = "https://shillerdata.com/"                             # follow link to ie_data.xls
FRENCH = ("https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
          "F-F_Research_Data_Factors_daily_CSV.zip")
RITTER = ["https://site.warrington.ufl.edu/ritter/files/IPO-Statistics.pdf",
          "https://site.warrington.ufl.edu/ritter/files/IPOs-Tech.pdf"]
FINRA_MARGIN = "https://www.finra.org/rules-guidance/key-topics/margin-accounts/margin-statistics"

MONTH_CODES = "FGHJKMNQUVXZ"


def get(url, binary=False):
    req = urllib.request.Request(url, headers={"User-Agent": "warning-system/1.0"})
    with urllib.request.urlopen(req, timeout=60) as r:
        data = r.read()
    return data if binary else data.decode("utf-8", errors="replace")


def save(out, name, blob):
    os.makedirs(out, exist_ok=True)
    path = os.path.join(out, name)
    mode = "wb" if isinstance(blob, bytes) else "w"
    with open(path, mode) as f:
        f.write(blob)
    print(f"  saved {path}")


def fred_csv(series):
    return get(f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series}")


def alfred_all_vintages(series):
    # ALFRED API without a key: use the fredgraph vintage-dates trick is limited;
    # with a (free) FRED API key set FRED_API_KEY, use:
    #   /fred/series/observations?series_id=X&realtime_start=1776-07-04&realtime_end=9999-12-31
    key = os.environ.get("FRED_API_KEY")
    if not key:
        print(f"  [skip] ALFRED {series}: set FRED_API_KEY for full vintages")
        return None
    url = ("https://api.stlouisfed.org/fred/series/observations"
           f"?series_id={series}&api_key={key}&file_type=json"
           "&realtime_start=1776-07-04&realtime_end=9999-12-31")
    return get(url)


def upsert_fred(conn, series, csv_text, pub_date):
    """pub_date semantics: the date the value became PUBLICLY KNOWABLE.

    For series that are never revised (series_meta.derivable_pub_date), that is
    obs_date + publication_lag -- deriving it reproduces exactly what ALFRED
    would return. Stamping the pull date instead makes every historical
    point-in-time read return NA, which is what broke the first S1 replay.

    For revisable series the pull date is retained: only true ALFRED vintages
    can say what the first print was, and inventing one would be fabrication.
    `pulled_at` (schema default) records when we actually fetched, either way.
    """
    try:
        from series_meta import derivable_pub_date, pub_lag_days
    except ImportError:                       # keep the script standalone-runnable
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                        "..", "warning"))
        from series_meta import derivable_pub_date, pub_lag_days

    derive = derivable_pub_date(series)
    lag = pub_lag_days(series)
    rows = list(csv.reader(io.StringIO(csv_text)))
    n = 0
    for obs_date, value in rows[1:]:
        if value in (".", ""):
            continue
        if derive:
            pd_ = (date.fromisoformat(obs_date) + timedelta(days=lag)).isoformat()
        else:
            pd_ = pub_date
        conn.execute("INSERT OR IGNORE INTO data_vintages "
                     "(series_id, obs_date, pub_date, value, source) VALUES (?,?,?,?,?)",
                     (series, obs_date, pd_, float(value), "FRED"))
        n += 1
    conn.commit()
    how = f"pub=obs+{lag}d derived" if derive else f"pub {pub_date} (revisable: needs ALFRED)"
    print(f"  FRED {series}: {n} obs ({how})")


def daily_page_range(out, start, end, sleep=0.4):
    """Scrape Cboe per-day archive pages. ~250 pages/yr. Store raw HTML;
    parse VXO OHLC + P/C volumes in a separate pass."""
    d = start
    while d <= end:
        if d.weekday() < 5:
            url = CBOE_DAILY_PAGE.format(d=d.isoformat())
            try:
                save(os.path.join(out, "cboe_daily"), f"{d.isoformat()}.html", get(url))
            except Exception as e:
                print(f"  [miss] {d}: {e}")   # holidays 404 — expected
            time.sleep(sleep)
        d += timedelta(days=1)


def cfe_all(out, y0=2004, y1=None):
    y1 = y1 or date.today().year
    for y in range(y0, y1 + 1):
        for m in MONTH_CODES:
            url = CFE_SETTLE.format(m=m, yy=str(y)[2:])
            try:
                save(os.path.join(out, "cfe"), f"CFE_{m}{str(y)[2:]}_VX.csv", get(url))
            except Exception:
                pass                            # contracts that never listed — expected
            time.sleep(0.2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="warning.db")
    ap.add_argument("--out", default="data/raw")
    ap.add_argument("--scrape-daily-pages", action="store_true",
                    help="also scrape Cboe per-day pages 1997-2019 (slow, one-shot)")
    ap.add_argument("--only", default="all",
                    help="comma-separated legs to run: fred,alfred,cboe,cfe,french,ritter "
                         "or 'all' (default). Phase 0 = --only fred; weekly cron = --only fred "
                         "(re-pulling Cboe/CFE weekly is wasteful and rude to the source).")
    args = ap.parse_args()
    legs = {l.strip().lower() for l in args.only.split(",")}
    run = lambda leg: ("all" in legs) or (leg in legs)
    conn = sqlite3.connect(args.db)
    today = date.today().isoformat()

    if run("fred"):
        print("[FRED]")
        for s in FRED:
            try:
                upsert_fred(conn, s, fred_csv(s), today)
            except Exception as e:
                print(f"  [fail] {s}: {e}")

    if run("alfred"):
        print("[ALFRED vintages]")
        for s in ALFRED:
            blob = alfred_all_vintages(s)
            if blob:
                save(args.out, f"alfred_{s}_{today}.json", blob)

    if run("cboe"):
        print("[Cboe csvs]")
        for name, url in CBOE_CSVS.items():
            try:
                save(os.path.join(args.out, "cboe"), name, get(url, binary=name.endswith(".xls")))
            except Exception as e:
                print(f"  [fail] {name}: {e}")

    if run("cfe"):
        print("[CFE settles]")
        cfe_all(args.out)

    if run("french"):
        print("[French daily factors]")
        try:
            save(args.out, "F-F_Research_Data_Factors_daily_CSV.zip", get(FRENCH, binary=True))
        except Exception as e:
            print(f"  [fail] French: {e}")

    if run("ritter"):
        print("[Ritter PDFs]")
        for url in RITTER:
            try:
                save(os.path.join(args.out, "ritter"), url.rsplit("/", 1)[-1], get(url, binary=True))
            except Exception as e:
                print(f"  [fail] {url}: {e}")

    print(f"[note] Shiller ie_data: download via {SHILLER} (link target moves)")
    print(f"[note] FINRA margin stats: xlsx linked from {FINRA_MARGIN}")

    if args.scrape_daily_pages:
        print("[Cboe per-day pages 1997-2019 — the 2000-era unlock; slow]")
        daily_page_range(args.out, date(1997, 1, 2), date(2019, 12, 31))


if __name__ == "__main__":
    main()
