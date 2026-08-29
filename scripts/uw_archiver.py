#!/usr/bin/env python3
"""
uw_archiver.py — daily Unusual Whales snapshot -> SQLite (uw_archive table).

WHY THIS EXISTS (roadmap item #1): your UW lookback is capped (~44 days on
your account). Your options history therefore BEGINS the day this cron first
runs. Every day it doesn't run is history you never get back.

RUN ON YOUR MACHINE (this was written in a sandbox with no network access —
endpoints listed below are the ones your Basic plan exposes; adjust
ENDPOINTS to match your actual entitlements before first run):

    export UW_API_KEY=...          # never hardcode (repo convention; .env already has it)
    python uw_archiver.py --db warning.db

Cron (Vietnam morning, beside check_kill_switches.py — market close ≈ 3am ICT):
    30 6 * * 2-6  cd ~/quant && /usr/bin/python3 uw_archiver.py --db warning.db >> logs/uw_archiver.log 2>&1

  06:30 ICT = 19:30 ET the PREVIOUS day, i.e. after the 16:00 ET close. Tue-Sat
  ICT therefore covers Mon-Fri ET. Rows are stamped with the ET date (see
  et_today below), so snapshot_date names the session actually captured.

  The payload also carries its own date -- market-tide rows include
  {"date": "2026-08-27", "timestamp": "...-04:00"}. A parse step should prefer
  that per-endpoint date over snapshot_date; the raw JSON is stored verbatim
  precisely so parsing stays separate and re-runnable.

Design rules:
  - INSERT OR IGNORE on (endpoint, params, snapshot_date): re-runs are safe.
  - Raw JSON stored verbatim; parsing is a separate, re-runnable step.
  - Failures per endpoint are logged and skipped — one bad endpoint must
    never cost the day's other snapshots.
  - Respect rate limits: simple sleep between calls; exponential backoff on 429.
"""

import argparse, json, os, sqlite3, sys, time, urllib.request, urllib.error
from datetime import date, datetime
from zoneinfo import ZoneInfo

# snapshot_date is the EASTERN date, not the local one.
#
# WHY (found 2026-08-28): date.today() returns the VN date, which runs a day
# ahead of ET. The cron fires 06:30 ICT = 19:30 ET the PREVIOUS day, so every run
# captures a US session and then labels it with tomorrow's date. Meanwhile every
# other series in warning.db is ET-dated -- data_vintages.obs_date, SPY_CLOSE,
# and pit.series_asof, which compares date strings directly. A join from
# uw_archive to any of them would be silently off by one: no error, just
# misalignment. uw_archive is append-only (rule #8), so a wrong label is
# permanent.
#
# ZoneInfo rather than a fixed offset: ET is UTC-4 or UTC-5 depending on DST, and
# hardcoding either would be wrong for half the year.
ET = ZoneInfo("America/New_York")


def et_today() -> str:
    return datetime.now(ET).date().isoformat()

BASE = "https://api.unusualwhales.com"          # verify path prefix in your docs
# Repo convention is UW_API_KEY (set in .env, used by monitor_ticker.py and every
# other UW caller). UW_TOKEN kept as a fallback only. Verified 2026-08-28: .env
# defines UW_API_KEY and NOT UW_TOKEN -- reading UW_TOKEN alone made this script
# exit "not set" on every cron run.
TOKEN = os.environ.get("UW_API_KEY") or os.environ.get("UW_TOKEN")

# (endpoint, params) pairs to snapshot daily. EDIT to your plan's entitlements.
# Keep index/ETF options context + market-wide aggregates + your universe.
UNIVERSE_FILE = "universe_tickers.txt"           # one ticker per line (optional)

ENDPOINTS = [
    # VERIFIED 2026-08-28 by scripts/uw_probe.py: all 13 returned HTTP 200 on the
    # live account (0 dropped). Measured payload ~0.9 MB/day at probe limits,
    # ~1.2 MB/day at the production limits below => ~0.4 GB/yr. Append-only
    # forever (rule #8), so this cadence is a permanent commitment; it was sized
    # deliberately, not by default.
    #
    # market-wide / index level (dashboard features F2-F9, F11)
    ("/api/market/market-tide", {}),
    ("/api/market/total-options-volume", {}),
    ("/api/darkpool/recent", {"limit": 200}),
    ("/api/option-trades/flow-alerts", {"limit": 200}),
    # per-index chains for skew/term/gamma features (F3, F4, F9)
    ("/api/stock/SPY/option-chains", {}),
    ("/api/stock/SPY/greek-exposure", {}),        # gamma proxy (F9) -- assumption-laden
    ("/api/stock/SPY/volatility/term-structure", {}),
    ("/api/stock/SPY/volatility/realized", {}),
    ("/api/stock/SPY/greeks", {}),
    ("/api/stock/SPY/options-volume", {}),
    ("/api/darkpool/SPY", {"limit": 200}),
    ("/api/stock/QQQ/option-chains", {}),
    ("/api/stock/IWM/option-chains", {}),
]

# PER_TICKER is DORMANT by design. This is an INDEX-LEVEL regime system; the
# per-name legs it would feed (S5-S8 breadth/concentration/rotation/epicenter)
# come from the Massive universe, not UW. UNIVERSE_FILE below is deliberately
# NOT tickers.txt -- creating universe_tickers.txt would fire 3 calls x N names
# every morning. Do that only as a conscious decision.
PER_TICKER = [                                    # applied to universe file if present
    "/api/stock/{t}/volatility/realized",
    "/api/stock/{t}/greeks",
    "/api/stock/{t}/options-volume",
]

SLEEP_BETWEEN = 0.6                               # seconds; tune to rate limit
MAX_RETRIES = 4


def http_get(url):
    req = urllib.request.Request(url, headers={
        "Authorization": f"Bearer {TOKEN}", "Accept": "application/json"})
    backoff = 2.0
    for attempt in range(MAX_RETRIES):
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                return r.read().decode()
        except urllib.error.HTTPError as e:
            if e.code == 429 and attempt < MAX_RETRIES - 1:
                time.sleep(backoff); backoff *= 2; continue
            raise
    raise RuntimeError("unreachable")


def canon(params):
    return json.dumps(dict(sorted(params.items())), separators=(",", ":"))


def snapshot(conn, endpoint, params, day):
    qs = "&".join(f"{k}={v}" for k, v in sorted(params.items()))
    url = f"{BASE}{endpoint}" + (f"?{qs}" if qs else "")
    payload = http_get(url)
    json.loads(payload)                            # validate it's JSON before storing
    conn.execute(
        "INSERT OR IGNORE INTO uw_archive (endpoint, query_params, snapshot_date, payload_json) "
        "VALUES (?,?,?,?)", (endpoint, canon(params), day, payload))
    conn.commit()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="warning.db")
    args = ap.parse_args()
    if not TOKEN:
        sys.exit("UW_API_KEY (or UW_TOKEN) not set -- check .env")
    conn = sqlite3.connect(args.db)
    conn.execute("""CREATE TABLE IF NOT EXISTS uw_archive(
        endpoint TEXT NOT NULL, query_params TEXT NOT NULL,
        snapshot_date TEXT NOT NULL, payload_json TEXT NOT NULL,
        pulled_at TEXT NOT NULL DEFAULT (datetime('now')),
        PRIMARY KEY (endpoint, query_params, snapshot_date, pulled_at))""")
    day = et_today()

    jobs = list(ENDPOINTS)
    if os.path.exists(UNIVERSE_FILE):
        tickers = [l.strip() for l in open(UNIVERSE_FILE) if l.strip()]
        jobs += [(tpl.format(t=t), {}) for t in tickers for tpl in PER_TICKER]

    ok = fail = 0
    for endpoint, params in jobs:
        try:
            snapshot(conn, endpoint, params, day)
            ok += 1
        except Exception as e:                     # log & continue — never abort the day
            fail += 1
            print(f"[uw_archiver] FAIL {endpoint} {params}: {e}", file=sys.stderr)
        time.sleep(SLEEP_BETWEEN)
    print(f"[uw_archiver] {day}: {ok} ok, {fail} failed")


if __name__ == "__main__":
    main()
