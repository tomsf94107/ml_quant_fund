#!/usr/bin/env python3
"""
add_ticker.py -- add ticker(s) to the universe and backfill data via the fund's
REAL backfill scripts (batch-oriented). See header notes in prior version.

BACKFILLS commands were INFERRED from argparse/docstring recon (2026-08-10).
Verify with --dry-run before a live batch. backfill_greeks interface is
UNCONFIRMED (options -> confirmed=False, won't auto-run).

TIME-CRITICAL: dark-pool history is a ~44-day perishable UW window -> runs FIRST.

--force re-backfills a name already in the CSV WITHOUT re-appending its row.
"""
import argparse, csv, os, shutil, sqlite3, subprocess, sys
from datetime import datetime, timezone

VERSION = "add_ticker v2.3"

ROOT = os.path.expanduser(os.environ.get("ML_QUANT_ROOT", "~/ML_Quant_Fund"))
META = os.path.join(ROOT, "tickers_metadata.csv")
WATCHLIST = os.path.join(ROOT, "tickers_watchlist.txt")
# THE universe file. daily_runner.load_tickers() reads tickers.txt.
# tickers_metadata.csv is METADATA ONLY -- appending there does NOT enrol a
# ticker into the daily prediction run. (Confirmed 2026-08-14: CYBR had 7,705
# rows in accuracy.db while absent from the CSV entirely.)
RUNNER = os.path.join(ROOT, "tickers.txt")
DBS = {
    "prices":         os.path.join(ROOT, "prices.db"),
    "monitor":        os.path.join(ROOT, "earnings_monitor.db"),
    "accuracy":       os.path.join(ROOT, "accuracy.db"),
    "short_interest": os.path.join(ROOT, "short_interest.db"),
    "institutional":  os.path.join(ROOT, "institutional_trades.db"),
}
TICKER_COLS = {"ticker", "symbol", "sym"}

APPLIES = {
    "darkpool":       {"equity", "etf", "adr"},
    "ohlcv":          {"equity", "etf", "adr"},
    "options":        {"equity", "etf", "adr"},
    "short_interest": {"equity", "adr"},
    "monitor":        {"equity"},
    "earnings":       {"equity", "adr"},
}

BACKFILLS = [
    dict(key="darkpool", confirmed=True, time_critical=True, db="monitor",
         cmd="python scripts/initiate_darkpool_universe.py --budget {dp_budget} --days {dp_days}",
         verify="SELECT COUNT(*) FROM darkpool_prints WHERE ticker=?"),
    dict(key="ohlcv", confirmed=True, db="prices",
         cmd="python scripts/backfill_raw_bars.py --tickers {tlist} --start {start}",
         verify="SELECT COUNT(*) FROM raw_bars WHERE ticker=?"),
    dict(key="options", confirmed=True, whole_universe=True, db="accuracy",
         # reads tickers.txt + tickers_watchlist.txt directly (patched 2026-08-15),
         # so new names are in scope automatically -- same shape as FINRA SI.
         cmd="python backfill_greeks.py",
         verify="SELECT COUNT(*) FROM options_greeks WHERE ticker=?"),
    dict(key="short_interest", confirmed=True, whole_universe=True, db="short_interest",
         cmd="python finra_short_interest.py",
         verify="SELECT COUNT(*) FROM short_interest WHERE ticker=?"),
    dict(key="monitor", confirmed=True, db="monitor",
         cmd="python scripts/monitor_ticker.py {tspace}",
         verify="SELECT COUNT(*) FROM form4_transactions WHERE ticker=?"),
    dict(key="earnings", confirmed=False, needs_build=True, db="accuracy",
         cmd="# backfill_earnings_uw_new_tickers.py hardcoded to 30 names -- generalize to --tickers",
         verify="SELECT COUNT(*) FROM earnings_calendar WHERE ticker=?"),
]


import re

SENTINELS = {"TICKER", "SYMBOL", "MACRO", "SKIP", "NONE", "NULL", "NA", "TEST",
             "ALL", "DEFAULT", "EXAMPLE"}
SYMBOL_RE = re.compile(r"^[A-Z][A-Z0-9.=-]{0,6}$")


def validate_symbol(t):
    """Reject non-symbols BEFORE they reach a fetcher.

    On 2026-08-15 five different kinds of junk had reached live fetchers:
      'macro'/'skip'  control keywords cached as splits (2026-07-16, 41s apart)
      'TICKER'        a placeholder line in tickers_watchlist.txt, sent to UW
                      on every greeks run
      'RZLZ'          a typo of RZLV, cached with an empty splits payload
      'WCCBYND'       WCC + BYND concatenated, and it received a fitness score
    None was caught by anything. This is the entry-point gate."""
    if t in SENTINELS:
        return f"sentinel/placeholder ({t.lower()})"
    if not SYMBOL_RE.match(t):
        return "does not look like a symbol (^[A-Z][A-Z0-9.=-]{0,6}$)"
    return None


def classify(t):
    tu = t.upper()
    if tu.endswith("=F"): return "future"
    if tu.endswith("-USD") or tu.endswith("-USDT"): return "crypto"
    if "." in tu and tu.rsplit(".", 1)[1] not in ("A", "B"): return "foreign"
    return "equity"


def load_meta():
    if not os.path.isfile(META):
        sys.exit(f"FATAL: universe file not found: {META}")
    with open(META, newline="") as f:
        rows = list(csv.reader(f))
    header = rows[0] if rows else ["ticker"]
    tcol = next((i for i, h in enumerate(header) if h.strip().lower() in TICKER_COLS), 0)
    existing = {r[tcol].strip().upper() for r in rows[1:] if r and len(r) > tcol and r[tcol].strip()}
    return header, tcol, existing


def append_meta(header, tcol, ticker, sector, cohort):
    shutil.copy2(META, META + ".bak")
    row = [""] * len(header)
    row[tcol] = ticker
    for i, h in enumerate(header):
        hl = h.strip().lower()
        if hl == "sector" and sector: row[i] = sector
        elif hl in ("cohort", "bucket") and cohort: row[i] = cohort
    with open(META, "a", newline="") as f:
        csv.writer(f).writerow(row)


def seed_prices(ticker, start, dry):
    """Seed a NEW ticker's price history.

    backfill_raw_bars.py only DEEPENS tickers already in raw_bars -- it reports
    'already deep, nothing to do' for a symbol with zero rows, so it cannot seed.
    price_cache only extends FORWARD from an existing last bar. Neither seeds.
    This is the path verified on JPM/IBM/ORCL and the 9 tickers added 2026-08-15.

    Runs BEFORE enrolment: a ticker in tickers.txt with no bars trips the
    stale-panel guard and logs REFUSING on every pipeline run."""
    if dry:
        return f"DRY: seed {start}..today"
    try:
        sys.path.insert(0, ROOT)
        from features import massive_client as mc
        from features import price_cache as pc
    except Exception as e:
        return f"IMPORT-FAIL {e.__class__.__name__}"
    try:
        raw = mc.download(ticker, start=start, end=None, auto_adjust=False)
    except Exception as e:
        return f"FETCH-FAIL {e.__class__.__name__}"
    n = 0 if raw is None else len(raw)
    if not n:
        return "NO-VENDOR-DATA (check symbol / corporate action)"
    try:
        con = pc._conn()
        pc._write_raw(con, ticker, raw)
        con.commit()
        con.close()
    except Exception as e:
        return f"WRITE-FAIL {e.__class__.__name__}"
    return f"SEEDED {n} rows"


def append_runner(ticker):
    """Append to tickers.txt -- this is what actually enrols the ticker."""
    existing = {l.strip().upper() for l in open(RUNNER)} if os.path.isfile(RUNNER) else set()
    if ticker in existing:
        return False
    if os.path.isfile(RUNNER):
        shutil.copy2(RUNNER, RUNNER + ".bak")
        tail_nl = open(RUNNER).read().endswith("\n")
    else:
        tail_nl = True
    with open(RUNNER, "a") as f:
        f.write(("" if tail_nl else "\n") + ticker + "\n")
    return True


def append_watchlist(ticker):
    existing = {l.strip().upper() for l in open(WATCHLIST)} if os.path.isfile(WATCHLIST) else set()
    if ticker not in existing:
        with open(WATCHLIST, "a") as f:
            f.write(ticker + "\n")


def verify(dbkey, sql, ticker):
    path = DBS[dbkey]
    if not os.path.isfile(path): return f"NO-DB({dbkey})"
    try:
        con = sqlite3.connect(path, timeout=30)
        try:
            n = con.execute(sql, (ticker,)).fetchone()[0]
        finally:
            con.close()
        return f"OK n={n}"
    except sqlite3.Error as e:
        return f"VERIFY-ERR({e.__class__.__name__})"


def run_backfill(bf, tickers, args):
    if bf.get("needs_build"):
        return bf["cmd"], {t: "NEEDS-BUILD" for t in tickers}
    cmd = bf["cmd"].format(tlist=",".join(tickers), tspace=" ".join(tickers),
                           start=args.start, dp_budget=args.dp_budget, dp_days=args.dp_days)
    if not bf["confirmed"]:
        return cmd, {t: "VERIFY-CMD" for t in tickers}
    if args.dry_run:
        return cmd, {t: "DRY" for t in tickers}
    try:
        r = subprocess.run(cmd, shell=True, cwd=ROOT, capture_output=True, text=True, timeout=7200)
    except subprocess.TimeoutExpired:
        return cmd, {t: "TIMEOUT" for t in tickers}
    if r.returncode != 0:
        tail = ((r.stderr or r.stdout).strip().splitlines() or [""])[-1][:50]
        return cmd, {t: f"FAIL rc={r.returncode} {tail}" for t in tickers}
    return cmd, {t: verify(bf["db"], bf["verify"], t) for t in tickers}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tickers", nargs="*")
    ap.add_argument("--from-file")
    ap.add_argument("--type", choices=["equity", "etf", "adr", "future", "crypto", "foreign"])
    ap.add_argument("--sector", default="")
    ap.add_argument("--cohort", default="")
    ap.add_argument("--only")
    ap.add_argument("--skip")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--watchlist", action="store_true",
                    help="also append tickers_watchlist.txt (predictions only, no accuracy scoring)")
    ap.add_argument("--no-runner", action="store_true",
                    help="do NOT append tickers.txt (ticker will not generate predictions)")
    ap.add_argument("--start", default="2016-01-01",
                    help="price seed / raw_bars backfill start date")
    ap.add_argument("--no-seed", action="store_true",
                    help="skip the price seed (NOT recommended for new tickers: "
                         "backfill_raw_bars cannot seed a ticker with zero rows)")
    ap.add_argument("--allow-any-symbol", action="store_true",
                    help="bypass symbol validation")
    ap.add_argument("--dp-budget", type=int, default=50000, dest="dp_budget")
    ap.add_argument("--dp-days", type=int, default=45, dest="dp_days")
    ap.add_argument("--root")
    args = ap.parse_args()

    global ROOT, META, WATCHLIST, RUNNER, DBS
    if args.root:
        ROOT = os.path.expanduser(args.root)
        META = os.path.join(ROOT, "tickers_metadata.csv")
        WATCHLIST = os.path.join(ROOT, "tickers_watchlist.txt")
        RUNNER = os.path.join(ROOT, "tickers.txt")
        DBS = {k: os.path.join(ROOT, os.path.basename(v)) for k, v in DBS.items()}

    tickers = [t.strip().upper() for t in args.tickers if t.strip()]
    if args.from_file:
        for line in open(args.from_file):
            tok = line.split(",")[0].strip().upper()
            if tok: tickers.append(tok)
    tickers = list(dict.fromkeys(tickers))
    if not tickers:
        ap.error("no tickers given")

    bad = [(t, validate_symbol(t)) for t in tickers]
    bad = [(t, r) for t, r in bad if r]
    if bad:
        for t, r in bad:
            print(f"REJECTED {t}: {r}")
        if not args.allow_any_symbol:
            sys.exit("FATAL: invalid symbol(s). Use --allow-any-symbol to override.")
        print("# --allow-any-symbol given; proceeding anyway\n")

    only = set(args.only.split(",")) if args.only else None
    skip = set(args.skip.split(",")) if args.skip else set()

    header, tcol, existing = load_meta()
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%MZ")
    print(f"# {VERSION}  root={ROOT}  meta_rows={len(existing)}  run={stamp}")
    if args.dry_run: print("# DRY-RUN: no writes, no fetches\n")

    meta, to_add, new_names = {}, [], []
    for t in tickers:
        it = args.type or classify(t)
        present = t in existing
        if it in ("future", "crypto", "foreign"):
            meta[t] = dict(type=it, universe="EXCLUDE(non-equity)")
            continue
        if present:
            meta[t] = dict(type=it, universe="REFRESH" if args.force else "EXISTS(skip)")
            if args.force: to_add.append(t)
            continue
        meta[t] = dict(type=it, universe="ADD")
        new_names.append(t); to_add.append(t)

    # PRICE SEED FIRST -- data before enrolment. Enrolling a ticker with no bars
    # trips the stale-panel guard on every run (learned the hard way 2026-08-15).
    seeded = {}
    if not args.no_seed:
        for t in to_add:
            if meta[t]["type"] in ("future", "crypto", "foreign"):
                continue
            seeded[t] = seed_prices(t, args.start, args.dry_run)
        if seeded:
            print("# price seed:")
            for t, r in seeded.items():
                print(f"#   {t:8s} {r}")
            print()

    enrolled = []
    if not args.dry_run:
        for t in new_names:
            append_meta(header, tcol, t, args.sector, args.cohort)
            if not args.no_runner and append_runner(t):
                enrolled.append(t)
            if args.watchlist: append_watchlist(t)
    # already-in-CSV names may still be missing from the runner file
    if not args.dry_run and not args.no_runner:
        for t in to_add:
            if t not in new_names and append_runner(t):
                enrolled.append(t)
    if enrolled:
        print(f"# enrolled in tickers.txt (will generate predictions): {', '.join(enrolled)}")
    elif not args.dry_run and args.no_runner:
        print("# --no-runner: NOT enrolled in tickers.txt -> no predictions will be generated")

    status = {t: {} for t in tickers}
    ran = []
    for bf in sorted(BACKFILLS, key=lambda b: (not b.get("time_critical"),)):
        k = bf["key"]
        if (only and k not in only) or k in skip: continue
        subset = [t for t in to_add if meta[t]["type"] in APPLIES[k]]
        if not subset:
            for t in tickers: status[t].setdefault(k, "N/A")
            continue
        cmd, res = run_backfill(bf, subset, args)
        ran.append((k, cmd))
        for t in tickers: status[t][k] = res.get(t, "N/A")

    if ran:
        print("# commands:")
        for k, c in ran: print(f"#   [{k}] {c}")
        print()

    cols = ["ticker", "type", "universe"] + [b["key"] for b in BACKFILLS]
    def cell(t, c):
        if c == "ticker": return t
        if c in ("type", "universe"): return meta[t][c]
        return status[t].get(c, "N/A")
    w = {c: max(len(c), *(len(str(cell(t, c))) for t in tickers)) for c in cols}
    line = "  ".join(c.ljust(w[c]) for c in cols)
    print(line); print("-" * len(line))
    for t in tickers:
        print("  ".join(str(cell(t, c)).ljust(w[c]) for c in cols))

    unconf = [b["key"] for b in BACKFILLS if not b["confirmed"]]
    if unconf: print(f"\n# unconfirmed (won't auto-run): {', '.join(unconf)}")


if __name__ == "__main__":
    main()
