#!/usr/bin/env python3
"""
uw_probe.py — READ-ONLY probe of Unusual Whales endpoints. Resolves OPEN ITEM 2
(which endpoints your plan actually exposes) empirically, before uw_archiver.py
is scheduled against a guessed list.

WRITES NOTHING. No database, no files. One small GET per endpoint, rate-limited.

WHY A CONTROL GROUP
  If auth or the base path is wrong, EVERY endpoint fails and a naive probe would
  report "no entitlement" for endpoints you actually own. So two endpoints proven
  to work in this repo's existing code (monitor/uw_client callers) are probed
  FIRST as controls. If the controls fail, the probe aborts and tells you the
  problem is auth/network -- not entitlement. Nothing is pruned on a bad control.

USAGE
    cd ~/Desktop/ML_Quant_Fund
    set -a && . ./.env && set +a
    python scripts/uw_probe.py

    python scripts/uw_probe.py --json probe_result.json     # machine-readable

VERDICTS
    OK              200, JSON parsed, non-empty            -> keep
    OK_EMPTY        200 but no rows (may be time-of-day)   -> keep, re-probe later
    NO_ENTITLEMENT  403                                    -> prune
    NOT_FOUND       404 (wrong path or not on this plan)   -> prune
    AUTH            401                                    -> fix token, do not prune
    RATE_LIMIT      429 after retries                      -> re-probe later
    ERROR           anything else
"""

import argparse, json, os, sys, time, urllib.request, urllib.error

BASE = "https://api.unusualwhales.com"
TOKEN = os.environ.get("UW_API_KEY") or os.environ.get("UW_TOKEN")

# Proven in this repo's working code -- used as controls, never pruned.
CONTROLS = [
    ("/api/market/market-tide", {}),
    ("/api/option-trades/flow-alerts", {"limit": 1}),
]

# Unproven guesses currently sitting in uw_archiver.ENDPOINTS, plus the
# per-ticker templates instantiated on SPY. These are what the probe decides.
CANDIDATES = [
    ("/api/market/total-options-volume", {}),
    ("/api/darkpool/recent", {"limit": 1}),
    ("/api/stock/SPY/option-chains", {}),
    ("/api/stock/SPY/greek-exposure", {}),
    ("/api/stock/SPY/volatility/term-structure", {}),
    ("/api/stock/QQQ/option-chains", {}),
    ("/api/stock/IWM/option-chains", {}),
    ("/api/stock/SPY/volatility/realized", {}),
    ("/api/stock/SPY/greeks", {}),
    ("/api/stock/SPY/options-volume", {}),
    # proven-by-analogy but with a ticker path -- confirm the shape the archiver needs
    ("/api/darkpool/SPY", {"limit": 1}),
]

SLEEP = 0.7
MAX_RETRIES = 3


def probe(endpoint, params):
    qs = "&".join(f"{k}={v}" for k, v in sorted(params.items()))
    url = f"{BASE}{endpoint}" + (f"?{qs}" if qs else "")
    req = urllib.request.Request(url, headers={
        "Authorization": f"Bearer {TOKEN}", "Accept": "application/json"})
    backoff = 2.0
    for attempt in range(MAX_RETRIES):
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                raw = r.read().decode()
            try:
                doc = json.loads(raw)
            except json.JSONDecodeError:
                return dict(status=r.status, verdict="ERROR", detail="non-JSON body",
                            keys=[], nbytes=len(raw))
            keys = list(doc.keys())[:6] if isinstance(doc, dict) else ["<list>"]
            body = doc.get("data", doc) if isinstance(doc, dict) else doc
            n = len(body) if isinstance(body, (list, dict)) else 0
            return dict(status=200, verdict="OK" if n else "OK_EMPTY",
                        detail=f"{n} items", keys=keys, nbytes=len(raw))
        except urllib.error.HTTPError as e:
            if e.code == 429 and attempt < MAX_RETRIES - 1:
                time.sleep(backoff); backoff *= 2; continue
            v = {401: "AUTH", 403: "NO_ENTITLEMENT",
                 404: "NOT_FOUND", 429: "RATE_LIMIT"}.get(e.code, "ERROR")
            return dict(status=e.code, verdict=v, detail=e.reason, keys=[], nbytes=0)
        except Exception as e:
            return dict(status=None, verdict="ERROR", detail=str(e)[:80],
                        keys=[], nbytes=0)
    return dict(status=None, verdict="ERROR", detail="unreachable", keys=[], nbytes=0)


def row(endpoint, r):
    return (f"  {r['verdict']:<15} {str(r['status'] or '-'):>4}  {endpoint:<48} "
            f"{r['detail'][:28]:<28} {','.join(map(str, r['keys']))[:34]}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default=None, help="write machine-readable result here")
    args = ap.parse_args()
    if not TOKEN:
        sys.exit("UW_API_KEY (or UW_TOKEN) not set -- run: set -a && . ./.env && set +a")

    results = {}

    print("== CONTROLS (proven in existing repo code) ==")
    ctrl_ok = 0
    for ep, p in CONTROLS:
        r = probe(ep, p); results[ep] = r
        print(row(ep, r))
        if r["verdict"] in ("OK", "OK_EMPTY"):
            ctrl_ok += 1
        time.sleep(SLEEP)

    if ctrl_ok == 0:
        print("\nABORT: every control failed. This is an AUTH / BASE-PATH / NETWORK "
              "problem, not an entitlement problem. Nothing pruned, nothing concluded.")
        print("Check: token value in .env, and that BASE is still", BASE)
        sys.exit(2)
    print(f"  -> {ctrl_ok}/{len(CONTROLS)} controls OK; entitlement verdicts below are meaningful.\n")

    print("== CANDIDATES (unproven guesses in uw_archiver.ENDPOINTS) ==")
    for ep, p in CANDIDATES:
        r = probe(ep, p); results[ep] = r
        print(row(ep, r))
        time.sleep(SLEEP)

    keep = [ep for ep, r in results.items() if r["verdict"] in ("OK", "OK_EMPTY")]
    drop = [(ep, r["verdict"]) for ep, r in results.items()
            if r["verdict"] in ("NO_ENTITLEMENT", "NOT_FOUND")]
    unclear = [(ep, r["verdict"]) for ep, r in results.items()
               if r["verdict"] in ("RATE_LIMIT", "ERROR", "AUTH")]

    print(f"\n== SUMMARY ==\n  keep {len(keep)}  drop {len(drop)}  unclear {len(unclear)}")
    if drop:
        print("  DROP:", ", ".join(f"{e} ({v})" for e, v in drop))
    if unclear:
        print("  UNCLEAR (re-probe; do NOT prune on this):",
              ", ".join(f"{e} ({v})" for e, v in unclear))

    print("\n== PASTE INTO uw_archiver.py (verified " +
          time.strftime("%Y-%m-%d") + ") ==")
    print("ENDPOINTS = [")
    for ep in keep:
        p = dict(CONTROLS + CANDIDATES).get(ep, {})
        print(f'    ("{ep}", {p!r}),')
    print("]")
    if unclear:
        print("# UNRESOLVED, left out pending re-probe:",
              ", ".join(e for e, _ in unclear))

    if args.json:
        with open(args.json, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
