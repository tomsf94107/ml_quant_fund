#!/usr/bin/env python3
"""
patch_spot_session.py -- stop the spot anchor landing on a PRE-MARKET print.

TWO DEFECTS, ONE SYMPTOM (found 2026-08-21 via MP report)
  MP's report: "Monitor set spot to pre-market $56.55; overridden to $58.51
  close." $56.55 is MP's pre-market open.

  (1) /api/stock/{t}/ohlc/1d returns THREE rows PER DATE, one per session:
        {'date':'2025-08-21','market_time':'pr','close':'68.45'}   pre
        {'date':'2025-08-21','market_time':'po','close':'68.70'}   post
        {'date':'2025-08-21','market_time':'r' ,'close':'68.28'}   regular
      The sort keyed on date ALONE, so all three tie; Python's sort is STABLE,
      so reverse=True returns the FIRST in API order -- the PRE-MARKET row.
      The code comment at ~2334 already recorded this ("the OHLC fallback was
      worse -- reverse-stable sort landed on the PRE-MARKET row") but only the
      symptom was worked around, not the sort.

  (2) The clock guard re-anchors spot to the live quote on >1% divergence. That
      is right post-earnings (AMZN Jul-31: close 235.50 vs live 257.96) but
      wrong pre-market, where a thin print is LESS reliable than the settled
      close. /stock-state carries "market_time" and the guard never read it.

VOCABULARY: the two endpoints disagree -- /stock-state says "regular",
/ohlc/1d says "r". Both handled.

The gate NAMES ONLY WHAT IT BLOCKS (pre/closed). Anything unrecognised keeps
current behaviour, so an unseen spelling of "post" cannot silently disable the
post-earnings re-anchor this guard exists for.
"""
import argparse, os, shutil, sys
from datetime import date

ROOT = os.path.expanduser(os.environ.get("ML_QUANT_ROOT", "~/ML_Quant_Fund"))
TARGET = os.path.join(ROOT, "scripts", "monitor_ticker.py")

SORT_OLD = '''            spot_rows.sort(key=lambda r: (r.get("date") or r.get("market_time") or
                                          r.get("start_time") or ""), reverse=True)'''
SORT_NEW = '''            # /ohlc/1d returns THREE rows per date -- 'pr' (pre), 'r' (regular),
            # 'po' (post). Keying on date alone made all three TIE, and a stable
            # sort under reverse=True returns the first in API order: PRE-MARKET.
            # Rank the session so the REGULAR row wins its date. (2026-08-21)
            _SESSION_RANK = {"r": 0, "regular": 0, "po": 1, "post": 1,
                             "pr": 2, "pre": 2, "premarket": 2}
            spot_rows.sort(key=lambda r: (
                r.get("date") or r.get("start_time") or "",
                -_SESSION_RANK.get(str(r.get("market_time") or "r").lower(), 3),
            ), reverse=True)'''

CTX_OLD = '''    _quote_ctx = spot'''
CTX_NEW = '''    _quote_ctx = spot
    # Session of the live quote. /stock-state -> "regular"; /ohlc/1d -> "r".
    try:
        _mkt_sess = str((rt or {}).get("market_time") or "").lower()
    except Exception:
        _mkt_sess = ""'''

GUARD_OLD = '''    _SPOT_CLOCK_TOL = 0.01'''
GUARD_NEW = '''    _SPOT_CLOCK_TOL = 0.01
    # SESSION GATE (2026-08-21). Re-anchoring is correct POST-event, when the
    # close is stale by the whole gap. It is wrong PRE-MARKET, where a thin
    # print is less reliable than the settled close -- MP anchored to a
    # pre-market $56.55 over a settled $58.51 (3.4%). Require a much larger
    # divergence before a pre-market/closed quote may override the close.
    # Blocks by NAME so an unrecognised spelling of "post" cannot silently
    # disable the post-earnings re-anchor this guard exists for.
    if _mkt_sess.startswith("pr") or _mkt_sess in ("closed", "c", "clsd"):
        _SPOT_CLOCK_TOL = 0.05
        print(f"  [note] {ticker}: live quote is {_mkt_sess or 'off-session'} "
              f"-- re-anchor tolerance widened 1% -> 5% (settled close preferred)")'''


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--path")
    a = ap.parse_args()
    t = a.path or TARGET
    if not os.path.isfile(t):
        sys.exit(f"FATAL: {t} not found")
    s = open(t).read()
    if "_SESSION_RANK" in s and "_mkt_sess" in s:
        print("# already patched"); return 0
    for name, old in (("ohlc sort", SORT_OLD), ("quote ctx", CTX_OLD),
                      ("clock tol", GUARD_OLD)):
        if s.count(old) != 1:
            sys.exit(f"FATAL: {name} anchor matched {s.count(old)} times, expected 1")
    if a.dry_run:
        print(f"# DRY-RUN {t}\n#   1 rank /ohlc/1d sessions so REGULAR wins its date"
              f"\n#   2 capture market_time into _mkt_sess"
              f"\n#   3 widen re-anchor tolerance 1%->5% pre-market/closed")
        return 0
    shutil.copy2(t, f"{t}.bak.session.{date.today().isoformat()}")
    s = s.replace(SORT_OLD, SORT_NEW).replace(CTX_OLD, CTX_NEW).replace(GUARD_OLD, GUARD_NEW)
    open(t, "w").write(s)
    import py_compile
    try:
        py_compile.compile(t, doraise=True)
    except Exception as e:
        sys.exit(f"FATAL: does not compile: {e}")
    print(f"# patched {t}\n#   backup {t}.bak.session.{date.today().isoformat()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
