#!/usr/bin/env python3
"""
daily_driver.py — one evaluation: builders -> engine -> composite_scores/alerts.

    python warning/daily_driver.py --db warning.db
    python warning/daily_driver.py --db warning.db --asof 2026-08-28
    python warning/daily_driver.py --db warning.db --dry-run

WHY THE FULL ROSTER IS EMITTED, INCLUDING UNBUILT SIGNALS
    warning_engine computes layer coverage from the readings it is GIVEN. Passing
    only the built signals would make every layer look 100% covered and produce a
    confident composite from 2 of 15 inputs. So the driver emits all 15 shortlist
    signals every day; unbuilt ones are state 'NA'. Coverage then falls below
    STALE_COVERAGE_MIN and the engine returns INSUFFICIENT_DATA, which is the
    honest answer until enough builders exist.

TWO-PASS, BUT NOT CIRCULAR
    The composite is the 15 SHORTLIST signals. F-features are tier 'dashboard' --
    report line 338: options data serve the Part VII dashboard, "forward only".
    So the engine runs first, then F-features are computed with the resulting L2
    layer score (F2's red condition is "<10th w/ L2>=0.5"). F-features are stored
    in signal_values for display and never feed the composite.

L4 HAS NO REGISTRY SIGNALS -- STRUCTURAL GAP, FLAGGED NOT PATCHED
    warning_engine assigns L4 weight 0.25 and gives it the crisis override (any
    L4 at 'B' -> CRISIS immediately, bypassing persistence). signal_registry.csv
    defines no L4 row; the engine's own tests invent L4A/L4B/L4C. Consequence:
    the funding-seizure override cannot fire against real data, and L4 is
    permanently NA. Since NA_LAYER_LIMIT is 1, a permanently-NA L4 consumes the
    entire allowance -- one more NA layer forces INSUFFICIENT_DATA forever.
    This needs a ruling before the system can ever produce a live composite.

ENGINE STATE PERSISTENCE
    EngineState (band, candidate, per-signal persistence counters) is carried in
    schema_meta under 'engine_state' as JSON. No schema change needed.
"""
import argparse
import json
import os
import hashlib
import sqlite3
import sys
from datetime import date, datetime
from zoneinfo import ZoneInfo

ET = ZoneInfo("America/New_York")

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
# Repo root as well. utils.market_calendar is the ONE holiday-aware US calendar
# in this repo -- rules-computed, not hardcoded, and validated against 2,549
# real raw_bars sessions with zero disagreements in either direction. Cron runs
# this file directly, so sys.path[0] is warning/ and utils/ is otherwise unseen.
sys.path.insert(0, os.path.dirname(_HERE))

from utils.market_calendar import (last_completed_session,   # noqa: E402
                                   is_trading_day)
from warning_engine import SignalReading, EngineState, step  # noqa: E402
from builders import s1_term_spread as S1                    # noqa: E402
from builders import s2_credit as S2                         # noqa: E402
from builders import f2_vix_percentile as F2                 # noqa: E402
from builders import s4_funding as S4                        # noqa: E402
from builders import f3_vix_term_slope as F3                 # noqa: E402
from builders import s14_vol_structure as S14                # noqa: E402
from builders import s7_defensive_rotation as S7            # noqa: E402
from builders import s8_epicenter_fracture as S8            # noqa: E402
from builders import s9_short_interest as S9                # noqa: E402
from builders import s5_breadth as S5                       # noqa: E402
from builders import s3_sloos as S3                         # noqa: E402
from builders import s11_issuance as S11                    # noqa: E402
from builders import s13_valuation as S13                   # noqa: E402
from builders import s10_margin_debt as S10                 # noqa: E402
from builders import s6_concentration as S6                 # noqa: E402
from builders import l4_propagation as L4                    # noqa: E402

# The 15 shortlist signals and their layers, from signal_registry.csv.
# "L1+trigger" (S10) scores as L1; the trigger role is a separate concern.
ROSTER = {
    "S11": "L1", "S13": "L1", "S15": "L1", "S10": "L1",
    "S1": "L2", "S2": "L2", "S3": "L2", "S5": "L2", "S6": "L2",
    "S7": "L2", "S8": "L2", "S9": "L2", "S12": "L2",
    "S4": "L3", "S14": "L3",
}
BUILT = {"S1": S1, "S2": S2, "S4": S4, "S14": S14, "S7": S7, "S8": S8,
         "S9": S9, "S5": S5, "S3": S3, "S11": S11, "S6": S6,
         "S13": S13, "S10": S10}
DASHBOARD = {"F2": F2, "F3": F3}             # tier 'dashboard', computed after


def load_state(con) -> EngineState:
    row = con.execute("SELECT value FROM schema_meta WHERE key='engine_state'").fetchone()
    if not row:
        return EngineState()
    d = json.loads(row[0])
    st = EngineState(band=d.get("band", "NORMAL"),
                     candidate_band=d.get("candidate_band"),
                     candidate_days=d.get("candidate_days", 0))
    st.persistence = {k: tuple(v) for k, v in d.get("persistence", {}).items()}
    return st


def save_state(con, st: EngineState):
    con.execute("INSERT OR REPLACE INTO schema_meta (key, value) VALUES ('engine_state', ?)",
                (json.dumps({"band": st.band, "candidate_band": st.candidate_band,
                             "candidate_days": st.candidate_days,
                             "persistence": {k: list(v) for k, v in st.persistence.items()}}),))
    con.commit()          # without this the write is discarded on close, and the
                          # engine restarts from NORMAL with empty persistence
                          # counters every single day (found 2026-08-28)


def build_readings(con, asof):
    """One reading per shortlist signal, plus the five L4 propagation conditions.
    Unbuilt -> NA so layer coverage stays honest. Returns (readings, details)."""
    readings, details = [], {}
    for sid, r in L4.compute_all(con, asof).items():
        readings.append(L4.to_reading(r))
        details[sid] = r
    for sid, layer in ROSTER.items():
        mod = BUILT.get(sid)
        if mod is None:
            readings.append(SignalReading(sid, layer, "NA", stale=True))
            details[sid] = {"state": "NA", "detail": {"reason": "builder not implemented"}}
            continue
        r = mod.compute(con, asof)
        readings.append(mod.to_reading(r))
        details[sid] = r
    return readings, details


def registry_version():
    """B4 interim. Every row carried the literal "unversioned", so no reading
    could be tied to the thresholds that produced it. The signal_registry table
    (B4/B9) remains the full fix; this closes the unenforceable half."""
    p = os.path.join(_HERE, "signal_registry.csv")
    if not os.path.exists(p):
        raise SystemExit(f"signal_registry.csv missing at {p}; refusing to run.")
    with open(p, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()[:12]


def missing_sessions(con, target: str) -> list[str]:
    """Trading sessions between the last stored asof_date and `target` with no row.

    A missing session is an OPERATIONAL failure -- a cron that did not fire, a
    machine asleep -- not a market event. The driver refuses rather than
    silently stepping over it, matching the FEED_STALE exit and
    market_calendar's refusal to guess past HORIZON_YEAR. Auto-backfill would
    be defensible, since pit.series_asof is genuinely PIT-honest and a re-step
    sees only what was visible then, but it would make the upstream failure
    invisible, and any cap on how far back to go would be an invented number.

    For whoever reads this during an outage: NOT re-stepping is also a choice.
    apply_persistence counts consecutive OBSERVATIONS, not calendar sessions,
    so a skipped day does not reset a run -- it is never counted. A signal at
    13 days that misses a session resumes at 14, one session later in
    wall-clock time than it should be.
    """
    row = con.execute("SELECT MAX(asof_date) FROM signal_values").fetchone()
    last = row[0] if row and row[0] else None
    if not last or last >= target:
        return []
    d = date.fromisoformat(last)
    end = date.fromisoformat(target)
    out = []
    while d < end:
        d = date.fromordinal(d.toordinal() + 1)
        if d < end and is_trading_day(d):
            out.append(d.isoformat())
    return out


def persist(con, asof, res, details, dash):
    rv = registry_version()
    con.execute("""INSERT OR REPLACE INTO composite_scores
      (asof_date,composite,band,path,do_nothing,l4_override,insufficient_data,
       l1_score,l2_score,l3_score,l4_score,l1_cov,l2_cov,l3_cov,l4_cov,na_layers,
       action_gross,action_hedge,action_carry_bps,candidate_band,candidate_days,
       registry_version)
      VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
      (str(asof), res.composite, res.band, res.path, int(res.do_nothing),
       int(res.l4_override), int(res.composite is None),
       res.layer_scores.get("L1"), res.layer_scores.get("L2"),
       res.layer_scores.get("L3"), res.layer_scores.get("L4"),
       res.layer_coverage.get("L1"), res.layer_coverage.get("L2"),
       res.layer_coverage.get("L3"), res.layer_coverage.get("L4"),
       json.dumps([L for L, s in res.layer_scores.items() if s is None]),
       res.action.get("gross"), res.action.get("hedge"),
       res.action.get("carry_bps_mo"), None, 0, rv))

    for sid, r in list(details.items()) + list(dash.items()):
        d = r.get("detail", {})
        con.execute("""INSERT OR REPLACE INTO signal_values
          (asof_date,signal_id,layer,raw_value,state,sub_score,stale,
           persistence_days,effective_state,source_asof,registry_version)
          VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
          (str(asof), sid, r.get("layer", ROSTER.get(sid, "L1")), r.get("raw_value"),
           r["state"], res.contributions.get(sid), int(bool(r.get("stale", True))),
           r.get("persistence_days", 1), None, r.get("source_asof"), rv))

    for a in res.alerts:
        # B3. Alerts were the only non-idempotent table: every other write uses
        # INSERT OR REPLACE, so re-stepping a date collapsed its rows while
        # alerts accumulated. 2026-08-28 reached 33 rows across six runs, and
        # 2026-09-04 held two contradictory sets -- one asserting L2 at 67% and
        # L3 at 50%, the other showing both passing, with only the second
        # surviving in signal_values. An alert log that disagrees with the
        # signal log is worse than no alert log.
        #
        # The unique index uses COALESCE because LAYER_NA carries to_state NULL
        # and SQLite treats NULLs as distinct in a unique index, so a plain
        # index would never collide on exactly the rows that duplicated most.
        # `reason` is deliberately outside the key: it varies between runs of
        # the same alert, and REPLACE keeping the latest matches signal_values.
        con.execute("""INSERT OR REPLACE INTO alerts (asof_date,alert_type,from_state,to_state,reason)
          VALUES (?,?,?,?,?)""", (str(asof), a[0], a[1], a[2], a[3]))
    con.commit()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=os.environ.get("WARNING_DB",
                                               "warning.db"))
    # ET, not local. date.today() returns the VN date, a day ahead of ET, and
    # the cron fires 06:00 VN = 19:00 ET the PREVIOUS day -- so an unqualified
    # today() would stamp Monday's US session with Tuesday's date. Every other
    # series in warning.db is ET-dated (data_vintages.obs_date, SPY_CLOSE,
    # pit.series_asof compares date strings directly), so the composite row
    # would carry a date one ahead of its own inputs.
    #
    # Same defect class as the uw_archive collision fixed 2026-08-30. ZoneInfo
    # rather than a fixed offset because ET is UTC-4 or UTC-5 depending on DST.
    #
    # B8 amendment 2026-09-08: the ET fix above was necessary and not
    # sufficient. Right timezone, wrong calendar -- an ET date still labels
    # weekends (a Sunday 2026-08-30 row exists), holidays, and the current day
    # mid-session (six 2026-08-28 rows written 09:31-13:47 ET). The default is
    # now the last COMPLETED session, which also carries a 17:00 ET guard.
    ap.add_argument("--asof", default=None,
                    help="evaluation date; defaults to the last completed "
                         "US session. Pass explicitly to re-step a past date.")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--allow-gap", action="store_true",
                    help="proceed despite missing sessions rather than exiting 3")
    args = ap.parse_args()

    con = sqlite3.connect(args.db)

    explicit_asof = args.asof is not None
    last_session = last_completed_session().isoformat()
    if not explicit_asof:
        args.asof = last_session

    # An explicit --asof exists to RE-STEP A PAST DATE. It must never reach
    # forward: on 2026-09-08 a --asof of that date wrote a row for a session
    # that had not begun in New York, since VN early morning is the previous
    # ET afternoon. The FEED_STALE check below is deliberately bypassed for
    # explicit dates, so without this there is no upper bound at all.
    if args.asof > last_session:
        sys.exit(f"--asof {args.asof} is beyond the last completed session "
                 f"({last_session}). Sessions are labelled after they close.")
    if not is_trading_day(args.asof):
        sys.exit(f"--asof {args.asof} is not a US trading session.")

    # The calendar says which session to LABEL; SPY_CLOSE says which session we
    # actually HOLD, PIT-honest via pub_date. Disagreement means a feed is
    # behind -- fail loudly. A driver whose asof silently stops advancing when
    # an upstream ingest dies is the same class of defect as the mislabelled
    # rows this replaces. An explicit --asof bypasses this, for deliberate
    # re-steps from archived vintages.
    row = con.execute(
        "SELECT MAX(obs_date) FROM data_vintages "
        "WHERE series_id='SPY_CLOSE' AND pub_date <= ?",
        (datetime.now(ET).date().isoformat(),)).fetchone()
    have = row[0] if row else None
    if not explicit_asof and (have is None or have < args.asof):
        print(f"FEED_STALE: calendar session {args.asof}, SPY_CLOSE available "
              f"{have}. Refusing to write. Run the ingests, or pass --asof to "
              f"re-step a past date deliberately.", file=sys.stderr)
        sys.exit(2)

    gaps = missing_sessions(con, args.asof)
    if gaps and not explicit_asof and not args.dry_run:
        print(f"GAP: {len(gaps)} trading session(s) between the last stored "
              f"asof_date and {args.asof} have no row:", file=sys.stderr)
        for g in gaps:
            print(f"  python warning/daily_driver.py --db {args.db} --asof {g}",
                  file=sys.stderr)
        print("Run those in order, then re-run this. Or pass --allow-gap to "
              "proceed and leave them unfilled -- persistence runs continue "
              "across a gap, they do not reset.", file=sys.stderr)
        if not args.allow_gap:
            sys.exit(3)
    elif gaps:
        print(f"GAP: {len(gaps)} unfilled session(s): {gaps}")

    st = load_state(con)
    readings, details = build_readings(con, args.asof)
    res = step(str(args.asof), readings, st)

    # dashboard features, downstream of the engine
    dash = {}
    for sid, mod in DASHBOARD.items():
        # F2's red condition needs the L2 layer score; F3 does not take one.
        try:
            dash[sid] = mod.compute(con, args.asof,
                                    l2_score=res.layer_scores.get("L2"))
        except TypeError:
            dash[sid] = mod.compute(con, args.asof)

    built = [s for s in ROSTER if s in BUILT]
    print(f"as of {args.asof}   composite inputs built: {len(built)}/{len(ROSTER)} "
          f"({', '.join(built)})")
    print(f"BAND {res.band}   composite "
          f"{'n/a' if res.composite is None else format(res.composite, '.1f')}"
          f"   path {res.path}   gross {res.action.get('gross')}"
          f"   hedge {res.action.get('hedge')}\n")

    print(f"  {'layer':<7}{'score':>8}{'coverage':>10}   signals")
    for L in ("L1", "L2", "L3", "L4"):
        s, c = res.layer_scores.get(L), res.layer_coverage.get(L)
        ids = ([k for k, v in ROSTER.items() if v == L]
               or (list(L4.compute_all.__doc__ and ["L4A", "L4B", "L4C", "L4D", "L4E"])
                   if L == "L4" else ["-- none --"]))
        print(f"  {L:<7}{'NA' if s is None else format(s, '.3f'):>8}{c:>9.0%}   "
              f"{','.join(ids)}")

    print("\n  L4 propagation conditions (derived; report line 601):")
    for sid in ("L4A", "L4B", "L4C", "L4D", "L4E"):
        r = details[sid]
        d = r["detail"]
        extra = d.get("condition", d.get("reason", ""))
        print(f"    {sid:<5} {r['state']:<3}  {extra[:70]}")

    print("\n  composite inputs:")
    for sid in ROSTER:
        r = details[sid]
        note = "" if sid in BUILT else "   (builder not implemented)"
        print(f"    {sid:<5} {r['state']:<3}{note}")

    print("\n  dashboard (not in the composite):")
    for sid, r in dash.items():
        d = r["detail"]
        if "reason" in d:
            print(f"    {sid:<5} {r['state']:<3}  {d['reason'][:70]}")
        elif sid == "F2":
            print(f"    {sid:<5} {r['state']:<3}  VIX {d.get('vix')} "
                  f"pct {d.get('percentile_504d')}  L2 passed = {d.get('l2_score')}")
        else:
            print(f"    {sid:<5} {r['state']:<3}  VIX {d.get('vix')} "
                  f"VIX3M {d.get('vix3m')}  slope {d.get('slope_pct')}%  "
                  f"inverted {d.get('inverted_run_days')}d")

    if res.alerts:
        print("\n  alerts:")
        for a in res.alerts:
            print(f"    {a[0]}  {a[1]} -> {a[2]}  ({a[3]})")

    if args.dry_run:
        print("\nDRY RUN -- nothing written.")
    else:
        persist(con, args.asof, res, details, dash)
        save_state(con, st)
        print(f"\nwrote composite_scores/signal_values/alerts for {args.asof}")
    con.close()


if __name__ == "__main__":
    main()
