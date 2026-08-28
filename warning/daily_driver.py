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
import sqlite3
import sys
from datetime import date

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from warning_engine import SignalReading, EngineState, step  # noqa: E402
from builders import s1_term_spread as S1                    # noqa: E402
from builders import s2_credit as S2                         # noqa: E402
from builders import f2_vix_percentile as F2                 # noqa: E402
from builders import s4_funding as S4                        # noqa: E402
from builders import f3_vix_term_slope as F3                 # noqa: E402
from builders import l4_propagation as L4                    # noqa: E402

# The 15 shortlist signals and their layers, from signal_registry.csv.
# "L1+trigger" (S10) scores as L1; the trigger role is a separate concern.
ROSTER = {
    "S11": "L1", "S13": "L1", "S15": "L1", "S10": "L1",
    "S1": "L2", "S2": "L2", "S3": "L2", "S5": "L2", "S6": "L2",
    "S7": "L2", "S8": "L2", "S9": "L2", "S12": "L2",
    "S4": "L3", "S14": "L3",
}
BUILT = {"S1": S1, "S2": S2, "S4": S4}       # composite inputs implemented so far
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


def persist(con, asof, res, details, dash):
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
       res.action.get("carry_bps_mo"), None, 0, "unversioned"))

    for sid, r in list(details.items()) + list(dash.items()):
        d = r.get("detail", {})
        con.execute("""INSERT OR REPLACE INTO signal_values
          (asof_date,signal_id,layer,raw_value,state,sub_score,stale,
           persistence_days,effective_state,source_asof,registry_version)
          VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
          (str(asof), sid, r.get("layer", ROSTER.get(sid, "L1")), r.get("raw_value"),
           r["state"], res.contributions.get(sid), int(bool(r.get("stale", True))),
           r.get("persistence_days", 1), None, r.get("source_asof"), "unversioned"))

    for a in res.alerts:
        con.execute("""INSERT INTO alerts (asof_date,alert_type,from_state,to_state,reason)
          VALUES (?,?,?,?,?)""", (str(asof), a[0], a[1], a[2], a[3]))
    con.commit()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="warning.db")
    ap.add_argument("--asof", default=date.today().isoformat())
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    con = sqlite3.connect(args.db)
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
