#!/usr/bin/env python3
"""cews_engine_json.py — read warning.db, write exports/engine-data.json for the Crash Odds Desk page.

Run from the repo root:  python exports/cews_engine_json.py [--db warning.db] [--out exports/engine-data.json]
Read-only on the DB. No network. Idempotent.

The dashboard renders the engine panel from this JSON verbatim; the scheduled refresh task only swaps the
block into the published page. Edit BLOCKERS / NAMES below when the situation changes — this file is the
editorial source for those sections.
"""
import os
import argparse, collections, hashlib, json, os, re, sqlite3, sys
from datetime import datetime, timezone, date

# Repo root, so utils/ is importable when this runs from cron with exports/ as
# sys.path[0]. Same insert daily_driver.py needs for the same reason.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.market_calendar import is_trading_day

GATE = 0.70  # STALE_COVERAGE_MIN in warning/warning_engine.py — keep in sync

NAMES = {"S2": "credit", "S3": "SLOOS", "S5": "breadth", "S6": "concentration", "S7": "defensive rotation",
         "S8": "epicenter fracture", "S9": "short interest", "S10": "margin-debt blowoff", "S11": "issuance (Ritter)",
         "S12": "EDGAR Form-4", "S13": "valuation gate (CAPE)", "S14": "vol structure", "S15": "credit-boom R-zone",
         "F2": "VIX percentile", "F3": "VIX term slope",
         "L4A": "funding seizure", "L4B": "spread blowout velocity", "L4C": "correlation spike",
         "L4D": "forced deleveraging", "L4E": "hedging feedback"}

# Editorial: open items. The composite no longer freezes -- all four layers
# passed their gates on 2026-09-08 -- so this is a list of what is not built
# and what is imprecise, not a list of blocks. Update as items close.
# effect_class: crit | warn | neutral | good
BLOCKERS = [
    {"item": "S15 not built", "evidence": "Inputs all exist and are reachable (BCNSDODNS, CMDEBT, GDP from 1945; Shiller real prices 1871; CSUSHPINSA 1975). Z.1 is revisable, so an honest ingest returns NA before ALFRED vintage coverage, and CSUSHPINSA's earliest vintage is 2014-11-25. The registry's verdicts are 2000 and 2008.",
     "effect": "Deliberate", "effect_class": "neutral",
     "clears": "D30. Not a defect and not scheduled: a PIT-honest S15 has ZERO testable episodes. L1 already passes at 75%, so 100% changes no gate, no band and no action. Reopen if ALFRED extends earlier or a vintage archive is acquired"},
    {"item": "S12 not built", "evidence": "EDGAR Form-4 insider breadth. The axis was closed twice -- CLOSED_AXIS_insider_selling_2026-09-03 and CLOSED_AXIS_insider_breadth_2026-09-07, the latter after seven constructions over 365,910 filings with nothing clearing NW t = 3.0.",
     "effect": "Deliberate", "effect_class": "neutral",
     "clears": "Not scheduled. Insider features stay wired inside models; no standalone insider signal exists. A future attempt should test INTERACTIONS, not an eighth univariate construction"},
    {"item": "L4E not built", "evidence": "Hedging feedback needs F3 (VIX curve, exists) and F9 (negative-gamma estimate, does not). F9 needs options open interest and dealer positioning with no feed in the stack.",
     "effect": "Data gap", "effect_class": "neutral",
     "clears": "A positioning data source. L4 passes at 80% without it (4 of 5)"},
    {"item": "Staleness semantics", "evidence": "pit.staleness_days measures asof - obs_date, i.e. time since the observation period ended, not since the data became knowable. For a lagged series those differ by the publication lag. S9 was discarded at 25 days on data that was 13 days old.",
     "effect": "Defect", "effect_class": "warn",
     "clears": "D31. S9 patched in isolation (commit cf89c73c) and is back in L2 at 8/9. The global fix -- measure from pub_date when series_meta.derivable_pub_date is True, from obs_date when not, since revisable series carry the pull date -- moves every signal at once and needs a limits review at a registry version bump. <strong>Watch S10: 39 days by obs against a 45 limit, 18 by pub. One late FINRA release drops L1 to 50%</strong>"},
    {"item": "Persistence left-censoring", "evidence": "S1 (term-spread inversion) and S6 (concentration) both read R. Both carry persistence_days = 21 and only seven asof_dates exist, so both contribute G and L2 scores 0.000.",
     "effect": "Structural", "effect_class": "warn",
     "clears": "D26. Roughly 2026-09-28, when the window fills. L2's 0.000 means NOT YET COUNTABLE, not nothing happening -- and both become eligible on the same day, so L2 may move sharply rather than gradually"},
    {"item": "F3 stale fallback", "evidence": "F3 has a primary (VIX3M/VIX) to fallback (CFE VX futures, last row 2018-02-23) chain, and the fallback has no freshness check. 2-4 Sep it served 2018 data.",
     "effect": "Defect", "effect_class": "warn",
     "clears": "A staleness check on every leg of the chain, not just the primary -- s14_vol_structure.py:146-154 is the reference implementation. Dashboard tier, so coverage is unaffected"},
    {"item": "Derived persistence", "evidence": "The engine accumulates persistence in a mutable counter in schema_meta rather than deriving it from signal_values. Rebuilt by hand after every re-step.",
     "effect": "Proposed", "effect_class": "neutral",
     "clears": "D25 Phase 2, unratified. It alters hysteresis and therefore crisis behaviour, which is the objection D10 raised. Until then rebuild_persistence.py must be re-run after any re-step"},
]

# Holidays come from utils.market_calendar, which computes NYSE rules rather
# than listing dates. The literal set that was here covered 2026 only and was
# labelled best effort -- it would have gone silently wrong on 2027-01-01, the
# exact failure mode that module was written to kill after 785 predictions and
# 2,324 outcomes were logged against Juneteenth and July 4. It also duplicated
# the calendar the driver now uses for asof_date, so the dashboard and the
# engine could have disagreed about which days were sessions.


def sig_key(s):
    m = re.match(r"([A-Z]+)(\d*)([A-Z]?)", s)
    return ({"S": 0, "F": 1, "L": 2}[m.group(1)[0]], int(m.group(2) or 0), m.group(3))


def read_note(latest_by_sig, composite=None):
    reds = sorted([s for s, r in latest_by_sig.items() if r["state"] == "R"], key=sig_key)
    ambers = sorted([s for s, r in latest_by_sig.items() if r["state"] == "Y"], key=sig_key)
    nm = lambda s: s + (" (" + NAMES[s] + ")" if s in NAMES else "")
    # Heading follows the state rather than asserting one. "Read despite the
    # freeze" was hardcoded from the period when every run printed
    # INSUFFICIENT_DATA. The composite has computed since 2026-09-08 and the
    # line survived into the PDF as a false claim.
    parts = ["<strong>Under the composite.</strong>" if composite is not None
             else "<strong>Read despite the freeze.</strong>"]
    if reds:
        parts.append("Red on the latest run: " + ", ".join("<strong>" + nm(s) + "</strong>" for s in reds) + ".")
    if ambers:
        parts.append("Amber: " + ", ".join(nm(s) for s in ambers) + ".")
    eff = sorted([s for s, r in latest_by_sig.items() if r.get("effective_state") not in (None, "", "G")], key=sig_key)
    parts.append(("Effective (persistence met): " + ", ".join(eff) + ".") if eff else
                 "Nothing has reached its persistence requirement, so no state is effective and the layer scores are 0.")
    # The thing a reader most needs and the signal table does not show:
    # which signals are red but not yet countable.
    pend = sorted([s for s, r in latest_by_sig.items()
                   if r["state"] == "R"
                   and r.get("effective_state") in (None, "", "G")], key=sig_key)
    if pend:
        many = len(pend) > 1
        parts.append("<strong>" + ", ".join(nm(s) for s in pend) + "</strong> "
                     + ("read R but are" if many else "reads R but is")
                     + " not yet countable -- the persistence window is not "
                       "full, so " + ("they contribute" if many
                                      else "it contributes") + " G. A layer "
                       "score of 0.000 here means NOT YET COUNTABLE, not "
                       "nothing happening.")
    parts.append("Per the brief, no single signal is acted on.")
    return " ".join(parts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=os.environ.get("WARNING_DB",
                                               "warning.db"))
    ap.add_argument("--out", default="exports/engine-data.json")
    ap.add_argument("--registry", default="warning/signal_registry.csv")
    a = ap.parse_args()

    con = sqlite3.connect(f"file:{a.db}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    q = lambda sql: [dict(r) for r in con.execute(sql)]
    comp = q("select * from composite_scores order by asof_date")
    if not comp:
        sys.exit("composite_scores is empty")
    latest = comp[-1]
    sv = q("select * from signal_values order by asof_date, signal_id")
    alerts = q("select * from alerts")
    dates = sorted({r["asof_date"] for r in sv})
    idx = {(r["signal_id"], r["asof_date"]): r for r in sv}
    latest_by_sig = {r["signal_id"]: r for r in sv if r["asof_date"] == dates[-1]}
    layers = ["L1", "L2", "L3", "L4"]
    sigs_by_layer = {L: sorted([s for s, r in latest_by_sig.items() if r["layer"] == L], key=sig_key) for L in layers}
    alerts_by_day = collections.Counter(x["asof_date"] for x in alerts)
    kinds = collections.Counter(x["alert_type"] for x in alerts)

    signals = []
    for L in layers:
        for s in sigs_by_layer[L]:
            lr = latest_by_sig[s]
            cells = []
            for dt in dates:
                r = idx.get((s, dt))
                cells.append(None if not r else {"s": r["state"] or "NA", "stale": int(r["stale"]), "raw": r["raw_value"], "src": r["source_asof"]})
            signals.append({"id": s, "layer": L, "name": NAMES.get(s, ""), "tier": "dashboard" if s.startswith("F") else "",
                            "cells": cells, "raw": lr["raw_value"], "src": lr["source_asof"],
                            "persist": lr["persistence_days"], "stale": int(lr["stale"]), "state": lr["state"]})
    lay = []
    for L in layers:
        cov = latest[f"{L.lower()}_cov"] or 0.0
        n_all = len([s for s in sigs_by_layer[L] if not s.startswith("F")])
        lay.append({"id": L, "cov": cov, "n_live": round(cov * n_all), "n_all": n_all,
                    "score": latest[f"{L.lower()}_score"], "hist": [c[f"{L.lower()}_cov"] or 0.0 for c in comp]})
    runlog = [{"asof": c["asof_date"], "band": c["band"], "na": json.loads(c["na_layers"] or "[]"),
               "cov": [c["l1_cov"] or 0, c["l2_cov"] or 0, c["l3_cov"] or 0, c["l4_cov"] or 0],
               "alerts": alerts_by_day.get(c["asof_date"], 0), "written": (c["created_at"] or "")[:16]} for c in comp]
    transitions = sum(1 for x in alerts if x["alert_type"] not in ("LAYER_NA", "INSUFFICIENT_DATA"))
    reg_hash = None
    if os.path.exists(a.registry):
        reg_hash = hashlib.sha256(open(a.registry, "rb").read()).hexdigest()[:12]

    out = {
        "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M"),
        "asof": latest["asof_date"], "run_utc": (latest["created_at"] or "")[:16],
        "runs": len(comp), "first_run": comp[0]["asof_date"],
        "band": latest["band"], "path": latest["path"], "composite": latest["composite"],
        "registry_version": latest["registry_version"] or "unversioned", "registry_sha256_12": reg_hash,
        "gate": GATE, "dates": dates, "layers": lay, "signals": signals, "runlog": runlog,
        "alerts_total": len(alerts), "alerts_layer_na": kinds.get("LAYER_NA", 0),
        "alerts_insufficient": kinds.get("INSUFFICIENT_DATA", 0), "alerts_transitions": transitions,
        "holiday_rows": [d for d in dates if not is_trading_day(d)],
        "blockers": BLOCKERS,
        "read_note": read_note(latest_by_sig, latest["composite"]),
        "runlog_note": "Run log is complete since 2026-09-08: every write carries a run_id "
                       "and the runs table records when each ran, under which code and registry, "
                       "and against which database (B2). Alerts are idempotent via a unique index "
                       "and INSERT OR REPLACE (B3), so re-stepping a date no longer accumulates "
                       "rows -- 28 Aug had held 30 from six dev re-runs. asof_date derives from "
                       "utils.market_calendar.last_completed_session() and the driver refuses a "
                       "future date, a non-trading date, an unfilled gap and a session whose "
                       "prices are not ingested (B8, D24). The Sunday and Labor Day rows were "
                       "deleted.",
        "actions": {"action_gross": latest["action_gross"], "action_hedge": latest["action_hedge"],
                    "action_carry_bps": latest["action_carry_bps"], "candidate_band": latest["candidate_band"],
                    "candidate_days": latest["candidate_days"], "do_nothing": latest["do_nothing"], "l4_override": latest["l4_override"]},
        "assumptions": "Gate = 0.70 read from warning_engine.py [confirmed]. Counting signals equals the engine's weight ratio: no builder "
                       "passes weight, so every reading uses the 1.0 default [confirmed 2026-09-08 by grep of warning/builders/]. "
                       "F-signals assumed uncounted from the ratios [unconfirmed]. Names from builder files, the driver log and "
                       "signal_registry.csv; S1, S4 unnamed.",
    }
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump(out, open(a.out, "w"), ensure_ascii=False, indent=1)
    print(f"wrote {a.out}: asof {out['asof']} run {out['run_utc']} band {out['band']} runs {out['runs']} signals {len(signals)}")


if __name__ == "__main__":
    main()
