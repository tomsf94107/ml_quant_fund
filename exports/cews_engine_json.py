#!/usr/bin/env python3
"""cews_engine_json.py — read warning.db, write exports/engine-data.json for the Crash Odds Desk page.

Run from the repo root:  python exports/cews_engine_json.py [--db warning.db] [--out exports/engine-data.json]
Read-only on the DB. No network. Idempotent.

The dashboard renders the engine panel from this JSON verbatim; the scheduled refresh task only swaps the
block into the published page. Edit BLOCKERS / NAMES below when the situation changes — this file is the
editorial source for those sections.
"""
import argparse, collections, hashlib, json, os, re, sqlite3, sys
from datetime import datetime, timezone

GATE = 0.70  # STALE_COVERAGE_MIN in warning/warning_engine.py — keep in sync

NAMES = {"S2": "credit", "S3": "SLOOS", "S5": "breadth", "S6": "concentration", "S7": "defensive rotation",
         "S8": "epicenter fracture", "S9": "short interest", "S10": "margin-debt blowoff", "S11": "issuance (Ritter)",
         "S12": "EDGAR Form-4", "S13": "valuation gate (CAPE)", "S14": "vol structure", "S15": "credit-boom R-zone",
         "F2": "VIX percentile", "F3": "VIX term slope",
         "L4A": "funding seizure", "L4B": "spread blowout velocity", "L4C": "correlation spike",
         "L4D": "forced deleveraging", "L4E": "hedging feedback"}

# Editorial: what keeps the composite NULL. Update as items close. effect_class: crit | warn | neutral | good
BLOCKERS = [
    {"item": "L1 unbuilt", "evidence": "S10, S13, S15 = NA in every run; only S11 live → L1 coverage 25%.",
     "effect": "Hard block", "effect_class": "crit",
     "clears": "Any two of S10 / S13 / S15 flowing (3 of 4 = 75% ≥ 70%). Order: S13 (Shiller CAPE) → S10 (FINRA margin xlsx, manual, VPN) → S15 later"},
    {"item": "L4 gaps", "evidence": "L4D (forced deleveraging) needs S10 margin data; L4E (hedging feedback) needs F3 and F9 — F9 is Phase 5. L4C (Cboe COR) stale until the Cboe leg runs daily. 4 of 5 needed.",
     "effect": "Hard block", "effect_class": "crit",
     "clears": "L4A + L4B + L4C + L4D live — Cboe leg daily for L4C, S10 for L4D. L4E stays unbuilt until F9 exists"},
    {"item": "Ingest schedule", "evidence": "ingest_spx.py / ingest_breadth.py were unscheduled until the 7 Sep manual backfill; FRED weekly leg timed out on 30 Aug and 6 Sep.",
     "effect": "Fixed by hand", "effect_class": "warn",
     "clears": "The sequenced wrapper (ingests → UW → driver) is in cron and has produced two consecutive clean runs"},
    {"item": "F3 fallback", "evidence": "2–4 Sep F3 used its CFE-futures leg (last row 23 Feb 2018) because the FRED spot leg aged past 3 days; the futures leg has no freshness check.",
     "effect": "Defect", "effect_class": "warn",
     "clears": "<code>_fresh()</code> required on the futures leg too, else NA; test added"},
    {"item": "Unversioned", "evidence": "daily_driver.py writes the literal \"unversioned\"; signal_registry.csv has no version column.",
     "effect": "Rule 3", "effect_class": "warn",
     "clears": "sha256[:12] of the CSV stamped per run; loader refuses to run without the file"},
    {"item": "S9 date stamp", "evidence": "s9_short_interest.py stamps the FINRA settlement date as source_asof; the registry's 20-day limit assumes the publication date.",
     "effect": "Defect", "effect_class": "warn",
     "clears": "Builder stamps pub_date; no registry change needed"},
]

# NYSE full-day closures 2026 (best effort; Good Friday 3 Apr, Independence Day observed 3 Jul)
NYSE_HOLIDAYS_2026 = {"2026-01-01", "2026-01-19", "2026-02-16", "2026-04-03", "2026-05-25", "2026-06-19",
                      "2026-07-03", "2026-09-07", "2026-11-26", "2026-12-25"}


def sig_key(s):
    m = re.match(r"([A-Z]+)(\d*)([A-Z]?)", s)
    return ({"S": 0, "F": 1, "L": 2}[m.group(1)[0]], int(m.group(2) or 0), m.group(3))


def read_note(latest_by_sig):
    reds = sorted([s for s, r in latest_by_sig.items() if r["state"] == "R"], key=sig_key)
    ambers = sorted([s for s, r in latest_by_sig.items() if r["state"] == "Y"], key=sig_key)
    nm = lambda s: s + (" (" + NAMES[s] + ")" if s in NAMES else "")
    parts = ["<strong>Read despite the freeze.</strong>"]
    if reds:
        parts.append("Red on the latest run: " + ", ".join("<strong>" + nm(s) + "</strong>" for s in reds) + ".")
    if ambers:
        parts.append("Amber: " + ", ".join(nm(s) for s in ambers) + ".")
    eff = sorted([s for s, r in latest_by_sig.items() if r.get("effective_state") not in (None, "", "G")], key=sig_key)
    parts.append(("Effective (persistence met): " + ", ".join(eff) + ".") if eff else
                 "Nothing has reached its persistence requirement, so no state is effective and the layer scores are 0.")
    parts.append("Per the brief, no single signal is acted on.")
    return " ".join(parts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="warning.db")
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
        "holiday_rows": [d for d in dates if d in NYSE_HOLIDAYS_2026 or datetime.strptime(d, "%Y-%m-%d").weekday() >= 5],
        "blockers": BLOCKERS,
        "read_note": read_note(latest_by_sig),
        "runlog_note": "Both defects this note described are fixed (2026-09-08). B3 gave alerts a unique index and INSERT OR REPLACE, "
                       "so re-stepping a date no longer accumulates rows; 28 Aug held 30 from six dev re-runs. B8 derives asof_date "
                       "from utils.market_calendar.last_completed_session(); the Sunday and Labor Day rows were deleted and no "
                       "non-trading date can be written.",
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
