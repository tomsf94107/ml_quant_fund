"""
l4_propagation.py — Layer 4, crash propagation.

REPORT Part VI, line 601 (verbatim):
    "Layer 4 -- Crash propagation (stress underway -> overrides composite to B).
     Weight 0.25. Funding seizure (S4 red + breadth-of-stress across >=2 funding
     markets), spread blowout velocity (S2 +150bp/21d), correlation spike (avg
     pairwise 63d corr top-decile jump), forced-deleveraging evidence (margin
     -10%/3m + vol clustering), hedging-feedback (VIX curve inverted +
     negative-gamma est. + widening realized ranges)."

WHY NO REGISTRY ROW IS L4
    These are DERIVED conditions over signals that already exist (S2, S4, S10,
    F3, F9), not independent measurements. signal_registry.csv lists the 15
    measured signals; L4 composes them. An earlier reading of this codebase
    treated the absence of L4 rows as a structural blocker -- it is not. L4
    coverage rises as S4 and S10 are built.

STATE MAPPING
    A fired propagation condition emits 'B'. warning_engine.l4_propagation_red
    escalates on any single 'B' (or two 'R'), which bypasses persistence and
    goes straight to CRISIS. That severity is the report's intent: these are
    stress-underway conditions, not warnings. Untriggered conditions emit 'G';
    conditions whose inputs do not exist emit 'NA' so layer coverage stays honest.

BUILT: L4B (spread blowout velocity), L4C (correlation spike).
NOT BUILT: L4A (needs S4), L4D (needs S10), L4E (needs F3 + F9).
"""

from __future__ import annotations
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pit import series_asof, staleness_bdays  # noqa: E402

LAYER = "L4"
MAX_STALENESS_DAYS = 3

# --- L4B: spread blowout velocity -----------------------------------------
BLOWOUT_SERIES = "BAMLH0A0HYM2"
BLOWOUT_BP = 1.50          # report: "S2 +150bp/21d"; series in percentage points
BLOWOUT_WINDOW = 21        # trading days

# --- L4C: correlation spike ------------------------------------------------
# Report: "avg pairwise 63d corr top-decile jump". Cboe's 3-month implied
# correlation index is the documented proxy (report Part VII F8: "from F7
# identity or Cboe COR files"), and 3 months ~= 63 trading days, so COR3M is the
# natural match for the stated horizon.
#
# DECISION D9 -- the report gives no lookback for "top-decile". 504 trading days
# is used, matching F2's own percentile window, so the two percentile-based
# conditions in this system share one convention. Flagged, not silently chosen.
CORR_SERIES = "CBOE_COR3M"
CORR_WINDOW = 504
CORR_TOP_DECILE = 90.0     # "top-decile"
CORR_JUMP_WINDOW = 21      # "jump", not merely a high level
CORR_JUMP_MIN = 0.0        # must have risen over the window


def compute_all(con, asof):
    """Returns {condition_id: result-dict} for all five L4 conditions."""
    return {
        "L4A": _na("L4A", asof, "funding seizure needs S4 (builder not implemented) "
                                "plus breadth-of-stress across >=2 funding markets"),
        "L4B": spread_blowout(con, asof),
        "L4C": correlation_spike(con, asof),
        "L4D": _na("L4D", asof, "forced deleveraging needs S10 margin data "
                                "(FINRA xlsx not ingested) + vol clustering"),
        "L4E": _na("L4E", asof, "hedging feedback needs F3 (VIX curve) and F9 "
                                "(negative-gamma estimate, experimental)"),
    }


def spread_blowout(con, asof):
    """L4B: HY OAS widened >= 150bp over 21 trading days."""
    rows = series_asof(con, BLOWOUT_SERIES, asof)
    if len(rows) < BLOWOUT_WINDOW + 1:
        return _na("L4B", asof, f"need {BLOWOUT_WINDOW + 1} obs of "
                                f"{BLOWOUT_SERIES}, have {len(rows)}")
    cur = rows[-1][1]
    prior = rows[-1 - BLOWOUT_WINDOW][1]
    delta = cur - prior
    stale = staleness_bdays(con, BLOWOUT_SERIES, asof)
    fired = delta >= BLOWOUT_BP
    return {
        "signal_id": "L4B", "layer": LAYER, "asof": str(asof),
        "state": "B" if fired else "G",
        "raw_value": delta, "zscore": None,
        "stale": (stale is None or stale > MAX_STALENESS_DAYS), "stale_days": stale,
        "persistence_days": 1,          # propagation conditions are immediate
        "source_asof": rows[-1][0],
        "detail": {"condition": "spread blowout velocity",
                   "series": BLOWOUT_SERIES,
                   "now": round(cur, 3), "prior_21d": round(prior, 3),
                   "delta_bp": round(delta * 100, 1),
                   "threshold_bp": BLOWOUT_BP * 100, "fired": fired},
    }


def correlation_spike(con, asof):
    """L4C: implied correlation in its top decile AND risen over 21 days."""
    rows = series_asof(con, CORR_SERIES, asof)
    if len(rows) < CORR_WINDOW:
        return _na("L4C", asof, f"need {CORR_WINDOW} obs of {CORR_SERIES}, "
                                f"have {len(rows)}")
    window = [v for _, v in rows[-CORR_WINDOW:]]
    cur = window[-1]
    pct = 100.0 * sum(1 for v in window if v <= cur) / len(window)
    prior = rows[-1 - CORR_JUMP_WINDOW][1] if len(rows) > CORR_JUMP_WINDOW else cur
    jumped = (cur - prior) > CORR_JUMP_MIN
    stale = staleness_bdays(con, CORR_SERIES, asof)
    fired = pct >= CORR_TOP_DECILE and jumped
    return {
        "signal_id": "L4C", "layer": LAYER, "asof": str(asof),
        "state": "B" if fired else "G",
        "raw_value": cur, "zscore": None,
        "stale": (stale is None or stale > MAX_STALENESS_DAYS), "stale_days": stale,
        "persistence_days": 1,
        "source_asof": rows[-1][0],
        "detail": {"condition": "correlation spike",
                   "series": CORR_SERIES, "note": "Cboe implied correlation is a "
                   "documented proxy for average pairwise realized correlation "
                   "(report Part VII F8); 3-month ~= the stated 63-day horizon",
                   "level": round(cur, 2),
                   f"percentile_{CORR_WINDOW}d": round(pct, 1),
                   "prior_21d": round(prior, 2), "jumped": jumped,
                   "top_decile": pct >= CORR_TOP_DECILE, "fired": fired},
    }


def _na(sid, asof, reason):
    return {"signal_id": sid, "layer": LAYER, "asof": str(asof), "state": "NA",
            "raw_value": None, "zscore": None, "stale": True, "stale_days": None,
            "persistence_days": 1, "source_asof": None,
            "detail": {"reason": reason}}


def to_reading(result):
    from warning_engine import SignalReading
    return SignalReading(
        signal_id=result["signal_id"], layer=result["layer"],
        state=result["state"], stale=bool(result["stale"]),
        min_persistence=result["persistence_days"])
