"""
s13_valuation.py — builder for S13, Valuation gate.

REGISTRY ROW:
    id S13 | layer L1 | tier shortlist | history_start 1871-01
    formula  CAPE percentile vs own history-to-date: amber 85-95th; red >95th.
             ERP check: 1/CAPE - 10y real
    source   Shiller                  series: ie_data.xls
    arm      85th pct                 red: 95th pct
    frequency monthly                 publication_lag ~1 month
    persistence_days 1                max_staleness_days 45
    direction high_val_bearish        role: GATE
    verdicts  44.19 Dec-99 (top of history) | 2008 DID NOT FIRE (~25-27)
              | 2022 fired (37-38.6)

PERCENTILES ARE EXPANDING, NOT FULL-SAMPLE
    Same construction as S11 and read the same way: at each evaluation month
    the percentile is computed against every month up to and including that
    one, and nothing after. Ranking against the full 1881-2026 sample would
    tell 1999 what 2021 looked like.

    This matters more here than anywhere else in the roster. CAPE has drifted
    upward for forty years, so a full-sample percentile would place the 1990s
    lower than a contemporary observer could have seen them, and would flatter
    every recent reading. The expanding form is what the registry says and it
    is what the Dec-99 verdict of "top of history" means -- top of history AS
    OF THEN.

THE SIGNAL IS STRUCTURALLY BLIND TO 2008, BY DESIGN
    The registry says so in its own notes, and the verdict column records that
    2008 DID NOT FIRE at CAPE 25-27. That is not a defect to tune away. 2008
    was a credit and leverage event, not a valuation event.

    But "not expensive" overstates it, and the measured figures are worth
    carrying: October 2007 reads CAPE 26.73 at the 94.0th percentile of its
    own history to that date -- AMBER, one point under the red line. The gate
    was not blind before the GFC; it was scaling exposure down and one
    percentile short of a red. What it could not see was the leverage. S13 is a GATE that scales exposure
    ceilings, not a crash predictor, and L1 carries S10 and S15 for exactly
    the risks S13 cannot see. A version of this that fired in 2007 would have
    been fitted to one episode.

ERP LEG NOT EVALUATED
    The registry's second clause is "ERP check: 1/CAPE - 10y real". The 10-year
    REAL rate needs a PIT inflation-expectation series; DGS10 is nominal, and
    subtracting realised CPI would be a different quantity computed with
    hindsight. Reported in detail as unavailable rather than approximated.
    S13 is currently the CAPE percentile leg alone -- the same treatment S11
    gives its unbuilt credit leg.

WHY MIN_MONTHS
    A percentile over a handful of observations is arithmetic, not
    information. CAPE begins about 1881 because it needs ten years of earnings
    to exist at all, and the floor here holds the signal at NA until the
    expanding history can support a 95th percentile.

SOURCE FRESHNESS IS A REAL RISK
    warning/ingest_shiller.py refuses a file whose newest complete month is
    too old, because the Yale mirror is frozen at 2023-08 and downloads and
    parses without error. This builder's staleness check is the second line:
    max_staleness_days 45 against a monthly series means one missed release
    turns the gate NA rather than freezing it at a stale reading. That is the
    L4C failure -- a signal reading G off week-old data is worse than one
    reading NA.
"""

from __future__ import annotations
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pit import series_asof, staleness_days  # noqa: E402

SIGNAL_ID = "S13"
LAYER = "L1"
MAX_STALENESS_DAYS = 45            # registry; monthly series
PERSISTENCE_DAYS = 1               # registry: "1 obs"
AMBER_STATE = "Y"                  # DECISIONS.md D1

CAPE_SERIES = "SHILLER_CAPE"
ARM_PCTILE = 85.0                  # registry threshold_arm
RED_PCTILE = 95.0                  # registry threshold_red
MIN_MONTHS = 120                   # ten years before a 95th pctile means anything


def _pctile(values, value):
    return 100.0 * sum(1 for v in values if v <= value) / len(values)


def compute(con, asof):
    rows = series_asof(con, CAPE_SERIES, asof)
    if not rows:
        return _na(asof, f"no visible observations for {CAPE_SERIES}; run "
                         f"warning/ingest_shiller.py")
    if len(rows) < MIN_MONTHS:
        return _na(asof, f"need {MIN_MONTHS} published months for an expanding "
                         f"95th percentile, have {len(rows)}")

    vals = [v for _, v in rows]
    cape = vals[-1]
    pct = _pctile(vals, cape)

    if pct > RED_PCTILE:
        state = "R"
    elif pct >= ARM_PCTILE:
        state = AMBER_STATE
    else:
        state = "G"

    stale = staleness_days(con, CAPE_SERIES, asof)
    return {
        "signal_id": SIGNAL_ID, "layer": LAYER, "asof": str(asof),
        "state": state, "raw_value": cape, "zscore": None,
        "stale": (stale is None or stale > MAX_STALENESS_DAYS),
        "stale_days": stale,
        "persistence_days": PERSISTENCE_DAYS,
        "source_asof": rows[-1][0],
        "detail": {
            "reference_month": rows[-1][0][:7],
            "cape": round(cape, 2),
            "cape_pctile": round(pct, 1),
            "arm_at": ARM_PCTILE, "red_at": RED_PCTILE,
            "months_in_history": len(vals),
            "months_above": sum(1 for v in vals if v > cape),
            "max_to_date": round(max(vals), 2),
            "pctile_note": "expanding: ranked against months published up to "
                           "this date only, never the full sample",
            "series_note": "standard CAPE (ie_data column 12), not the "
                           "total-return variant; the registry's Dec-99 "
                           "verdict of 44.19 is the standard series",
            "erp_leg": None,
            "erp_note": "the registry's second clause, 1/CAPE - 10y real, is "
                        "NOT evaluated: the 10y REAL rate needs a PIT "
                        "inflation-expectation series, and DGS10 is nominal. "
                        "S13 is the CAPE percentile leg alone.",
            "blind_spot": "does not fire RED on 2008-type events. Measured: "
                          "2007-10 CAPE 26.73, 94.0th percentile to date = "
                          "AMBER, one point under red. Not blind -- it was "
                          "scaling exposure down. What it cannot see is "
                          "leverage, which is S10 and S15. Registry verdict "
                          "DID NOT FIRE means no red, not no signal.",
        },
    }


def _na(asof, reason):
    return {"signal_id": SIGNAL_ID, "layer": LAYER, "asof": str(asof),
            "state": "NA", "raw_value": None, "zscore": None,
            "stale": True, "stale_days": None,
            "persistence_days": PERSISTENCE_DAYS, "source_asof": None,
            "detail": {"reason": reason}}


def to_reading(result):
    from warning_engine import SignalReading
    return SignalReading(signal_id=result["signal_id"], layer=result["layer"],
                         state=result["state"], stale=bool(result["stale"]),
                         min_persistence=result["persistence_days"])
