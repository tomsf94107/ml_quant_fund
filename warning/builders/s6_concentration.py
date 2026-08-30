"""
s6_concentration.py — builder for S6, Concentration / equal-weight lag.

REGISTRY ROW:
    id S6 | layer L2 | tier shortlist
    formula  EW_minus_CW 126td relative return < -4% while SPX within 3% of high
    arm      -2%            red: -4%
    source   RSP/SPXEW 2003+; French size pre-2003
    series   RSP;SPXEW;French          history_start: 2003-01-08

WHAT IT MEASURES
    The equal-weight index minus the cap-weight index over 126 trading days.
    When the cap-weight index keeps making highs while the equal-weight version
    lags badly, the advance has narrowed to a handful of large names -- the
    index is being carried rather than led.

    Like S2, S5 and S7, the gate is the point. Equal-weight lagging in a DECLINE
    is ordinary: large caps are defensive, so EW underperforms on the way down
    without that meaning anything. The signal is narrowing WHILE the index is
    still at its high.

DATA ROUTE, AND WHY IT DIFFERS FROM THE OTHER PRICE SIGNALS
    prices.db carries 443 tickers and RSP is not among them, so RSP_CLOSE is
    fetched from Massive directly (ingest_spx.py --from-massive) rather than via
    prices.db like SPY and the sector ETFs. The provenance is recorded in
    data_vintages.source so the two routes are never confused.

    Massive returns 2,511 RSP bars against SPY's 2,544, so the two series do not
    align perfectly; pit.align intersects them and the overlap count is reported
    in every reading.

COVERAGE
    2016-07-18 onward, not the registry's 2003. SPXEW and the French size
    portfolios would extend it and neither is ingested. As with every other
    price-derived signal here, the sample contains no credit crisis (D17).
"""

from __future__ import annotations
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pit import series_asof, align, staleness_bdays  # noqa: E402

SIGNAL_ID = "S6"
LAYER = "L2"
MAX_STALENESS_DAYS = 7
PERSISTENCE_DAYS = 10
AMBER_STATE = "Y"                  # DECISIONS.md D1

EW = "RSP_CLOSE"                   # equal-weight S&P 500
CW = "SPY_CLOSE"                   # cap-weight; PROXY for SPX
RS_WINDOW = 126                    # registry: "126td"
ARM_LEVEL = -0.02                  # registry threshold_arm: "-2%"
RED_LEVEL = -0.04                  # registry threshold_red: "-4%"
NEAR_HIGH_PCT = 0.03               # registry: "within 3% of high"
HIGH_WINDOW = 252


def compute(con, asof):
    ew = series_asof(con, EW, asof)
    cw = series_asof(con, CW, asof)
    if not ew or not cw:
        missing = [s for s, r in ((EW, ew), (CW, cw)) if not r]
        return _na(asof, f"no visible observations for {','.join(missing)}")
    if len(cw) < HIGH_WINDOW:
        return _na(asof, f"need {HIGH_WINDOW} obs of {CW} for the near-high "
                         f"gate, have {len(cw)}")

    joined = align(ew, cw)
    if len(joined) < RS_WINDOW + 1:
        return _na(asof, f"need {RS_WINDOW + 1} overlapping {EW}/{CW} obs, "
                         f"have {len(joined)}")

    _d0, e0, c0 = joined[-1 - RS_WINDOW]
    _d1, e1, c1 = joined[-1]
    if not e0 or not c0:
        return _na(asof, "zero price at the window start")
    rel = ((e1 - e0) / e0) - ((c1 - c0) / c0)

    hi = max(v for _, v in cw[-HIGH_WINDOW:])
    last = cw[-1][1]
    near_high = last >= hi * (1.0 - NEAR_HIGH_PCT)

    if near_high and rel < RED_LEVEL:
        state = "R"
    elif near_high and rel < ARM_LEVEL:
        state = AMBER_STATE
    else:
        state = "G"

    stale = max(x for x in (staleness_bdays(con, EW, asof),
                            staleness_bdays(con, CW, asof)) if x is not None)

    return {
        "signal_id": SIGNAL_ID, "layer": LAYER, "asof": str(asof),
        "state": state, "raw_value": rel, "zscore": None,
        "stale": stale > MAX_STALENESS_DAYS, "stale_days": stale,
        "persistence_days": PERSISTENCE_DAYS,
        "source_asof": joined[-1][0],
        "detail": {
            "ew_minus_cw_126d_pct": round(rel * 100, 2),
            "arm_below_pct": ARM_LEVEL * 100, "red_below_pct": RED_LEVEL * 100,
            "ew_126d_pct": round((e1 - e0) / e0 * 100, 2),
            "cw_126d_pct": round((c1 - c0) / c0 * 100, 2),
            "overlapping_obs": len(joined),
            "index_pct_below_high": round(100 * (hi - last) / hi, 2),
            "near_high": near_high,
            "note": "equal-weight lagging in a DECLINE is ordinary -- large caps "
                    "are defensive. The signal is narrowing WHILE the index "
                    "holds its high.",
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
