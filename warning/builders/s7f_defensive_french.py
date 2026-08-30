"""
s7f_defensive_french.py — S7's formula on French data, 1926-2026.

WHY THIS EXISTS
    S7 runs on XLP/XLU/XLV against SPY and has fired exactly ONCE in ten years
    (2024-09-30). One observation cannot tell us whether the registry's +3%/+5%
    thresholds are calibrated well. The same problem afflicted S6 and S8, and in
    both cases a century of French data settled it -- vindicating S6's
    thresholds (D19) and indicting S8's leader definition (D15). The answers
    went opposite ways, which is exactly why the test is worth running rather
    than reasoning about.

    French's 12 industries include the defensive trio almost directly:
        NoDur  consumer non-durables  ~ XLP
        Utils  utilities              ~ XLU
        Hlth   healthcare             ~ XLV
    against the CRSP value-weighted market (Mkt-RF + RF) rather than SPY.

    S7 keeps running on SPDRs. This is a calibration companion, not a
    replacement, so the two can be compared -- which is the only reason D19 and
    D15 could be resolved at all.

CAVEATS, same as S8F
    French industries are not SPDR sectors. Returns are compounded into an index
    from an arbitrary base, so every test applied must be scale-invariant -- both
    of S7's are (a relative return over a window, and a level against a trailing
    high). And D20: French restates history when CRSP is revised, so this is
    calibration-grade, NOT a real-time replay. "S7F would have fired in 1973" is
    not a claim this data supports.

THRESHOLDS ARE S7'S, IMPORTED NOT RESTATED
    So the two cannot drift apart if S7's numbers are ever ratified differently.
"""

from __future__ import annotations
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pit import series_asof  # noqa: E402
from builders.s7_defensive_rotation import (  # noqa: E402
    RS_WINDOW, RS_ARM, RS_RED, NEAR_HIGH_PCT, HIGH_WINDOW, MIN_DEFENSIVE_ETFS,
)
from builders.s8f_epicenter_french import _compound, _market  # noqa: E402

SIGNAL_ID = "S7F"
LAYER = "L2"
AMBER_STATE = "Y"

DEFENSIVE = {"NoDur": "~XLP", "Utils": "~XLU", "Hlth": "~XLV"}
IND_PREFIX = "FR_I12_VW:"


def compute(con, asof):
    mkt = _market(con, asof)
    if len(mkt) < max(HIGH_WINDOW, RS_WINDOW + 1):
        return _na(asof, f"market needs {max(HIGH_WINDOW, RS_WINDOW + 1)} obs, "
                         f"have {len(mkt)}")
    m0, m1 = mkt[-1 - RS_WINDOW][1], mkt[-1][1]
    mkt_ret = (m1 - m0) / m0

    rs, missing = {}, []
    for name in DEFENSIVE:
        rows = series_asof(con, IND_PREFIX + name, asof)
        if len(rows) < RS_WINDOW + 1:
            missing.append(f"{name}:{len(rows)}")
            continue
        idx = _compound(rows)
        r = (idx[-1][1] - idx[-1 - RS_WINDOW][1]) / idx[-1 - RS_WINDOW][1]
        rs[name] = r - mkt_ret

    if len(rs) < MIN_DEFENSIVE_ETFS:
        return _na(asof, f"need >={MIN_DEFENSIVE_ETFS} defensive industries, "
                         f"have {len(rs)} ({';'.join(missing) or 'none'})")

    mean_rs = sum(rs.values()) / len(rs)
    hi = max(v for _, v in mkt[-HIGH_WINDOW:])
    near_high = m1 >= hi * (1.0 - NEAR_HIGH_PCT)

    if near_high and mean_rs > RS_RED:
        state = "R"
    elif near_high and mean_rs > RS_ARM:
        state = AMBER_STATE
    else:
        state = "G"

    return {
        "signal_id": SIGNAL_ID, "layer": LAYER, "asof": str(asof),
        "state": state, "raw_value": mean_rs, "zscore": None,
        "stale": False, "stale_days": 0, "persistence_days": 10,
        "source_asof": mkt[-1][0],
        "detail": {
            "mean_rs_63d_pct": round(mean_rs * 100, 2),
            "arm_pct": RS_ARM * 100, "red_pct": RS_RED * 100,
            "per_industry_rs_pct": {k: round(v * 100, 2) for k, v in rs.items()},
            "mapping": DEFENSIVE,
            "n_defensive": len(rs),
            "market_pct_below_high": round(100 * (hi - m1) / hi, 2),
            "near_high": near_high,
            "CAVEAT": "French industries, not SPDR sectors; compounded index; "
                      "restated history (D20) -- calibration-grade, not a replay.",
        },
    }


def _na(asof, reason):
    return {"signal_id": SIGNAL_ID, "layer": LAYER, "asof": str(asof),
            "state": "NA", "raw_value": None, "zscore": None,
            "stale": True, "stale_days": None, "persistence_days": 10,
            "source_asof": None, "detail": {"reason": reason}}
