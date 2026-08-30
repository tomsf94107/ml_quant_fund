"""
s8f_epicenter_french.py — S8's formula on French data, 1926-2026.

A COMPANION TO S8, NOT A REPLACEMENT
    S8 runs on 11 SPDR sector ETFs against SPY, and prices.db starts 2016-07-18
    because that is Massive's plan floor (tested 2026-08-30, D19). In that decade
    S8 fired exactly once -- 2023-05-31, a false positive driven by XLE's
    post-energy-cycle decay (D15).

    One fire is not a record. This module runs the SAME thresholds on Ken
    French's 12 industry portfolios against the CRSP value-weighted market,
    daily from 1926 -- a century containing 1929, 1937, 1973, 1987, 2000 and
    2008. The question it answers is the one D19 answered for S6: are the
    registry's -15% / 200DMA / 5% thresholds sensible, or is the single SPDR-era
    fire telling us the specification is wrong?

    S8 keeps running on SPDRs. Both exist so the comparison is possible; D19
    only became answerable because the French and RSP versions could be set side
    by side.

WHAT IS DIFFERENT FROM S8, AND WHY IT MATTERS
    1. INDUSTRIES, NOT SECTORS. French's 12 (NoDur, Durbl, Manuf, Enrgy, Chems,
       BusEq, Telcm, Utils, Shops, Hlth, Money, Other) map loosely onto SPDRs --
       BusEq~XLK, Money~XLF, Enrgy~XLE -- but Manuf spans XLI and XLB, and Other
       is a catch-all. This is a RELATED signal, not the same one.

    2. RETURNS, NOT PRICES. French publishes daily returns, so the 252-day high
       and the 200-day average are computed from a COMPOUNDED INDEX built here
       from a base of 100. Both tests are ratios (drawdown from high, price vs
       its own average), so they are scale-invariant and the arbitrary base does
       not matter -- but the index is a construction, not a traded price, and
       dividends are included where a price index would exclude them.

    3. THE MARKET IS Mkt-RF + RF, the CRSP value-weighted total return, not SPX.

    4. NOT POINT-IN-TIME (D20). French restates history when CRSP is revised, so
       these readings are CALIBRATION-GRADE, not replay-grade. "S8F would have
       fired in 1973" is not a claim this data supports. "The threshold fires X%
       of the time across a century" is.

THRESHOLDS ARE S8'S, UNCHANGED
    Imported from the S8 module rather than restated, so the two cannot drift
    apart. If S8's numbers are ever ratified differently, this follows.
"""

from __future__ import annotations
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pit import series_asof, align  # noqa: E402
from builders.s8_epicenter_fracture import (  # noqa: E402
    RS_WINDOW, HIGH_WINDOW, DMA_WINDOW, ARM_DRAWDOWN, RED_DRAWDOWN,
    INDEX_NEAR_HIGH, MIN_SECTORS,
)

SIGNAL_ID = "S8F"
LAYER = "L2"
AMBER_STATE = "Y"

INDUSTRIES = ["NoDur", "Durbl", "Manuf", "Enrgy", "Chems", "BusEq",
              "Telcm", "Utils", "Shops", "Hlth", "Money", "Other"]
IND_PREFIX = "FR_I12_VW:"
MKT_EXCESS = "FR_F:Mkt-RF"
RISK_FREE = "FR_F:RF"


def _compound(rows):
    """[(date, pct_return)] -> [(date, index_level)] from a base of 100.

    The base is arbitrary. Every test applied to this series is a ratio --
    drawdown from a trailing high, level versus a trailing average -- so the
    base cancels. Stated because a compounded return series looks like a price
    series and is not one: it includes dividends, and it is not tradeable.
    """
    lvl, out = 100.0, []
    for d, r in rows:
        lvl *= (1.0 + r / 100.0)
        out.append((d, lvl))
    return out


def _market(con, asof):
    ex = series_asof(con, MKT_EXCESS, asof)
    rf = series_asof(con, RISK_FREE, asof)
    if not ex or not rf:
        return []
    return _compound([(d, a + b) for d, a, b in align(ex, rf)])


def compute(con, asof):
    mkt = _market(con, asof)
    if len(mkt) < max(HIGH_WINDOW, RS_WINDOW + 1):
        return _na(asof, f"market needs {max(HIGH_WINDOW, RS_WINDOW + 1)} obs, "
                         f"have {len(mkt)}")

    m0 = mkt[-1 - RS_WINDOW][1]
    m1 = mkt[-1][1]
    mkt_ret = (m1 - m0) / m0

    rs, skipped = {}, []
    levels = {}
    for name in INDUSTRIES:
        rows = series_asof(con, IND_PREFIX + name, asof)
        if len(rows) < max(RS_WINDOW + 1, HIGH_WINDOW, DMA_WINDOW):
            skipped.append(f"{name}:{len(rows)}")
            continue
        idx = _compound(rows)
        levels[name] = idx
        r = (idx[-1][1] - idx[-1 - RS_WINDOW][1]) / idx[-1 - RS_WINDOW][1]
        rs[name] = r - mkt_ret

    if len(rs) < MIN_SECTORS:
        return _na(asof, f"need >={MIN_SECTORS} industries, have {len(rs)} "
                         f"({';'.join(skipped) or 'none'})")

    leader = max(rs, key=rs.get)
    lidx = levels[leader]
    lhigh = max(v for _, v in lidx[-HIGH_WINDOW:])
    llast = lidx[-1][1]
    ldd = (lhigh - llast) / lhigh
    ldma = sum(v for _, v in lidx[-DMA_WINDOW:]) / DMA_WINDOW
    lbelow = llast < ldma

    mhigh = max(v for _, v in mkt[-HIGH_WINDOW:])
    near_high = m1 >= mhigh * (1.0 - INDEX_NEAR_HIGH)

    if near_high and lbelow and ldd >= RED_DRAWDOWN:
        state = "R"
    elif near_high and lbelow and ldd >= ARM_DRAWDOWN:
        state = AMBER_STATE
    else:
        state = "G"

    return {
        "signal_id": SIGNAL_ID, "layer": LAYER, "asof": str(asof),
        "state": state, "raw_value": ldd, "zscore": None,
        "stale": False, "stale_days": 0, "persistence_days": 10,
        "source_asof": mkt[-1][0],
        "detail": {
            "leader": leader,
            "leader_rs_2y_pct": round(rs[leader] * 100, 1),
            "rs_ranking_pct": {k: round(v * 100, 1)
                               for k, v in sorted(rs.items(), key=lambda x: -x[1])},
            "leader_drawdown_pct": round(ldd * 100, 2),
            "leader_below_200dma": lbelow,
            "market_pct_below_high": round(100 * (mhigh - m1) / mhigh, 2),
            "market_near_high": near_high,
            "n_industries": len(rs),
            "CAVEAT": "French industries, not SPDR sectors; compounded index, "
                      "not a traded price; restated history (D20), so this is "
                      "calibration-grade and NOT a real-time replay.",
        },
    }


def _na(asof, reason):
    return {"signal_id": SIGNAL_ID, "layer": LAYER, "asof": str(asof),
            "state": "NA", "raw_value": None, "zscore": None,
            "stale": True, "stale_days": None, "persistence_days": 10,
            "source_asof": None, "detail": {"reason": reason}}
