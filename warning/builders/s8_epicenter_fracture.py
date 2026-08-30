"""
s8_epicenter_fracture.py — builder for S8, Epicenter fracture.

REGISTRY ROW:
    id S8 | layer L2 | tier shortlist
    formula  leader := sector/theme with top trailing-2y RS.
             FIRE if leader -15% from its high AND < its 200DMA
             while SPX within 5% of its high
    arm      -10% & <200DMA        red: -15% & <200DMA
    source   own universe / sector ETFs / French industries
    history  full (French)

THE IDEA
    Every mania has an epicenter -- the sector that led on the way up. It breaks
    first, and it breaks while the index still looks fine, because the index is
    an average and the epicenter is one part of it. 2000 had technology; 2007 had
    financials and homebuilders. The signal is not "a sector fell"; it is "the
    sector that led is now broken WHILE the index has not noticed".

    The leader is not named in advance. It is whichever sector has the top
    trailing 2-year relative strength as of the evaluation date -- so the signal
    identifies its own epicenter from the data, point-in-time, and cannot be
    accused of knowing in hindsight which sector mattered.

COVERAGE
    Eleven SPDR sector ETFs from prices.db, 2016-07-18 (XLC from 2018-06-19).
    A 2-year RS window plus a 252-day high means S8 runs from roughly 2018-08.
    The registry's "full (French)" history assumes French industry portfolios
    the stack does not hold; before 2018 the signal reports NA rather than
    pretending to a longer record.

    SPY stands in for SPX in the near-high gate, labelled in every reading.

WHY THE INDEX GATE IS 5% HERE AND 3% IN S2/S7
    The registry says so: S8 specifies "within 5% of its high", S2 and S7 say 3%.
    The looser gate is consistent with the idea -- an epicenter can crack while
    the index has already drifted a few percent, and the point is that the index
    has NOT confirmed. Not harmonized; the frozen numbers are used as written.
"""

from __future__ import annotations
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pit import series_asof, align, staleness_bdays  # noqa: E402

SIGNAL_ID = "S8"
LAYER = "L2"
MAX_STALENESS_DAYS = 7
PERSISTENCE_DAYS = 10
AMBER_STATE = "Y"                    # DECISIONS.md D1

SECTORS = ["XLB_CLOSE", "XLC_CLOSE", "XLE_CLOSE", "XLF_CLOSE", "XLI_CLOSE",
           "XLK_CLOSE", "XLP_CLOSE", "XLRE_CLOSE", "XLU_CLOSE", "XLV_CLOSE",
           "XLY_CLOSE"]
BENCH = "SPY_CLOSE"                  # PROXY for SPX

RS_WINDOW = 504                      # registry: "trailing-2y RS"
HIGH_WINDOW = 252                    # the leader's own 52-week high
DMA_WINDOW = 200                     # registry: "< its 200DMA"
ARM_DRAWDOWN = 0.10                  # registry threshold_arm: "-10% & <200DMA"
RED_DRAWDOWN = 0.15                  # registry threshold_red: "-15% & <200DMA"
INDEX_NEAR_HIGH = 0.05               # registry: "SPX within 5% of its high"
MIN_SECTORS = 6                      # a "leader" among three sectors is noise


def _total_return(rows, window):
    if len(rows) < window + 1:
        return None
    p0, p1 = rows[-1 - window][1], rows[-1][1]
    return None if not p0 else (p1 - p0) / p0


def compute(con, asof):
    bench = series_asof(con, BENCH, asof)
    if len(bench) < max(HIGH_WINDOW, RS_WINDOW + 1):
        return _na(asof, f"need {max(HIGH_WINDOW, RS_WINDOW + 1)} obs of {BENCH}, "
                         f"have {len(bench)}")
    bench_ret = _total_return(bench, RS_WINDOW)
    if bench_ret is None:
        return _na(asof, f"{BENCH} has no {RS_WINDOW}-day return")

    # Point-in-time leader: top trailing-2y relative strength, chosen from data
    # visible at `asof` only. The epicenter is never named in advance.
    rs, skipped = {}, []
    for name in SECTORS:
        s = series_asof(con, name, asof)
        if len(s) < max(RS_WINDOW + 1, HIGH_WINDOW, DMA_WINDOW):
            skipped.append(f"{name.replace('_CLOSE','')}: {len(s)} obs")
            continue
        st = staleness_bdays(con, name, asof)
        if st is None or st > MAX_STALENESS_DAYS:
            skipped.append(f"{name.replace('_CLOSE','')}: stale {st}d")
            continue
        r = _total_return(s, RS_WINDOW)
        if r is None:
            continue
        rs[name] = r - bench_ret

    if len(rs) < MIN_SECTORS:
        return _na(asof, f"need >={MIN_SECTORS} sectors with {RS_WINDOW}d history, "
                         f"have {len(rs)} ({'; '.join(skipped) or 'none'})")

    leader = max(rs, key=rs.get)
    lrows = series_asof(con, leader, asof)
    lhigh = max(v for _, v in lrows[-HIGH_WINDOW:])
    llast = lrows[-1][1]
    ldd = (lhigh - llast) / lhigh
    ldma = sum(v for _, v in lrows[-DMA_WINDOW:]) / DMA_WINDOW
    lbelow = llast < ldma

    bhigh = max(v for _, v in bench[-HIGH_WINDOW:])
    blast = bench[-1][1]
    index_near_high = blast >= bhigh * (1.0 - INDEX_NEAR_HIGH)

    if index_near_high and lbelow and ldd >= RED_DRAWDOWN:
        state = "R"
    elif index_near_high and lbelow and ldd >= ARM_DRAWDOWN:
        state = AMBER_STATE
    else:
        state = "G"

    stale = max(x for x in (staleness_bdays(con, BENCH, asof),
                            staleness_bdays(con, leader, asof))
                if x is not None)

    return {
        "signal_id": SIGNAL_ID, "layer": LAYER, "asof": str(asof),
        "state": state, "raw_value": ldd, "zscore": None,
        "stale": stale > MAX_STALENESS_DAYS, "stale_days": stale,
        "persistence_days": PERSISTENCE_DAYS,
        "source_asof": bench[-1][0],
        "detail": {
            "leader": leader.replace("_CLOSE", ""),
            "leader_rs_2y_pct": round(rs[leader] * 100, 1),
            "rs_ranking_pct": {k.replace("_CLOSE", ""): round(v * 100, 1)
                               for k, v in sorted(rs.items(), key=lambda x: -x[1])},
            "leader_last": round(llast, 2),
            "leader_52w_high": round(lhigh, 2),
            "leader_drawdown_pct": round(ldd * 100, 2),
            "leader_200dma": round(ldma, 2),
            "leader_below_200dma": lbelow,
            "arm_at_pct": ARM_DRAWDOWN * 100, "red_at_pct": RED_DRAWDOWN * 100,
            "bench": BENCH + " (PROXY for SPX)",
            "bench_pct_below_high": round(100 * (bhigh - blast) / bhigh, 2),
            "index_near_high": index_near_high,
            "n_sectors": len(rs), "skipped": skipped or None,
            "note": "leader chosen point-in-time from trailing-2y RS; the "
                    "epicenter is never named in advance",
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
