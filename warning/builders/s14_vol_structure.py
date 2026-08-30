"""
s14_vol_structure.py — builder for S14, Vol-structure regime.

REGISTRY ROW:
    id S14 | layer L3 | tier shortlist | history_start 1990 (RV) / 2004 (futures)
    formula  (a) 21d RV crosses top quartile of 2y w/ >=10d persistence
                 while SPX < 200DMA
             (b) VIX futures front > second >= 5d (2004+)
    source   own calc + CBOE/CFE       series: VIX csv; CFE settles

TWO LEGS, DELIBERATELY NOT AVERAGED
    The registry gives (a) and (b) as separate conditions, not as inputs to a
    blend. They measure different things: (a) is realized volatility entering a
    high regime with the index already below trend -- confirmation that a decline
    is underway. (b) is the futures curve inverting, which is positioning stress
    and can occur with the index at a high.

    Either alone arms; both together fire. Averaging them would produce a number
    matching neither, the same error avoided in F3 (DECISIONS.md D14) where the
    two legs pointed in opposite directions at the COVID bottom.

LEG (a) NEEDS AN EQUITY SERIES AND IS THEREFORE 2016+
    Realized vol and the 200-day moving average both come from SPY_CLOSE, which
    prices.db carries from 2016-07-18. Before that leg (a) reports NA. The
    registry's "1990 (RV full)" assumes an SPX series the stack does not yet
    have -- the same Shiller gap that blocks S13 and D12.

LEG (b) IS 2004-03-26 .. 2018-02-23
    VX_FRONT/VX_SECOND come from the CFE archive, scale-normalized per
    DECISIONS.md D13. CFE reorganized the archive after Feb 2018, so the futures
    leg has no modern coverage. Both legs are reported with their own
    availability; a reading never implies coverage it does not have.

    Consequence worth stating: the two legs BARELY OVERLAP (2016-07 to 2018-02).
    S14 is effectively leg (b) before 2016 and leg (a) after 2018. That is a
    structural discontinuity, not a resolution change, and any walk-forward
    evaluation spanning it must treat the eras separately -- the same caution
    recorded for S2 at its 2024 instrument switch.
"""

from __future__ import annotations
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pit import series_asof, align, staleness_bdays  # noqa: E402

SIGNAL_ID = "S14"
LAYER = "L3"
MAX_STALENESS_DAYS = 3
PERSISTENCE_DAYS = 5
AMBER_STATE = "Y"                 # DECISIONS.md D1

EQUITY_SERIES = "SPY_CLOSE"       # proxy for SPX, labelled in every reading
FUT_FRONT = "VX_FRONT"
FUT_SECOND = "VX_SECOND"

RV_WINDOW = 21                    # registry: "21d RV"
RV_LOOKBACK = 504                 # registry: "top quartile of 2y"
RV_QUARTILE = 75.0                # top quartile
RV_PERSIST = 10                   # registry: ">=10d persistence"
DMA_WINDOW = 200                  # registry: "SPX < 200DMA"
FUT_INVERT_DAYS = 5               # registry: "front > second >= 5d"


def _returns(closes):
    out = []
    for i in range(1, len(closes)):
        p0, p1 = closes[i - 1][1], closes[i][1]
        if p0:
            out.append((closes[i][0], (p1 - p0) / p0))
    return out


def _realized_vol(rets, window=RV_WINDOW):
    """Annualized realized vol over a trailing window, per date."""
    out = []
    for i in range(window, len(rets) + 1):
        w = [r for _, r in rets[i - window:i]]
        m = sum(w) / len(w)
        var = sum((x - m) ** 2 for x in w) / (len(w) - 1)
        out.append((rets[i - 1][0], math.sqrt(var * 252)))
    return out


def _pctile(window, value):
    return 100.0 * sum(1 for v in window if v <= value) / len(window)


def leg_a(con, asof):
    """Realized-vol regime: RV in the top quartile of 2y for >=10d, index < 200DMA."""
    closes = series_asof(con, EQUITY_SERIES, asof)
    need = DMA_WINDOW + RV_LOOKBACK + RV_WINDOW
    if len(closes) < need:
        return None, {"reason": f"leg (a) needs {need} obs of {EQUITY_SERIES}, "
                                f"have {len(closes)}"}
    rets = _returns(closes)
    rv = _realized_vol(rets)
    if len(rv) < RV_LOOKBACK + RV_PERSIST:
        return None, {"reason": f"leg (a) needs {RV_LOOKBACK + RV_PERSIST} RV obs, "
                                f"have {len(rv)}"}

    # consecutive days ending now with RV in the top quartile of its trailing 2y
    run = 0
    for i in range(len(rv) - 1, RV_LOOKBACK - 2, -1):
        w = [v for _, v in rv[i - RV_LOOKBACK + 1:i + 1]]
        if _pctile(w, rv[i][1]) >= RV_QUARTILE:
            run += 1
        else:
            break

    last_px = closes[-1][1]
    dma = sum(v for _, v in closes[-DMA_WINDOW:]) / DMA_WINDOW
    below = last_px < dma
    fired = run >= RV_PERSIST and below
    return fired, {
        "rv_annualized": round(rv[-1][1], 4),
        "rv_pctile_2y": round(_pctile([v for _, v in rv[-RV_LOOKBACK:]],
                                      rv[-1][1]), 1),
        "top_quartile_run_days": run, "needs_days": RV_PERSIST,
        "price": round(last_px, 2), "dma200": round(dma, 2),
        "last_obs": closes[-1][0],
        "below_200dma": below, "fired": fired,
        "source": EQUITY_SERIES + " (PROXY for SPX; 2016-07-18+)",
    }


def leg_b(con, asof):
    """Futures curve inverted: front > second for >= 5 consecutive sessions."""
    f = series_asof(con, FUT_FRONT, asof)
    s = series_asof(con, FUT_SECOND, asof)
    if not f or not s:
        return None, {"reason": f"leg (b) needs {FUT_FRONT}/{FUT_SECOND}; "
                                f"have {len(f)}/{len(s)} obs"}
    joined = align(f, s)
    if len(joined) < FUT_INVERT_DAYS:
        return None, {"reason": f"leg (b) needs {FUT_INVERT_DAYS} overlapping obs, "
                                f"have {len(joined)}"}

    # STALENESS IS PER LEG. series_asof returns the last row published on or
    # before asof, so once CFE coverage ended (2018-02-23) leg (b) happily
    # reported that day's curve for every later date -- identical numbers at
    # 2020-03-20, 2022-06-16 and 2026-08-28, an eight-year-old reading dressed
    # as current. The signal-level staleness check missed it because it took the
    # MINIMUM across series and fresh SPY_CLOSE masked a stale VX_FRONT.
    fs = staleness_bdays(con, FUT_FRONT, asof)
    if fs is None or fs > MAX_STALENESS_DAYS:
        return None, {"reason": f"leg (b) stale: last {FUT_FRONT} obs "
                                f"{joined[-1][0]}, {fs} business days before "
                                f"{asof} (limit {MAX_STALENESS_DAYS}). CFE "
                                f"archive coverage ends 2018-02-23.",
                      "last_obs": joined[-1][0], "stale_days": fs}
    run = 0
    for _, fr, se in reversed(joined):
        if fr > se:
            run += 1
        else:
            break
    fired = run >= FUT_INVERT_DAYS
    return fired, {
        "front": round(joined[-1][1], 2), "second": round(joined[-1][2], 2),
        "inverted": joined[-1][1] > joined[-1][2],
        "inverted_run_days": run, "needs_days": FUT_INVERT_DAYS,
        "fired": fired, "last_obs": joined[-1][0],
        "source": "CFE, D13-normalized; coverage 2004-03-26..2018-02-23",
    }


def compute(con, asof):
    a, da = leg_a(con, asof)
    b, db = leg_b(con, asof)

    if a is None and b is None:
        return _na(asof, f"neither leg available -- (a) {da.get('reason')}; "
                         f"(b) {db.get('reason')}")

    fired = [x for x in (a, b) if x is True]
    avail = [n for n, x in (("a", a), ("b", b)) if x is not None]
    if len(fired) == 2:
        state = "R"                    # both conditions -> regime confirmed
    elif fired:
        state = AMBER_STATE            # one leg
    else:
        state = "G"

    # Staleness of the legs actually in use. Taking the minimum across all
    # candidate series would let one fresh feed vouch for a dead one.
    used = []
    if a is not None:
        used.append(EQUITY_SERIES)
    if b is not None:
        used.append(FUT_FRONT)
    stale = None
    for sname in used:
        d = staleness_bdays(con, sname, asof)
        if d is not None:
            stale = d if stale is None else max(stale, d)

    return {
        "signal_id": SIGNAL_ID, "layer": LAYER, "asof": str(asof),
        "state": state,
        "raw_value": da.get("rv_annualized") if a is not None else None,
        "zscore": None,
        "stale": (stale is None or stale > MAX_STALENESS_DAYS),
        "stale_days": stale,
        "persistence_days": PERSISTENCE_DAYS,
        "source_asof": (db.get("last_obs") if b is not None
                        else (da.get("last_obs") if a is not None else None)),
        "detail": {
            "legs_available": avail, "legs_fired": len(fired),
            "leg_a_rv_regime": da, "leg_b_futures_inversion": db,
            "note": "legs are never averaged; either arms, both fire "
                    "(see module docstring)",
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
