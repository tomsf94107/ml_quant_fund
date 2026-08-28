"""
s4_funding.py — builder for S4, Funding stress.

REGISTRY ROW:
    id S4 | layer L3 | tier shortlist | persistence 5 | max_staleness 3
    formula  hist: TED z>2 (1y) AND >100bp for >=5d
             modern: F10 composite (CP-Tbill; SOFR-IORB; ABCP 4wk delta)
    series   TEDRATE(1986-2022); ABCOMP; CP series TBD
    arm      z>1.5          red: z>2 & level
    direction funding_stress_bearish

WHY S4 MATTERS DISPROPORTIONATELY
    It is the trigger layer of the four-layer design -- "rupture triggers them"
    (report line 41) -- and the one signal that led the 2007 peak from inside the
    funding system rather than from price. It is also L4A's precondition:
    "funding seizure (S4 red + breadth-of-stress across >=2 funding markets)".

TWO MODES, NEVER MIXED (each reading declares its own)
    historic  TEDRATE, 1986-01 .. 2022-01. DISCONTINUED with LIBOR; a missing
              value after 2022-01 is not staleness, it is the end of the series.
    modern    mean z of the available funding legs:
                CP - Tbill   RIFSPPFAAD90NB - DTB3   (1997+)   [OPEN ITEM 1]
                SOFR - IORB  SOFR - IORB             (2021-07+)
                ABCP 4wk d   ABCOMP 28-day change    (2001+, REVISABLE)
              Legs with no data are omitted and the count is reported; the
              composite is never computed from fewer than MIN_MODERN_LEGS.

D11 -- threshold_red's "& level" has no modern equivalent
    The historic rule is explicit: z>2 AND the TED level above 100bp for >=5 days.
    The modern composite is a mean of z-scores and has no natural basis-point
    level, and the registry does not supply one. Modern red therefore uses the
    z>2 condition alone, sustained over the same 5 days. This is a narrowing of
    the rule, not a loosening: it drops a conjunct rather than adding one. It
    means modern red is easier to reach than historic red, which is a real
    asymmetry between the two eras and must be carried into any evaluation that
    spans 2022.
"""

from __future__ import annotations
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pit import series_asof, align, staleness_bdays  # noqa: E402

SIGNAL_ID = "S4"
LAYER = "L3"
MAX_STALENESS_DAYS = 3
PERSISTENCE_DAYS = 5
AMBER_STATE = "Y"                 # DECISIONS.md D1

Z_WINDOW = 252                    # registry: "z ... (1y)"
ARM_Z = 1.5                       # registry: "z>1.5"
RED_Z = 2.0                       # registry: "z>2"
RED_LEVEL_PP = 1.00               # registry: ">100bp", historic mode only
RED_SUSTAIN_DAYS = 5              # registry: "for >=5d"
MIN_MODERN_LEGS = 2               # never composite a single funding market
ABCP_DELTA_DAYS = 28              # registry: "ABCP 4wk delta"


def _z(values):
    """z of the last value against the trailing window. None if degenerate."""
    w = values[-Z_WINDOW:]
    n = len(w)
    if n < Z_WINDOW:
        return None
    mu = sum(w) / n
    var = sum((x - mu) ** 2 for x in w) / (n - 1)
    sd = var ** 0.5
    return None if sd == 0 else (w[-1] - mu) / sd


def _series_z(rows):
    return _z([v for _, v in rows]) if rows else None


def compute(con, asof, mode: str = "auto"):
    ted = series_asof(con, "TEDRATE", asof)
    ted_stale = staleness_bdays(con, "TEDRATE", asof)
    ted_usable = bool(ted) and ted_stale is not None and ted_stale <= MAX_STALENESS_DAYS
    use_hist = (mode == "historic") or (mode == "auto" and ted_usable)

    if use_hist:
        return _historic(con, asof, ted, ted_stale)
    return _modern(con, asof)


def _historic(con, asof, ted, stale_days):
    if len(ted) < Z_WINDOW:
        return _na(asof, f"historic mode needs {Z_WINDOW} TEDRATE obs, "
                         f"have {len(ted)}", "historic")
    vals = [v for _, v in ted]
    z = _z(vals)
    if z is None:
        return _na(asof, "TEDRATE z undefined (zero variance)", "historic")

    level = vals[-1]
    # ">100bp for >=5d" -- the LEVEL must hold, per the registry's wording
    sustained = all(v > RED_LEVEL_PP for v in vals[-RED_SUSTAIN_DAYS:])
    if z > RED_Z and sustained:
        state = "R"
    elif z > ARM_Z:
        state = AMBER_STATE
    else:
        state = "G"
    return _out(asof, state, level, z, stale_days, {
        "mode": "historic", "series": "TEDRATE",
        "ted_pp": round(level, 4), "z": round(z, 2),
        f"above_{int(RED_LEVEL_PP*100)}bp_for_{RED_SUSTAIN_DAYS}d": sustained,
        "legs": None,
    }, ted[-1][0])


def _modern(con, asof):
    legs, detail_legs = {}, {}

    cp = series_asof(con, "RIFSPPFAAD90NB", asof)
    tb = series_asof(con, "DTB3", asof)
    if cp and tb:
        joined = align(cp, tb)
        if len(joined) >= Z_WINDOW:
            spread = [(d, a - b) for d, a, b in joined]
            z = _series_z(spread)
            if z is not None:
                legs["cp_tbill"] = z
                detail_legs["cp_tbill"] = {"level_pp": round(spread[-1][1], 4),
                                           "z": round(z, 2), "asof": spread[-1][0]}

    sofr = series_asof(con, "SOFR", asof)
    iorb = series_asof(con, "IORB", asof)
    if sofr and iorb:
        joined = align(sofr, iorb)
        if len(joined) >= Z_WINDOW:
            spread = [(d, a - b) for d, a, b in joined]
            z = _series_z(spread)
            if z is not None:
                legs["sofr_iorb"] = z
                detail_legs["sofr_iorb"] = {"level_pp": round(spread[-1][1], 4),
                                            "z": round(z, 2), "asof": spread[-1][0]}

    ab = series_asof(con, "ABCOMP", asof)
    if len(ab) > ABCP_DELTA_DAYS:
        deltas = [(ab[i][0], ab[i][1] - ab[i - ABCP_DELTA_DAYS][1])
                  for i in range(ABCP_DELTA_DAYS, len(ab))]
        if len(deltas) >= Z_WINDOW:
            z = _series_z(deltas)
            if z is not None:
                # ABCP CONTRACTION is the stress direction: a shrinking market is
                # funding withdrawal. Sign flipped so every leg points the same way.
                legs["abcp_4wk"] = -z
                detail_legs["abcp_4wk"] = {"delta": round(deltas[-1][1], 2),
                                           "z_raw": round(z, 2),
                                           "z_stress": round(-z, 2),
                                           "asof": deltas[-1][0],
                                           "note": "sign flipped: contraction = stress"}

    if len(legs) < MIN_MODERN_LEGS:
        return _na(asof, f"modern mode needs >={MIN_MODERN_LEGS} funding legs with "
                         f"{Z_WINDOW} obs, have {len(legs)} ({sorted(legs) or 'none'})",
                   "modern")

    z = sum(legs.values()) / len(legs)
    if z > RED_Z:
        state = "R"          # D11: no modern "& level" conjunct exists
    elif z > ARM_Z:
        state = AMBER_STATE
    else:
        state = "G"
    return _out(asof, state, z, z, 0, {
        "mode": "modern", "series": "CP-Tbill; SOFR-IORB; ABCP 4wk delta",
        "composite_z": round(z, 2), "n_legs": len(legs), "legs": detail_legs,
        "d11": "modern red is z>2 alone; the historic '& level >100bp' conjunct "
               "has no modern equivalent in the registry",
    }, max((v["asof"] for v in detail_legs.values()), default=None))


def _out(asof, state, raw, z, stale_days, detail, src):
    return {"signal_id": SIGNAL_ID, "layer": LAYER, "asof": str(asof),
            "state": state, "raw_value": raw, "zscore": z,
            "stale": (stale_days is None or stale_days > MAX_STALENESS_DAYS),
            "stale_days": stale_days, "persistence_days": PERSISTENCE_DAYS,
            "source_asof": src, "detail": detail}


def _na(asof, reason, mode):
    return {"signal_id": SIGNAL_ID, "layer": LAYER, "asof": str(asof), "state": "NA",
            "raw_value": None, "zscore": None, "stale": True, "stale_days": None,
            "persistence_days": PERSISTENCE_DAYS, "source_asof": None,
            "detail": {"reason": reason, "mode": mode}}


def to_reading(result):
    from warning_engine import SignalReading
    return SignalReading(signal_id=result["signal_id"], layer=result["layer"],
                         state=result["state"], stale=bool(result["stale"]),
                         min_persistence=result["persistence_days"])
