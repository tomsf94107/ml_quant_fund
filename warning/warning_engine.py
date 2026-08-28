"""
warning_engine.py — runnable core of the crash early-warning system.

This is the REAL implementation of the Part VI state machine from
Crash_Warning_Forensics_2000_2008.md: persistence filtering, layer
aggregation with staleness coverage, composite scoring, hysteresis band
transitions, path classification, INSUFFICIENT_DATA and DO_NOTHING states,
and the action mapping. No network, no data vendor calls — you feed it
signal readings (your builder computes those per signal_registry.csv),
it returns auditable state.

Tested by test_warning_engine.py (synthetic 2007-style credit sequence,
1998-style false alarm, staleness, hysteresis). Run: pytest -q
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional

# ---------------------------------------------------------------------------
# Configuration (Part VI values; frozen — change only with a version bump)
# ---------------------------------------------------------------------------

LAYER_WEIGHTS = {"L1": 0.25, "L2": 0.35, "L3": 0.15, "L4": 0.25}
L1_BOOST_WHEN_L2 = 1.3          # L1 weight multiplier when L2 >= 0.5
BANDS = [                        # (name, entry floor)
    ("NORMAL", 0.0),
    ("WATCH", 20.0),
    ("ELEVATED", 40.0),
    ("DEFENSIVE", 60.0),
    ("CRISIS", 80.0),
]
EXIT_OFFSET = 15.0               # exit only 15 pts below the entry floor
PERSIST_DAYS_DEFAULT = 10        # consecutive days before a band change sticks
PERSIST_DAYS_DEFENSIVE = 21      # stricter for DEFENSIVE and above
STALE_COVERAGE_MIN = 0.70        # layer coverage below this -> layer NA
SATURATION_K = {"L1": 3, "L2": 3, "L3": 2, "L4": 2}
# ^ evidence saturation: K weight-units of fully-red signals make the layer
#   fully red; further reds add nothing. Chosen so realistic episodes (never
#   all 15 firing at once) map to the report's band anchors: quiet layers
#   still drag the composite (that is the point — 1998's missing L1 is what
#   caps it below DEFENSIVE), but a layer doesn't need unanimity to scream.
NA_LAYER_LIMIT = 1               # more NA layers than this -> INSUFFICIENT_DATA
DO_NOTHING_GAP = 0.5             # max layer disagreement inside 20-59 band
PATH_MARGIN = 0.2                # classifier margin

CREDIT_PATH_SIGNALS = {"S15", "S3", "S2", "S4"}
SPEC_PATH_SIGNALS = {"S11", "S13", "S5", "S6", "S8", "S10"}

ACTION_TABLE = {
    "NORMAL":            {"gross": 1.00, "hedge": "none", "carry_bps_mo": 0},
    "WATCH":             {"gross": 1.00, "hedge": "precompute_tickets", "carry_bps_mo": 0},
    "ELEVATED":          {"gross": 0.85, "hedge": "put_spread_3m_5_15_25pct", "carry_bps_mo": 25},
    "DEFENSIVE":         {"gross": 0.65, "hedge": "puts_6m_10otm_50pct_plus_collars", "carry_bps_mo": 60},
    "CRISIS":            {"gross": 0.40, "hedge": "maintain_ladder_no_new_shorts", "carry_bps_mo": 100},
    "INSUFFICIENT_DATA": {"gross": None, "hedge": "freeze", "carry_bps_mo": 0},
}

STATE_SCORE = {"G": 0.0, "Y": 0.33, "O": 0.66, "R": 1.0, "B": 1.0, "NA": None}


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------

@dataclass
class SignalReading:
    """One signal on one as-of date, already computed by your builder."""
    signal_id: str
    layer: str                    # 'L1' | 'L2' | 'L3' | 'L4'
    state: str                    # 'G'|'Y'|'O'|'R'|'B'|'NA'
    weight: float = 1.0           # within-layer weight
    stale: bool = False
    min_persistence: int = 1      # trading days state must hold to count


@dataclass
class EngineState:
    """Carried between days (persist to composite_scores/state table)."""
    band: str = "NORMAL"
    candidate_band: Optional[str] = None
    candidate_days: int = 0
    # signal_id -> (state, consecutive days at that state)
    persistence: dict = field(default_factory=dict)


@dataclass
class DayResult:
    asof: str
    layer_scores: dict            # L -> float | None
    layer_coverage: dict          # L -> float
    composite: Optional[float]   # 0-100 | None if insufficient
    band: str
    path: str                     # SPECULATIVE | CREDIT | UNCLASSIFIED
    do_nothing: bool
    l4_override: bool
    action: dict
    contributions: dict           # signal_id -> effective score used
    alerts: list


# ---------------------------------------------------------------------------
# Core steps
# ---------------------------------------------------------------------------

def apply_persistence(readings: list[SignalReading], st: EngineState) -> dict:
    """R4: a signal's state counts only after min_persistence consecutive days.
    Until then the last persistent state (or 'G') is used. NA passes through.
    Returns signal_id -> effective_state, and updates st.persistence in place."""
    effective = {}
    for r in readings:
        if r.state == "NA":
            effective[r.signal_id] = "NA"
            continue
        prev_state, prev_days, prev_effective = st.persistence.get(
            r.signal_id, (r.state, 0, "G"))
        days = prev_days + 1 if r.state == prev_state else 1
        eff = r.state if days >= r.min_persistence else prev_effective
        st.persistence[r.signal_id] = (r.state, days, eff)
        effective[r.signal_id] = eff
    return effective


def layer_score(readings, effective, layer):
    """Saturating evidence sum over non-stale, non-NA signals in the layer:
    score = min(1, sum(w_i * state_i) / SATURATION_K[layer]).
    Returns (score|None, coverage). Coverage = usable weight / total weight."""
    total_w = usable_w = acc = 0.0
    for r in readings:
        if r.layer != layer:
            continue
        total_w += r.weight
        eff = effective[r.signal_id]
        if r.stale or eff == "NA":
            continue
        usable_w += r.weight
        acc += r.weight * STATE_SCORE[eff]
    if total_w == 0:
        return None, 0.0
    coverage = usable_w / total_w
    if coverage < STALE_COVERAGE_MIN:
        return None, coverage
    return min(1.0, acc / SATURATION_K[layer]), coverage


def classify_path(readings, effective):
    """CREDIT vs SPECULATIVE vs UNCLASSIFIED by mean state score of each set."""
    def mean_of(ids):
        vals = [STATE_SCORE[effective[r.signal_id]]
                for r in readings
                if r.signal_id in ids and effective[r.signal_id] != "NA"
                and not r.stale]
        return sum(vals) / len(vals) if vals else None
    credit, spec = mean_of(CREDIT_PATH_SIGNALS), mean_of(SPEC_PATH_SIGNALS)
    if credit is None or spec is None:
        return "UNCLASSIFIED"
    if credit - spec > PATH_MARGIN:
        return "CREDIT"
    if spec - credit > PATH_MARGIN:
        return "SPECULATIVE"
    return "UNCLASSIFIED"


def l4_propagation_red(readings, effective) -> bool:
    """L4 override: any L4 signal at B, or >=2 L4 signals at R."""
    b = r_count = 0
    for r in readings:
        if r.layer != "L4" or r.stale:
            continue
        eff = effective[r.signal_id]
        if eff == "B":
            b += 1
        elif eff == "R":
            r_count += 1
    return b >= 1 or r_count >= 2


def band_floor(name):
    return dict(BANDS)[name]


def target_band(composite):
    tb = BANDS[0][0]
    for name, floor in BANDS:
        if composite >= floor:
            tb = name
    return tb


def hysteresis_step(st: EngineState, composite: Optional[float],
                    l4_red: bool, alerts: list) -> str:
    """R5: enter at floor; exit only 15 below floor; persistence gate.
    L4 red overrides straight to CRISIS (no persistence wait)."""
    if l4_red:
        if st.band != "CRISIS":
            alerts.append(("BAND_UP", st.band, "CRISIS", "L4 propagation override"))
        st.band, st.candidate_band, st.candidate_days = "CRISIS", None, 0
        return st.band
    if composite is None:                      # insufficient data: freeze band
        st.candidate_band, st.candidate_days = None, 0
        return st.band
    cur = st.band
    tgt = target_band(composite)
    order = [b for b, _ in BANDS]
    if order.index(tgt) > order.index(cur):
        candidate = tgt                        # upgrade path
    elif composite < band_floor(cur) - EXIT_OFFSET:
        candidate = tgt                        # downgrade only past exit level
    else:
        candidate = cur                        # inside hysteresis corridor
    if candidate == cur:
        st.candidate_band, st.candidate_days = None, 0
        return cur
    if st.candidate_band == candidate:
        st.candidate_days += 1
    else:
        st.candidate_band, st.candidate_days = candidate, 1
    need = (PERSIST_DAYS_DEFENSIVE
            if candidate in ("DEFENSIVE", "CRISIS") else PERSIST_DAYS_DEFAULT)
    if st.candidate_days >= need:
        alerts.append((
            "BAND_UP" if order.index(candidate) > order.index(cur) else "BAND_DOWN",
            cur, candidate, f"persisted {st.candidate_days}d"))
        st.band, st.candidate_band, st.candidate_days = candidate, None, 0
    return st.band


def step(asof: str, readings: list[SignalReading], st: EngineState) -> DayResult:
    """One daily evaluation. Mutates st (carry it forward day to day)."""
    alerts: list = []
    effective = apply_persistence(readings, st)

    scores, coverage = {}, {}
    for L in LAYER_WEIGHTS:
        scores[L], coverage[L] = layer_score(readings, effective, L)
        if scores[L] is None and coverage[L] < STALE_COVERAGE_MIN:
            alerts.append(("LAYER_NA", L, None, f"coverage {coverage[L]:.0%}"))

    na_layers = [L for L, s in scores.items() if s is None]
    insufficient = len(na_layers) > NA_LAYER_LIMIT

    composite = None
    if not insufficient:
        w = dict(LAYER_WEIGHTS)
        if scores.get("L2") is not None and scores["L2"] >= 0.5:
            w["L1"] = w["L1"] * L1_BOOST_WHEN_L2
        used = {L: s for L, s in scores.items() if s is not None}
        wsum = sum(w[L] for L in used)
        composite = 100.0 * sum(w[L] * used[L] for L in used) / wsum

    l4_red = l4_propagation_red(readings, effective)
    path = classify_path(readings, effective)

    if insufficient:
        band = st.band                      # frozen
        alerts.append(("INSUFFICIENT_DATA", st.band, st.band,
                       f"NA layers: {na_layers}"))
        action = dict(ACTION_TABLE["INSUFFICIENT_DATA"])
    else:
        band = hysteresis_step(st, composite, l4_red, alerts)
        action = dict(ACTION_TABLE[band])

    do_nothing = (composite is not None and 20 <= composite < 60
                  and path == "UNCLASSIFIED"
                  and _max_layer_gap(scores) > DO_NOTHING_GAP)
    if do_nothing:
        alerts.append(("DO_NOTHING_CONFLICT", band, band,
                       "layers disagree; hold state"))

    contributions = {r.signal_id: STATE_SCORE.get(effective[r.signal_id])
                     for r in readings}
    return DayResult(asof, scores, coverage,
                     None if insufficient else composite,
                     "INSUFFICIENT_DATA" if insufficient else band,
                     path, do_nothing, l4_red, action, contributions, alerts)


def _max_layer_gap(scores):
    vals = [s for s in scores.values() if s is not None]
    return (max(vals) - min(vals)) if len(vals) >= 2 else 0.0


# ---------------------------------------------------------------------------
# Re-entry ladder (from CRISIS) — evaluated by the caller with market inputs
# ---------------------------------------------------------------------------

def reentry_steps(funding_ok_21d: bool, spreads_off_peak_25pct: bool,
                  breadth_thrust: bool, above_200dma_21d: bool) -> float:
    """Returns fraction of gross to restore (25% per satisfied step, in order:
    a later step only counts if all earlier ones hold — the ladder is ordered)."""
    steps = [funding_ok_21d, spreads_off_peak_25pct, breadth_thrust,
             above_200dma_21d]
    frac = 0.0
    for ok in steps:
        if not ok:
            break
        frac += 0.25
    return frac
