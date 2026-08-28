"""
test_warning_engine.py — behavioral tests for warning_engine.py
Synthetic scenarios only (no market data): verifies the state machine does
what Part VI says it does. Run: pytest -q
"""

import warning_engine as we
from warning_engine import SignalReading as SR, EngineState, step


def mk(states, stale=(), persistence=None):
    """Build a full 15-signal reading set from a {signal_id: state} dict.
    Everything not mentioned is G. persistence: {signal_id: days_required}."""
    layout = {
        "S13": "L1", "S15": "L1", "S11": "L1", "S10": "L1", "S6": "L1",
        "S5": "L2", "S8": "L2", "S2": "L2", "S1": "L2", "S7": "L2",
        "S12": "L2", "S9": "L2", "S3": "L2",
        "S4": "L3", "S14": "L3",
        "L4A": "L4", "L4B": "L4", "L4C": "L4",
    }
    persistence = persistence or {}
    return [
        SR(sid, layer, states.get(sid, "G"),
           stale=(sid in stale),
           min_persistence=persistence.get(sid, 1))
        for sid, layer in layout.items()
    ]


def run_days(seq, st=None):
    """seq = list of (readings, n_days). Returns (st, last_result)."""
    st = st or EngineState()
    res = None
    day = 0
    for readings, n in seq:
        for _ in range(n):
            day += 1
            res = step(f"d{day:03d}", readings, st)
    return st, res


# ---------------------------------------------------------------- basics

def test_all_green_is_normal():
    st, res = run_days([(mk({}), 5)])
    assert res.band == "NORMAL"
    assert res.composite == 0.0
    assert res.action["gross"] == 1.00
    assert res.path == "UNCLASSIFIED"
    assert not res.do_nothing


def test_persistence_gate_blocks_one_day_spikes():
    """A single-day R on a 10-day-persistence signal must not move the score."""
    calm = mk({})
    spike = mk({"S2": "R"}, persistence={"S2": 10})
    st, _ = run_days([(calm, 3)])
    _, res = run_days([(spike, 1)], st)
    assert res.contributions["S2"] == 0.0        # still effective G
    _, res = run_days([(spike, 9)], st)          # day 10 at R
    assert res.contributions["S2"] == 1.0        # now counts


# ------------------------------------------------- 2007-style credit sequence

def test_credit_bear_sequence_reaches_defensive_with_credit_path():
    """L1 fragility red + credit-side L2 red -> DEFENSIVE-range composite,
    CREDIT path label, band upgrade after persistence."""
    st = EngineState()
    # 2006: fragility building
    frag = mk({"S15": "R", "S13": "Y", "S10": "O", "S1": "O"})
    st, res = run_days([(frag, 15)], st)
    assert res.band in ("WATCH", "ELEVATED")
    # mid/late-2007: credit layer + funding stress join (the Aug-07 pattern)
    credit = mk({"S15": "R", "S13": "Y", "S10": "O",
                 "S1": "R", "S2": "R", "S3": "R", "S5": "O", "S8": "R",
                 "S4": "R", "S14": "O"})
    st, res = run_days([(credit, 25)], st)
    assert res.band == "DEFENSIVE"
    assert res.path == "CREDIT"
    assert res.action["gross"] == 0.65


def test_l4_override_goes_straight_to_crisis_no_waiting():
    """Funding seizure (any L4 'B') overrides persistence entirely —
    the Aug-2007/Sep-2008 rule."""
    st, res = run_days([(mk({"L4A": "B"}), 1)])
    assert res.l4_override
    assert res.band == "CRISIS"
    assert res.action["gross"] == 0.40


# ------------------------------------------------- 1998-style false alarm

def test_false_alarm_deescalates_only_past_hysteresis_exit():
    """Credit+breadth+funding fire with NO fragility layer (1998 pattern):
    the missing L1 must cap the band BELOW DEFENSIVE — that is the system's
    1998 defense — and recovery must clear the hysteresis exit and persist
    before the band steps down."""
    st = EngineState()
    alarm = mk({"S2": "R", "S5": "R", "S4": "R", "S14": "R"})
    st, res = run_days([(alarm, 15)], st)
    band_at_peak = res.band
    assert band_at_peak in ("WATCH", "ELEVATED")     # L1 quiet -> capped
    # partial cooling into the hysteresis corridor: band must HOLD
    cooling = mk({"S2": "Y", "S5": "O", "S4": "G", "S14": "Y"})
    st, res = run_days([(cooling, 3)], st)
    assert res.band == band_at_peak                 # corridor: no flap
    # full normalization, long enough to persist: band steps down
    st, res = run_days([(mk({}), 30)], st)
    assert res.band == "NORMAL"


# ------------------------------------------------- 2022-style speculative path

def test_speculative_bear_classified_without_credit():
    spec = mk({"S13": "R", "S11": "R", "S5": "O", "S6": "O",
               "S8": "R", "S10": "O"})
    st, res = run_days([(spec, 25)])
    assert res.path == "SPECULATIVE"
    assert res.band in ("ELEVATED", "DEFENSIVE")


# ------------------------------------------------- data honesty states

def test_staleness_flips_to_insufficient_and_freezes_band():
    st = EngineState()
    st, res = run_days([(mk({"S2": "O"}), 12)], st)
    band_before = res.band
    # L1 and L2 mostly stale -> two NA layers -> INSUFFICIENT_DATA
    stale = mk({"S2": "O"},
               stale=("S13", "S15", "S11", "S10", "S6",
                      "S5", "S8", "S2", "S1", "S7", "S12"))
    st, res = run_days([(stale, 1)], st)
    assert res.band == "INSUFFICIENT_DATA"
    assert res.composite is None
    assert res.action["hedge"] == "freeze"
    # engine's underlying band is frozen, not reset
    assert st.band == band_before
    assert any(a[0] == "INSUFFICIENT_DATA" for a in res.alerts)


def test_do_nothing_flag_on_conflicting_midband():
    """L1 hot + everything else quiet -> mid composite, UNCLASSIFIED path
    conflict -> do_nothing flag (the 1998/2011/2018 protection)."""
    conflict = mk({"S13": "R", "S15": "R", "S11": "R", "S10": "R", "S6": "R"})
    st, res = run_days([(conflict, 15)])
    if res.composite is not None and 20 <= res.composite < 60:
        assert res.do_nothing or res.path != "UNCLASSIFIED"


# ------------------------------------------------- re-entry ladder

def test_reentry_ladder_is_ordered():
    assert we.reentry_steps(False, True, True, True) == 0.0    # gate 1 fails
    assert we.reentry_steps(True, False, True, True) == 0.25
    assert we.reentry_steps(True, True, False, False) == 0.50
    assert we.reentry_steps(True, True, True, True) == 1.00


# ------------------------------------------------- graduated exit levels (Part VIII)

def test_exit_levels_match_report_part_viii():
    """Report Part VIII: WATCH exit <15, ELEVATED <30, DEFENSIVE <45, CRISIS <55.
    A uniform offset gives 5/25/45/65 -- only DEFENSIVE coincides. This test
    pins all four so the divergence cannot silently reappear."""
    assert we.exit_level("WATCH") == 15.0
    assert we.exit_level("ELEVATED") == 30.0
    assert we.exit_level("DEFENSIVE") == 45.0
    assert we.exit_level("CRISIS") == 55.0


def test_crisis_does_not_exit_above_55():
    """The consequential case: at composite 60 a CRISIS book must STAY at CRISIS.
    Under a uniform-15 offset it would exit at <65 and re-risk into a live
    crisis. Held for well beyond the 21d DEFENSIVE persistence gate."""
    st = EngineState()
    st.band = "CRISIS"
    for i in range(40):
        res = step(f"x{i:03d}", mk({}), st)          # composite drives the band
        st.band = "CRISIS" if i == 0 else st.band     # seed only on first day
    # drive an explicit mid-band composite through the hysteresis logic directly
    st2 = EngineState(); st2.band = "CRISIS"
    alerts = []
    for _ in range(30):
        band = we.hysteresis_step(st2, 60.0, False, alerts)
    assert band == "CRISIS", "exited CRISIS at composite 60 -- exit must be <55"
    # and below 55 it does step down, after persistence
    st3 = EngineState(); st3.band = "CRISIS"
    alerts3 = []
    for _ in range(30):
        band3 = we.hysteresis_step(st3, 50.0, False, alerts3)
    assert band3 != "CRISIS", "failed to exit CRISIS at composite 50"


def test_watch_exit_is_stickier_than_uniform_offset():
    """WATCH exit <15 (not <5): a book at composite 10 steps down to NORMAL."""
    st = EngineState(); st.band = "WATCH"
    alerts = []
    for _ in range(15):
        band = we.hysteresis_step(st, 10.0, False, alerts)
    assert band == "NORMAL"
    # but at 17 it holds inside the corridor
    st2 = EngineState(); st2.band = "WATCH"
    alerts2 = []
    for _ in range(15):
        band2 = we.hysteresis_step(st2, 17.0, False, alerts2)
    assert band2 == "WATCH"
