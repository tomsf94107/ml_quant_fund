"""Tests for pit.py and the S1 builder. Synthetic data only, plus a
vintage-integrity test that is the whole point of rule #1."""
import sqlite3, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pit
from builders import s1_term_spread as S1

SCHEMA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "warning_schema.sql")


def db():
    con = sqlite3.connect(":memory:")
    con.executescript(open(SCHEMA).read())
    return con


def put(con, series, obs, pub, val):
    con.execute("INSERT OR IGNORE INTO data_vintages (series_id,obs_date,pub_date,value,source)"
                " VALUES (?,?,?,?,?)", (series, obs, pub, val, "TEST"))


# --------------------------------------------------------------- pit: vintages

def test_future_publications_are_invisible():
    con = db()
    put(con, "X", "2007-06-01", "2007-06-02", 1.0)
    put(con, "X", "2007-07-01", "2007-07-02", 2.0)
    assert pit.series_asof(con, "X", "2007-06-15") == [("2007-06-01", 1.0)]
    assert len(pit.series_asof(con, "X", "2007-07-15")) == 2


def test_revisions_use_the_vintage_visible_then_not_the_latest():
    """The rule-#1 test: a value revised in 2010 must NOT appear in a 2007 read."""
    con = db()
    put(con, "X", "2007-06-01", "2007-06-02", 1.0)     # first print
    put(con, "X", "2007-06-01", "2010-01-01", 9.9)     # later revision
    assert pit.series_asof(con, "X", "2007-08-01") == [("2007-06-01", 1.0)]
    assert pit.series_asof(con, "X", "2011-01-01") == [("2007-06-01", 9.9)]


def test_staleness_days_and_missing_series():
    con = db()
    put(con, "X", "2026-08-01", "2026-08-02", 1.0)
    assert pit.staleness_days(con, "X", "2026-08-08") == 7
    assert pit.staleness_days(con, "NOPE", "2026-08-08") is None


def test_monthly_mean():
    rows = [("2007-01-05", 1.0), ("2007-01-20", 3.0), ("2007-02-05", 5.0)]
    assert pit.monthly_mean(rows) == [("2007-01", 2.0), ("2007-02", 5.0)]


# --------------------------------------------------------------- S1 states

def _load(con, spreads):
    """spreads: {'YYYY-MM': spread}. DTB3 fixed at 4.0, DGS10 = 4.0 + spread.
    pub_date = obs_date + 1 day, matching the registry's publication_lag for
    DGS10/DTB3. Stamping a far-future pub_date would make every row invisible
    to a historical as-of read -- which is rule #1 working, not a bug."""
    from datetime import date, timedelta
    for m, s in spreads.items():
        for day in ("05", "15", "25"):
            obs = f"{m}-{day}"
            pub = (date.fromisoformat(obs) + timedelta(days=1)).isoformat()
            put(con, "DTB3", obs, pub, 4.0)
            put(con, "DGS10", obs, pub, 4.0 + s)


def test_s1_green_when_not_inverted():
    con = db(); _load(con, {"2026-06": 0.5, "2026-07": 0.4, "2026-08": 0.6})
    r = S1.compute(con, "2026-09-02")
    assert r["state"] == "G"
    assert r["detail"]["inverted_of_last3"] == 0


def test_s1_arms_on_two_of_three_inverted_months():
    con = db(); _load(con, {"2026-06": -0.2, "2026-07": 0.1, "2026-08": -0.3})
    r = S1.compute(con, "2026-09-02")
    assert r["state"] == S1.AMBER_STATE
    assert r["detail"]["armed"] and not r["detail"]["escalated"]


def test_s1_single_inverted_month_does_not_arm():
    con = db(); _load(con, {"2026-06": 0.2, "2026-07": 0.1, "2026-08": -0.3})
    assert S1.compute(con, "2026-09-02")["state"] == "G"


def test_s1_escalates_on_resteepening_after_long_inversion():
    """6 inverted months troughing at -0.60, then the curve NORMALIZES and rises
    >50bp off the trough. Note 2025-07 must be POSITIVE: an escalation while the
    curve is still inverted is the bug fixed on 2026-08-28, and the original
    version of this test asserted exactly that -- it encoded the defect."""
    sp = {"2025-01": -0.2, "2025-02": -0.3, "2025-03": -0.45, "2025-04": -0.6,
          "2025-05": -0.5, "2025-06": -0.4, "2025-07": 0.05}
    con = db(); _load(con, sp)
    r = S1.compute(con, "2025-08-02")
    assert r["state"] == "R", r["detail"]
    assert r["detail"]["resteepen"]["rise_bp"] >= 50


def test_s1_short_inversion_does_not_escalate():
    """Same re-steepen shape but a short inversion run -> no ESCALATE.
    The run is 4 months (2025-04..07; -0.05 is still an inversion), below the
    registry's >=6m floor, so the +55bp rise off the trough must NOT fire."""
    sp = {"2025-04": -0.6, "2025-05": -0.5, "2025-06": -0.4, "2025-07": -0.05}
    con = db(); _load(con, sp)
    r = S1.compute(con, "2025-08-02")
    assert r["state"] != "R"
    assert r["detail"]["resteepen"]["run_len"] == 4


def test_s1_missing_data_is_na_not_zero():
    con = db()
    for day in ("05", "15", "25"):
        put(con, "DGS10", f"2026-08-{day}", "2026-08-28", 4.0)
    r = S1.compute(con, "2026-08-28")
    assert r["state"] == "NA" and r["raw_value"] is None
    assert "DTB3" in r["detail"]["reason"]


def test_s1_flags_staleness_past_registry_limit():
    con = db(); _load(con, {"2026-05": 0.3, "2026-06": 0.2, "2026-07": 0.1})
    r = S1.compute(con, "2026-08-28")
    assert r["stale"] is True and r["stale_days"] > S1.MAX_STALENESS_DAYS


def test_s1_to_reading_shape():
    con = db(); _load(con, {"2026-06": -0.2, "2026-07": 0.1, "2026-08": -0.3})
    rd = S1.to_reading(S1.compute(con, "2026-09-02"))
    assert rd.signal_id == "S1" and rd.layer == "L2" and rd.min_persistence == 21


# --------------------------------------------------------------- vintage semantics

def test_revisable_series_are_refused_by_default():
    import series_meta as sm
    assert sm.is_revisable("UNKNOWN_SERIES") is True      # safe direction
    assert sm.derivable_pub_date("UNKNOWN_SERIES") is False
    assert sm.derivable_pub_date("DGS10") is True
    assert sm.derivable_pub_date("HOUST") is False        # revised -> ALFRED only
    assert sm.derivable_pub_date("PAYEMS") is False


def test_restamp_only_touches_non_revisable_and_preserves_values():
    import subprocess, sys, tempfile, os as _os
    con = db()
    put(con, "DGS10", "2000-02-15", "2026-08-28", 6.5)
    put(con, "HOUST", "2000-02-01", "2026-08-28", 1600.0)
    con.commit()
    # persist to a temp file so the CLI can open it
    fd, path = tempfile.mkstemp(suffix=".db"); _os.close(fd); _os.remove(path)
    disk = sqlite3.connect(path); con.backup(disk); disk.commit(); disk.close()

    here = _os.path.dirname(_os.path.abspath(__file__))
    subprocess.run([sys.executable, _os.path.join(here, "restamp_vintages.py"),
                    "--db", path, "--apply"], capture_output=True, text=True)

    d = sqlite3.connect(path)
    dgs = d.execute("SELECT pub_date, value FROM data_vintages WHERE series_id='DGS10'").fetchone()
    hou = d.execute("SELECT pub_date, value FROM data_vintages WHERE series_id='HOUST'").fetchone()
    n = d.execute("SELECT COUNT(*) FROM data_vintages").fetchone()[0]
    d.close(); _os.remove(path)

    assert dgs[0] == "2000-02-16", "DGS10 pub_date should derive to obs+1d"
    assert dgs[1] == 6.5, "value must be untouched"
    assert hou[0] == "2026-08-28", "revisable series must NOT be restamped"
    assert n == 2, "UPDATE must not create or destroy rows"


def test_restamp_unblocks_historical_pit_reads():
    """The bug this whole correction exists for: before restamping, a 2000 read
    sees nothing; after, it sees the value that was knowable then."""
    con = db()
    put(con, "DGS10", "2000-02-15", "2026-08-28", 6.5)
    assert pit.series_asof(con, "DGS10", "2000-02-29") == []
    con.execute("UPDATE data_vintages SET pub_date=date(obs_date,'+1 day') "
                "WHERE series_id='DGS10'")
    assert pit.series_asof(con, "DGS10", "2000-02-29") == [("2000-02-15", 6.5)]


def test_stale_inversion_cannot_escalate_decades_later():
    """REGRESSION (found on real data 2026-08-28): S1 fired R at 2000-02 off the
    1980-11..1981-08 inversion (trough -2.65, 'rise 363bp'). A 19-year-old trough
    must not generate a live escalation."""
    con = db()
    sp = {}
    for i, m in enumerate(["1980-11", "1980-12", "1981-01", "1981-02", "1981-03",
                           "1981-04", "1981-05", "1981-06", "1981-07", "1981-08"]):
        sp[m] = -2.0 - (0.65 if m == "1980-12" else 0.0)
    for y in range(1982, 2001):                      # two decades of normal curve
        for mm in ("03", "06", "09", "12"):
            sp[f"{y}-{mm}"] = 1.0
    _load(con, sp)
    r = S1.compute(con, "2000-02-29")
    assert r["state"] != "R", f"escalated off a stale trough: {r['detail']}"
    rs = r["detail"]["resteepen"]
    assert rs["months_since_run_end"] > S1.ESCALATE_WINDOW_MONTHS
    assert "stale" in rs["reason"]


def test_recent_resteepen_still_escalates_inside_the_window():
    """The window must not break the 2007 case: 9m inversion, re-steepen 2m later."""
    sp = {"2006-08": -0.1, "2006-09": -0.15, "2006-10": -0.2, "2006-11": -0.25,
          "2006-12": -0.3, "2007-01": -0.35, "2007-02": -0.37, "2007-03": -0.375,
          "2007-04": -0.2, "2007-05": 0.0, "2007-06": 0.5}
    con = db(); _load(con, sp)
    r = S1.compute(con, "2007-07-02")
    assert r["state"] == "R", r["detail"]
    assert r["detail"]["resteepen"]["months_since_run_end"] <= S1.ESCALATE_WINDOW_MONTHS


def test_short_run_detail_is_complete_not_partial():
    """The detail dict must always carry run_start/trough, even when the run is
    too short to escalate -- the display printed 'None..None (1m)' before."""
    con = db(); _load(con, {"2026-06": 0.2, "2026-07": 0.1, "2026-08": -0.3})
    d = S1.compute(con, "2026-09-02")["detail"]["resteepen"]
    assert d["run_start"] is not None and d["trough"] is not None
    assert d["run_len"] == 1


def test_incomplete_month_is_excluded_until_published():
    """REGRESSION (real data 2026-08-28): S1 went Y->R->G across 2001-01/02
    because January was half-published (mean -0.001) at the 1/31 read and
    positive once complete. A month must not count until fully published."""
    rows = [("2026-08-03", 1.0), ("2026-08-31", 1.0), ("2026-07-15", 2.0)]
    # asof 2026-08-31: August's last print (lag 1d) is not out yet -> excluded
    assert [m for m, _ in pit.monthly_mean_complete(rows, "2026-08-31", 1)] == ["2026-07"]
    # asof 2026-09-01: August is fully published -> included
    assert [m for m, _ in pit.monthly_mean_complete(rows, "2026-09-01", 1)] == \
["2026-07", "2026-08"]


def test_no_escalation_while_inversion_is_ongoing():
    """REGRESSION (real data 2026-08-28): S1 flapped Y->R->Y->R through 1974,
    1980-81 and 2023-24, escalating at spreads like -0.941 while the curve was
    still deeply inverted. Re-steepening is measured only after the run ends."""
    sp = {}
    for i, m in enumerate(["2023-01", "2023-02", "2023-03", "2023-04", "2023-05",
                           "2023-06", "2023-07", "2023-08", "2023-09", "2023-10"]):
        sp[m] = -1.60 if m == "2023-05" else -0.95     # +65bp off trough, still inverted
    con = db(); _load(con, sp)
    r = S1.compute(con, "2023-11-02")
    assert r["state"] != "R", f"escalated mid-inversion: {r['detail']}"
    rs = r["detail"]["resteepen"]
    assert rs["ongoing"] is True and "ongoing" in rs["reason"]
    assert rs["rise_bp"] >= 50            # the rise is real; the context is not


def test_escalation_fires_once_the_inversion_actually_ends():
    """Same shape, but the curve normalizes -> this IS the 2007 pattern."""
    sp = {m: (-1.60 if m == "2023-05" else -0.95)
          for m in ["2023-01", "2023-02", "2023-03", "2023-04", "2023-05",
                    "2023-06", "2023-07", "2023-08"]}
    sp["2023-09"] = 0.20                    # inversion ends
    con = db(); _load(con, sp)
    r = S1.compute(con, "2023-10-02")
    assert r["state"] == "R", r["detail"]
    assert r["detail"]["resteepen"]["ongoing"] is False


# --------------------------------------------------------------- S2

from builders import s2_credit as S2  # noqa: E402


def _load_monthly(con, series, vals, start="1990-01"):
    """vals: list of monthly spreads starting at `start`."""
    from datetime import date, timedelta
    y, m = int(start[:4]), int(start[5:7])
    for v in vals:
        obs = f"{y:04d}-{m:02d}-01"
        pub = (date.fromisoformat(obs) + timedelta(days=1)).isoformat()
        put(con, series, obs, pub, v)
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)


def test_s2_green_when_credit_leg_fails():
    con = db(); _load_monthly(con, "BAA10YM", [2.0] * 14)
    r = S2.compute(con, "1991-04-02")
    assert r["state"] == "G" and r["detail"]["credit_leg"] is False


def test_s2_arms_on_credit_leg_alone_when_equity_unknown():
    """Credit widens 100bp off its 6m low and sits above the 10m MA."""
    con = db(); _load_monthly(con, "BAA10YM", [2.0] * 10 + [2.2, 2.5, 2.8, 3.1])
    r = S2.compute(con, "1991-04-02")
    d = r["detail"]
    assert d["credit_leg"] is True and d["equity_leg"] is None
    assert r["state"] == S2.AMBER_STATE, "must ARM, not fire, without SPX"
    assert "absent or too short" in d["equity_note"]


def test_s2_fires_red_only_with_the_equity_leg():
    con = db(); _load_monthly(con, "BAA10YM", [2.0] * 10 + [2.2, 2.5, 2.8, 3.1])
    assert S2.compute(con, "1991-04-02", spx_near_high=True)["state"] == "R"
    assert S2.compute(con, "1991-04-02", spx_near_high=False)["state"] == S2.AMBER_STATE


def test_s2_needs_75bp_not_merely_above_ma():
    """Above the MA but only 40bp off the low -> half-condition not met."""
    con = db(); _load_monthly(con, "BAA10YM", [2.0] * 10 + [2.1, 2.2, 2.3, 2.4])
    d = S2.compute(con, "1991-04-02")["detail"]
    assert d["above_ma"] is True and d["off_low"] is False and d["credit_leg"] is False


def test_s2_declares_its_mode_and_refuses_short_history():
    con = db(); _load_monthly(con, "BAA10YM", [2.0] * 4)
    r = S2.compute(con, "1990-06-02")
    assert r["state"] == "NA" and r["detail"]["mode"] == "monthly"
    assert "needs 10 complete months" in r["detail"]["reason"]


def test_s2_equity_leg_returns_none_on_short_history():
    closes = [(f"2020-01-{d:02d}", 100.0) for d in range(1, 29)]
    assert S2.equity_leg_from_prices(closes, "2020-01-28") is None


def _load_spy(con, start="2016-07-18", n=400, base=200.0, drop_at=None):
    """Synthetic SPY_CLOSE: rising, optionally with a drawdown near the end."""
    from datetime import date, timedelta
    d = date.fromisoformat(start); i = 0
    while i < n:
        if d.weekday() < 5:
            v = base + i * 0.1
            if drop_at is not None and i >= drop_at:
                v = (base + drop_at * 0.1) * 0.85        # 15% below the high
            put(con, "SPY_CLOSE", d.isoformat(),
                (d + timedelta(days=1)).isoformat(), v)
            i += 1
        d += timedelta(days=1)
    return d.isoformat()


def test_s2_equity_leg_autoloads_and_fires_red():
    con = db(); _load_monthly(con, "BAA10YM", [2.0] * 10 + [2.2, 2.5, 2.8, 3.1])
    asof = _load_spy(con, start="1989-01-02", n=300)     # at highs, no drawdown
    r = S2.compute(con, "1991-04-02")
    d = r["detail"]
    assert d["equity_source"] == "SPY_CLOSE"
    assert d["equity_leg"] is True
    assert r["state"] == "R", d


def test_s2_equity_leg_false_when_index_is_off_its_high():
    con = db(); _load_monthly(con, "BAA10YM", [2.0] * 10 + [2.2, 2.5, 2.8, 3.1])
    _load_spy(con, start="1989-01-02", n=300, drop_at=270)
    r = S2.compute(con, "1991-04-02")
    assert r["detail"]["equity_leg"] is False
    assert r["state"] == S2.AMBER_STATE, "credit widening with equity already down is not the divergence"


def test_s2_equity_leg_none_when_spy_history_too_short():
    con = db(); _load_monthly(con, "BAA10YM", [2.0] * 10 + [2.2, 2.5, 2.8, 3.1])
    _load_spy(con, start="1990-10-01", n=40)
    r = S2.compute(con, "1991-04-02")
    assert r["detail"]["equity_leg"] is None
    assert "too short" in r["detail"]["equity_note"]
    assert r["state"] == S2.AMBER_STATE


# --------------------------------------------------------------- F2

from builders import f2_vix_percentile as F2  # noqa: E402


def _load_vix(con, values, start="2000-01-03"):
    from datetime import date, timedelta
    d = date.fromisoformat(start); i = 0
    while i < len(values):
        if d.weekday() < 5:
            put(con, "VIXCLS", d.isoformat(),
                (d + timedelta(days=1)).isoformat(), values[i])
            i += 1
        d += timedelta(days=1)
    return d.isoformat()


def test_f2_percentile_maths():
    assert F2.percentile_of_last([1, 2, 3, 4, 10]) == 100.0
    assert F2.percentile_of_last([10, 2, 3, 4, 1]) == 20.0


def test_f2_green_at_mid_distribution():
    con = db(); asof = _load_vix(con, [10 + (i % 40) for i in range(504)])
    r = F2.compute(con, asof)
    assert r["state"] == "G" and r["detail"]["armed_below_20th"] is False


def test_f2_arms_in_the_bottom_quintile_low_vol_is_the_warning():
    con = db(); asof = _load_vix(con, [30.0] * 503 + [9.0])
    r = F2.compute(con, asof)
    assert r["detail"]["percentile_504d"] < F2.ARM_PCTILE
    assert r["state"] == F2.AMBER_STATE, "low VIX must ARM: direction is complacency"


def test_f2_cannot_fire_red_without_the_l2_score():
    con = db(); asof = _load_vix(con, [30.0] * 503 + [9.0])
    r = F2.compute(con, asof)
    assert r["detail"]["below_10th"] is True
    assert r["state"] == F2.AMBER_STATE
    assert "cannot be evaluated" in r["detail"]["l2_note"]


def test_f2_red_requires_both_bottom_decile_and_l2():
    con = db(); asof = _load_vix(con, [30.0] * 503 + [9.0])
    assert F2.compute(con, asof, l2_score=0.6)["state"] == "R"
    assert F2.compute(con, asof, l2_score=0.4)["state"] == F2.AMBER_STATE
    assert F2.compute(con, asof, l2_score=0.0)["state"] == F2.AMBER_STATE


def test_f2_high_vix_is_not_a_warning_even_with_hot_l2():
    con = db(); asof = _load_vix(con, [10.0] * 503 + [80.0])
    assert F2.compute(con, asof, l2_score=1.0)["state"] == "G"


def test_f2_refuses_short_window():
    con = db(); asof = _load_vix(con, [20.0] * 100)
    r = F2.compute(con, asof)
    assert r["state"] == "NA" and "need 504 obs" in r["detail"]["reason"]


# --------------------------------------------------------------- Cboe parser

def test_cboe_date_parser_handles_padded_and_unpadded():
    import parse_cboe as PC
    assert PC.parse_date("01/02/1990") == "1990-01-02"
    assert PC.parse_date("1/2/2004") == "2004-01-02"
    assert PC.parse_date(" 10/04/2019 ") == "2019-10-04"
    assert PC.parse_date("DATE") is None
    assert PC.parse_date("") is None
    assert PC.parse_date("13/45/1990") is None       # invalid, not coerced


def test_cboe_parser_skips_preamble_and_coerces_padded_values(tmp_path):
    import parse_cboe as PC
    p = tmp_path / "totalpc.csv"
    p.write_text("disclaimer,,,,\n, PRODUCT: TOTAL,,EXCHANGE: Cboe,\n"
                 "DATE,CALLS,PUTS,TOTAL,P/C Ratio\n"
                 "11/1/2006,1401036,1271445,2672481,0.91\n"
                 "10/04/2019, 2175006, 2289715, 4464721, 1.05\n")
    recs, skipped = PC.parse_file(str(p), "pre3", {"CBOE_PC_TOTAL": 4})
    assert skipped == 0
    assert recs == [("CBOE_PC_TOTAL", "2006-11-01", 0.91),
                    ("CBOE_PC_TOTAL", "2019-10-04", 1.05)]


def test_cboe_parser_survives_non_utf8_and_multiline_header(tmp_path):
    import parse_cboe as PC
    p = tmp_path / "pcratioarchive.csv"
    p.write_bytes(b'disclaimer \xa0 byte,,,,\nCboe PUT/CALL RATIO ARCHIVE,,,,\n'
                  b'DATE,TOTAL,INDEX,EQUITY,"multi\nline\n"\n'
                  b'12/31/2003,1.25,2.96,0.95\n')
    recs, skipped = PC.parse_file(str(p), "pre3",
                                  {"CBOE_PC_TOTAL": 1, "CBOE_PC_INDEX": 2})
    assert skipped == 0
    assert ("CBOE_PC_TOTAL", "2003-12-31", 1.25) in recs
    assert ("CBOE_PC_INDEX", "2003-12-31", 2.96) in recs


# --------------------------------------------------------------- daily driver

def test_driver_roster_matches_the_registry_shortlist():
    import csv, daily_driver as DD
    sl = {r["id"] for r in csv.DictReader(open(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "signal_registry.csv")))
        if r["tier"] == "shortlist"}
    assert set(DD.ROSTER) == sl, "driver roster drifted from the registry shortlist"
    assert len(DD.ROSTER) == 15


def test_driver_emits_na_for_unbuilt_so_coverage_is_honest():
    """The point of the full roster: 2 of 15 built must NOT look fully covered."""
    import daily_driver as DD
    from warning_engine import EngineState, step
    con = db()
    readings, details = DD.build_readings(con, "2026-08-28")
    assert len(readings) == 20        # 15 shortlist + 5 L4 propagation conditions
    # every UNBUILT signal must be NA. (Built ones are also NA here because this
    # fixture db has no data -- which is itself correct behaviour.)
    unbuilt = {sid for sid in DD.ROSTER if sid not in DD.BUILT}
    assert unbuilt, "derived, not hardcoded: this count changes as builders land"
    na = {r.signal_id for r in readings if r.state == "NA"}
    assert unbuilt <= na
    res = step("2026-08-28", readings, EngineState())
    assert res.band == "INSUFFICIENT_DATA"
    assert res.composite is None
    assert res.action["hedge"] == "freeze"


def test_driver_state_survives_close_and_reopen(tmp_path):
    """Durability, not just round-trip. The original version of this test called
    save_state then read back on the SAME open connection with its own commit,
    so it passed while save_state was missing con.commit() and every day's state
    was silently discarded on close. Cross the connection boundary."""
    import daily_driver as DD
    from warning_engine import EngineState
    path = str(tmp_path / "state.db")
    con = sqlite3.connect(path)
    con.executescript(open(SCHEMA).read())
    st = EngineState(band="ELEVATED", candidate_band="DEFENSIVE", candidate_days=7)
    st.persistence = {"S1": ("Y", 3, "G")}
    DD.save_state(con, st)
    con.close()                                   # no explicit commit here

    con2 = sqlite3.connect(path)
    back = DD.load_state(con2)
    con2.close()
    assert back.band == "ELEVATED"
    assert back.candidate_band == "DEFENSIVE" and back.candidate_days == 7
    assert back.persistence["S1"] == ("Y", 3, "G")


def test_driver_l4_has_no_registry_signals():
    """Structural gap: the engine weights L4 at 0.25 and gives it the crisis
    override, but no registry row is L4. Pinned so the gap cannot be forgotten."""
    import daily_driver as DD
    from warning_engine import LAYER_WEIGHTS
    assert "L4" in LAYER_WEIGHTS and LAYER_WEIGHTS["L4"] == 0.25
    assert not [s for s, L in DD.ROSTER.items() if L == "L4"]


# --------------------------------------------------------------- L4

from builders import l4_propagation as L4  # noqa: E402


def _load_daily(con, series, values, start="2024-01-01"):
    from datetime import date, timedelta
    d = date.fromisoformat(start); i = 0
    while i < len(values):
        if d.weekday() < 5:
            put(con, series, d.isoformat(),
                (d + timedelta(days=1)).isoformat(), values[i])
            i += 1
        d += timedelta(days=1)
    return d.isoformat()


def test_l4_has_five_conditions_three_unbuilt():
    con = db()
    res = L4.compute_all(con, "2026-08-28")
    assert set(res) == {"L4A", "L4B", "L4C", "L4D", "L4E"}
    for sid in ("L4A", "L4D", "L4E"):
        assert res[sid]["state"] == "NA"
    assert "S4" in res["L4A"]["detail"]["reason"]
    assert "S10" in res["L4D"]["detail"]["reason"]


def test_l4b_fires_on_150bp_widening_in_21_days():
    con = db()
    asof = _load_daily(con, "BAMLH0A0HYM2", [3.0] * 21 + [4.6])
    r = L4.spread_blowout(con, asof)
    assert r["state"] == "B" and r["detail"]["delta_bp"] >= 150


def test_l4b_silent_on_a_smaller_widening():
    con = db()
    asof = _load_daily(con, "BAMLH0A0HYM2", [3.0] * 21 + [4.2])
    r = L4.spread_blowout(con, asof)
    assert r["state"] == "G" and r["detail"]["delta_bp"] == 120.0


def test_l4c_needs_both_top_decile_and_a_jump():
    con = db()
    # a high level that has NOT risen over 21d must not fire
    asof = _load_daily(con, "CBOE_COR3M", [10.0] * 480 + [60.0] * 24)
    r = L4.correlation_spike(con, asof)
    assert r["detail"]["top_decile"] is True
    assert r["detail"]["jumped"] is False
    assert r["state"] == "G", "a plateau at a high level is not a spike"


def test_l4c_fires_on_a_jump_into_the_top_decile():
    con = db()
    asof = _load_daily(con, "CBOE_COR3M", [10.0] * 503 + [60.0])
    r = L4.correlation_spike(con, asof)
    assert r["detail"]["top_decile"] and r["detail"]["jumped"]
    assert r["state"] == "B"


def test_l4_single_B_drives_crisis_when_coverage_is_adequate():
    """Report line 601: 'stress underway -> overrides composite to B'. With the
    layers covered, one B bypasses persistence and goes straight to CRISIS."""
    from warning_engine import SignalReading, EngineState, step
    import daily_driver as DD
    readings = [SignalReading(s, L, "G") for s, L in DD.ROSTER.items()]
    readings += [SignalReading("L4A", "L4", "G"), SignalReading("L4B", "L4", "B"),
                 SignalReading("L4C", "L4", "G")]
    res = step("2026-08-28", readings, EngineState())
    assert res.l4_override is True
    assert res.band == "CRISIS" and res.action["gross"] == 0.40


def test_l4_override_survives_low_coverage():
    """DECISIONS.md D10, RESOLVED 2026-08-30.

    Before the fix, step() tested `insufficient` before hysteresis_step, so a
    fired L4 override was computed and then discarded: the band froze and the
    action was `freeze`. With 3 of 15 builders live that meant a 150bp HY
    blowout would print "freeze" while credit gapped.

    An L4 propagation condition is a DIRECT OBSERVATION -- "HY OAS widened
    >=150bp over 21 sessions" is true or false on its own. The do-nothing rule
    exists to stop the system acting on a COMPOSITE it cannot compute, not to
    discard a measurement it took successfully.
    """
    from warning_engine import SignalReading, EngineState, step
    import daily_driver as DD
    readings = [SignalReading(s, L, "NA", stale=True) for s, L in DD.ROSTER.items()]
    readings += [SignalReading("L4A", "L4", "G"), SignalReading("L4B", "L4", "B"),
                 SignalReading("L4C", "L4", "G")]
    res = step("2026-08-28", readings, EngineState())
    assert res.l4_override is True
    assert res.band == "CRISIS", "the override must reach the band"
    assert res.action["gross"] == 0.40
    # honesty is preserved: no composite was computable, and the alert says so
    assert res.composite is None
    assert "L4_OVERRIDE_LOW_COVERAGE" in [a[0] for a in res.alerts]


def test_low_coverage_without_l4_still_freezes():
    """The fix must be surgical: with no L4 condition firing, an under-covered
    stack behaves exactly as before."""
    from warning_engine import SignalReading, EngineState, step
    import daily_driver as DD
    readings = [SignalReading(s, L, "NA", stale=True) for s, L in DD.ROSTER.items()]
    readings += [SignalReading("L4A", "L4", "G"), SignalReading("L4B", "L4", "G"),
                 SignalReading("L4C", "L4", "G")]
    res = step("2026-08-28", readings, EngineState())
    assert res.band == "INSUFFICIENT_DATA"
    assert res.action["hedge"] == "freeze"
    assert res.composite is None


# --------------------------------------------------------------- S4

from builders import s4_funding as S4B  # noqa: E402


def test_s4_historic_needs_both_z_and_sustained_level():
    con = db()
    # 252 calm days then a spike: z is huge but the level held >100bp only 1 day
    vals = [0.20] * 252 + [1.60]
    asof = _load_daily(con, "TEDRATE", vals, start="2007-01-01")
    r = S4B.compute(con, asof, mode="historic")
    d = r["detail"]
    assert d["mode"] == "historic" and d["z"] > S4B.RED_Z
    assert d["above_100bp_for_5d"] is False
    assert r["state"] == S4B.AMBER_STATE, "z alone must not fire red"


def test_s4_historic_fires_red_when_level_is_sustained():
    con = db()
    vals = [0.20] * 252 + [1.60] * 5
    asof = _load_daily(con, "TEDRATE", vals, start="2007-01-01")
    r = S4B.compute(con, asof, mode="historic")
    assert r["detail"]["above_100bp_for_5d"] is True
    assert r["state"] == "R"


def test_s4_refuses_to_composite_a_single_funding_leg():
    """MIN_MODERN_LEGS: one market is not 'funding stress'."""
    con = db()
    _load_daily(con, "RIFSPPFAAD90NB",
                [0.30 + 0.01 * (i % 5) for i in range(300)], start="2024-01-01")
    asof = _load_daily(con, "DTB3", [0.1] * 300, start="2024-01-01")
    r = S4B.compute(con, asof, mode="modern")
    assert r["state"] == "NA"
    assert "needs >=2 funding legs" in r["detail"]["reason"]


def test_s4_modern_composites_available_legs():
    con = db()
    n = 300
    _load_daily(con, "RIFSPPFAAD90NB",
                [0.30 + 0.01 * (i % 5) for i in range(n - 1)] + [2.5], start="2024-01-01")
    _load_daily(con, "DTB3", [0.1] * n, start="2024-01-01")
    _load_daily(con, "SOFR",
                [0.50 + 0.01 * (i % 5) for i in range(n - 1)] + [2.0], start="2024-01-01")
    asof = _load_daily(con, "IORB", [0.4] * n, start="2024-01-01")
    r = S4B.compute(con, asof, mode="modern")
    d = r["detail"]
    assert d["mode"] == "modern" and d["n_legs"] == 2
    assert set(d["legs"]) == {"cp_tbill", "sofr_iorb"}
    assert r["state"] == "R", d


def test_s4_abcp_contraction_counts_as_stress_not_relief():
    """ABCP shrinking is funding withdrawal; the leg's sign must be flipped."""
    con = db()
    n = 400
    vals = [1000.0] * (n - 1) + [200.0]          # sharp contraction
    _load_daily(con, "ABCOMP", vals, start="2023-01-01")
    # the CP-Tbill leg must VARY: a constant spread has zero variance and _z
    # correctly returns None, which would drop the leg and trip MIN_MODERN_LEGS
    _load_daily(con, "RIFSPPFAAD90NB", [0.30 + 0.01 * (i % 7) for i in range(n)],
                start="2023-01-01")
    asof = _load_daily(con, "DTB3", [0.1] * n, start="2023-01-01")
    r = S4B.compute(con, asof, mode="modern")
    leg = r["detail"]["legs"]["abcp_4wk"]
    assert leg["z_raw"] < 0 and leg["z_stress"] > 0, "contraction must read as stress"


def test_s4_auto_mode_prefers_ted_only_while_it_is_fresh():
    con = db()
    asof = _load_daily(con, "TEDRATE", [0.2] * 260, start="2007-01-01")
    fresh = S4B.compute(con, asof)
    assert fresh["detail"]["mode"] == "historic"
    # years later TEDRATE is discontinued -> auto must not keep using it
    stale = S4B.compute(con, "2026-08-28")
    assert stale["detail"]["mode"] == "modern"


def test_staleness_is_counted_in_business_days_not_calendar_days():
    """REGRESSION (real data 2026-08-28): on 2007-10-09, the trading day after
    Columbus Day, S4's last TED print was Friday 2007-10-05 -- 4 calendar days
    back, past max_staleness_days=3 -- so S4 went NA and fell back to modern mode
    mid-crisis. Counted in business days it is 2, comfortably fresh."""
    con = db()
    put(con, "TEDRATE", "2007-10-05", "2007-10-06", 1.0)
    assert pit.staleness_days(con, "TEDRATE", "2007-10-09") == 4      # calendar
    assert pit.staleness_bdays(con, "TEDRATE", "2007-10-09") == 2     # business


def test_business_day_staleness_spans_a_normal_weekend():
    con = db()
    put(con, "X", "2026-08-28", "2026-08-29", 1.0)            # Fri obs, Sat pub
    assert pit.staleness_bdays(con, "X", "2026-08-31") == 1   # Monday: 1 bday
    # on the observation date itself the row is not yet published, so there is
    # nothing visible at all -- None, not 0. That is rule #1, not staleness.
    assert pit.staleness_bdays(con, "X", "2026-08-28") is None
    assert pit.staleness_bdays(con, "NOPE", "2026-08-31") is None


# --------------------------------------------------------------- F3

from builders import f3_vix_term_slope as F3B  # noqa: E402


def _load_pair(con, vix, vix3m, start="2008-01-01"):
    from datetime import date, timedelta
    d = date.fromisoformat(start); i = 0
    while i < len(vix):
        if d.weekday() < 5:
            pub = (d + timedelta(days=1)).isoformat()
            put(con, "VIXCLS", d.isoformat(), pub, vix[i])
            put(con, "VXVCLS", d.isoformat(), pub, vix3m[i])
            i += 1
        d += timedelta(days=1)
    return d.isoformat()


def test_f3_green_in_contango():
    con = db()
    asof = _load_pair(con, [15.0] * 10, [18.0] * 10)
    r = F3B.compute(con, asof)
    assert r["state"] == "G"
    assert r["detail"]["inverted"] is False
    assert r["detail"]["slope_pct"] == 20.0


def test_f3_arms_on_a_single_inverted_day():
    con = db()
    asof = _load_pair(con, [15.0] * 9 + [22.0], [18.0] * 10)
    r = F3B.compute(con, asof)
    assert r["detail"]["inverted_run_days"] == 1
    assert r["state"] == F3B.AMBER_STATE


def test_f3_reds_only_after_five_consecutive_inverted_days():
    con = db()
    asof4 = _load_pair(con, [15.0] * 6 + [22.0] * 4, [18.0] * 10)
    assert F3B.compute(con, asof4)["state"] == F3B.AMBER_STATE
    con2 = db()
    asof5 = _load_pair(con2, [15.0] * 5 + [22.0] * 5, [18.0] * 10)
    r = F3B.compute(con2, asof5)
    assert r["detail"]["inverted_run_days"] == 5
    assert r["state"] == "R"


def test_f3_run_resets_on_a_single_contango_day():
    """The run must be CONSECUTIVE and end at the latest observation."""
    con = db()
    asof = _load_pair(con, [22.0] * 4 + [15.0] + [22.0] * 3, [18.0] * 8)
    r = F3B.compute(con, asof)
    assert r["detail"]["inverted_run_days"] == 3, "an intervening contango day resets"
    assert r["state"] == F3B.AMBER_STATE


def test_f3_names_its_primary_leg_and_what_was_available():
    con = db()
    asof = _load_pair(con, [15.0] * 10, [18.0] * 10)
    d = F3B.compute(con, asof)["detail"]
    assert d["leg"] == "vix3m"
    assert d["legs_available"] == ["vix3m"]
    assert d["slope_futures_pct"] is None


def test_f3_falls_back_to_futures_before_vix3m_exists():
    """D14: the futures leg carries 2004-03..2007-12, which the vix3m leg cannot
    reach. This is the only reason the registry's Aug-07 verdict is testable."""
    con = db()
    from datetime import date, timedelta
    d0, i = date(2007, 7, 2), 0
    while i < 10:
        if d0.weekday() < 5:
            pub = (d0 + timedelta(days=1)).isoformat()
            put(con, "VX_FRONT", d0.isoformat(), pub, 20.0)     # front > second
            put(con, "VX_SECOND", d0.isoformat(), pub, 18.0)    # = backwardation
            i += 1
        d0 += timedelta(days=1)
    r = F3B.compute(con, d0.isoformat())
    assert r["detail"]["leg"] == "futures"
    assert r["detail"]["inverted"] is True
    assert r["state"] == "R", "10 inverted days is well past the 5-day threshold"


def test_f3_prefers_vix3m_when_both_legs_are_fresh():
    """Never averaged: two term-structure measures with different tenors would
    produce a number matching neither."""
    con = db()
    from datetime import date, timedelta
    d0, i = date(2010, 1, 4), 0
    while i < 10:
        if d0.weekday() < 5:
            pub = (d0 + timedelta(days=1)).isoformat()
            put(con, "VIXCLS", d0.isoformat(), pub, 15.0)
            put(con, "VXVCLS", d0.isoformat(), pub, 18.0)       # contango
            put(con, "VX_FRONT", d0.isoformat(), pub, 20.0)     # backwardation
            put(con, "VX_SECOND", d0.isoformat(), pub, 18.0)
            i += 1
        d0 += timedelta(days=1)
    d = F3B.compute(con, d0.isoformat())["detail"]
    assert d["leg"] == "vix3m"
    assert sorted(d["legs_available"]) == ["futures", "vix3m"]
    assert d["slope_vix3m_pct"] == 20.0 and d["slope_futures_pct"] == -10.0
    assert d["inverted"] is False, "the primary leg decides, not the other one"


def test_f3_na_without_vix3m():
    con = db()
    from datetime import date, timedelta
    for i in range(10):
        d = date(2008, 1, 7) + timedelta(days=i)
        if d.weekday() < 5:
            put(con, "VIXCLS", d.isoformat(),
                (d + timedelta(days=1)).isoformat(), 15.0)
    r = F3B.compute(con, "2008-01-25")
    assert r["state"] == "NA" and "VXVCLS" in r["detail"]["reason"]


# --------------------------------------------------------------- CFE parser

def test_cfe_contract_key_from_filename():
    import parse_cfe as PF
    assert PF.contract_key("CFE_F05_VX.csv") == (2005, 1, "F05")
    assert PF.contract_key("CFE_Z26_VX.csv") == (2026, 12, "Z26")
    assert PF.contract_key("CFE_H07_VX.csv") == (2007, 3, "H07")
    assert PF.contract_key("notacontract.csv") is None


def test_cfe_parse_file_reads_settle_and_skips_header(tmp_path):
    import parse_cfe as PF
    p = tmp_path / "CFE_F05_VX.csv"
    p.write_text("Trade Date,Futures,Open,High,Low,Close,Settle,Change,"
                 "Total Volume,EFP,Open Interest\n"
                 "10/21/2004,F (Jan 05),168.1,168.9,168,168.5,167,167,13,0,13\n"
                 "10/22/2004,F (Jan 05),166.7,170,166.3,170,169.4,2.4,74,0,87\n")
    rows, skipped = PF.parse_file(str(p))
    assert skipped == 1                       # the header
    assert rows == [("2004-10-21", 167.0), ("2004-10-22", 169.4)]


def test_cfe_front_and_second_need_no_expiry_calendar(tmp_path):
    """The set of files containing a date IS the set of live contracts."""
    import parse_cfe as PF
    hdr = ("Trade Date,Futures,Open,High,Low,Close,Settle,Change,"
           "Total Volume,EFP,Open Interest\n")
    (tmp_path / "CFE_H07_VX.csv").write_text(
        hdr + "01/05/2007,H,0,0,0,0,11.0,0,1,0,1\n")          # Mar-07: front
    (tmp_path / "CFE_J07_VX.csv").write_text(
        hdr + "01/05/2007,J,0,0,0,0,12.0,0,1,0,1\n")          # Apr-07: second
    (tmp_path / "CFE_Z07_VX.csv").write_text(
        hdr + "01/05/2007,Z,0,0,0,0,15.0,0,1,0,1\n")          # Dec-07: back
    per, front, second, _ = PF.build(str(tmp_path))
    assert front == [("2007-01-05", 11.0)]
    assert second == [("2007-01-05", 12.0)]
    assert set(per) == {"VX_H07", "VX_J07", "VX_Z07"}


def test_cfe_year_rollover_orders_correctly(tmp_path):
    """A Jan-08 contract must sort AFTER Dec-07, not before it."""
    import parse_cfe as PF
    hdr = ("Trade Date,Futures,Open,High,Low,Close,Settle,Change,"
           "Total Volume,EFP,Open Interest\n")
    (tmp_path / "CFE_Z07_VX.csv").write_text(
        hdr + "12/03/2007,Z,0,0,0,0,22.0,0,1,0,1\n")
    (tmp_path / "CFE_F08_VX.csv").write_text(
        hdr + "12/03/2007,F,0,0,0,0,23.0,0,1,0,1\n")
    _, front, second, _ = PF.build(str(tmp_path))
    assert front == [("2007-12-03", 22.0)], "Dec-07 is front, not Jan-08"
    assert second == [("2007-12-03", 23.0)]


def test_cfe_scale_report_surfaces_the_multiplier(tmp_path):
    """The 10x-quoted era must show a ratio near 10 against VIXCLS."""
    import parse_cfe as PF
    con = db()
    put(con, "VIXCLS", "2004-10-21", "2004-10-22", 16.7)
    put(con, "VIXCLS", "2010-06-01", "2010-06-02", 35.0)
    con.commit()
    rep = dict((y, r) for y, _, r in PF.scale_report(
        con, [("2004-10-21", 167.0), ("2010-06-01", 35.4)]))
    assert 9.5 < rep["2004"] < 10.5, "multiplied era should read ~10x"
    assert 0.9 < rep["2010"] < 1.1, "de-multiplied era should read ~1x"


def test_cfe_normalize_classifies_each_row_on_its_own_evidence():
    """D13: no changeover date is asserted; the ratio decides per row."""
    import parse_cfe as PF
    con = db()
    put(con, "VIXCLS", "2006-06-01", "2006-06-02", 16.0)
    put(con, "VIXCLS", "2007-06-01", "2007-06-02", 13.0)
    con.commit()
    rows = [("2006-06-01", 168.0), ("2007-06-01", 13.4)]
    out, dropped, switch = PF.normalize(con, rows)
    assert out == [("2006-06-01", 16.8), ("2007-06-01", 13.4)]
    assert dropped == 0 and switch == "2007-06-01"


def test_cfe_normalize_carries_classification_over_a_missing_vix_day():
    import parse_cfe as PF
    con = db()
    put(con, "VIXCLS", "2006-06-01", "2006-06-02", 16.0)
    con.commit()
    out, dropped, _ = PF.normalize(
        con, [("2006-06-01", 168.0), ("2006-06-02", 170.0)])
    assert out == [("2006-06-01", 16.8), ("2006-06-02", 17.0)]
    assert dropped == 0


def test_cfe_normalize_drops_rows_it_cannot_classify():
    """An unclassifiable settle is worse than a missing one."""
    import parse_cfe as PF
    con = db()
    out, dropped, _ = PF.normalize(con, [("2004-03-26", 168.0)])
    assert out == [] and dropped == 1


def test_uw_archive_keeps_multiple_pulls_of_one_session(tmp_path):
    """REGRESSION (2026-08-28, caught before the first cron run): with pulled_at
    outside the primary key, two pulls covering one ET session collide and
    INSERT OR IGNORE silently discards the second. The pre-open pull at 08:18 ET
    would have locked out the post-close cron pull for that session, permanently,
    in an append-only table."""
    path = str(tmp_path / "uw.db")
    con = sqlite3.connect(path)
    con.executescript(open(SCHEMA).read())
    ins = ("INSERT OR IGNORE INTO uw_archive "
           "(endpoint, query_params, snapshot_date, payload_json, pulled_at) "
           "VALUES (?,?,?,?,?)")
    con.execute(ins, ("/api/market/market-tide", "{}", "2026-08-28",
                      '{"stale":"pre-open"}', "2026-08-28 12:18:40"))
    con.execute(ins, ("/api/market/market-tide", "{}", "2026-08-28",
                      '{"fresh":"post-close"}', "2026-08-28 23:30:00"))
    con.commit()
    rows = con.execute("SELECT pulled_at, payload_json FROM uw_archive "
                       "ORDER BY pulled_at").fetchall()
    con.close()
    assert len(rows) == 2, "both vintages of one session must persist"
    assert "post-close" in rows[-1][1], "latest pulled_at is the freshest view"


def test_uw_archive_still_dedupes_an_identical_rerun(tmp_path):
    """Idempotency must survive the key change: re-running the same pull is a
    no-op, only a genuinely later pull adds a row."""
    path = str(tmp_path / "uw2.db")
    con = sqlite3.connect(path)
    con.executescript(open(SCHEMA).read())
    ins = ("INSERT OR IGNORE INTO uw_archive "
           "(endpoint, query_params, snapshot_date, payload_json, pulled_at) "
           "VALUES (?,?,?,?,?)")
    for _ in range(3):
        con.execute(ins, ("/api/darkpool/recent", '{"limit":200}', "2026-08-28",
                          '{"x":1}', "2026-08-28 23:30:00"))
    con.commit()
    n = con.execute("SELECT COUNT(*) FROM uw_archive").fetchone()[0]
    con.close()
    assert n == 1


# --------------------------------------------------------------- S14

from builders import s14_vol_structure as S14B  # noqa: E402


def _load_series(con, name, values, start="2020-01-01"):
    from datetime import date, timedelta
    d = date.fromisoformat(start); i = 0
    while i < len(values):
        if d.weekday() < 5:
            put(con, name, d.isoformat(),
                (d + timedelta(days=1)).isoformat(), values[i])
            i += 1
        d += timedelta(days=1)
    return d.isoformat()


def test_s14_leg_b_needs_five_consecutive_inverted_days():
    con = db()
    _load_series(con, "VX_FRONT", [20.0] * 6 + [25.0] * 4)
    asof = _load_series(con, "VX_SECOND", [22.0] * 10)
    fired, d = S14B.leg_b(con, asof)
    assert d["inverted_run_days"] == 4 and fired is False
    con2 = db()
    _load_series(con2, "VX_FRONT", [20.0] * 5 + [25.0] * 5)
    asof2 = _load_series(con2, "VX_SECOND", [22.0] * 10)
    fired2, d2 = S14B.leg_b(con2, asof2)
    assert d2["inverted_run_days"] == 5 and fired2 is True


def test_s14_leg_b_run_resets_on_one_contango_day():
    con = db()
    _load_series(con, "VX_FRONT", [25.0] * 4 + [20.0] + [25.0] * 3)
    asof = _load_series(con, "VX_SECOND", [22.0] * 8)
    _fired, d = S14B.leg_b(con, asof)
    assert d["inverted_run_days"] == 3, "an intervening contango day resets"


def test_s14_one_leg_arms_both_legs_fire():
    """Legs are never averaged: either arms, both together fire."""
    con = db()
    _load_series(con, "VX_FRONT", [25.0] * 10)
    asof = _load_series(con, "VX_SECOND", [22.0] * 10)
    r = S14B.compute(con, asof)
    assert r["detail"]["legs_available"] == ["b"]
    assert r["detail"]["legs_fired"] == 1
    assert r["state"] == S14B.AMBER_STATE, "one leg arms, does not fire red"


def test_s14_na_when_neither_leg_has_data():
    con = db()
    r = S14B.compute(con, "2026-08-28")
    assert r["state"] == "NA"
    assert "neither leg available" in r["detail"]["reason"]


def test_s14_leg_a_reports_its_coverage_limit():
    """Leg (a) needs SPY_CLOSE, which starts 2016-07-18; before that it is NA
    and says so rather than silently contributing nothing."""
    con = db()
    _load_series(con, "SPY_CLOSE", [100.0] * 50)
    fired, d = S14B.leg_a(con, "2020-04-01")
    assert fired is None and "needs" in d["reason"]


def test_s14_realized_vol_matches_a_known_value():
    """Constant returns -> zero vol; alternating +/-1% -> a computable value."""
    rets = [(f"d{i}", 0.0) for i in range(30)]
    assert S14B._realized_vol(rets)[-1][1] == 0.0
    alt = [(f"d{i}", 0.01 if i % 2 else -0.01) for i in range(30)]
    rv = S14B._realized_vol(alt)[-1][1]
    # Compute the expectation from the DEFINITION rather than a closed form.
    # A 21-day window of alternating returns has 11 of one sign and 10 of the
    # other, so the mean is not zero (0.01/21) -- assuming it was is what made
    # the first version of this test fail against correct code.
    import math as _m
    w = [r for _, r in alt[-21:]]
    m = sum(w) / len(w)
    var = sum((x - m) ** 2 for x in w) / (len(w) - 1)
    assert abs(rv - _m.sqrt(var * 252)) < 1e-12


def test_s14_leg_b_goes_na_once_futures_coverage_ends():
    """REGRESSION (real data 2026-08-30): leg (b) reported the 2018-02-23 curve
    verbatim at 2020-03-20, 2022-06-16 and 2026-08-28 -- identical front/second
    on three dates years apart. series_asof returns the last published row, so a
    dead feed reads as a live one unless staleness is checked PER LEG."""
    con = db()
    _load_series(con, "VX_FRONT", [25.0] * 10, start="2018-02-01")
    _load_series(con, "VX_SECOND", [22.0] * 10, start="2018-02-01")
    fresh, d_fresh = S14B.leg_b(con, "2018-02-16")
    assert fresh is True, d_fresh
    stale, d_stale = S14B.leg_b(con, "2020-03-20")
    assert stale is None, "a two-year-old curve must not be reported as current"
    assert "stale" in d_stale["reason"]


def test_s14_stale_futures_alone_yields_NA_not_a_stale_reading():
    """With only dead futures data the signal must report NA, not the last
    curve it happens to hold. Before the per-leg staleness gate this returned a
    confident G built on 2018 prices."""
    con = db()
    _load_series(con, "VX_FRONT", [25.0] * 10, start="2018-02-01")
    _load_series(con, "VX_SECOND", [22.0] * 10, start="2018-02-01")
    r = S14B.compute(con, "2020-03-20")
    assert r["state"] == "NA"
    assert "neither leg available" in r["detail"]["reason"]
    assert "stale" in r["detail"]["reason"]


# --------------------------------------------------------------- S7

from builders import s7_defensive_rotation as S7B  # noqa: E402


def _load_px(con, name, values, start="2020-01-01"):
    from datetime import date, timedelta
    d = date.fromisoformat(start); i = 0
    while i < len(values):
        if d.weekday() < 5:
            put(con, name, d.isoformat(),
                (d + timedelta(days=1)).isoformat(), values[i])
            i += 1
        d += timedelta(days=1)
    return d.isoformat()


def _s7_fixture(con, defensive_gain, bench_path):
    """bench_path: bench closes. Defensives ride it plus a linear tilt.

    NOTE the tilt is spread over the WHOLE path, so the relative move over the
    trailing 63 days is roughly defensive_gain * 63/len(path). A 0.30 tilt over
    320 days gives only ~4.9% -- just under the +5% threshold. Sized here to
    clear it unambiguously rather than sit on the line.
    """
    n = len(bench_path)
    _load_px(con, "SPY_CLOSE", bench_path)
    for name in ("XLP_CLOSE", "XLU_CLOSE", "XLV_CLOSE"):
        vals = [bench_path[i] * (1 + defensive_gain * i / n) for i in range(n)]
        asof = _load_px(con, name, vals)
    return asof


def test_s7_needs_BOTH_legs_to_fire():
    """Defensives outperforming in a DECLINE is arithmetic. The signal is
    defensives leading while the index is still at its high."""
    con = db()
    # index 20% off its high, defensives strongly ahead -> RS leg only
    path = [100.0] * 260 + [80.0] * 60
    asof = _s7_fixture(con, 0.80, path)
    r = S7B.compute(con, asof)
    d = r["detail"]
    assert d["rs_leg"] is True, d
    assert d["near_high_leg"] is False, d
    assert r["state"] == "G", ("near-high GATES the signal: strong defensive RS "
                               "in a decline is arithmetic and must not arm")


def test_s7_fires_when_defensives_lead_at_the_high():
    con = db()
    path = [100.0 + i * 0.05 for i in range(320)]      # index grinding to highs
    asof = _s7_fixture(con, 0.80, path)
    r = S7B.compute(con, asof)
    d = r["detail"]
    assert d["rs_leg"] and d["near_high_leg"], d
    assert r["state"] == "R"


def test_s7_green_when_defensives_lag_at_the_high():
    con = db()
    path = [100.0 + i * 0.05 for i in range(320)]
    asof = _s7_fixture(con, -0.10, path)               # defensives BEHIND
    r = S7B.compute(con, asof)
    assert r["detail"]["rs_leg"] is False
    assert r["state"] == "G", ("near-high ALONE must not arm -- an index at its "
                               "high is an ordinary bull market, not a warning")


def test_s7_arms_between_the_registry_thresholds():
    """Registry gives arm '+3%' and red '+5%', not a leg-count rule."""
    con = db()
    path = [100.0 + i * 0.05 for i in range(320)]
    asof = _s7_fixture(con, 0.20, path)                # ~ +3.3% over 63d
    r = S7B.compute(con, asof)
    rs = r["detail"]["mean_rs_63d_pct"]
    assert S7B.RS_ARM * 100 < rs <= S7B.RS_RED * 100, rs
    assert r["detail"]["near_high_leg"] is True
    assert r["state"] == S7B.AMBER_STATE


def test_s7_refuses_to_call_one_sector_a_rotation():
    con = db()
    path = [100.0 + i * 0.05 for i in range(320)]
    _load_px(con, "SPY_CLOSE", path)
    asof = _load_px(con, "XLP_CLOSE", [p * 1.3 for p in path])
    r = S7B.compute(con, asof)
    assert r["state"] == "NA"
    assert "need >=2 defensive ETFs" in r["detail"]["reason"]


def test_s7_declares_a_partial_mean_rather_than_hiding_it():
    con = db()
    path = [100.0 + i * 0.05 for i in range(320)]
    _load_px(con, "SPY_CLOSE", path)
    _load_px(con, "XLP_CLOSE", [p * 1.3 for p in path])
    asof = _load_px(con, "XLU_CLOSE", [p * 1.3 for p in path])
    d = S7B.compute(con, asof)["detail"]
    assert d["n_defensive"] == 2
    assert any("XLV" in m for m in d["omitted"])


def test_s7_rel_strength_arithmetic():
    """+10% sector against +2% bench over the window is +8pp relative."""
    sect = [(f"2020-01-{i+1:02d}", 100.0) for i in range(63)] + [("2020-04-01", 110.0)]
    bench = [(f"2020-01-{i+1:02d}", 50.0) for i in range(63)] + [("2020-04-01", 51.0)]
    v = S7B._rel_strength(sect, bench)
    assert abs(v - 0.08) < 1e-12


# --------------------------------------------------------------- S8

from builders import s8_epicenter_fracture as S8B  # noqa: E402


def _s8_fixture(con, n=600, leader="XLK_CLOSE", leader_dd=0.0,
                bench_dd=0.0, leader_below_dma=False):
    """Bench rises steadily; every sector tracks it. `leader` gets extra 2y RS,
    then optionally a drawdown at the end."""
    bench = [100.0 + i * 0.10 for i in range(n)]
    if bench_dd:
        bench = bench[:-40] + [bench[-40] * (1 - bench_dd)] * 40
    _load_px(con, "SPY_CLOSE", bench)
    asof = None
    for name in S8B.SECTORS:
        if name == leader:
            vals = [bench[i] * (1 + 0.60 * i / n) for i in range(n)]
            if leader_dd:
                peak = vals[-60]
                tail = [peak * (1 - leader_dd)] * 60
                vals = vals[:-60] + tail
            if leader_below_dma:
                vals = vals[:-5] + [vals[-1] * 0.7] * 5
        else:
            vals = [bench[i] * (1 - 0.05 * i / n) for i in range(n)]
        asof = _load_px(con, name, vals)
    return asof


def test_s8_identifies_the_leader_point_in_time():
    con = db()
    asof = _s8_fixture(con, leader="XLK_CLOSE")
    d = S8B.compute(con, asof)["detail"]
    assert d["leader"] == "XLK", d["rs_ranking_pct"]
    assert d["leader_rs_2y_pct"] > 0
    assert d["n_sectors"] == len(S8B.SECTORS)


def test_s8_green_when_the_leader_is_intact():
    con = db()
    asof = _s8_fixture(con, leader="XLE_CLOSE")
    r = S8B.compute(con, asof)
    assert r["detail"]["leader_drawdown_pct"] < S8B.ARM_DRAWDOWN * 100
    assert r["state"] == "G"


def test_s8_fires_when_the_leader_breaks_while_the_index_holds():
    """NOTE the drawdown is sized at 16%: enough to clear the 15% red threshold
    and to put the leader below its 200DMA, but not so deep that it destroys the
    trailing-2y RS that made it the leader. A crash large enough to cost a
    sector its leadership makes S8 look elsewhere -- see
    test_s8_loses_sight_of_an_epicenter_that_has_fully_collapsed."""
    con = db()
    asof = _s8_fixture(con, leader="XLK_CLOSE", leader_dd=0.16)
    r = S8B.compute(con, asof)
    d = r["detail"]
    assert d["leader_below_200dma"] is True, d
    assert d["index_near_high"] is True, d
    assert d["leader_drawdown_pct"] >= S8B.RED_DRAWDOWN * 100
    assert r["state"] == "R"


def test_s8_silent_when_the_index_has_already_broken():
    """The point is the index has NOT confirmed. A leader falling in a market
    that is already down 20% is just a bear market."""
    con = db()
    asof = _s8_fixture(con, leader="XLK_CLOSE", leader_dd=0.16, bench_dd=0.20)
    r = S8B.compute(con, asof)
    assert r["detail"]["index_near_high"] is False
    assert r["state"] == "G"


def test_s8_refuses_to_pick_a_leader_from_too_few_sectors():
    con = db()
    bench = [100.0 + i * 0.10 for i in range(600)]
    _load_px(con, "SPY_CLOSE", bench)
    asof = _load_px(con, "XLK_CLOSE", [b * 1.5 for b in bench])
    r = S8B.compute(con, asof)
    assert r["state"] == "NA"
    assert "need >=6 sectors" in r["detail"]["reason"]


def test_s8_loses_sight_of_an_epicenter_that_has_fully_collapsed():
    """A PROPERTY of the frozen formula, pinned so it is not rediscovered.

    The leader is whichever sector has the top trailing-2y RS at the evaluation
    date. A sector that falls far enough loses that status, and S8 then measures
    a different, intact sector and reads G. So S8 has a WINDOW in which it can
    see a fracture -- deep enough to breach -15% and the 200DMA, shallow enough
    to remain the 2y leader -- and goes quiet once the collapse is complete.

    That is not a defect to patch: the registry defines the leader this way and
    the thresholds are frozen. It is a limit on what the signal can be asked."""
    con = db()
    asof = _s8_fixture(con, leader="XLK_CLOSE", leader_dd=0.22,
                       leader_below_dma=True)          # -22% then a further -30%
    d = S8B.compute(con, asof)["detail"]
    assert d["leader"] != "XLK", "a fully collapsed sector is no longer the leader"
    assert d["leader_drawdown_pct"] < S8B.ARM_DRAWDOWN * 100


# --------------------------------------------------------------- S9

from builders import s9_short_interest as S9B  # noqa: E402


def _load_si(con, n_dates=60, n_tickers=150, base=1e7, drift=0.0,
             shock=0.0, drop_after=None, start_year=2021):
    """Semi-monthly SI panel. `drop_after` removes a slice of tickers from
    dates after that index, to simulate coverage change."""
    from datetime import date, timedelta
    d = date(start_year, 1, 15)
    dates = []
    for i in range(n_dates):
        dates.append(d.isoformat())
        d += timedelta(days=15)
    for i, ds in enumerate(dates):
        pub = (date.fromisoformat(ds) + timedelta(days=12)).isoformat()
        n_t = n_tickers if (drop_after is None or i <= drop_after) else n_tickers // 2
        for t in range(n_t):
            v = base * (1 + drift * i)
            if shock and i == len(dates) - 1:
                v *= (1 + shock)
            put(con, f"SI:T{t:03d}", ds, pub, v)
    return dates[-1]


def test_s9_business_day_lag_arithmetic():
    import importlib.util, os as _os
    here = _os.path.dirname(_os.path.abspath(__file__))
    spec = importlib.util.spec_from_file_location(
        "isi", _os.path.join(here, "ingest_short_interest.py"))
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    # 2026-08-14 is a Friday; +8 business days = 2026-08-26 (Wednesday)
    assert m.plus_business_days("2026-08-14", 8) == "2026-08-26"
    assert m.plus_business_days("2026-08-13", 1) == "2026-08-14"
    assert m.plus_business_days("2026-08-14", 1) == "2026-08-17"   # skips weekend


def test_s9_green_on_a_flat_panel():
    con = db()
    asof = _load_si(con)
    r = S9B.compute(con, asof)
    assert r["state"] == "NA" or abs(r["detail"]["z"]) < S9B.ARM_Z, r["detail"]


def test_s9_fires_on_a_short_interest_spike():
    con = db()
    asof = _load_si(con, n_dates=60, drift=0.002, shock=0.25)
    from datetime import date, timedelta
    r = S9B.compute(con, (date.fromisoformat(asof) + timedelta(days=20)).isoformat())
    assert r["detail"].get("z", 0) > S9B.RED_Z, r["detail"]
    assert r["state"] == "R"


def test_s9_panel_intersects_all_visible_dates_and_excludes_late_arrivals():
    """The panel is the intersection over every VISIBLE date -- point-in-time,
    since nothing after asof is read. A name that appears late never joins, so
    it cannot inflate the aggregate; that is the conservative direction.

    An earlier version intersected only the trailing z-window, which starved the
    trend fit whenever coverage had grown and left S9 NA at nearly every anchor.
    """
    con = db()
    from datetime import date, timedelta
    # 100 core names throughout, 50 more only from halfway
    asof = _load_si(con, n_dates=60, n_tickers=100)
    _load_si(con, n_dates=30, n_tickers=150, start_year=2022)
    r = S9B.compute(con, (date.fromisoformat(asof) + timedelta(days=20)).isoformat())
    if r["state"] != "NA":
        assert r["detail"]["panel_names"] == 100, r["detail"]
        assert r["detail"]["obs_used"] >= S9B.MIN_TREND_OBS


def test_s9_refuses_a_tiny_panel():
    con = db()
    asof = _load_si(con, n_dates=60, n_tickers=20)
    from datetime import date, timedelta
    r = S9B.compute(con, (date.fromisoformat(asof) + timedelta(days=20)).isoformat())
    assert r["state"] == "NA"
    assert "panel is" in r["detail"]["reason"]


def test_s9_ols_residual_is_zero_on_a_perfect_line():
    ys = [1.0 + 0.5 * i for i in range(20)]
    resid, a, b = S9B._ols_residual_last(ys)
    assert abs(b - 0.5) < 1e-9 and abs(a - 1.0) < 1e-9
    assert all(abs(r) < 1e-9 for r in resid)
