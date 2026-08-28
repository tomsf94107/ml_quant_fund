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
    assert len(unbuilt) == 13
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


def test_insufficient_data_currently_suppresses_the_l4_crisis_override():
    """DECISIONS.md D10 -- SPEC CONFLICT, pinned so it cannot be forgotten.

    step() tests `insufficient` BEFORE hysteresis_step, so when coverage is poor
    the band freezes and the L4 override never reaches it: l4_override is True
    while the band reads INSUFFICIENT_DATA and the action is 'freeze'.

    Report line 601 says L4 'overrides composite to B'; the Part VI honesty rule
    says an under-covered composite makes Part VIII's do-nothing rule bind. Both
    cannot hold at once. This test asserts CURRENT behaviour, not desired
    behaviour -- with 2 of 15 builders live, a 150bp HY blowout would print
    'freeze' rather than 'CRISIS'. Awaiting a ruling."""
    from warning_engine import SignalReading, EngineState, step
    import daily_driver as DD
    readings = [SignalReading(s, L, "NA", stale=True) for s, L in DD.ROSTER.items()]
    readings += [SignalReading("L4A", "L4", "G"), SignalReading("L4B", "L4", "B"),
                 SignalReading("L4C", "L4", "G")]
    res = step("2026-08-28", readings, EngineState())
    assert res.l4_override is True              # the override DID fire
    assert res.band == "INSUFFICIENT_DATA"      # ...and was suppressed
    assert res.action["hedge"] == "freeze"
