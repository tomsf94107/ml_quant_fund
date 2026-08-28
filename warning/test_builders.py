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
