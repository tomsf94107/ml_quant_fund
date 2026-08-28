"""
pit.py — point-in-time reads from data_vintages.

RULE #1 (non-negotiable): every historical join is on PUBLICATION date, never
reference date. This module is the only sanctioned way to read a macro series;
a builder that queries data_vintages directly can silently see the future.

The vintage semantics: for a given as-of date, the value of an observation is
the one from the LATEST pub_date that is <= as-of. Later revisions are invisible.
This is what makes a 2007 backtest see 2007's numbers, not today's revised ones.

    from pit import series_asof, staleness_days
    spread = series_asof(con, "DGS10", "2007-08-15")
"""

from __future__ import annotations
import sqlite3
from datetime import date, datetime, timedelta


def _d(x) -> str:
    return x if isinstance(x, str) else x.isoformat()


def series_asof(con: sqlite3.Connection, series_id: str, asof,
                start=None, end=None) -> list[tuple[str, float]]:
    """Observations of `series_id` as they were visible on `asof`.

    Returns [(obs_date, value)] sorted by obs_date. For each obs_date the value
    comes from the latest pub_date <= asof; observations first published after
    `asof` are absent entirely (that is the point).
    """
    asof = _d(asof)
    sql = """
        SELECT dv.obs_date, dv.value
        FROM data_vintages dv
        WHERE dv.series_id = ?
          AND dv.pub_date <= ?
          AND dv.pub_date = (
              SELECT MAX(d2.pub_date) FROM data_vintages d2
              WHERE d2.series_id = dv.series_id
                AND d2.obs_date  = dv.obs_date
                AND d2.pub_date <= ?
          )
    """
    params = [series_id, asof, asof]
    if start:
        sql += " AND dv.obs_date >= ?"; params.append(_d(start))
    if end:
        sql += " AND dv.obs_date <= ?"; params.append(_d(end))
    sql += " ORDER BY dv.obs_date"
    return [(r[0], r[1]) for r in con.execute(sql, params)]


def latest_obs_asof(con, series_id: str, asof):
    """(obs_date, value) of the most recent observation visible on `asof`."""
    rows = series_asof(con, series_id, asof)
    return rows[-1] if rows else None


def staleness_days(con, series_id: str, asof) -> int | None:
    """Calendar days between `asof` and the newest observation visible then.

    Compared against the registry's max_staleness_days to set SignalReading.stale.
    None = the series has no visible observations at all (a finding, not a zero).
    """
    last = latest_obs_asof(con, series_id, asof)
    if last is None:
        return None
    a = datetime.fromisoformat(_d(asof)).date()
    o = datetime.fromisoformat(last[0]).date()
    return (a - o).days


def monthly_mean(rows: list[tuple[str, float]]) -> list[tuple[str, float]]:
    """Collapse daily [(obs_date, value)] to [('YYYY-MM', mean)], sorted.

    Used by any registry row whose frequency is 'daily->monthly' (e.g. S1).
    Partial months are returned as-is; the caller decides whether the current
    (incomplete) month is usable.
    """
    buckets: dict[str, list[float]] = {}
    for d, v in rows:
        buckets.setdefault(d[:7], []).append(v)
    return sorted((m, sum(vs) / len(vs)) for m, vs in buckets.items())


def align(a: list[tuple[str, float]],
          b: list[tuple[str, float]]) -> list[tuple[str, float, float]]:
    """Inner-join two series on obs_date. Returns [(obs_date, a_val, b_val)]."""
    bd = dict(b)
    return [(d, v, bd[d]) for d, v in a if d in bd]
