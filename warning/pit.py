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


def month_end(ym: str) -> str:
    """Last calendar day of 'YYYY-MM'."""
    y, m = int(ym[:4]), int(ym[5:7])
    ny, nm = (y + 1, 1) if m == 12 else (y, m + 1)
    return (date(ny, nm, 1) - timedelta(days=1)).isoformat()


def monthly_mean_complete(rows, asof, pub_lag_days: int = 1):
    """monthly_mean, but ONLY for months whose observations are all published.

    A month is usable when asof >= month_end + publication_lag. Anything later is
    dropped, INCLUDING a month that has ended on the calendar but whose final
    prints have not yet been released.

    WHY THIS IS NOT COSMETIC (found on real data 2026-08-28):
      At the 2001-01-31 read, January's last observation was not yet published,
      so the January mean was -0.001 -- fractionally inverted. S1 counted January
      in the inversion run, reached the 6-month floor, and fired R. By the
      2001-02-28 read the full month was in, January was positive, the run shrank
      to 5 months, and the escalation vanished. A signal must not enter and leave
      a state because a month was half-counted.

    Cost: one extra month of lag. Immaterial for S1, whose documented lead is
    14-16 months, and the registry's verdicts name DATA months, not read dates.
    """
    asof = _d(asof)
    out = []
    for ym, v in monthly_mean(rows):
        ready = (datetime.fromisoformat(month_end(ym)).date()
                 + timedelta(days=pub_lag_days)).isoformat()
        if asof >= ready:
            out.append((ym, v))
    return out


def align(a: list[tuple[str, float]],
          b: list[tuple[str, float]]) -> list[tuple[str, float, float]]:
    """Inner-join two series on obs_date. Returns [(obs_date, a_val, b_val)]."""
    bd = dict(b)
    return [(d, v, bd[d]) for d, v in a if d in bd]
