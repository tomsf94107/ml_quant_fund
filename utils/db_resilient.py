"""
utils/db_resilient.py — shared SQLite resilience.

connect_resilient(path): connection with WAL + busy_timeout + connect timeout.
run_with_io_retry(fn, ...): runs a callable, retrying ONLY on transient SQLite
    faults (disk I/O error / locked / busy) with exponential backoff. After the
    final attempt it RE-RAISES the original error. It never swallows a fault.
"""
from __future__ import annotations
import sqlite3, time, logging

log = logging.getLogger(__name__)
_TRANSIENT = ("disk i/o error", "database is locked", "database is busy")


def _is_transient(err):
    return isinstance(err, sqlite3.OperationalError) and any(
        m in str(err).lower() for m in _TRANSIENT)


def connect_resilient(path, timeout: float = 30.0) -> sqlite3.Connection:
    conn = sqlite3.connect(path, timeout=timeout)
    try:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA busy_timeout=30000")
    except sqlite3.OperationalError as e:
        log.warning("connect_resilient: PRAGMA setup failed on %s: %s", path, e)
    return conn


def run_with_io_retry(fn, *args, attempts: int = 6, base: float = 1.0, **kwargs):
    last = None
    for i in range(attempts):
        try:
            return fn(*args, **kwargs)
        except sqlite3.OperationalError as e:
            if not _is_transient(e):
                raise
            last = e
            if i == attempts - 1:
                break
            wait = base * (2 ** i)
            log.warning("transient DB fault (%s) - attempt %d/%d failed, retry in %.1fs",
                        e, i + 1, attempts, wait)
            time.sleep(wait)
    assert last is not None
    raise last
