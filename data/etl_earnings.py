# data/etl_earnings.py
# ─────────────────────────────────────────────────────────────────────────────
# Earnings surprise ETL. Fetches EPS and revenue beat/miss from yfinance.
#
# Why this matters:
#   - EPS surprise is the #1 predictor of next-day price movement
#   - A 20% EPS beat → stock gaps up ~70% of the time
#   - Price-only models are BLIND to this — they see the price reaction
#     but not the cause, making them late
#   - Adding surprise features lets the model be early
#
# Features added to builder.py:
#   eps_surprise       : (actual - estimate) / |estimate|  — normalized
#   eps_surprise_pct   : raw % beat/miss
#   rev_surprise       : revenue beat/miss normalized
#   days_to_earnings   : how many days until next earnings (urgency signal)
#   post_earnings_drift: momentum effect — stocks that beat keep rising 3-5d
#
# Data source: yfinance .earnings_history + .calendar (free, no API key)
# Limitation: yfinance has ~4 quarters of history, not years
# Solution: cache to SQLite and accumulate over time
# ─────────────────────────────────────────────────────────────────────────────

from __future__ import annotations

import os
import sqlite3
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import yfinance as yf

from utils.pit_helpers import knowable_at

DB_PATH = Path(os.getenv("EARNINGS_DB_PATH", "earnings.db"))
ACCURACY_DB_PATH = Path(os.getenv("ACCURACY_DB_PATH", "accuracy.db"))


# ══════════════════════════════════════════════════════════════════════════════
#  DATABASE
# ══════════════════════════════════════════════════════════════════════════════

def _init_db(db_path: Path = DB_PATH) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS earnings_surprises (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker          TEXT    NOT NULL,
            report_date     TEXT    NOT NULL,
            eps_actual      REAL,
            eps_estimate    REAL,
            eps_surprise    REAL,
            eps_surprise_pct REAL,
            rev_actual      REAL,
            rev_estimate    REAL,
            rev_surprise    REAL,
            created_at      TEXT    NOT NULL,
            UNIQUE(ticker, report_date)
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS earnings_calendar (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker          TEXT    NOT NULL,
            next_date       TEXT    NOT NULL,
            updated_at      TEXT    NOT NULL,
            UNIQUE(ticker)
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_earn_ticker_date ON earnings_surprises(ticker, report_date)")
    conn.commit()
    return conn


# ══════════════════════════════════════════════════════════════════════════════
#  FETCH
# ══════════════════════════════════════════════════════════════════════════════

def _fetch_earnings(ticker: str) -> tuple[pd.DataFrame, Optional[date]]:
    """Fetch earnings via yf_resilient (May 5 2026 — DNS leak fix)."""
    try:
        from features.yf_resilient import _retry_yf_call
        def _make_ticker():
            return yf.Ticker(ticker)
        t = _retry_yf_call(_make_ticker, label=f"yf.Ticker({ticker}) earnings")
        if t is None:
            return pd.DataFrame(), None

        # Earnings history
        hist = t.earnings_history
        if hist is None or (hasattr(hist, 'empty') and hist.empty):
            hist = pd.DataFrame()

        # Next earnings date from calendar
        next_date = None
        try:
            cal = t.calendar
            if cal is not None and not cal.empty:
                # Calendar format varies by yfinance version
                if isinstance(cal, pd.DataFrame):
                    if "Earnings Date" in cal.index:
                        raw = cal.loc["Earnings Date"].iloc[0]
                        next_date = pd.to_datetime(raw).date()
                    elif "Earnings Date" in cal.columns:
                        next_date = pd.to_datetime(cal["Earnings Date"].iloc[0]).date()
                elif isinstance(cal, dict) and "Earnings Date" in cal:
                    raw = cal["Earnings Date"]
                    if isinstance(raw, (list, tuple)) and raw:
                        next_date = pd.to_datetime(raw[0]).date()
        except Exception:
            next_date = None

        return hist, next_date

    except Exception as e:
        print(f"  ⚠ yfinance earnings fetch failed for {ticker}: {e}")
        return pd.DataFrame(), None


def _parse_earnings_history(ticker: str, hist: pd.DataFrame) -> list[dict]:
    """Parse yfinance earnings history into normalized rows."""
    if hist is None or hist.empty:
        return []

    rows = []
    now  = datetime.utcnow().isoformat()

    # yfinance column names vary — handle both formats
    col_map = {
        "epsActual":      ["epsActual", "EPS Actual", "Reported EPS"],
        "epsEstimate":    ["epsEstimate", "EPS Estimate", "EPS Estimate"],
        "epsSurprise":    ["epsSurprise", "EPS Difference", "Surprise(%)"],
        "revActual":      ["revenueActual", "Revenue Actual"],
        "revEstimate":    ["revenueEstimate", "Revenue Estimate"],
    }

    def _get_col(df, aliases):
        for a in aliases:
            if a in df.columns:
                return df[a]
        return None

    for idx, row in hist.iterrows():
        try:
            report_date = pd.to_datetime(idx).date() if not isinstance(idx, date) else idx

            eps_actual   = float(row.get("epsActual",   row.get("Reported EPS",   np.nan)) or np.nan)
            eps_estimate = float(row.get("epsEstimate", row.get("EPS Estimate",   np.nan)) or np.nan)

            # Normalized surprise: (actual - estimate) / |estimate|
            if not np.isnan(eps_actual) and not np.isnan(eps_estimate) and eps_estimate != 0:
                eps_surprise     = (eps_actual - eps_estimate) / abs(eps_estimate)
                eps_surprise_pct = eps_surprise * 100
            else:
                eps_surprise = eps_surprise_pct = np.nan

            # yfinance revenue removed May 22 2026 — fields don't exist in
            # earnings_history. Polygon (data/etl_polygon_revenue.py) is the
            # sole writer of rev_actual / rev_estimate / rev_surprise.

            rows.append({
                "ticker":           ticker.upper(),
                "report_date":      str(report_date),
                "eps_actual":       eps_actual,
                "eps_estimate":     eps_estimate,
                "eps_surprise":     eps_surprise,
                "eps_surprise_pct": eps_surprise_pct,
                "created_at":       knowable_at(report_date),
            })
        except Exception:
            continue

    return rows


# ══════════════════════════════════════════════════════════════════════════════
#  PUBLIC API — WRITE
# ══════════════════════════════════════════════════════════════════════════════

def run_earnings_etl(
    tickers: list[str],
    db_path: Path = DB_PATH,
    verbose: bool = True,
) -> dict[str, int]:
    """Fetch earnings data for all tickers and cache to SQLite."""
    conn    = _init_db(db_path)
    results = {}
    now     = datetime.utcnow().isoformat()

    for ticker in tickers:
        ticker = ticker.upper().strip()
        if verbose:
            print(f"  {ticker} — fetching earnings...")

        hist, next_date = _fetch_earnings(ticker)
        rows = _parse_earnings_history(ticker, hist)

        if rows:
            conn.executemany("""
                -- yfinance ETL only writes EPS columns. Revenue columns are
                -- owned by Polygon (data/etl_polygon_revenue.py).
                -- Using ON CONFLICT to preserve existing rev_* values.
                INSERT INTO earnings_surprises
                    (ticker, report_date, eps_actual, eps_estimate,
                     eps_surprise, eps_surprise_pct, created_at)
                VALUES
                    (:ticker, :report_date, :eps_actual, :eps_estimate,
                     :eps_surprise, :eps_surprise_pct, :created_at)
                ON CONFLICT(ticker, report_date) DO UPDATE SET
                    eps_actual       = excluded.eps_actual,
                    eps_estimate     = excluded.eps_estimate,
                    eps_surprise     = excluded.eps_surprise,
                    eps_surprise_pct = excluded.eps_surprise_pct
            """, rows)
            conn.commit()

        if next_date:
            conn.execute("""
                INSERT OR REPLACE INTO earnings_calendar (ticker, next_date, updated_at)
                VALUES (?, ?, ?)
            """, (ticker, str(next_date), now))
            conn.commit()

        results[ticker] = len(rows)
        if verbose:
            nd = f"next={next_date}" if next_date else "next=unknown"
            print(f"    ✓ {len(rows)} quarters cached  {nd}")

    conn.close()
    return results


# ══════════════════════════════════════════════════════════════════════════════
#  PUBLIC API — READ (called by builder.py)
# ══════════════════════════════════════════════════════════════════════════════

def load_earnings_features(
    ticker:     str,
    date_index: pd.Index,       # DatetimeIndex or date index from builder.py
    db_path:    Path = DB_PATH,
    as_of:      str | date | None = None,
) -> pd.DataFrame:
    """
    Build a daily earnings feature DataFrame aligned to date_index.

    For each date:
      - eps_surprise / rev_surprise: value from the most recent earnings report
        (forward-filled — represents what the market knows on that date)
      - days_to_earnings: days until next known earnings date
      - post_earnings_1d/3d/5d: flag for 1/3/5 days after an earnings report

    Returns DataFrame with index=date_index and columns:
        eps_surprise, rev_surprise, days_to_earnings,
        post_earnings_1d, post_earnings_3d, post_earnings_5d
    """
    default = pd.DataFrame({
        "eps_surprise":     0.0,
        "rev_surprise":     0.0,
        "days_to_earnings": 999.0,
        "post_earnings_1d": 0.0,
        "post_earnings_3d": 0.0,
        "post_earnings_5d": 0.0,
    }, index=date_index)

    if not db_path.exists():
        return default

    try:
        conn = sqlite3.connect(db_path)

        # Load surprise history (point-in-time honest when as_of is set)
        # PIT REWIRE (Jul 11 2026).
        # WAS: earnings_surprises.report_date = the FISCAL PERIOD END, not the
        # announcement date. 75% of those rows land on a quarter-end, 35% on a
        # weekend. MU reads 2026-05-31; MU announced 2026-06-24.
        # created_at = knowable_at(fiscal_end) = fiscal_end + 2BD, so the PIT
        # filter admitted eps_surprise 15-29 DAYS EARLY and the forward-fill
        # stamped it from the fiscal end onward.
        # PROOF (MSFT, pre-fix): eps_surprise changed and post_earnings_1d fired
        # 1.0 on 2026-03-31 -- the day the QUARTER ended -- while MSFT did not
        # announce until 2026-04-29. In training the model saw the surprise ~4
        # weeks before the market did; in production the row does not exist yet.
        # Measured worth of that window: SUE predicts the announcement-day move
        # at IC +0.2612 (t=+30). A quantified train/serve leak.
        #
        # NOW: earnings_events.announce_date = the SEC 8-K Item 2.02 filing date
        # (verified 8/8 against our own eightk_items). The announcement date IS
        # the knowable date -- no created_at proxy needed.
        # rev_surprise joins back on fiscal_end. NOTE it is NULL in practice --
        # Polygon gives rev_actual but no rev_estimate -- so that feature has been
        # a constant 0.0 in production. Kept wired so nothing changes shape.
        _PIT_SQL = """
            SELECT e.announce_date AS report_date,
                   e.eps_surprise  AS eps_surprise,
                   s.rev_surprise  AS rev_surprise
            FROM earnings_events e
            LEFT JOIN earnings_surprises s
              ON s.ticker = e.ticker AND DATE(s.report_date) = e.fiscal_end
            WHERE e.ticker = ?
              AND e.eps_surprise IS NOT NULL
              AND e.announce_date <= ?
            ORDER BY e.announce_date
        """
        _cut = str(as_of) if as_of is not None else datetime.utcnow().strftime("%Y-%m-%d")
        hist = pd.read_sql(_PIT_SQL, conn, params=(ticker.upper(), _cut))
        # RESIDUAL-LEAK FIX: most large caps announce AMC (after the close). A number
        # released at 16:05 on day D is NOT actionable against a close(D)->close(D+1)
        # label -- a real trader can only act at the open of D+1, by which time the
        # gap has happened. That overnight gap is exactly where the value is
        # (SUE vs the announcement move: IC +0.2612, t=+30), so treating the surprise
        # as known on D would re-import the very leak this rewire removes.
        # Make it usable from the NEXT session. BMO reporters lose one day of
        # staleness on a slow feature -- a cheap price for a hard guarantee.
        if not hist.empty:
            hist["report_date"] = (pd.to_datetime(hist["report_date"])
                                   + pd.tseries.offsets.BDay(1))
        conn.close()

        # Next earnings date — sourced from accuracy.db.earnings_cache (the
        # source of truth maintained by daily_uw_snapshot.py). The legacy
        # earnings.db.earnings_calendar table is empty (yfinance-era artifact).
        # DAYS_TO_EARNINGS FIX (Jul 11 2026), part 2.
        # The query filtered `report_date > as_of`, so it only ever returned
        # announcements AFTER THE END of the whole window -- any announcement
        # INSIDE the window was invisible. On 2026-04-20 MSFT therefore read 100
        # (counting to the July print) while it actually announced 9 days later, on
        # 2026-04-29. Over a 2022->today training window `as_of` is today, so every
        # historical date counted to a print 1000+ days out and clipped to 999:
        # the feature was a CONSTANT across almost all of training, while production
        # (as_of = today) got real values. Train/serve skew that killed the feature.
        # Now: pull ALL announcement dates; pick the next one PER DATE (searchsorted).
        # PIT note: the next announcement date is normally confirmed 2-6 weeks ahead,
        # so using it earlier is mild look-ahead -- but earnings TIMING is near-
        # deterministic (same fiscal week each year), so the date itself carries
        # almost no information. The original design already accepted this.
        cal = pd.DataFrame()
        try:
            cal_conn = sqlite3.connect(ACCURACY_DB_PATH)
            cal = pd.read_sql(
                "SELECT report_date AS next_date FROM earnings_cache "
                "WHERE ticker = ? AND report_date IS NOT NULL ORDER BY report_date",
                cal_conn, params=(ticker.upper(),))
            cal_conn.close()
        except Exception:
            cal = pd.DataFrame()

        if hist.empty:
            return default

        hist["report_date"] = pd.to_datetime(hist["report_date"])
        hist = hist.set_index("report_date").sort_index()

        # Build daily date range
        dates = pd.to_datetime(date_index)
        result = pd.DataFrame(index=dates)

        # Forward-fill surprise values (model knows latest surprise on each date)
        result["eps_surprise"] = np.nan
        result["rev_surprise"] = np.nan

        for rdate, row in hist.iterrows():
            mask = result.index >= rdate
            result.loc[mask, "eps_surprise"] = row["eps_surprise"]
            result.loc[mask, "rev_surprise"] = row["rev_surprise"]

        result["eps_surprise"] = result["eps_surprise"].fillna(0.0)
        result["rev_surprise"] = result["rev_surprise"].fillna(0.0)

        # Clamp to [-3, +3] to prevent extreme values from dominating
        result["eps_surprise"] = result["eps_surprise"].clip(-3.0, 3.0)
        result["rev_surprise"] = result["rev_surprise"].clip(-3.0, 3.0)

        # Days to earnings
        # DAYS_TO_EARNINGS FIX (Jul 11 2026). The old code took ONE next_date --
        # the first announcement after `as_of` (the END of the whole window) -- and
        # applied it to EVERY date in the index, skipping straight over any
        # announcement that fell inside the window. Result: on 2026-04-20 MSFT read
        # 100 (counting to the July print) when it actually announced 9 days later,
        # on 2026-04-29. Over a 2022->today training window the single next_date sits
        # 1000+ days from early rows, so the value clipped to 999 across most of
        # training. The feature never meant what its name says.
        # Now: for each date d, the next announcement strictly after d.
        _ann = pd.Series(dtype="datetime64[ns]")
        if not cal.empty:
            _ann = pd.to_datetime(cal["next_date"], errors="coerce").dropna().sort_values()
        next_date = None   # retained for the fallback branch below

        if len(_ann) > 0:
            _pos = np.searchsorted(_ann.values, result.index.values, side="left")
            _nxt = np.where(_pos < len(_ann),
                            _ann.values[np.clip(_pos, 0, len(_ann) - 1)],
                            np.datetime64("NaT"))
            _d = ((pd.to_datetime(_nxt) - result.index)
                  .total_seconds().values / 86400.0)
            result["days_to_earnings"] = np.nan_to_num(np.clip(_d, 0, 999), nan=999.0)
        else:
            result["days_to_earnings"] = 999.0

        # Post-earnings drift flags
        report_dates = set(hist.index.date)
        for col, window in [("post_earnings_1d", 1),
                             ("post_earnings_3d", 3),
                             ("post_earnings_5d", 5)]:
            flags = []
            for d in result.index.date:
                in_window = any(
                    0 <= (d - rd).days <= window
                    for rd in report_dates
                )
                flags.append(1.0 if in_window else 0.0)
            result[col] = flags

        result.index = date_index
        return result

    except Exception as e:
        print(f"  ⚠ Earnings feature load failed for {ticker}: {e}")
        return default


# ══════════════════════════════════════════════════════════════════════════════
#  CLI
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse
    from models.train_all import DEFAULT_TICKERS

    parser = argparse.ArgumentParser(description="Earnings surprise ETL")
    parser.add_argument("--tickers", nargs="+", default=None)
    args = parser.parse_args()

    tickers = args.tickers or DEFAULT_TICKERS
    print(f"\nFetching earnings for {len(tickers)} tickers...")
    results = run_earnings_etl(tickers)
    total = sum(results.values())
    print(f"\nDone. {total} total earnings records cached.")


# ══════════════════════════════════════════════════════════════════════════════
#  UW EARNINGS FEATURES — richer than yfinance
# ══════════════════════════════════════════════════════════════════════════════

def load_uw_earnings_features(
    ticker:     str,
    date_index: pd.Index,
) -> pd.DataFrame:
    """
    Load earnings features from Unusual Whales API.
    Adds 3 features not available in yfinance:
      - expected_move_perc  : options-implied expected move % for next earnings
      - pre_earnings_drift  : avg pre-earnings move 3d (historical)
      - post_earnings_drift : avg post-earnings move 3d (historical)

    Falls back to zeros if UW unavailable.
    """
    import os, requests

    default = pd.DataFrame({
        "expected_move_perc":  0.0,
        "pre_earnings_drift":  0.0,
        "post_earnings_drift": 0.0,
        "is_earnings_week":    0.0,
    }, index=date_index)

    try:
        key     = os.getenv("UW_API_KEY", "")
        if not key:
            return default

        headers = {"Authorization": f"Bearer {key}"}
        r = requests.get(
            f"https://api.unusualwhales.com/api/earnings/{ticker}",
            headers=headers, timeout=10
        )
        if r.status_code != 200:
            return default

        data = r.json().get("data", [])
        if not data:
            return default

        # Get next earnings date and expected move
        today = pd.Timestamp.today().normalize()
        future = [d for d in data if pd.Timestamp(d["report_date"]) >= today]
        next_report = future[0] if future else None

        # Historical pre/post earnings moves
        historical = [d for d in data
                      if d.get("pre_earnings_move_3d") is not None
                      and d.get("post_earnings_move_3d") is not None]

        avg_pre  = 0.0
        avg_post = 0.0
        if historical:
            avg_pre  = float(sum(float(d["pre_earnings_move_3d"])  for d in historical[-8:]) / len(historical[-8:]))
            avg_post = float(sum(float(d["post_earnings_move_3d"]) for d in historical[-8:]) / len(historical[-8:]))

        result = pd.DataFrame(index=date_index)
        result["pre_earnings_drift"]  = avg_pre
        result["post_earnings_drift"] = avg_post

        # Expected move for upcoming earnings
        if next_report:
            exp_move = float(next_report.get("expected_move_perc", 0) or 0)
            next_ts  = pd.Timestamp(next_report["report_date"])
            # Apply expected move only in the 10 days before earnings
            result["expected_move_perc"] = 0.0
            dates = pd.to_datetime(date_index)
            mask  = (dates >= next_ts - pd.Timedelta(days=10)) & (dates <= next_ts)
            result.loc[mask, "expected_move_perc"] = exp_move

            # is_earnings_week — within 5 days of next earnings
            result["is_earnings_week"] = 0.0
            mask2 = (dates >= next_ts - pd.Timedelta(days=5)) & (dates <= next_ts + pd.Timedelta(days=2))
            result.loc[mask2, "is_earnings_week"] = 1.0
        else:
            result["expected_move_perc"] = 0.0
            result["is_earnings_week"]   = 0.0

        result.index = date_index
        return result

    except Exception as e:
        return default
