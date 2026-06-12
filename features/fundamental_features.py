"""
features/fundamental_features.py - PIT fundamental features from fundamentals.db.

PIT discipline: a fact is usable strictly AFTER its filed_date (merge_asof with
allow_exact_matches=False). Vectorized fast path (no per-row loops).

Features (per ticker, daily index, step-function between filings):
  fund_gp_assets  = (revenue - cogs) / total_assets     [Novy-Marx quality]
  fund_op_equity  = operating_income / equity           [operating profitability]
  fund_ni_margin  = net_income / revenue                [margin quality]
  fund_bm         = equity / (close * shares_out)       [value, needs close series]
  fund_ep         = net_income / (close * shares_out)   [earnings yield]

API: load_fundamental_features_pit(ticker, date_index, close=None, db_path=None)
  -> DataFrame indexed by date_index. NaN where no filing yet (PIT-honest unknown).
"""
import os
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

DB_PATH = Path(os.environ.get("FUNDAMENTALS_DB_PATH", "fundamentals.db"))
CONCEPTS = ["revenue", "cogs", "total_assets", "equity",
            "operating_income", "net_income", "shares_out"]
_cache = {}


def _facts_for(ticker, db_path):
    key = (str(db_path), ticker)
    if key in _cache:
        return _cache[key]
    conn = sqlite3.connect(str(db_path))
    df = pd.read_sql(
        "SELECT concept, filed_date, period_end, value FROM xbrl_facts "
        "WHERE ticker = ? AND concept IN (%s)" % ",".join("?" * len(CONCEPTS)),
        conn, params=[ticker] + CONCEPTS)
    conn.close()
    df["filed_date"] = pd.to_datetime(df["filed_date"])
    _cache[key] = df
    return df


def _pit_series(facts, concept, date_index):
    """Step series: at each date, the value of the most recent filing strictly before it."""
    d = facts[facts.concept == concept]
    if d.empty:
        return pd.Series(np.nan, index=date_index)
    # for same filed_date keep the most recent period_end (latest quarter wins)
    d = d.sort_values(["filed_date", "period_end"]).drop_duplicates("filed_date", keep="last")
    left = pd.DataFrame({"date": pd.to_datetime(date_index)})
    m = pd.merge_asof(left, d[["filed_date", "value"]].rename(columns={"filed_date": "date"}),
                      on="date", allow_exact_matches=False)
    return pd.Series(m["value"].values, index=date_index)


def load_fundamental_features_pit(ticker, date_index, close=None, db_path=None):
    db = Path(db_path) if db_path else DB_PATH
    out = pd.DataFrame(index=date_index)
    try:
        facts = _facts_for(ticker.upper().strip(), db)
    except Exception as e:
        import logging
        logging.getLogger(__name__).error(f"fundamental_features load fail {ticker}: {e!r}")
        for c in ("fund_gp_assets", "fund_op_equity", "fund_ni_margin", "fund_bm", "fund_ep"):
            out[c] = np.nan
        return out

    rev = _pit_series(facts, "revenue", date_index)
    cogs = _pit_series(facts, "cogs", date_index)
    assets = _pit_series(facts, "total_assets", date_index)
    eq = _pit_series(facts, "equity", date_index)
    oi = _pit_series(facts, "operating_income", date_index)
    ni = _pit_series(facts, "net_income", date_index)
    sh = _pit_series(facts, "shares_out", date_index)

    assets_pos = assets.where(assets > 0)
    eq_pos = eq.where(eq > 0)
    rev_pos = rev.where(rev != 0)
    out["fund_gp_assets"] = (rev - cogs) / assets_pos
    out["fund_op_equity"] = oi / eq_pos
    out["fund_ni_margin"] = ni / rev_pos
    if close is not None:
        px = pd.Series(close, index=date_index) if not isinstance(close, pd.Series) else close.reindex(date_index)
        mc = (px * sh).where(lambda x: x > 0)
        out["fund_bm"] = eq / mc
        out["fund_ep"] = ni / mc
    else:
        out["fund_bm"] = np.nan
        out["fund_ep"] = np.nan
    return out
