"""
scripts/momentum_shadow.py — Pipeline C Stage 3: momentum signal in SHADOW MODE.

Runs ONCE after all daily_runner batches (it needs the full universe at once —
momentum is cross-sectional). Computes today's momentum BUY candidates via the
VALIDATED signals/momentum_signal.py and writes them to momentum_shadow_predictions
(a SEPARATE table) — NOT the live predictions table. Live BUYs stay disabled.

WHY SHADOW: momentum passed strict purged-WF (4/4 OOS folds, net ~+1.0), but the
whole lesson of the rebuild is that backtest != live. This logs daily picks so we
can measure LIVE momentum hit-rate vs the backtest over several weeks BEFORE any
real BUY fires. Promotion to the live path + re-enable of BUYs is a later,
deliberate step gated on live confirmation — not granted by assumption.

Idempotent: INSERT OR REPLACE on (prediction_date, ticker, kind).
Non-fatal in the pipeline: shadow failing must never break signal publish.
"""
from __future__ import annotations
import sqlite3, sys
from datetime import datetime, timezone
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import pandas as pd
from signals.momentum_signal import rank_signal, MOM_HORIZON

DB = ROOT / "accuracy.db"
KINDS = ["mom_6_1", "mom_12_1"]
HISTORY_START = "2024-01-01"   # enough for 252d lookback + buffer


DDL = """
CREATE TABLE IF NOT EXISTS momentum_shadow_predictions (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    prediction_date  TEXT NOT NULL,
    ticker           TEXT NOT NULL,
    kind             TEXT NOT NULL,
    horizon          INTEGER NOT NULL,
    mom_score        REAL,
    mom_pct_rank     REAL,
    is_buy_candidate INTEGER NOT NULL,
    weight           REAL,
    created_at       TEXT NOT NULL,
    UNIQUE(prediction_date, ticker, kind)
);
CREATE INDEX IF NOT EXISTS idx_momshadow_date ON momentum_shadow_predictions(prediction_date);
"""


def load_universe():
    return [t.strip().upper() for t in (ROOT / "tickers.txt").read_text().splitlines()
            if t.strip() and not t.startswith("#")]


def build_panel(tickers, start=HISTORY_START):
    # Read closes from the local prices.db cache instead of live-fetching via
    # features.builder._download. WHY: XProtect 5347 (Jun 2026) flags certain
    # ETF fetches (SPY, SLV, ...) as malware at the network layer, killing the
    # whole run. Reading cached adj_close runs zero network fetches, so nothing
    # trips XProtect. adj_close is the right series for momentum (split/div
    # adjusted).
    # CORRECTED Aug 1 2026: this comment used to claim index/commodity ETFs were
    # "correctly excluded" as uncached. FALSE -- prices.db has carried SPY/QQQ/
    # SLV since 2022-01-03, and all 9 ETFs in tickers.txt have been in the shadow
    # book since its first date (2026-05-29). Measured impact: 2.3% of the
    # cross-section, ONE ETF BUY candidate in the entire history (SLV, 06-12).
    # analysis/momentum_18yr_test.py pivots the SAME daily_prices with no ticker
    # filter, so backtest and shadow share the universe -- no parity gap, no
    # action needed. The comment was the only defect.
    import sqlite3
    PRICES_DB = ROOT / "prices.db"
    want = [t.strip().upper() for t in tickers if t and t.strip()]
    con = sqlite3.connect(f"file:{PRICES_DB}?mode=ro", uri=True, timeout=30)
    placeholders = ",".join("?" * len(want))
    rows = con.execute(
        f"SELECT ticker, date, adj_close FROM daily_prices "
        f"WHERE ticker IN ({placeholders}) AND date >= ? AND adj_close IS NOT NULL "
        f"ORDER BY date",
        (*want, start),
    ).fetchall()
    con.close()
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows, columns=["ticker", "date", "adj_close"])
    panel = df.pivot(index="date", columns="ticker", values="adj_close").sort_index()
    panel.index = pd.to_datetime(panel.index)
    _have = set(panel.columns)
    _missing = [t for t in want if t not in _have]
    if _missing:
        print(f"momentum_shadow: {len(_missing)} tickers not in prices.db cache, excluded: {_missing}")
    return panel


def main(as_of=None):
    con = sqlite3.connect(str(DB), timeout=120)
    con.execute("PRAGMA busy_timeout=120000")  # wait out concurrent accuracy.db writers (Fix Jul 1 2026)
    con.executescript(DDL)
    tickers = load_universe()
    panel = build_panel(tickers)
    if as_of is not None:
        cutoff = pd.Timestamp(as_of)
        panel = panel[panel.index <= cutoff]
    if panel.empty:
        print("momentum_shadow: empty panel, nothing to write")
        return 0
    # Trim ragged trailing edge: drop trailing dates where cross-sectional
    # coverage is sparse (e.g. only a few tickers backfilled past the rest).
    # Anchor pred_date to the last date with >=50% of tickers present, so the
    # momentum rank computes on a dense cross-section, not a 4-name tail.
    _ncols = panel.shape[1]
    if _ncols > 0:
        _cov = panel.notna().sum(axis=1) / _ncols
        _dense = _cov[_cov >= 0.5]
        if len(_dense) > 0:
            _last_dense = _dense.index[-1]
            _dropped = (panel.index > _last_dense).sum()
            if _dropped > 0:
                print(f"momentum_shadow: trimming {_dropped} sparse trailing date(s) "
                      f"after {str(_last_dense.date())} (coverage <50%)")
                panel = panel.loc[panel.index <= _last_dense]
    if panel.empty:
        print("momentum_shadow: empty panel after trim, nothing to write")
        return 0
    pred_date = str(panel.index[-1].date())
    if as_of is not None and pred_date != str(pd.Timestamp(as_of).date()):
        print(f"momentum_shadow: WARN requested {as_of} but panel ends {pred_date} (no bar that day?) — skipping")
        con.close()
        return 0
    now = datetime.now(timezone.utc).isoformat()
    total = 0
    for kind in KINDS:
        try:
            sig = rank_signal(panel, kind, bucket_cap=3)
        except Exception as e:
            print(f"momentum_shadow: {kind} failed: {type(e).__name__}: {e}")
            continue
        rows = [(pred_date, r.ticker, kind, MOM_HORIZON,
                 float(r.mom_score), float(r.mom_pct_rank),
                 int(r.is_buy_candidate), float(getattr(r, "weight", 0.0) or 0.0), now)
                for r in sig.itertuples()]
        con.executemany(
            """INSERT OR REPLACE INTO momentum_shadow_predictions
               (prediction_date, ticker, kind, horizon, mom_score, mom_pct_rank,
                is_buy_candidate, weight, created_at)
               VALUES (?,?,?,?,?,?,?,?,?)""", rows)
        n_buy = sum(r[6] for r in rows)
        print(f"momentum_shadow: {kind} — {len(rows)} names, {n_buy} BUY candidates written for {pred_date}")
        total += len(rows)
    con.commit(); con.close()
    print(f"momentum_shadow: wrote {total} shadow rows (SHADOW ONLY — no live BUYs)")
    return 0


if __name__ == "__main__":
    # 2026-07-14: as_of used to default to None, which DISARMED the guard at
    # line ~118 (`if as_of is not None and pred_date != as_of: skip`). During the
    # 429 storm daily_prices froze at 2026-06-23; momentum_shadow trimmed the
    # sparse tail, fell back to 06-23, and INSERT OR REPLACE'd the SAME picks
    # every night from 07-06 to 07-09 -- exit 0, four 436-byte logs, no alarm.
    # Anchoring to the real last session makes a frozen feed a LOUD skip.
    from utils.market_calendar import last_completed_session as _lcs
    _as_of = sys.argv[1] if len(sys.argv) > 1 else str(_lcs())
    sys.exit(main(_as_of))
