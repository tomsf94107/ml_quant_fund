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
    from features.builder import _download
    closes = {}
    for tk in tickers:
        try:
            closes[tk] = _download(tk, start, None).set_index("date")["close"]
        except Exception:
            pass
    panel = pd.DataFrame(closes).sort_index()
    panel.index = pd.to_datetime(panel.index)
    return panel


def main(as_of=None):
    con = sqlite3.connect(str(DB), timeout=30)
    con.executescript(DDL)
    tickers = load_universe()
    panel = build_panel(tickers)
    if as_of is not None:
        cutoff = pd.Timestamp(as_of)
        panel = panel[panel.index <= cutoff]
    if panel.empty:
        print("momentum_shadow: empty panel, nothing to write")
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
    _as_of = sys.argv[1] if len(sys.argv) > 1 else None
    sys.exit(main(_as_of))
