"""
signals/momentum_signal.py — cross-sectional momentum signal (VALIDATED Step 2b).

This is the signal that survived the full gauntlet: full-history + per-regime +
strict purged-WF (positive 4/4 OOS folds, mean net Sharpe ~+1.0, strongest in the
current 2025-26 regime). It REPLACES the broken per-ticker direction model whose
DOWN calls were inverted.

Construction is lifted EXACTLY from analysis/momentum_purged_wf.py so live matches
backtest:
    mom_6_1  = 126d return - 21d return   (6-month momentum, ex most-recent month)
    mom_12_1 = 252d return - 21d return   (12-month momentum, ex most-recent month)
Cross-sectional: rank ALL names each day, LONG the top decile (highest momentum),
avoid the bottom. This is a UNIVERSE-LEVEL signal — it must see every name at once,
so it CANNOT run inside the per-ticker daily_runner loop; it runs as a post-pass.

Pure functions — no DB, no side effects. Tested against the backtest numbers.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

MOM_HORIZON = 20          # 20-trading-day hold, matches validation
DECILE = 0.10             # top decile = BUY candidates
LOOKBACKS = {"mom_6_1": 126, "mom_12_1": 252}
SKIP = 21                 # skip most-recent month (standard momentum)


def compute_momentum(close_panel: pd.DataFrame, kind: str = "mom_6_1") -> pd.Series:
    """Cross-sectional momentum score for the LAST date in the panel.

    Args:
        close_panel: date x ticker DataFrame of close prices (sorted ascending).
        kind: 'mom_6_1' or 'mom_12_1'.
    Returns:
        Series ticker -> momentum score (higher = stronger momentum), NaNs dropped.
    """
    if kind not in LOOKBACKS:
        raise ValueError(f"kind must be one of {list(LOOKBACKS)}, got {kind!r}")
    lb = LOOKBACKS[kind]
    if len(close_panel) < lb + 1:
        raise ValueError(f"need >= {lb+1} rows for {kind}, got {len(close_panel)}")
    long_ret  = close_panel.iloc[-1] / close_panel.iloc[-1 - lb]  - 1.0
    short_ret = close_panel.iloc[-1] / close_panel.iloc[-1 - SKIP] - 1.0
    score = long_ret - short_ret
    return score.dropna()


def _load_bucket_map():
    """ticker -> bucket from tickers_metadata.csv (the sector-like grouping)."""
    try:
        import pandas as _pd
        meta = _pd.read_csv(ROOT / "tickers_metadata.csv")
        return dict(zip(meta["ticker"].str.upper(), meta["bucket"].fillna("UNK")))
    except Exception:
        return {}


def rank_signal(close_panel: pd.DataFrame, kind: str = "mom_6_1",
                decile: float = DECILE, bucket_cap: int | None = None) -> pd.DataFrame:
    """Rank the universe by momentum; mark the top decile as BUY candidates.

    Returns a DataFrame [ticker, mom_score, mom_pct_rank, is_buy_candidate],
    sorted best-to-worst. BUY candidate = top `decile` of the cross-section.

    bucket_cap: if set (e.g. 3), enforce at most that many BUY candidates per
    bucket (tickers_metadata.csv 'bucket' column). Walks the ranked list and
    skips a name once its bucket is full, refilling from the next-best — keeps
    the basket size but caps single-sector concentration. Validated May 31 2026:
    cap=3 is equal-or-better Sharpe in 4/5 years, cuts 2022 bear concentration
    34%->22%, near-zero return cost. None = no cap (unchanged behavior).
    """
    score = compute_momentum(close_panel, kind)
    if len(score) < 10:
        return pd.DataFrame(columns=["ticker", "mom_score", "mom_pct_rank", "is_buy_candidate"])
    pct = score.rank(pct=True)
    k = max(1, int(len(score) * decile))
    ranked = score.sort_values(ascending=False)  # best first
    if bucket_cap is None:
        top = set(ranked.head(k).index)
    else:
        bmap = _load_bucket_map()
        top = set(); per = {}
        for t in ranked.index:
            b = bmap.get(t, "UNK")
            if per.get(b, 0) < bucket_cap:
                top.add(t); per[b] = per.get(b, 0) + 1
            if len(top) >= k:
                break
    out = pd.DataFrame({
        "ticker": score.index,
        "mom_score": score.values,
        "mom_pct_rank": pct.values,
        "is_buy_candidate": [t in top for t in score.index],
    }).sort_values("mom_score", ascending=False).reset_index(drop=True)
    return out


if __name__ == "__main__":
    # quick self-test against the validation data path
    from features.builder import _download
    tickers = [t.strip().upper() for t in (ROOT / "tickers.txt").read_text().splitlines()
               if t.strip() and not t.startswith("#")]
    closes = {}
    for tk in tickers:
        try:
            closes[tk] = _download(tk, "2024-01-01", None).set_index("date")["close"]
        except Exception:
            pass
    panel = pd.DataFrame(closes).sort_index()
    panel.index = pd.to_datetime(panel.index)
    sig = rank_signal(panel, "mom_6_1")
    n_buy = int(sig["is_buy_candidate"].sum())
    print(f"momentum signal as of {panel.index[-1].date()}: {len(sig)} names, {n_buy} BUY candidates")
    print(sig.head(n_buy).to_string(index=False))
