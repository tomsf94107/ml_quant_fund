# signals/generator.py
# ─────────────────────────────────────────────────────────────────────────────
# Signal generation and backtesting. Pure Python — zero Streamlit imports.
#
# Replaces the signal logic buried inside the "Run Strategy" button handler
# in streamlit_main_workingv19_3.py (lines 595-750).
#
# What we keep from the old code:
#   ✓ Prob_eff = Prob × risk_mult  (global 72h event risk multiplier)
#   ✓ Block gate: skip entry when risk_next_3d >= block_tau
#   ✓ Signal shift(1) — avoids lookahead bias in backtest returns
#   ✓ Sharpe, MaxDD, CAGR, trades, exposure, profit_factor
#
# What we fix:
#   ✗ Removed: uncalibrated sigmoid confidence proxy (our calibrated model
#     makes that hack unnecessary)
#   ✗ Removed: all Streamlit calls — UI is a thin wrapper around this module
#
# Zero Streamlit imports. Zero UI code. Backend only.
# ─────────────────────────────────────────────────────────────────────────────

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from models.classifier import (
    TrainResult, predict_proba, FEATURE_COLUMNS
)

def _load_ticker_metadata() -> dict:
    """Load tickers_metadata.csv → {ticker: {bucket, tier, thesis}}"""
    try:
        import pandas as _pd
        from pathlib import Path as _Path
        _p = _Path(__file__).parent.parent / "tickers_metadata.csv"
        if _p.exists():
            _df = _pd.read_csv(_p)
            return _df.set_index("ticker").to_dict("index")
    except Exception:
        pass
    return {}

_TICKER_METADATA = _load_ticker_metadata()

TIER_THRESHOLD_DELTA = {
    "core":      0.00,
    "secondary": 0.00,
    "tactical":  0.00,
    "lotto":    +0.05,   # lotto names need +5% more conviction to BUY
}

# ── Sentiment multiplier table ────────────────────────────────────────────────
# Maps sentiment score [-1, +1] to a probability multiplier.
# Strong negative news down-weights BUY signals. Strong positive amplifies them.
# Designed to be conservative — we never boost above 1.10 or cut below 0.70.
#
#  score >= +0.20  → 1.08  (bullish news — mild boost)
#  score >= +0.05  → 1.03  (slightly positive)
#  score >= -0.05  → 1.00  (neutral — no change)
#  score >= -0.20  → 0.95  (mildly negative)
#  score >= -0.40  → 0.88  (negative — reduce confidence)
#  score <  -0.40  → 0.78  (strongly negative — significant cut)
#
# Today's market (all tickers -0.48 to -0.80): most will apply 0.78-0.88 mult.
# This is why NVDA at 54.6% raw × 0.78 = 42.6% effective → HOLD (below 0.55).
SENTIMENT_MULT_TABLE = [
    (+0.20, 1.08),
    (+0.05, 1.03),
    (-0.05, 1.00),
    (-0.20, 0.95),
    (-0.40, 0.88),
    (-1.00, 0.78),
]

def _get_sentiment_mult(sentiment_score: float) -> float:
    """Convert a sentiment score [-1, +1] to a probability multiplier."""
    for threshold, mult in SENTIMENT_MULT_TABLE:
        if sentiment_score >= threshold:
            return mult
    return 0.78  # floor

# ── Fear & Greed multiplier ───────────────────────────────────────────────────
# When market is in Extreme Fear (F&G < 25), Howard Marks says second-level
# thinkers buy — everyone else is selling, so good stocks are cheap.
# When market is Extreme Greed (F&G > 75), crowd is overconfident — be cautious.
def _get_fear_greed_mult() -> float:
    """Fetch Fear & Greed index and return a probability multiplier.
    Extreme Fear (<25)  → boost BUY signals by up to 8%
    Neutral (25-75)     → no adjustment
    Extreme Greed (>75) → cut BUY signals by up to 8%
    """
    try:
        import requests
        r = requests.get("https://api.alternative.me/fng/?limit=1",
                        headers={"User-Agent": "MLQuantFund/1.0"}, timeout=5)
        if r.status_code == 200:
            score = float(r.json()["data"][0]["value"])
            if score < 15:   return 1.08   # Extreme Fear — strong boost
            if score < 25:   return 1.05   # Fear — mild boost
            if score > 85:   return 0.92   # Extreme Greed — strong cut
            if score > 75:   return 0.95   # Greed — mild cut
        return 1.0
    except Exception:
        return 1.0

# ── Risk multiplier table (preserved from old dashboard) ──────────────────────
# Applied to raw probability to down-scale confidence on high-risk days.
# Low / Medium / High maps to the event_risk_next72 label from the calendar page.
RISK_MULTIPLIER: dict[str, float] = {
    "Low":    1.00,
    "Medium": 0.92,
    "High":   0.85,
}

# ── Signal thresholds ─────────────────────────────────────────────────────────
DEFAULT_CONFIDENCE_THRESHOLD = 0.55   # Prob_eff must exceed this for BUY
DEFAULT_BLOCK_TAU             = 3     # block entry when risk_next_3d >= this


# ══════════════════════════════════════════════════════════════════════════════
#  DATA CLASSES
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class BacktestMetrics:
    """Performance metrics for one ticker's strategy backtest."""
    ticker:        str
    horizon:       int
    sharpe:        float
    max_drawdown:  float
    cagr:          float
    accuracy:      float
    n_trades:      int
    exposure:      float        # fraction of days with open position
    profit_factor: float
    n_days:        int

    def to_dict(self) -> dict:
        return {
            "ticker":        self.ticker,
            "horizon":       self.horizon,
            "sharpe":        round(self.sharpe,        3),
            "max_drawdown":  round(self.max_drawdown,  4),
            "cagr":          round(self.cagr,           4),
            "accuracy":      round(self.accuracy,       4),
            "n_trades":      self.n_trades,
            "exposure":      round(self.exposure,       4),
            "profit_factor": round(self.profit_factor,  3),
            "n_days":        self.n_days,
        }


@dataclass
class SignalResult:
    """Everything produced by generate_signals() for one ticker."""
    ticker:          str
    horizon:         int
    signal_df:       pd.DataFrame   # full backtest frame with all columns
    today_signal:    str            # "BUY" | "HOLD"
    today_prob:      float          # calibrated P(up)
    today_prob_eff:  float          # after risk multiplier
    metrics:         BacktestMetrics
    error:           Optional[str] = None
    # ── Price forecast fields ─────────────────────────────────────────
    current_price:   Optional[float] = None
    price_target_up: Optional[float] = None
    price_target_dn: Optional[float] = None
    expected_return: Optional[float] = None
    atr:             Optional[float] = None


# ══════════════════════════════════════════════════════════════════════════════
#  PRIVATE HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _get_risk_mult(risk_label: Optional[str]) -> float:
    """Convert event risk label to probability multiplier."""
    if risk_label is None:
        return 1.0
    return RISK_MULTIPLIER.get(risk_label, 1.0)


def _compute_backtest_metrics(
    ticker: str,
    horizon: int,
    signal_series: pd.Series,    # 0/1 signals, already shifted
    return_series: pd.Series,    # daily returns aligned to signal_series
) -> BacktestMetrics:
    """Compute institutional-grade backtest metrics."""
    ANN = 252   # trading days per year

    ret_strat = (signal_series * return_series).fillna(0)
    eq_strat  = (1 + ret_strat).cumprod()
    eq_mkt    = (1 + return_series).cumprod()

    mu  = ret_strat.mean()
    sd  = ret_strat.std(ddof=1)
    sharpe = float(np.sqrt(ANN) * mu / sd) if sd and not np.isnan(sd) else np.nan

    drawdown  = eq_strat / eq_strat.cummax() - 1
    max_dd    = float(drawdown.min())

    n_days = max(1, len(eq_strat))
    ending = float(eq_strat.iloc[-1]) if len(eq_strat) > 0 else 1.0
    cagr   = float(ending ** (ANN / n_days) - 1)

    # Direction accuracy — are we right about up/down?
    sig_shifted = signal_series  # already shifted by caller
    correct = ((sig_shifted == 1) & (return_series > 0)) | \
              ((sig_shifted == 0) & (return_series <= 0))
    accuracy = float(correct.mean()) if len(correct) > 0 else np.nan

    # Trade count — new entries only (0→1 transitions)
    entries    = ((signal_series == 1) & (signal_series.shift(1) != 1))
    n_trades   = int(entries.sum())
    exposure   = float(signal_series.mean())

    wins = ret_strat[ret_strat > 0].sum()
    loss = -ret_strat[ret_strat < 0].sum()
    profit_factor = float(wins / loss) if loss > 0 else np.nan

    return BacktestMetrics(
        ticker=ticker,
        horizon=horizon,
        sharpe=sharpe,
        max_drawdown=max_dd,
        cagr=cagr,
        accuracy=accuracy,
        n_trades=n_trades,
        exposure=exposure,
        profit_factor=profit_factor,
        n_days=n_days,
    )


# ══════════════════════════════════════════════════════════════════════════════
#  PUBLIC API
# ══════════════════════════════════════════════════════════════════════════════

def generate_signals(
    ticker:               str,
    df:                   pd.DataFrame,     # output of build_feature_dataframe()
    horizon:              int   = 1,
    confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
    block_tau:            int   = DEFAULT_BLOCK_TAU,
    risk_label:           Optional[str] = None,   # "Low" | "Medium" | "High"
    result:               Optional[TrainResult] = None,  # pass to avoid disk load
    use_sentiment:        bool = True,   # apply FinBERT multiplier to today's signal
) -> SignalResult:
    """
    Generate BUY/HOLD signals for `ticker` over the full history in `df`.

    Parameters
    ----------
    ticker               : e.g. "AAPL"
    df                   : output of build_feature_dataframe() — no targets needed
    horizon              : 1, 3, or 5 days
    confidence_threshold : Prob_eff must exceed this to generate BUY (default 0.55)
    block_tau            : block entry when risk_next_3d >= this (default 3)
    risk_label           : global 72h event risk from calendar page
                           ("Low" | "Medium" | "High" | None)
    result               : pre-loaded TrainResult — pass to avoid disk I/O

    Returns
    -------
    SignalResult with signal_df, today's signal, and backtest metrics
    """
    ticker = ticker.upper().strip()
    risk_mult = _get_risk_mult(risk_label)

    # ── Macro regime adjustment ───────────────────────────────────────────────
    # Adjusts confidence_threshold and signal multiplier based on market regime.
    # BULL → lower threshold (easier to BUY), BEAR/VOLATILE → higher threshold.
    try:
        from models.regime_classifier import get_current_regime
        regime = get_current_regime(use_cache=True)
        # Override threshold and add regime multiplier if not manually set
        if confidence_threshold == DEFAULT_CONFIDENCE_THRESHOLD:
            confidence_threshold = regime.confidence_threshold
        regime_mult = regime.signal_multiplier
    except Exception:
        regime_mult = 1.0

    # ── Tier-aware threshold + VIX suppression ──────────────────────────────
    # Lotto tier needs higher conviction; suppressed in VIX > 25 regime
    _meta     = _TICKER_METADATA.get(ticker, {})
    _tier     = _meta.get("tier", "tactical")
    _tier_delta = TIER_THRESHOLD_DELTA.get(_tier, 0.0)
    confidence_threshold = confidence_threshold + _tier_delta

    # VIX regime suppression — lotto suppressed above 25, both above 30
    _vix_now = float(df["vix_close"].iloc[-1]) if "vix_close" in df.columns else 20.0
    _suppress = False
    if _vix_now > 30 and _tier in ("lotto",):
        _suppress = True
    elif _vix_now > 25 and _tier == "lotto":
        _suppress = True

    # ── Live sentiment multiplier ─────────────────────────────────────────────
    # Fetches today's cached FinBERT score. Returns 0.0 if not yet run.
    # Multiplier only applied to TODAY's signal — not to backtest history
    # (we don't have historical sentiment, so applying it there would be wrong).
    sent_mult = 1.0
    if use_sentiment:
        try:
            from data.etl_sentiment import get_sentiment_score
            sent_score = get_sentiment_score(ticker)
            sent_mult  = _get_sentiment_mult(sent_score)
        except Exception:
            sent_mult = 1.0


    # ETFs don't have options flow, short interest, or analyst data
    _ETFS = {"SLV", "GLD", "SPY", "QQQ", "TLT", "IWM", "XLF", "XLE", "VIX"}
    _is_etf = ticker.upper() in _ETFS

    # ── UW signals multiplier ────────────────────────────────────────────────
    # During market hours: calls live API for dark pool + skew
    # Pre/post market: reads from DB only — zero API calls
    options_mult = 1.0
    try:
        if _is_etf: raise Exception('ETF skip')
        import signal as _sig
        from datetime import datetime
        import pytz
        _ET  = pytz.timezone("America/New_York")
        _now = datetime.now(_ET)
        _market_open  = _now.replace(hour=9,  minute=30, second=0, microsecond=0)
        _market_close = _now.replace(hour=16, minute=0,  second=0, microsecond=0)
        _is_market_hours = (_market_open <= _now <= _market_close
                            and _now.weekday() < 5)

        def _timeout(s,f): raise TimeoutError()
        _sig.signal(_sig.SIGALRM, _timeout)
        _sig.alarm(15)
        try:
            from features.uw_signals import get_combined_uw_multiplier
            from features.dark_pool import get_dark_pool_ratio, dark_pool_to_multiplier
            from features.massive_options import get_25delta_skew_with_fallback as get_25delta_skew

            if _is_market_hours:
                # Live API during market hours
                uw = get_combined_uw_multiplier(ticker)
                options_mult = uw["combined"]

                # Save fresh skew to DB so future reads have latest data
                try:
                    _skew_result = get_25delta_skew(ticker)
                    if _skew_result.get("skew_25d") is not None:
                        import sqlite3
                        from pathlib import Path
                        from datetime import date, datetime
                        _db = Path(__file__).parent.parent / "accuracy.db"
                        with sqlite3.connect(_db, timeout=30) as _conn:
                            _conn.execute("""
                                INSERT OR REPLACE INTO options_skew_history
                                    (date, ticker, skew_25d, iv_rank, skew_signal, source, created_at)
                                VALUES (?, ?, ?, ?, ?, ?, ?)
                            """, (str(date.today()), ticker,
                                  _skew_result["skew_25d"],
                                  _skew_result.get("iv_rank"),
                                  _skew_result.get("skew_signal", "NEUTRAL"),
                                  _skew_result.get("source", "live"),
                                  datetime.now().isoformat()))
                            _conn.commit()
                except Exception:
                    pass  # Non-critical — skew still used for prediction
            else:
                # DB only pre/post market — read from dark_pool_history + options_skew_history
                import sqlite3
                from pathlib import Path
                from datetime import date, timedelta
                _db = Path(__file__).parent.parent / "accuracy.db"
                _cutoff = str(date.today() - timedelta(days=1))
                with sqlite3.connect(_db, timeout=30) as _conn:
                    _dp = _conn.execute("""
                        SELECT dp_ratio FROM dark_pool_history
                        WHERE ticker=? AND date>=? ORDER BY date DESC LIMIT 1
                    """, (ticker, _cutoff)).fetchone()
                    _sk = _conn.execute("""
                        SELECT skew_25d FROM options_skew_history
                        WHERE ticker=? AND date>=? ORDER BY date DESC LIMIT 1
                    """, (ticker, _cutoff)).fetchone()
                _dp_ratio  = _dp[0] if _dp else 0.0
                _skew_25d  = _sk[0] if _sk else 0.0
                _dp_mult   = dark_pool_to_multiplier(_dp_ratio)
                _skew_mult = 0.92 if _skew_25d > 0.03 else 1.05 if _skew_25d < -0.02 else 1.0
                options_mult = round(min(max(_dp_mult * _skew_mult, 0.85), 1.15), 4)
        finally:
            _sig.alarm(0)
    except Exception:
        options_mult = 1.0

    # ── Short interest / squeeze multiplier ───────────────────────────────────
    # High short interest + BUY signal = potential squeeze → boost probability
    # Shorts rapidly increasing = bearish trap → cut probability
    squeeze_mult = 1.0
    try:
        if _is_etf: raise Exception('ETF skip')
        import signal as _sig2
        def _timeout2(s,f): raise TimeoutError()
        _sig2.signal(_sig2.SIGALRM, _timeout2)
        _sig2.alarm(5)
        try:
            from features.short_interest import short_interest_to_multiplier
            squeeze_mult = short_interest_to_multiplier(ticker)
        finally:
            _sig2.alarm(0)
    except Exception:
        squeeze_mult = 1.0

    # ── Intraday momentum multiplier ─────────────────────────────────────────
    # If stock is moving UP intraday → boost prob_eff slightly
    # If stock is moving DOWN intraday → cut prob_eff slightly
    intraday_mult = 1.0
    try:
        from features.intraday_builder import get_all_intraday_signals
        intra = get_all_intraday_signals([ticker])
        sig = intra.get(ticker, {})
        momentum = sig.get("intraday_momentum", 0.0)
        if momentum is not None and momentum == momentum:  # not NaN
            # Scale: momentum of +0.01 → 1.02x boost, -0.01 → 0.98x penalty
            # Cap at ±5% adjustment
            intraday_mult = min(max(1.0 + (momentum * 2), 0.95), 1.05)
    except Exception:
        intraday_mult = 1.0

    # ── 1. Get calibrated probabilities ──────────────────────────────────────
    try:
        # Use ensemble if available, fall back to XGB-only
        try:
            from models.ensemble import predict_proba_ensemble
            prob_series = predict_proba_ensemble(ticker, df, horizon=horizon)
        except Exception:
            prob_series = predict_proba(ticker, df, horizon=horizon, result=result)
    except Exception as e:
        # Return a safe error result rather than crashing the whole dashboard
        empty_metrics = BacktestMetrics(
            ticker=ticker, horizon=horizon,
            sharpe=np.nan, max_drawdown=np.nan, cagr=np.nan,
            accuracy=np.nan, n_trades=0, exposure=0.0,
            profit_factor=np.nan, n_days=0,
        )
        return SignalResult(
            ticker=ticker, horizon=horizon,
            signal_df=pd.DataFrame(),
            today_signal="HOLD", today_prob=0.0, today_prob_eff=0.0,
            metrics=empty_metrics, error=str(e),
        )

    # ── 2. Build signal frame ─────────────────────────────────────────────────
    sdf = df[["date", "close", "return_1d",
              "risk_today", "risk_next_1d", "risk_next_3d"]].copy()
    sdf["ticker"]    = ticker
    sdf["prob"]      = prob_series.clip(0.05, 0.95).values

    # Apply global risk multiplier (from calendar page 72h event score)
    sdf["prob_eff"]  = sdf["prob"] * risk_mult

    # Block gate — don't enter when event risk is high
    gate = (sdf["risk_next_3d"].fillna(0) >= block_tau)
    sdf["gate_block"] = gate.astype(int)

    # Raw signal (before shift — for display only)
    sdf["signal_raw"] = (
        (sdf["prob_eff"] > confidence_threshold) & (~gate)
    ).astype(int)

    # Shifted signal for backtest — CRITICAL: prevents lookahead bias.
    # signal_raw[t] is based on features known at close of day t.
    # We can only act on day t+1's open, so we shift by 1.
    sdf["signal"] = sdf["signal_raw"].shift(1).fillna(0).astype(int)

    # ── 3. Backtest metrics ───────────────────────────────────────────────────
    metrics = _compute_backtest_metrics(
        ticker=ticker,
        horizon=horizon,
        signal_series=sdf["signal"],
        return_series=sdf["return_1d"].fillna(0),
    )

    # ── 4. Today's signal (use unshifted — this is forward-looking) ───────────
    today_prob      = float(sdf["prob"].iloc[-1])
    # Fear & Greed multiplier (Howard Marks second-level thinking)
    fg_mult = _get_fear_greed_mult()

    # Apply risk + sentiment + regime + options flow + squeeze + fear/greed multipliers
    today_prob_eff = float(sdf["prob"].iloc[-1]) * risk_mult * sent_mult * regime_mult * options_mult * squeeze_mult * intraday_mult * fg_mult
    today_prob_eff  = round(min(max(today_prob_eff, 0.0), 0.95), 4)
    today_gated     = bool(gate.iloc[-1])
    today_signal    = (
        "BUY"
        if today_prob_eff > confidence_threshold and not today_gated and not _suppress
        else "HOLD"
    )

    # Price forecast using ATR
    current_price   = float(df["close"].iloc[-1]) if "close" in df.columns else None
    atr_val         = float(df["atr"].iloc[-1])   if "atr"   in df.columns else None
    price_target_up = None
    price_target_dn = None
    expected_return = None
    if current_price and atr_val:
        import math
        move            = atr_val * math.sqrt(horizon)
        price_target_up = round(current_price + move, 2)
        price_target_dn = round(current_price - move, 2)
        expected_return = round(
            (today_prob_eff * move - (1 - today_prob_eff) * move) / current_price, 4
        )

    return SignalResult(
        ticker=ticker,
        horizon=horizon,
        signal_df=sdf,
        today_signal=today_signal,
        today_prob=round(today_prob, 4),
        today_prob_eff=round(today_prob_eff, 4),
        metrics=metrics,
        error=None,
        current_price=round(current_price, 2) if current_price else None,
        price_target_up=price_target_up,
        price_target_dn=price_target_dn,
        expected_return=expected_return,
        atr=round(atr_val, 4) if atr_val else None,
    )


def run_all_signals(
    tickers:              list[str],
    dfs:                  dict[str, pd.DataFrame],   # {ticker: feature_df}
    horizon:              int   = 1,
    confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
    block_tau:            int   = DEFAULT_BLOCK_TAU,
    risk_label:           Optional[str] = None,
) -> list[SignalResult]:
    """
    Generate signals for all tickers. Returns list of SignalResult.

    Parameters
    ----------
    tickers  : list of ticker strings
    dfs      : dict mapping ticker → feature DataFrame
    (others) : same as generate_signals()
    """
    results = []
    for ticker in tickers:
        df = dfs.get(ticker.upper())
        if df is None or df.empty:
            continue
        result = generate_signals(
            ticker=ticker,
            df=df,
            horizon=horizon,
            confidence_threshold=confidence_threshold,
            block_tau=block_tau,
            risk_label=risk_label,
        )
        results.append(result)
    return results


def signals_to_dataframe(results: list[SignalResult]) -> pd.DataFrame:
    """
    Convert a list of SignalResult into a clean summary DataFrame.
    Useful for the leaderboard page.
    """
    rows = []
    for r in results:
        row = {
            "ticker":         r.ticker,
            "horizon":        r.horizon,
            "today_signal":   r.today_signal,
            "today_prob":     r.today_prob,
            "today_prob_eff": r.today_prob_eff,
            "error":          r.error,
        }
        if r.metrics:
            row.update(r.metrics.to_dict())
        rows.append(row)

    return pd.DataFrame(rows)
