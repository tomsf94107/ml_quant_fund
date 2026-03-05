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

    # ── 1. Get calibrated probabilities ──────────────────────────────────────
    try:
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
    # Apply BOTH risk multiplier AND sentiment multiplier to today's signal.
    # Backtest uses only risk_mult (no historical sentiment available).
    today_prob_eff  = float(sdf["prob"].iloc[-1]) * risk_mult * sent_mult
    today_prob_eff  = round(min(max(today_prob_eff, 0.0), 0.95), 4)
    today_gated     = bool(gate.iloc[-1])
    today_signal    = (
        "BUY"
        if today_prob_eff > confidence_threshold and not today_gated
        else "HOLD"
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
