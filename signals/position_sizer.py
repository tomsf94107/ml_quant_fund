"""
signals/position_sizer.py
─────────────────────────────────────────────────────────────────────────────
Dynamic position sizing for ML Quant Fund.

Uses fractional Kelly criterion calibrated from live accuracy data,
adjusted for regime, volatility, VIX level, and portfolio heat.

Target: 80-100% annual return on $200-400K portfolio.
Strategy: Long + Short, concentrated positions (5-30% per trade).

Kelly Formula:
    f* = (W/L * win_rate - loss_rate) / (W/L)
    where W = avg win %, L = avg loss %, win_rate = historical accuracy

Fractional Kelly: use 25-50% of full Kelly to reduce variance.

Position size pipeline:
    base_kelly → confidence_mult → regime_mult → vix_mult → vol_mult → heat_mult → clamp

Usage:
    from signals.position_sizer import get_position_size, get_portfolio_plan

    # Single ticker
    size = get_position_size("AAPL", prob_eff=0.82, confidence="HIGH",
                              portfolio_value=300000)
    print(size)  # {"ticker": "AAPL", "direction": "LONG", "pct": 0.22, "dollars": 66000, ...}

    # Full portfolio plan from today's signals
    plan = get_portfolio_plan(signals, portfolio_value=300000)
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path
from typing import Optional
import json
import numpy as np

# ── Config ────────────────────────────────────────────────────────────────────
DB_PATH         = Path("accuracy.db")
PORTFOLIO_CONFIG = Path("config/portfolio.json")
KELLY_FRACTION  = 0.35      # use 35% of full Kelly (conservative)
MIN_POSITION    = 0.05      # 5% minimum position
MAX_POSITION    = 0.30      # 30% maximum position
MAX_LONG_HEAT   = 1.00      # 100% max long exposure
MAX_SHORT_HEAT  = 0.50      # 50% max short exposure
MIN_SAMPLES     = 5         # minimum data points to use live Kelly
DEFAULT_WIN_RATE = 0.55     # fallback win rate (slightly above random)
DEFAULT_AVG_WIN  = 1.5      # fallback avg win %
DEFAULT_AVG_LOSS = 1.2      # fallback avg loss %

# Regime multipliers — scale down in bad regimes
REGIME_MULT = {
    "BULL":     1.00,
    "NEUTRAL":  0.85,
    "BEAR":     0.65,
    "VOLATILE": 0.50,
}

# Confidence multipliers
CONFIDENCE_MULT = {
    "HIGH":   1.00,
    "MEDIUM": 0.60,
    "LOW":    0.00,   # never trade LOW confidence
}

# High-volatility tickers get smaller positions
HIGH_VOL_TICKERS = {"TSLA", "PLTR", "SMCI", "NVDA", "AMD", "MRNA", "CRWD",
                    "SNOW", "DDOG", "OPEN", "GME", "ASTS", "OKLO", "QUBT"}
HIGH_VOL_MULT = 0.75   # reduce by 25% for high-vol tickers


@dataclass
class PositionSize:
    ticker:          str
    direction:       str        # "LONG" or "SHORT"
    prob_eff:        float
    confidence:      str
    kelly_full:      float      # raw Kelly fraction
    kelly_fractional: float     # after Kelly fraction applied
    regime_mult:     float
    vix_mult:        float
    vol_mult:        float
    heat_mult:       float
    final_pct:       float      # final position as % of portfolio
    dollars:         float      # final position in dollars
    shares:          Optional[int] = None   # shares to buy (if price provided)
    rationale:       str = ""


@dataclass
class PortfolioPlan:
    date:            str
    portfolio_value: float
    regime:          str
    vix_level:       float
    positions:       list[PositionSize] = field(default_factory=list)
    total_long_pct:  float = 0.0
    total_short_pct: float = 0.0
    total_heat:      float = 0.0
    expected_return: float = 0.0   # weighted avg expected return
    notes:           list[str] = field(default_factory=list)



def get_portfolio_value() -> float:
    """Read portfolio value from config/portfolio.json. Falls back to 300000."""
    try:
        if PORTFOLIO_CONFIG.exists():
            cfg = json.loads(PORTFOLIO_CONFIG.read_text())
            return float(cfg.get("portfolio_value", 300000))
    except Exception:
        pass
    return 300000.0

def get_open_positions() -> list[dict]:
    """Read open positions from config/portfolio.json."""
    try:
        if PORTFOLIO_CONFIG.exists():
            cfg = json.loads(PORTFOLIO_CONFIG.read_text())
            return cfg.get("open_positions", {}).get("positions", [])
    except Exception:
        pass
    return []

def _get_ticker_stats(ticker: str, lookback_days: int = 90) -> dict:
    """
    Fetch historical win rate, avg win, avg loss for a ticker from accuracy.db.
    Falls back to defaults if insufficient data.
    """
    if not DB_PATH.exists():
        return {
            "win_rate": DEFAULT_WIN_RATE,
            "avg_win":  DEFAULT_AVG_WIN / 100,
            "avg_loss": DEFAULT_AVG_LOSS / 100,
            "n":        0,
        }

    cutoff = (date.today() - timedelta(days=lookback_days)).isoformat()
    try:
        conn = sqlite3.connect(DB_PATH)
        row = conn.execute("""
            SELECT
                COUNT(*) as n,
                AVG(CASE WHEN (prob_up>0.5 AND actual_up=1) OR
                              (prob_up<=0.5 AND actual_up=0)
                         THEN 1.0 ELSE 0.0 END) as win_rate,
                AVG(CASE WHEN actual_up=1 THEN actual_return ELSE NULL END) as avg_win,
                AVG(CASE WHEN actual_up=0 THEN ABS(actual_return) ELSE NULL END) as avg_loss
            FROM predictions p
            JOIN outcomes o ON p.ticker=o.ticker
                AND p.prediction_date=o.prediction_date
                AND p.horizon=o.horizon
            WHERE p.ticker=? AND p.horizon=1
              AND p.prediction_date >= ?
        """, (ticker, cutoff)).fetchone()
        conn.close()

        n = row[0] or 0
        if n < MIN_SAMPLES:
            return {
                "win_rate": DEFAULT_WIN_RATE,
                "avg_win":  DEFAULT_AVG_WIN / 100,
                "avg_loss": DEFAULT_AVG_LOSS / 100,
                "n":        n,
                "source":   "default (insufficient data)",
            }

        win_rate = float(row[1]) if row[1] else DEFAULT_WIN_RATE
        avg_win  = float(row[2]) if row[2] else DEFAULT_AVG_WIN / 100
        avg_loss = float(row[3]) if row[3] else DEFAULT_AVG_LOSS / 100

        return {
            "win_rate": win_rate,
            "avg_win":  avg_win,
            "avg_loss": avg_loss,
            "n":        n,
            "source":   f"live data ({n} samples)",
        }
    except Exception:
        return {
            "win_rate": DEFAULT_WIN_RATE,
            "avg_win":  DEFAULT_AVG_WIN / 100,
            "avg_loss": DEFAULT_AVG_LOSS / 100,
            "n":        0,
            "source":   "default (db error)",
        }


def _compute_kelly(win_rate: float, avg_win: float, avg_loss: float) -> float:
    """
    Kelly criterion: f* = (W/L * p - q) / (W/L)
    where p = win_rate, q = 1 - win_rate, W = avg_win, L = avg_loss
    """
    if avg_loss <= 0:
        return 0.0
    win_loss_ratio = avg_win / avg_loss
    loss_rate = 1 - win_rate
    kelly = (win_loss_ratio * win_rate - loss_rate) / win_loss_ratio
    return max(0.0, kelly)  # Kelly can be negative = don't trade


def _get_vix_mult(vix_level: float) -> float:
    """Scale position size based on VIX level."""
    if vix_level >= 35:   return 0.40
    if vix_level >= 30:   return 0.55
    if vix_level >= 25:   return 0.70
    if vix_level >= 20:   return 0.85
    if vix_level >= 15:   return 1.00
    return 1.10  # low VIX = slight boost


def _get_current_regime() -> tuple[str, float]:
    """Get current regime label and VIX from cache."""
    import json
    cache_path = Path("models/saved/regime_cache.json")
    if cache_path.exists():
        try:
            cache = json.loads(cache_path.read_text())
            return cache.get("label", "NEUTRAL"), float(cache.get("vix_level", 20))
        except Exception:
            pass
    return "NEUTRAL", 20.0


def get_position_size(
    ticker:          str,
    prob_eff:        float,
    confidence:      str,
    portfolio_value: float,
    current_price:   Optional[float] = None,
    regime:          Optional[str] = None,
    vix_level:       Optional[float] = None,
    current_heat:    float = 0.0,   # current total long exposure as fraction
) -> PositionSize:
    """
    Compute optimal position size for a single signal.

    Parameters
    ----------
    ticker           : e.g. "AAPL"
    prob_eff         : effective probability (after all multipliers)
    confidence       : "HIGH" | "MEDIUM" | "LOW"
    portfolio_value  : total portfolio value in dollars
    current_price    : current stock price (for share count)
    regime           : override regime label
    vix_level        : override VIX level
    current_heat     : current portfolio long exposure (0.0-1.0)
    """
    ticker = ticker.upper()

    # Determine direction
    direction = "LONG" if prob_eff >= 0.5 else "SHORT"

    # Get regime and VIX
    if regime is None or vix_level is None:
        reg, vix = _get_current_regime()
        regime    = regime    or reg
        vix_level = vix_level or vix

    # Get ticker stats
    stats = _get_ticker_stats(ticker)
    win_rate = stats["win_rate"]
    avg_win  = stats["avg_win"]
    avg_loss = stats["avg_loss"]

    # For SHORT signals, use inverted stats
    if direction == "SHORT":
        win_rate = 1 - win_rate
        avg_win, avg_loss = avg_loss, avg_win

    # Compute Kelly
    kelly_full       = _compute_kelly(win_rate, avg_win, avg_loss)
    kelly_fractional = kelly_full * KELLY_FRACTION

    # Confidence multiplier — LOW = 0 (never trade)
    conf_mult = CONFIDENCE_MULT.get(confidence, 0.0)
    if conf_mult == 0.0:
        return PositionSize(
            ticker=ticker, direction=direction,
            prob_eff=prob_eff, confidence=confidence,
            kelly_full=0.0, kelly_fractional=0.0,
            regime_mult=0.0, vix_mult=0.0, vol_mult=0.0, heat_mult=0.0,
            final_pct=0.0, dollars=0.0,
            rationale="LOW confidence — no position"
        )

    # Regime multiplier
    reg_mult = REGIME_MULT.get(regime, 0.85)

    # VIX multiplier
    vix_mult = _get_vix_mult(vix_level)

    # Volatility multiplier for high-vol tickers
    vol_mult = HIGH_VOL_MULT if ticker in HIGH_VOL_TICKERS else 1.0

    # Portfolio heat multiplier — reduce size as portfolio fills up
    remaining_capacity = max(0.0, MAX_LONG_HEAT - current_heat)
    heat_mult = min(1.0, remaining_capacity / 0.30)  # starts reducing when < 30% capacity left

    # Compute final position
    raw_pct = kelly_fractional * conf_mult * reg_mult * vix_mult * vol_mult * heat_mult

    # Clamp to min/max
    if raw_pct < MIN_POSITION:
        final_pct = MIN_POSITION
    else:
        final_pct = min(raw_pct, MAX_POSITION)

    dollars = round(portfolio_value * final_pct, 2)
    shares  = int(dollars / current_price) if current_price and current_price > 0 else None

    # Build rationale string
    rationale = (
        f"Kelly={kelly_full:.1%} × {KELLY_FRACTION:.0%}frac={kelly_fractional:.1%} "
        f"× conf={conf_mult:.0%} × regime={reg_mult:.0%} × vix={vix_mult:.0%} "
        f"× vol={vol_mult:.0%} → {final_pct:.1%}"
    )

    return PositionSize(
        ticker=ticker,
        direction=direction,
        prob_eff=prob_eff,
        confidence=confidence,
        kelly_full=kelly_full,
        kelly_fractional=kelly_fractional,
        regime_mult=reg_mult,
        vix_mult=vix_mult,
        vol_mult=vol_mult,
        heat_mult=heat_mult,
        final_pct=final_pct,
        dollars=dollars,
        shares=shares,
        rationale=rationale,
    )


def get_portfolio_plan(
    signals:         list[dict],
    portfolio_value: float,
    regime:          Optional[str] = None,
    vix_level:       Optional[float] = None,
) -> PortfolioPlan:
    """
    Build a full portfolio plan from today's BUY signals.

    Parameters
    ----------
    signals         : list of signal dicts from daily_runner
                      each must have: ticker, signal, prob_eff, confidence, current_price
    portfolio_value : total portfolio value in dollars
    """
    if regime is None or vix_level is None:
        reg, vix = _get_current_regime()
        regime    = regime    or reg
        vix_level = vix_level or vix

    plan = PortfolioPlan(
        date=date.today().isoformat(),
        portfolio_value=portfolio_value,
        regime=regime,
        vix_level=vix_level,
    )

    # Filter to actionable signals only (BUY or future SELL)
    actionable = [s for s in signals if s.get("signal") in ("BUY", "SELL")]

    # Sort by prob_eff descending — highest conviction first
    actionable.sort(key=lambda s: abs(s.get("prob_eff", 0.5) - 0.5), reverse=True)

    current_long_heat  = 0.0
    current_short_heat = 0.0

    for sig in actionable:
        ticker     = sig.get("ticker", "")
        prob_eff   = float(sig.get("prob_eff", 0.5))
        confidence = sig.get("confidence", "LOW")
        price      = sig.get("current_price", None)

        # Check capacity
        direction = "LONG" if sig.get("signal") == "BUY" else "SHORT"
        if direction == "LONG" and current_long_heat >= MAX_LONG_HEAT:
            plan.notes.append(f"Skipped {ticker} LONG — portfolio at max long exposure")
            continue
        if direction == "SHORT" and current_short_heat >= MAX_SHORT_HEAT:
            plan.notes.append(f"Skipped {ticker} SHORT — portfolio at max short exposure")
            continue

        pos = get_position_size(
            ticker=ticker,
            prob_eff=prob_eff,
            confidence=confidence,
            portfolio_value=portfolio_value,
            current_price=price,
            regime=regime,
            vix_level=vix_level,
            current_heat=current_long_heat if direction == "LONG" else current_short_heat,
        )

        if pos.final_pct > 0:
            plan.positions.append(pos)
            if direction == "LONG":
                current_long_heat  += pos.final_pct
            else:
                current_short_heat += pos.final_pct

    plan.total_long_pct  = round(current_long_heat, 3)
    plan.total_short_pct = round(current_short_heat, 3)
    plan.total_heat      = round(current_long_heat + current_short_heat, 3)

    # Expected return = weighted avg of (prob_eff - 0.5) * 2 * avg_win per position
    if plan.positions:
        weighted_returns = []
        for pos in plan.positions:
            stats = _get_ticker_stats(pos.ticker)
            edge = (pos.prob_eff - 0.5) * 2  # 0 to 1 scale
            exp_ret = edge * stats["avg_win"] * pos.final_pct
            weighted_returns.append(exp_ret)
        plan.expected_return = round(sum(weighted_returns), 4)

    return plan


def format_plan(plan: PortfolioPlan) -> str:
    """Pretty-print a portfolio plan."""
    lines = []
    lines.append("=" * 65)
    lines.append(f"  PORTFOLIO PLAN — {plan.date}")
    lines.append(f"  Portfolio: ${plan.portfolio_value:,.0f} | Regime: {plan.regime} | VIX: {plan.vix_level:.1f}")
    lines.append("=" * 65)

    if not plan.positions:
        lines.append("  No actionable positions today.")
        return "\n".join(lines)

    lines.append(f"  {'TICKER':<8} {'DIR':<6} {'CONF':<8} {'SIZE%':<8} {'$AMOUNT':<12} {'SHARES':<8} {'PROB':<6}")
    lines.append("  " + "-" * 60)

    for pos in plan.positions:
        shares_str = str(pos.shares) if pos.shares else "N/A"
        lines.append(
            f"  {pos.ticker:<8} {pos.direction:<6} {pos.confidence:<8} "
            f"{pos.final_pct*100:>5.1f}%  ${pos.dollars:>10,.0f}  {shares_str:<8} {pos.prob_eff:.2f}"
        )

    lines.append("  " + "-" * 60)
    lines.append(f"  Total long  : {plan.total_long_pct*100:.1f}%  (max {MAX_LONG_HEAT*100:.0f}%)")
    lines.append(f"  Total short : {plan.total_short_pct*100:.1f}%  (max {MAX_SHORT_HEAT*100:.0f}%)")
    lines.append(f"  Total heat  : {plan.total_heat*100:.1f}%")
    lines.append(f"  Exp. return : {plan.expected_return*100:.3f}% (portfolio-weighted)")

    if plan.notes:
        lines.append("\n  Notes:")
        for note in plan.notes:
            lines.append(f"    - {note}")

    lines.append("=" * 65)
    return "\n".join(lines)


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Position sizer for ML Quant Fund")
    parser.add_argument("--portfolio", type=float, default=300000, help="Portfolio value in dollars")
    parser.add_argument("--ticker",    type=str,   default=None,   help="Single ticker to size")
    parser.add_argument("--prob",      type=float, default=0.75,   help="prob_eff for single ticker")
    parser.add_argument("--conf",      type=str,   default="HIGH", help="Confidence: HIGH/MEDIUM")
    parser.add_argument("--price",     type=float, default=None,   help="Current price for share count")
    args = parser.parse_args()

    if args.ticker:
        pos = get_position_size(
            ticker=args.ticker,
            prob_eff=args.prob,
            confidence=args.conf,
            portfolio_value=args.portfolio,
            current_price=args.price,
        )
        print(f"\nPosition Size for {pos.ticker}:")
        print(f"  Direction  : {pos.direction}")
        print(f"  Size       : {pos.final_pct*100:.1f}% = ${pos.dollars:,.0f}")
        if pos.shares:
            print(f"  Shares     : {pos.shares}")
        print(f"  Rationale  : {pos.rationale}")
    else:
        # Demo with sample signals
        sample_signals = [
            {"ticker": "AAPL",  "signal": "BUY", "prob_eff": 0.82, "confidence": "HIGH",   "current_price": 254.23},
            {"ticker": "NVDA",  "signal": "BUY", "prob_eff": 0.75, "confidence": "HIGH",   "current_price": 181.93},
            {"ticker": "TSLA",  "signal": "BUY", "prob_eff": 0.71, "confidence": "HIGH",   "current_price": 399.27},
            {"ticker": "MSFT",  "signal": "BUY", "prob_eff": 0.68, "confidence": "MEDIUM", "current_price": 399.41},
            {"ticker": "CRWD",  "signal": "BUY", "prob_eff": 0.65, "confidence": "MEDIUM", "current_price": 423.78},
        ]
        plan = get_portfolio_plan(sample_signals, portfolio_value=args.portfolio)
        print(format_plan(plan))
