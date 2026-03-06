# features/short_interest.py
# ─────────────────────────────────────────────────────────────────────────────
# Short interest signal using yfinance ticker.info (free, no API key).
#
# Key metrics:
#   short_ratio       : days to cover (short interest / avg daily volume)
#                       High = crowded short = squeeze risk if stock rallies
#   short_pct_float   : % of float sold short
#                       >10% = heavily shorted, >20% = extreme
#   squeeze_score     : combined signal — high short + BUY signal = squeeze setup
#
# How this helps:
#   1. HIGH short interest + BUY signal → potential SHORT SQUEEZE (amplified upside)
#   2. LOW short interest + BUY signal → clean breakout (no headwind)
#   3. HIGH short interest + HOLD → avoid (shorts know something, bearish)
#
# The squeeze_multiplier boosts ML probability when squeeze conditions are met.
# ─────────────────────────────────────────────────────────────────────────────

from __future__ import annotations

import warnings
from datetime import datetime
from typing import Optional
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ── Thresholds ────────────────────────────────────────────────────────────────
SHORT_RATIO_HIGH     = 5.0    # >5 days to cover = heavily shorted
SHORT_RATIO_EXTREME  = 10.0   # >10 days = extreme short interest
SHORT_PCT_HIGH       = 0.10   # >10% of float shorted = high
SHORT_PCT_EXTREME    = 0.20   # >20% of float = extreme (squeeze candidate)

# ── Multiplier bounds ─────────────────────────────────────────────────────────
SQUEEZE_BOOST_MAX    = 1.15   # max +15% boost on squeeze setup
BEARISH_CUT_MAX      = 0.90   # max -10% cut when shorts are piling in


def get_short_interest(ticker: str) -> dict:
    """
    Fetch short interest data for a ticker via yfinance.

    Returns dict with:
        short_ratio         : days to cover (float)
        short_pct_float     : % of float shorted (0-1)
        shares_short        : absolute shares short
        squeeze_score       : 0-1 score (higher = more squeeze potential)
        squeeze_signal      : "HIGH_SQUEEZE" | "MODERATE" | "LOW" | "BEARISH_TRAP"
        short_signal        : "BULLISH" | "NEUTRAL" | "BEARISH"
        squeeze_multiplier  : probability multiplier for signal generator
        error               : error string if failed
    """
    ticker = ticker.upper().strip()
    result = {
        "ticker":             ticker,
        "short_ratio":        None,
        "short_pct_float":    None,
        "shares_short":       None,
        "squeeze_score":      0.0,
        "squeeze_signal":     "MODERATE",
        "short_signal":       "NEUTRAL",
        "squeeze_multiplier": 1.0,
        "timestamp":          datetime.utcnow().isoformat(),
        "error":              None,
    }

    try:
        import yfinance as yf
        info = yf.Ticker(ticker).info

        # Extract short interest fields
        short_ratio      = info.get("shortRatio")         # days to cover
        short_pct_float  = info.get("shortPercentOfFloat") # 0.0 - 1.0
        shares_short     = info.get("sharesShort")
        shares_short_prior = info.get("sharesShortPriorMonth")

        if short_ratio is not None:
            result["short_ratio"] = round(float(short_ratio), 2)

        if short_pct_float is not None:
            result["short_pct_float"] = round(float(short_pct_float), 4)

        if shares_short is not None:
            result["shares_short"] = int(shares_short)

        # ── Short interest change (momentum) ──────────────────────────────────
        short_change = None
        if shares_short and shares_short_prior and shares_short_prior > 0:
            short_change = (shares_short - shares_short_prior) / shares_short_prior
            result["short_change_pct"] = round(float(short_change), 4)

        # ── Squeeze score (0-1) ───────────────────────────────────────────────
        score = 0.0
        pct   = result["short_pct_float"] or 0.0
        ratio = result["short_ratio"] or 0.0

        # Float % component (50% weight)
        if pct >= SHORT_PCT_EXTREME:
            score += 0.50
        elif pct >= SHORT_PCT_HIGH:
            score += 0.25
        elif pct >= 0.05:
            score += 0.10

        # Days to cover component (30% weight)
        if ratio >= SHORT_RATIO_EXTREME:
            score += 0.30
        elif ratio >= SHORT_RATIO_HIGH:
            score += 0.15
        elif ratio >= 2.0:
            score += 0.05

        # Short interest increasing = bearish momentum (penalty)
        if short_change and short_change > 0.10:
            score -= 0.10   # shorts piling in = bearish signal
        elif short_change and short_change < -0.10:
            score += 0.10   # shorts covering = squeeze accelerating

        score = max(0.0, min(1.0, score))
        result["squeeze_score"] = round(score, 3)

        # ── Classify ──────────────────────────────────────────────────────────
        if score >= 0.60:
            result["squeeze_signal"] = "HIGH_SQUEEZE"
            result["short_signal"]   = "BULLISH"   # squeeze = potential rocket
            # High short + covering = strong squeeze multiplier
            mult = min(SQUEEZE_BOOST_MAX, 1.0 + score * 0.20)
        elif score >= 0.30:
            result["squeeze_signal"] = "MODERATE"
            result["short_signal"]   = "NEUTRAL"
            mult = 1.05
        elif short_change and short_change > 0.20 and pct >= SHORT_PCT_HIGH:
            # Shorts rapidly increasing — bearish trap
            result["squeeze_signal"] = "BEARISH_TRAP"
            result["short_signal"]   = "BEARISH"
            mult = BEARISH_CUT_MAX
        else:
            result["squeeze_signal"] = "LOW"
            result["short_signal"]   = "NEUTRAL"
            mult = 1.0

        result["squeeze_multiplier"] = round(mult, 3)

    except Exception as e:
        result["error"] = str(e)

    return result


def get_short_interest_batch(
    tickers: list[str],
    verbose: bool = True,
) -> pd.DataFrame:
    """Fetch short interest for multiple tickers. Returns DataFrame."""
    rows = []
    for ticker in tickers:
        if verbose:
            print(f"  {ticker}...", end=" ", flush=True)
        sig = get_short_interest(ticker)
        rows.append(sig)
        if verbose:
            pct = f"{sig['short_pct_float']:.1%}" if sig["short_pct_float"] else "N/A"
            print(f"{sig['squeeze_signal']:15s} pct={pct}  ratio={sig['short_ratio']}  mult={sig['squeeze_multiplier']:.2f}x")
    return pd.DataFrame(rows)


def short_interest_to_multiplier(ticker: str) -> float:
    """
    Quick helper — returns squeeze multiplier for a ticker.
    Returns 1.0 on any error.
    """
    try:
        sig = get_short_interest(ticker)
        return sig.get("squeeze_multiplier", 1.0)
    except Exception:
        return 1.0


if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")

    tickers = ["AAPL", "NVDA", "TSLA", "AMD", "GME", "PLTR", "SMCI"]
    print(f"\nShort Interest Signals — {datetime.today().strftime('%Y-%m-%d %H:%M')}\n")
    print(f"{'─'*75}")

    for ticker in tickers:
        sig = get_short_interest(ticker)
        if sig["error"]:
            print(f"{ticker:6s}  ERROR: {sig['error']}")
            continue
        pct   = f"{sig['short_pct_float']:.1%}" if sig["short_pct_float"] else "N/A"
        ratio = sig["short_ratio"] or "N/A"
        print(
            f"{ticker:6s}  "
            f"{sig['squeeze_signal']:15s}  "
            f"Float={pct:6s}  "
            f"DaysToCover={ratio}  "
            f"Score={sig['squeeze_score']:.2f}  "
            f"Mult={sig['squeeze_multiplier']:.2f}x  "
            f"Signal={sig['short_signal']}"
        )
