# features/massive_options.py
# ─────────────────────────────────────────────────────────────────────────────
# Massive API integration for options data.
# Uses Options Starter plan ($29/mo) with real-time Greeks and IV.
#
# Primary use: TRUE 25-delta skew calculation using actual delta values.
# Fallback to yfinance if Massive fails.
# ─────────────────────────────────────────────────────────────────────────────

from __future__ import annotations
import os
from datetime import date, timedelta
from typing import Optional

import requests

BASE_URL    = "https://api.massive.com"
API_KEY     = os.getenv("MASSIVE_API_KEY", "")


def _get_snapshot(ticker: str, contract_type: str = None,
                  strike_min: float = None, strike_max: float = None,
                  exp_min: str = None, exp_max: str = None,
                  limit: int = 100) -> list:
    """
    Get options snapshot for a ticker from Massive.
    Filters by contract type, strike range, expiration range.
    """
    if not API_KEY:
        return []

    params = {"apiKey": API_KEY, "limit": limit}
    if contract_type:
        params["contract_type"] = contract_type
    if strike_min is not None:
        params["strike_price.gte"] = strike_min
    if strike_max is not None:
        params["strike_price.lte"] = strike_max
    if exp_min:
        params["expiration_date.gte"] = exp_min
    if exp_max:
        params["expiration_date.lte"] = exp_max

    try:
        r = requests.get(f"{BASE_URL}/v3/snapshot/options/{ticker}",
                         params=params, timeout=10)
        if r.status_code != 200:
            return []
        return r.json().get("results", [])
    except Exception:
        return []


def get_25delta_skew_massive(ticker: str, current_price: float = None,
                              target_days: int = 30) -> dict:
    """
    Calculate TRUE 25-delta skew using actual delta values from Massive.

    25-delta skew = IV(25-delta put) - IV(25-delta call)
    Positive = puts more expensive = bearish
    Negative = calls more expensive = bullish
    """
    result = {
        "ticker":       ticker,
        "skew_25d":     None,
        "iv_25d_put":   None,
        "iv_25d_call":  None,
        "iv_rank":      None,
        "skew_signal":  "NEUTRAL",
        "source":       "massive",
        "error":        None,
    }

    if not API_KEY:
        result["error"] = "MASSIVE_API_KEY not set"
        return result

    try:
        # Get current price if not provided (use yfinance as free lookup)
        if current_price is None:
            import yfinance as yf
            t = yf.Ticker(ticker)
            current_price = t.info.get("regularMarketPrice") or t.info.get("currentPrice")
            if not current_price:
                hist = t.history(period="1d")
                if not hist.empty:
                    current_price = float(hist["Close"].iloc[-1])
            if not current_price:
                result["error"] = "Could not get current price"
                return result

        # Target expiration ~30 days out
        today       = date.today()
        exp_min     = (today + timedelta(days=target_days - 10)).isoformat()
        exp_max     = (today + timedelta(days=target_days + 10)).isoformat()

        # Strike range: ±30% of current price
        strike_min  = current_price * 0.7
        strike_max  = current_price * 1.3

        # Fetch calls and puts in parallel
        calls = _get_snapshot(ticker, "call", strike_min, strike_max, exp_min, exp_max, limit=200)
        puts  = _get_snapshot(ticker, "put",  strike_min, strike_max, exp_min, exp_max, limit=200)

        if not calls or not puts:
            result["error"] = f"No options data (calls={len(calls)} puts={len(puts)})"
            return result

        # Find contract closest to 25-delta for each side
        def _closest_to_25_delta(contracts: list, is_call: bool) -> Optional[dict]:
            """Find contract with delta closest to ±0.25."""
            target_delta = 0.25 if is_call else -0.25
            best     = None
            best_dif = 999
            for c in contracts:
                greeks = c.get("greeks", {})
                delta  = greeks.get("delta")
                if delta is None:
                    continue
                dif = abs(delta - target_delta)
                if dif < best_dif:
                    best_dif = dif
                    best     = c
            return best

        # Need IV — comes from day.implied_volatility or greeks.implied_volatility
        def _get_iv(contract: dict) -> Optional[float]:
            """Extract IV from contract."""
            iv = contract.get("implied_volatility")
            if iv is not None:
                return float(iv)
            iv = contract.get("greeks", {}).get("iv") or contract.get("greeks", {}).get("implied_volatility")
            if iv is not None:
                return float(iv)
            return None

        call_25d = _closest_to_25_delta(calls, is_call=True)
        put_25d  = _closest_to_25_delta(puts,  is_call=False)

        if not call_25d or not put_25d:
            result["error"] = "No 25-delta contracts found"
            return result

        iv_call = _get_iv(call_25d)
        iv_put  = _get_iv(put_25d)

        if iv_call is None or iv_put is None:
            result["error"] = f"Missing IV (call_iv={iv_call}, put_iv={iv_put})"
            return result

        skew = iv_put - iv_call

        result["skew_25d"]    = round(skew, 4)
        result["iv_25d_put"]  = round(iv_put, 4)
        result["iv_25d_call"] = round(iv_call, 4)

        if skew > 0.03:
            result["skew_signal"] = "BEARISH"
        elif skew < -0.02:
            result["skew_signal"] = "BULLISH"
        else:
            result["skew_signal"] = "NEUTRAL"

    except Exception as e:
        result["error"] = str(e)

    return result


def get_25delta_skew_with_fallback(ticker: str, current_price: float = None) -> dict:
    """
    Primary: Massive (real Greeks).
    Fallback: UW if Massive fails.
    Final fallback: yfinance.
    """
    # Try Massive first
    result = get_25delta_skew_massive(ticker, current_price)
    if result.get("skew_25d") is not None and result.get("error") is None:
        return result

    # Fallback to UW
    try:
        from features.options_flow import get_25delta_skew as get_uw_skew
        uw_result = get_uw_skew(ticker)
        if uw_result.get("skew_25d") is not None and uw_result.get("error") is None:
            uw_result["source"] = "unusual_whales"
            return uw_result
    except Exception:
        pass

    # Final fallback to yfinance
    try:
        from features.options_flow import get_options_signal
        yf_result = get_options_signal(ticker)
        if yf_result.get("iv_skew") is not None:
            return {
                "ticker":      ticker,
                "skew_25d":    round(yf_result["iv_skew"], 4),
                "iv_rank":     yf_result.get("iv_rank"),
                "skew_signal": "BEARISH" if yf_result["iv_skew"] > 0.03 else "BULLISH" if yf_result["iv_skew"] < -0.02 else "NEUTRAL",
                "source":      "yfinance",
                "error":       None,
            }
    except Exception:
        pass

    return result  # Return Massive error result if all fallbacks failed


def skew_to_multiplier(skew_25d: float) -> float:
    """Convert skew to probability multiplier.

    Bearish skew (positive) → suppress BUY signals
    Bullish skew (negative) → boost BUY signals
    """
    if skew_25d is None:
        return 1.0
    if skew_25d > 0.05:   return 0.92  # Strong bearish
    if skew_25d > 0.03:   return 0.96  # Mild bearish
    if skew_25d < -0.03:  return 1.05  # Strong bullish
    if skew_25d < -0.02:  return 1.03  # Mild bullish
    return 1.0
