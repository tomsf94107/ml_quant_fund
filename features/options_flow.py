# features/options_flow.py
# ─────────────────────────────────────────────────────────────────────────────
# Options flow signal using yfinance options chain data.
# Computes put/call ratio, IV skew, unusual volume, and max pain.
#
# These signals approximate "smart money" positioning:
#   - Unusual CALL volume spike → institutions buying upside
#   - Put/Call ratio < 0.5 → bullish positioning
#   - IV skew negative → calls more expensive than puts (rare, bullish)
#   - Price near max pain → market makers will pin price here at expiry
#
# Usage:
#   from features.options_flow import get_options_signal
#   sig = get_options_signal("AAPL")
#
# Free via yfinance — no API key needed.
# ─────────────────────────────────────────────────────────────────────────────

from __future__ import annotations

import warnings
from datetime import datetime, timedelta
from typing import Optional
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ── Thresholds ────────────────────────────────────────────────────────────────
PC_RATIO_BULLISH   = 0.70   # put/call < 0.7 → bullish
PC_RATIO_BEARISH   = 1.20   # put/call > 1.2 → bearish
VOL_OI_SPIKE       = 2.0    # options volume / open interest > 2x = unusual
IV_SKEW_BULLISH    = -0.02  # call IV > put IV by 2% → rare bullish signal
UNUSUAL_CALL_MULT  = 2.0    # call volume > 2x avg = unusual call activity


# ══════════════════════════════════════════════════════════════════════════════
#  HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _get_nearest_expiries(ticker_obj, n: int = 3) -> list[str]:
    """Get next N expiry dates from yfinance ticker."""
    try:
        expiries = ticker_obj.options
        if not expiries:
            return []
        # Filter to expiries within 45 days (most liquid)
        today = datetime.today().date()
        cutoff = today + timedelta(days=45)
        near = [e for e in expiries
                if datetime.strptime(e, "%Y-%m-%d").date() <= cutoff]
        return near[:n] if near else list(expiries[:n])
    except Exception:
        return []


def _fetch_chain(ticker_obj, expiry: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fetch calls and puts for a given expiry. Returns (calls, puts)."""
    try:
        chain = ticker_obj.option_chain(expiry)
        return chain.calls, chain.puts
    except Exception:
        return pd.DataFrame(), pd.DataFrame()


def _compute_max_pain(calls: pd.DataFrame, puts: pd.DataFrame) -> Optional[float]:
    """
    Max pain = strike price where total options value (for writers) is maximized.
    Market makers are incentivized to pin price near max pain at expiry.
    """
    try:
        strikes = sorted(set(calls["strike"].tolist() + puts["strike"].tolist()))
        pain = {}
        for s in strikes:
            call_pain = sum(
                max(0, s - k) * oi
                for k, oi in zip(calls["strike"], calls["openInterest"])
                if not np.isnan(oi)
            )
            put_pain = sum(
                max(0, k - s) * oi
                for k, oi in zip(puts["strike"], puts["openInterest"])
                if not np.isnan(oi)
            )
            pain[s] = call_pain + put_pain
        return min(pain, key=pain.get) if pain else None
    except Exception:
        return None


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN SIGNAL
# ══════════════════════════════════════════════════════════════════════════════

def get_options_signal(
    ticker:        str,
    current_price: Optional[float] = None,
) -> dict:
    """
    Compute options flow signal for a ticker.

    Returns dict with:
        put_call_ratio     : total put volume / total call volume
        pc_signal          : "BULLISH" | "BEARISH" | "NEUTRAL"
        iv_skew            : avg put IV - avg call IV (positive = fear)
        iv_skew_signal     : "BULLISH" | "BEARISH" | "NEUTRAL"
        unusual_calls      : True if call volume >> open interest
        unusual_puts       : True if put volume >> open interest
        flow_signal        : combined signal "BULLISH"|"BEARISH"|"NEUTRAL"
        flow_score         : -1.0 to +1.0 (positive = bullish)
        max_pain           : max pain strike price
        max_pain_distance  : % distance of current price from max pain
        call_vol_total     : total call volume across near expiries
        put_vol_total      : total put volume across near expiries
        error              : error message if failed
    """
    ticker = ticker.upper().strip()
    result = {
        "ticker":           ticker,
        "put_call_ratio":   None,
        "pc_signal":        "NEUTRAL",
        "iv_skew":          None,
        "iv_skew_signal":   "NEUTRAL",
        "unusual_calls":    False,
        "unusual_puts":     False,
        "flow_signal":      "NEUTRAL",
        "flow_score":       0.0,
        "max_pain":         None,
        "max_pain_distance": None,
        "call_vol_total":   0,
        "put_vol_total":    0,
        "error":            None,
        "timestamp":        datetime.utcnow().isoformat(),
    }

    try:
        import yfinance as yf
        tkr = yf.Ticker(ticker)

        # Get current price if not provided
        if current_price is None:
            try:
                hist = tkr.history(period="1d")
                current_price = float(hist["Close"].iloc[-1]) if not hist.empty else None
            except Exception:
                pass

        expiries = _get_nearest_expiries(tkr, n=3)
        if not expiries:
            result["error"] = "No options data available"
            try: _cf.write_text(json.dumps(result))
            except: pass
            return result

        # Aggregate across nearest expiries
        all_calls, all_puts = [], []
        max_pain_prices     = []

        for expiry in expiries:
            calls, puts = _fetch_chain(tkr, expiry)
            if calls.empty or puts.empty:
                continue

            # Filter to near-the-money strikes (within 10% of current price)
            if current_price:
                lo = current_price * 0.90
                hi = current_price * 1.10
                calls = calls[(calls["strike"] >= lo) & (calls["strike"] <= hi)]
                puts  = puts[ (puts["strike"]  >= lo) & (puts["strike"]  <= hi)]

            all_calls.append(calls)
            all_puts.append(puts)

            # Max pain for this expiry
            mp = _compute_max_pain(calls, puts)
            if mp:
                max_pain_prices.append(mp)

        if not all_calls:
            result["error"] = "No near-the-money options data"
            return result

        calls_df = pd.concat(all_calls, ignore_index=True)
        puts_df  = pd.concat(all_puts,  ignore_index=True)

        # ── Put/Call ratio ─────────────────────────────────────────────────────
        call_vol = calls_df["volume"].fillna(0).sum()
        put_vol  = puts_df["volume"].fillna(0).sum()
        result["call_vol_total"] = int(call_vol)
        result["put_vol_total"]  = int(put_vol)

        if call_vol > 0:
            pc_ratio = put_vol / call_vol
            result["put_call_ratio"] = round(float(pc_ratio), 3)
            if pc_ratio < PC_RATIO_BULLISH:
                result["pc_signal"] = "BULLISH"
            elif pc_ratio > PC_RATIO_BEARISH:
                result["pc_signal"] = "BEARISH"
            else:
                result["pc_signal"] = "NEUTRAL"

        # ── IV Skew ────────────────────────────────────────────────────────────
        call_iv = calls_df["impliedVolatility"].replace(0, np.nan).median()
        put_iv  = puts_df["impliedVolatility"].replace(0, np.nan).median()
        if not np.isnan(call_iv) and not np.isnan(put_iv):
            skew = float(put_iv - call_iv)
            result["iv_skew"] = round(skew, 4)
            if skew < IV_SKEW_BULLISH:
                result["iv_skew_signal"] = "BULLISH"
            elif skew > 0.05:
                result["iv_skew_signal"] = "BEARISH"
            else:
                result["iv_skew_signal"] = "NEUTRAL"

        # ── Unusual volume ─────────────────────────────────────────────────────
        call_oi  = calls_df["openInterest"].fillna(0).sum()
        put_oi   = puts_df["openInterest"].fillna(0).sum()
        if call_oi > 0 and call_vol / call_oi > VOL_OI_SPIKE:
            result["unusual_calls"] = True
        if put_oi > 0 and put_vol / put_oi > VOL_OI_SPIKE:
            result["unusual_puts"] = True

        # ── Max pain ───────────────────────────────────────────────────────────
        if max_pain_prices:
            mp = float(np.median(max_pain_prices))
            result["max_pain"] = round(mp, 2)
            if current_price and mp:
                result["max_pain_distance"] = round((current_price - mp) / mp, 4)

        # ── Combined flow score ────────────────────────────────────────────────
        # Score from -1 (bearish) to +1 (bullish)
        score = 0.0
        signal_map = {"BULLISH": 1, "NEUTRAL": 0, "BEARISH": -1}

        score += signal_map[result["pc_signal"]]    * 0.40   # PC ratio weighted most
        score += signal_map[result["iv_skew_signal"]] * 0.30  # IV skew
        if result["unusual_calls"]: score += 0.20             # unusual call activity
        if result["unusual_puts"]:  score -= 0.20             # unusual put activity

        result["flow_score"] = round(score, 3)
        if score >= 0.30:
            result["flow_signal"] = "BULLISH"
        elif score <= -0.30:
            result["flow_signal"] = "BEARISH"
        else:
            result["flow_signal"] = "NEUTRAL"

    except Exception as e:
        result["error"] = str(e)

    return result


def get_options_signals_batch(
    tickers: list[str],
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Fetch options signals for multiple tickers.
    Returns DataFrame with one row per ticker.
    """
    rows = []
    for ticker in tickers:
        if verbose:
            print(f"  {ticker}...", end=" ", flush=True)
        sig = get_options_signal(ticker)
        rows.append(sig)
        if verbose:
            print(f"{sig['flow_signal']} (score={sig['flow_score']:+.2f})")
    return pd.DataFrame(rows)


def options_score_to_multiplier(flow_score: float) -> float:
    """
    Convert options flow score to a probability multiplier.
    Used by signal generator to adjust ML model probability.

    Score +1.0 (max bullish) → multiplier 1.10  (+10% boost)
    Score  0.0 (neutral)     → multiplier 1.00  (no change)
    Score -1.0 (max bearish) → multiplier 0.85  (-15% cut)
    """
    if flow_score >= 0:
        return 1.0 + (flow_score * 0.10)   # up to +10%
    else:
        return 1.0 + (flow_score * 0.15)   # up to -15%


# ══════════════════════════════════════════════════════════════════════════════
#  POLYGON-BASED: TRUE 25-DELTA SKEW + IV RANK
#  Requires Polygon Options add-on. Falls back to yfinance if 403.
# ══════════════════════════════════════════════════════════════════════════════

import os as _os
import requests as _req

_UW_KEY     = _os.getenv("UW_API_KEY", "")
_UW_HEADERS = {"Authorization": f"Bearer {_UW_KEY}"}
_UW_BASE    = "https://api.unusualwhales.com"


def get_25delta_skew(ticker: str) -> dict:
    """
    Compute true 25-delta put/call IV skew via Unusual Whales API.
    skew_25d = IV(25-delta put) - IV(25-delta call)
    Positive skew = puts more expensive = bearish lean = smart money hedging.

    Returns dict with:
        skew_25d     : float (positive = bearish, negative = bullish)
        put_iv_25d   : implied vol of 25-delta put
        call_iv_25d  : implied vol of 25-delta call
        iv_rank      : where current ATM IV sits vs 52-week range (0-100)
        skew_signal  : "BEARISH" | "NEUTRAL" | "BULLISH"
        error        : error message if failed
    """
    result = {
        "ticker":      ticker.upper(),
        "skew_25d":    None,
        "put_iv_25d":  None,
        "call_iv_25d": None,
        "iv_rank":     None,
        "skew_signal": "NEUTRAL",
        "error":       None,
    }

    try:
        # Get options chain snapshot from UW
        url = f"{_UW_BASE}/api/stock/{ticker}/option-contracts"
        r = _req.get(url, headers=_UW_HEADERS, timeout=10)

        if r.status_code == 401:
            result["error"] = "Invalid UW API key"
            return result
        if r.status_code == 403:
            result["error"] = "UW API plan does not include options"
            return result
        if r.status_code != 200:
            result["error"] = f"UW API error: {r.status_code}"
            return result

        chain = r.json().get("data", [])
        if not chain:
            result["error"] = "Empty options chain"
            return result

        # Parse call/put from option symbol (e.g. AAPL260417C00270000)
        def _is_call(c):
            sym = c.get("option_symbol", "")
            return "C" in sym[10:] if len(sym) > 10 else False

        calls = [c for c in chain if _is_call(c) and c.get("implied_volatility")]
        puts  = [c for c in chain if not _is_call(c) and c.get("implied_volatility")]

        if not calls or not puts:
            result["error"] = "No calls or puts with IV data"
            return result

        # Sort by volume to find most active strikes (proxy for ~25 delta)
        # Most active OTM options tend to cluster around 25 delta
        calls_sorted = sorted(calls, key=lambda x: int(x.get("volume", 0)), reverse=True)
        puts_sorted  = sorted(puts,  key=lambda x: int(x.get("volume", 0)), reverse=True)

        # Use top 5 most active and take median IV
        import statistics
        top_call_ivs = [float(c["implied_volatility"]) for c in calls_sorted[:5]]
        top_put_ivs  = [float(p["implied_volatility"]) for p in puts_sorted[:5]]

        call_iv = statistics.median(top_call_ivs)
        put_iv  = statistics.median(top_put_ivs)
        skew    = round(put_iv - call_iv, 4)

        result["skew_25d"]    = skew
        result["put_iv_25d"]  = round(put_iv, 4)
        result["call_iv_25d"] = round(call_iv, 4)

        if skew > 0.03:
            result["skew_signal"] = "BEARISH"
        elif skew < -0.02:
            result["skew_signal"] = "BULLISH"
        else:
            result["skew_signal"] = "NEUTRAL"

        # IV rank — ATM call IV vs all calls
        all_ivs = [float(c["implied_volatility"]) for c in calls if c.get("implied_volatility")]
        if len(all_ivs) >= 5:
            atm_iv  = statistics.median(all_ivs)
            iv_min  = min(all_ivs)
            iv_max  = max(all_ivs)
            if iv_max > iv_min:
                result["iv_rank"] = round(
                    (atm_iv - iv_min) / (iv_max - iv_min) * 100, 1
                )

    except Exception as e:
        result["error"] = str(e)

    return result


def get_enhanced_options_signal(ticker: str, current_price=None) -> dict:
    """
    Enhanced options signal combining:
    1. yfinance-based flow signal (always available)
    2. Polygon 25-delta skew + IV rank (when Polygon Options available)

    Returns merged signal dict with all fields from both sources.
    """
    # Get base yfinance signal
    base = get_options_signal(ticker, current_price)

    # Try Polygon enhancement
    poly = get_25delta_skew(ticker)

    if poly.get("error") is None and poly.get("skew_25d") is not None:
        # Polygon succeeded — override iv_skew with delta-adjusted version
        base["iv_skew_25d"]   = poly["skew_25d"]
        base["put_iv_25d"]    = poly["put_iv_25d"]
        base["call_iv_25d"]   = poly["call_iv_25d"]
        base["iv_rank"]       = poly["iv_rank"]
        base["skew_25d_signal"] = poly["skew_signal"]

        # Update flow score with 25-delta skew signal
        skew_signal_map = {"BULLISH": 1, "NEUTRAL": 0, "BEARISH": -1}
        poly_adj = skew_signal_map[poly["skew_signal"]] * 0.15
        base["flow_score"] = round(
            min(max(base["flow_score"] + poly_adj, -1.0), 1.0), 3
        )
    else:
        base["iv_skew_25d"]     = None
        base["put_iv_25d"]      = None
        base["call_iv_25d"]     = None
        base["iv_rank"]         = None
        base["skew_25d_signal"] = "NEUTRAL"
        base["polygon_error"]   = poly.get("error")

    return base


if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")

    tickers = ["AAPL", "NVDA", "TSLA", "AMD"]
    print(f"\nOptions Flow Signals — {datetime.today().strftime('%Y-%m-%d %H:%M')}\n")
    print(f"{'─'*70}")

    for ticker in tickers:
        sig = get_options_signal(ticker)
        if sig["error"]:
            print(f"{ticker:6s}  ERROR: {sig['error']}")
            continue
        print(
            f"{ticker:6s}  "
            f"Flow={sig['flow_signal']:8s}  "
            f"Score={sig['flow_score']:+.2f}  "
            f"P/C={sig['put_call_ratio'] or 'N/A'}  "
            f"IV Skew={sig['iv_skew'] or 'N/A'}  "
            f"UnusualCalls={sig['unusual_calls']}  "
            f"MaxPain={sig['max_pain'] or 'N/A'}"
        )
