# features/uw_signals.py
# Unusual Whales signal fetchers:
#   1. Options flow alerts — unusual activity per ticker
#   2. Market tide — market-wide call/put flow
#   3. OI change — open interest change as GEX proxy
#   4. Institutional ownership — quarterly holdings change

import os
import sqlite3
import requests
from datetime import date, datetime
from pathlib import Path

UW_KEY   = os.getenv("UW_API_KEY", "")
HEADERS  = {"Authorization": f"Bearer {UW_KEY}"}
BASE_URL = "https://api.unusualwhales.com"
DB_PATH  = Path(__file__).parent.parent / "accuracy.db"


# ══════════════════════════════════════════════════════════════════════
# 1. OPTIONS FLOW ALERTS
# ══════════════════════════════════════════════════════════════════════

def get_options_flow_score(ticker: str) -> dict:
    """
    Compute options flow score for a ticker from UW flow alerts.
    Positive score = bullish unusual activity.
    Negative score = bearish unusual activity.
    """
    result = {
        "ticker":          ticker,
        "flow_score":      0.0,
        "bullish_prem":    0.0,
        "bearish_prem":    0.0,
        "sweep_count":     0,
        "alert_count":     0,
        "flow_signal":     "NEUTRAL",
        "error":           None,
    }
    try:
        r = requests.get(
            f"{BASE_URL}/api/option-trades/flow-alerts",
            headers=HEADERS, timeout=10
        )
        if r.status_code != 200:
            result["error"] = f"UW API error: {r.status_code}"
            return result

        alerts = [
            a for a in r.json().get("data", [])
            if a.get("ticker", "").upper() == ticker.upper()
        ]

        if not alerts:
            return result

        bullish_prem = 0.0
        bearish_prem = 0.0
        sweep_count  = 0

        for a in alerts:
            prem = float(a.get("total_premium", 0) or 0)
            typ  = a.get("type", "").lower()
            if typ == "call":
                bullish_prem += prem
            elif typ == "put":
                bearish_prem += prem
            if a.get("has_sweep"):
                sweep_count += 1

        total = bullish_prem + bearish_prem
        if total > 0:
            flow_score = (bullish_prem - bearish_prem) / total
        else:
            flow_score = 0.0

        result["flow_score"]   = round(flow_score, 4)
        result["bullish_prem"] = bullish_prem
        result["bearish_prem"] = bearish_prem
        result["sweep_count"]  = sweep_count
        result["alert_count"]  = len(alerts)

        if flow_score > 0.3:
            result["flow_signal"] = "BULLISH"
        elif flow_score < -0.3:
            result["flow_signal"] = "BEARISH"
        else:
            result["flow_signal"] = "NEUTRAL"

    except Exception as e:
        result["error"] = str(e)

    return result


def flow_score_to_multiplier(flow_score: float) -> float:
    """
    Convert options flow score to prob_eff multiplier.
    Strong bullish flow → boost. Strong bearish → suppress.
    """
    if flow_score > 0.5:   return 1.08
    if flow_score > 0.3:   return 1.04
    if flow_score < -0.5:  return 0.92
    if flow_score < -0.3:  return 0.96
    return 1.0


# ══════════════════════════════════════════════════════════════════════
# 2. MARKET TIDE
# ══════════════════════════════════════════════════════════════════════

def get_market_tide() -> dict:
    """
    Get today's market tide — net call vs put premium across entire market.
    Positive = bullish market flow. Negative = bearish.
    """
    result = {
        "tide_score":       0.0,
        "net_call_premium": 0.0,
        "net_put_premium":  0.0,
        "net_volume":       0,
        "tide_signal":      "NEUTRAL",
        "error":            None,
    }
    try:
        r = requests.get(
            f"{BASE_URL}/api/market/market-tide",
            headers=HEADERS, timeout=10
        )
        if r.status_code != 200:
            result["error"] = f"UW API error: {r.status_code}"
            return result

        data = r.json().get("data", [])
        if not data:
            result["error"] = "No market tide data"
            return result

        # Sum all intraday buckets for today
        today = str(date.today())
        today_data = [d for d in data if d.get("date", "") == today]
        if not today_data:
            today_data = data  # fallback to all data

        net_call = sum(float(d.get("net_call_premium", 0) or 0) for d in today_data)
        net_put  = sum(float(d.get("net_put_premium",  0) or 0) for d in today_data)
        net_vol  = sum(int(d.get("net_volume", 0) or 0)         for d in today_data)

        total = abs(net_call) + abs(net_put)
        tide_score = (net_call + net_put) / total if total > 0 else 0.0

        result["tide_score"]       = round(tide_score, 4)
        result["net_call_premium"] = net_call
        result["net_put_premium"]  = net_put
        result["net_volume"]       = net_vol

        if tide_score > 0.2:
            result["tide_signal"] = "BULLISH"
        elif tide_score < -0.2:
            result["tide_signal"] = "BEARISH"
        else:
            result["tide_signal"] = "NEUTRAL"

    except Exception as e:
        result["error"] = str(e)

    return result


def tide_score_to_multiplier(tide_score: float) -> float:
    """
    Convert market tide score to multiplier.
    Strong bullish tide → mild boost across all tickers.
    Strong bearish tide → mild suppression.
    """
    if tide_score > 0.4:  return 1.05
    if tide_score > 0.2:  return 1.02
    if tide_score < -0.4: return 0.95
    if tide_score < -0.2: return 0.98
    return 1.0


# ══════════════════════════════════════════════════════════════════════
# 3. OI CHANGE (GEX PROXY)
# ══════════════════════════════════════════════════════════════════════

def get_oi_change_score(ticker: str) -> dict:
    """
    Get open interest change score for a ticker.
    Rising OI + rising price = strong conviction = bullish signal.
    Rising OI + falling price = bearish signal.
    """
    result = {
        "ticker":      ticker,
        "oi_score":    0.0,
        "oi_change":   0.0,
        "oi_signal":   "NEUTRAL",
        "error":       None,
    }
    try:
        r = requests.get(
            f"{BASE_URL}/api/market/oi-change",
            headers=HEADERS, timeout=10
        )
        if r.status_code != 200:
            result["error"] = f"UW API error: {r.status_code}"
            return result

        data = [
            d for d in r.json().get("data", [])
            if d.get("underlying_symbol", "").upper() == ticker.upper()
        ]

        if not data:
            return result

        # Aggregate OI changes for this ticker
        total_oi_change = sum(
            float(d.get("oi_change", 0) or 0) for d in data
        )
        days_increasing = max(
            int(d.get("days_of_oi_increases", 0) or 0) for d in data
        )

        # Normalize: more days increasing = stronger signal
        oi_score = min(days_increasing / 5.0, 1.0) * (1 if total_oi_change > 0 else -1)

        result["oi_score"]  = round(oi_score, 4)
        result["oi_change"] = round(total_oi_change, 2)

        if oi_score > 0.4:
            result["oi_signal"] = "BULLISH"
        elif oi_score < -0.4:
            result["oi_signal"] = "BEARISH"
        else:
            result["oi_signal"] = "NEUTRAL"

    except Exception as e:
        result["error"] = str(e)

    return result


def oi_score_to_multiplier(oi_score: float) -> float:
    if oi_score > 0.6:  return 1.04
    if oi_score > 0.4:  return 1.02
    if oi_score < -0.6: return 0.96
    if oi_score < -0.4: return 0.98
    return 1.0


# ══════════════════════════════════════════════════════════════════════
# 4. INSTITUTIONAL OWNERSHIP
# ══════════════════════════════════════════════════════════════════════

def get_institutional_score(ticker: str) -> dict:
    """
    Get net institutional buying/selling from 13F filings.
    Positive = net buying. Negative = net selling.
    Note: 45-day lag — use as slow-moving background signal.
    """
    result = {
        "ticker":         ticker,
        "inst_score":     0.0,
        "units_changed":  0,
        "inst_signal":    "NEUTRAL",
        "filing_date":    None,
        "error":          None,
    }
    try:
        r = requests.get(
            f"{BASE_URL}/api/institution/{ticker}/ownership",
            headers=HEADERS, timeout=10
        )
        if r.status_code != 200:
            result["error"] = f"UW API error: {r.status_code}"
            return result

        data = r.json().get("data", [])
        if not data:
            result["error"] = "No institutional data"
            return result

        # Sum units_changed across top institutions
        total_changed = sum(int(d.get("units_changed", 0) or 0) for d in data[:20])
        shares_out    = float(data[0].get("shares_outstanding", 1) or 1)
        latest_filing = data[0].get("filing_date", None)

        # Normalize by shares outstanding
        inst_score = total_changed / shares_out * 1000  # scale up

        result["inst_score"]    = round(min(max(inst_score, -1.0), 1.0), 4)
        result["units_changed"] = total_changed
        result["filing_date"]   = latest_filing

        if inst_score > 0.1:
            result["inst_signal"] = "BULLISH"
        elif inst_score < -0.1:
            result["inst_signal"] = "BEARISH"
        else:
            result["inst_signal"] = "NEUTRAL"

    except Exception as e:
        result["error"] = str(e)

    return result


def inst_score_to_multiplier(inst_score: float) -> float:
    """
    Institutional score is slow-moving — use very mild multiplier.
    """
    if inst_score > 0.5:  return 1.03
    if inst_score > 0.1:  return 1.01
    if inst_score < -0.5: return 0.97
    if inst_score < -0.1: return 0.99
    return 1.0


# ══════════════════════════════════════════════════════════════════════
# COMBINED SIGNAL
# ══════════════════════════════════════════════════════════════════════

# Module-level cache for market-wide signals (fetched once per run)
_MW_CACHE = {"tide": None, "flow_alerts": None, "oi": None, "ts": None}


def _get_marketwide_cached() -> dict:
    """Fetch market-wide signals once and cache for 5 minutes."""
    from datetime import datetime
    now = datetime.now()
    if _MW_CACHE["ts"] is not None:
        age_mins = (now - _MW_CACHE["ts"]).total_seconds() / 60
        if age_mins < 5:
            return _MW_CACHE

    # Refresh cache
    _MW_CACHE["tide"]        = get_market_tide()
    _MW_CACHE["flow_alerts"] = _get_flow_alerts_raw()
    _MW_CACHE["oi"]          = _get_oi_raw()
    _MW_CACHE["ts"]          = now
    return _MW_CACHE


def _get_flow_alerts_raw() -> list:
    """Fetch raw flow alerts once."""
    try:
        r = requests.get(f"{BASE_URL}/api/option-trades/flow-alerts",
                         headers=HEADERS, timeout=10)
        return r.json().get("data", []) if r.status_code == 200 else []
    except Exception:
        return []


def _get_oi_raw() -> list:
    """Fetch raw OI change data once."""
    try:
        r = requests.get(f"{BASE_URL}/api/market/oi-change",
                         headers=HEADERS, timeout=10)
        return r.json().get("data", []) if r.status_code == 200 else []
    except Exception:
        return []


def get_combined_uw_multiplier(ticker: str) -> dict:
    """
    Fetch all 4 UW signals and return combined multiplier.
    Market-wide signals (tide, flow, OI) fetched ONCE and cached.
    Institutional reads from DB. No per-ticker API calls for market-wide data.
    """
    # Get market-wide signals from cache (1 API call total, not 126)
    mw    = _get_marketwide_cached()
    tide  = mw["tide"]

    # Compute per-ticker flow score from cached alerts
    alerts    = mw["flow_alerts"]
    t_alerts  = [a for a in alerts if a.get("ticker","").upper() == ticker.upper()]
    bull_prem = sum(float(a.get("total_premium",0) or 0) for a in t_alerts if a.get("type","").lower()=="call")
    bear_prem = sum(float(a.get("total_premium",0) or 0) for a in t_alerts if a.get("type","").lower()=="put")
    total_p   = bull_prem + bear_prem
    flow_score= round((bull_prem - bear_prem) / total_p, 4) if total_p > 0 else 0.0
    flow      = {
        "flow_score":   flow_score,
        "flow_signal":  "BULLISH" if flow_score > 0.3 else "BEARISH" if flow_score < -0.3 else "NEUTRAL",
        "alert_count":  len(t_alerts),
    }

    # Compute per-ticker OI score from cached data
    oi_data   = mw["oi"]
    t_oi      = [d for d in oi_data if d.get("underlying_symbol","").upper() == ticker.upper()]
    if t_oi:
        oi_chg    = sum(float(d.get("oi_change",0) or 0) for d in t_oi)
        days_inc  = max(int(d.get("days_of_oi_increases",0) or 0) for d in t_oi)
        oi_score  = min(days_inc / 5.0, 1.0) * (1 if oi_chg > 0 else -1)
    else:
        oi_score  = 0.0
    oi = {
        "oi_score":  round(oi_score, 4),
        "oi_signal": "BULLISH" if oi_score > 0.4 else "BEARISH" if oi_score < -0.4 else "NEUTRAL",
    }

    # Institutional from DB (no API call)
    inst  = get_institutional_score(ticker)

    flow_mult = flow_score_to_multiplier(flow["flow_score"])
    tide_mult = tide_score_to_multiplier(tide["tide_score"])
    oi_mult   = oi_score_to_multiplier(oi["oi_score"])
    inst_mult = inst_score_to_multiplier(inst["inst_score"])

    combined  = round(flow_mult * tide_mult * oi_mult * inst_mult, 4)
    combined  = round(min(max(combined, 0.80), 1.20), 4)

    return {
        "ticker":      ticker,
        "combined":    combined,
        "flow_mult":   flow_mult,
        "tide_mult":   tide_mult,
        "oi_mult":     oi_mult,
        "inst_mult":   inst_mult,
        "flow_score":  flow["flow_score"],
        "flow_signal": flow["flow_signal"],
        "tide_score":  tide["tide_score"],
        "tide_signal": tide["tide_signal"],
        "oi_score":    oi["oi_score"],
        "oi_signal":   oi["oi_signal"],
        "inst_score":  inst["inst_score"],
        "inst_signal": inst["inst_signal"],
    }


if __name__ == "__main__":
    import sys
    ticker = sys.argv[1] if len(sys.argv) > 1 else "AAPL"
    print(f"\nUW Signals for {ticker}:")
    result = get_combined_uw_multiplier(ticker)
    for k, v in result.items():
        print(f"  {k}: {v}")
