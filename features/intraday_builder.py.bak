"""
Intraday feature builder — fetches 5-min bars and computes
momentum, VWAP deviation, RSI, volume surge for 1hr/2hr/4hr horizons.
"""
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime
import pytz

ET = pytz.timezone("America/New_York")

import os, sqlite3
from pathlib import Path

_DB_PATH = Path(__file__).parent.parent / "accuracy.db"
_UW_KEY  = os.getenv("UW_API_KEY", "")
_UW_HDRS = {"Authorization": f"Bearer {_UW_KEY}"}


def _fetch_and_save_uw_today(ticker: str) -> dict:
    """
    Fetch fresh dark pool + skew for today from UW API.
    Saves to DB overwriting today's entry.
    Returns {"dp_ratio": float, "skew_25d": float}
    """
    from datetime import date as _date
    import requests as _req

    today     = str(_date.today())
    dp_ratio  = 0.0
    skew_25d  = 0.0

    try:
        r = _req.get(
            f"https://api.unusualwhales.com/api/darkpool/{ticker}",
            headers=_UW_HDRS, params={"date": today}, timeout=8
        )
        if r.status_code == 200:
            trades    = r.json().get("data", [])
            total_vol = float(trades[0].get("volume", 0)) if trades else 0
            dp_vol    = sum(float(t.get("size", 0)) for t in trades)
            if total_vol > 0:
                dp_ratio = round(dp_vol / total_vol, 4)

            from datetime import datetime as _dt
            now = _dt.now().isoformat()
            with sqlite3.connect(_DB_PATH) as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO dark_pool_history
                        (date, ticker, dp_ratio, dp_volume, total_volume, dp_signal, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (today, ticker, dp_ratio, int(dp_vol), int(total_vol),
                      "HIGH" if dp_ratio > 0.50 else "LOW" if dp_ratio < 0.25 else "NORMAL",
                      now))
                conn.commit()
    except Exception:
        pass

    try:
        import statistics as _stat
        r2 = _req.get(
            f"https://api.unusualwhales.com/api/stock/{ticker}/option-contracts",
            headers=_UW_HDRS, timeout=8
        )
        if r2.status_code == 200:
            chain = r2.json().get("data", [])

            def _is_call(c):
                sym = c.get("option_symbol", "")
                for i, ch in enumerate(sym):
                    if ch in ("C", "P") and i > 2:
                        return ch == "C"
                return False

            calls = [c for c in chain if _is_call(c) and c.get("implied_volatility")
                     and float(c.get("implied_volatility", 0)) > 0]
            puts  = [c for c in chain if not _is_call(c) and c.get("implied_volatility")
                     and float(c.get("implied_volatility", 0)) > 0]

            if calls and puts:
                calls_s  = sorted(calls, key=lambda x: int(x.get("volume", 0)), reverse=True)
                puts_s   = sorted(puts,  key=lambda x: int(x.get("volume", 0)), reverse=True)
                call_iv  = _stat.median([float(c["implied_volatility"]) for c in calls_s[:5]])
                put_iv   = _stat.median([float(p["implied_volatility"]) for p in puts_s[:5]])
                skew_25d = round(put_iv - call_iv, 4)

                signal = "BEARISH" if skew_25d > 0.03 else "BULLISH" if skew_25d < -0.02 else "NEUTRAL"
                all_ivs = [float(c["implied_volatility"]) for c in calls if c.get("implied_volatility")]
                iv_rank = None
                if len(all_ivs) >= 5:
                    import statistics as _s
                    atm_iv = _s.median(all_ivs)
                    iv_min, iv_max = min(all_ivs), max(all_ivs)
                    if iv_max > iv_min:
                        iv_rank = round((atm_iv - iv_min) / (iv_max - iv_min) * 100, 1)

                from datetime import datetime as _dt2
                now2 = _dt2.now().isoformat()
                with sqlite3.connect(_DB_PATH) as conn:
                    conn.execute("""
                        INSERT OR REPLACE INTO options_skew_history
                            (date, ticker, skew_25d, put_iv_25d, call_iv_25d,
                             iv_rank, skew_signal, created_at)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """, (today, ticker, skew_25d, round(put_iv, 4),
                          round(call_iv, 4), iv_rank, signal, now2))
                    conn.commit()
    except Exception:
        pass

    return {"dp_ratio": dp_ratio, "skew_25d": skew_25d}


def _dp_skew_to_mult(dp_ratio: float, skew_25d: float) -> float:
    """
    Convert dark pool ratio + skew to a combined intraday multiplier.
    Applied to prob_eff after momentum-based probability.
    """
    dp_mult   = 1.05 if dp_ratio > 0.60 else 1.02 if dp_ratio > 0.40 else 0.98 if dp_ratio < 0.20 else 1.0
    skew_mult = 0.92 if skew_25d > 0.03 else 1.05 if skew_25d < -0.02 else 1.0
    combined  = dp_mult * skew_mult
    return round(min(max(combined, 0.85), 1.15), 4)


def is_market_open() -> bool:
    now = datetime.now(ET)
    if now.weekday() >= 5:
        return False
    market_open  = now.replace(hour=9,  minute=30, second=0, microsecond=0)
    market_close = now.replace(hour=16, minute=0,  second=0, microsecond=0)
    return market_open <= now <= market_close


def minutes_since_open() -> int:
    now = datetime.now(ET)
    open_time = now.replace(hour=9, minute=30, second=0, microsecond=0)
    return max(0, int((now - open_time).total_seconds() / 60))


def build_intraday_features(ticker: str) -> pd.DataFrame:
    """
    Fetch today's 5-min bars and compute intraday features.
    Returns DataFrame with one row per 5-min bar.
    """
    data = yf.download(
        ticker, period="2d", interval="5m",
        progress=False, auto_adjust=True
    )
    if data.empty:
        return pd.DataFrame()

    # Fix MultiIndex — extract only the fields we need explicitly
    if isinstance(data.columns, pd.MultiIndex):
        # Build a flat DataFrame by extracting each field directly
        flat = {}
        for field in ["Close", "High", "Low", "Open", "Volume"]:
            # Try (field, ticker) format
            if (field, ticker) in data.columns:
                flat[field] = data[(field, ticker)]
            # Try (ticker, field) format
            elif (ticker, field) in data.columns:
                flat[field] = data[(ticker, field)]
            # Try field in level 0
            elif field in data.columns.get_level_values(0):
                flat[field] = data.xs(field, axis=1, level=0).iloc[:, 0]
            # Try field in level 1
            elif field in data.columns.get_level_values(1):
                flat[field] = data.xs(field, axis=1, level=1).iloc[:, 0]
        data = pd.DataFrame(flat, index=data.index)

    df = data.copy()
    df.index = pd.to_datetime(df.index)

    # Keep only most recent trading day's bars
    if df.index.tzinfo:
        df.index = df.index.tz_convert(ET)
    # Get the last date that has data
    last_date = df.index[-1].date()
    df = df[df.index.date == last_date]

    if df.empty or len(df) < 5:
        return pd.DataFrame()

    # ── Features ─────────────────────────────────────────────────────────────
    df = df.copy()
    df["return_5m"]    = df["Close"].pct_change()
    df["return_30m"]   = df["Close"].pct_change(6)    # 6 × 5min = 30min
    df["return_1hr"]   = df["Close"].pct_change(12)   # 12 × 5min = 1hr
    df["return_2hr"]   = df["Close"].pct_change(24)
    df["vol_surge"]    = df["Volume"] / df["Volume"].rolling(12).mean()

    # VWAP
    df["vwap"] = (df["Close"] * df["Volume"]).cumsum() / df["Volume"].cumsum()
    df["vwap_dev"] = (df["Close"] - df["vwap"]) / df["vwap"]

    # RSI 14 on 5-min bars
    delta = df["Close"].diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    rs    = gain / loss.replace(0, np.nan)
    df["rsi_14"] = 100 - (100 / (1 + rs))

    # Momentum score: combines return, vwap_dev, rsi
    df["momentum_score"] = (
        df["return_1hr"].fillna(0) * 0.4 +
        df["vwap_dev"].fillna(0)   * 0.3 +
        ((df["rsi_14"].fillna(50) - 50) / 50) * 0.3
    )

    # Volume-weighted momentum
    df["vol_momentum"] = df["momentum_score"] * df["vol_surge"].fillna(1)

    return df.dropna(subset=["Close"])


def get_intraday_signal(ticker: str) -> dict:
    """
    Returns intraday signals for 1hr, 2hr, 4hr horizons.
    """
    result = {
        "ticker": ticker,
        "signal_1hr": "NEUTRAL",
        "signal_2hr": "NEUTRAL",
        "signal_4hr": "NEUTRAL",
        "prob_1hr": 0.5,
        "prob_2hr": 0.5,
        "prob_4hr": 0.5,
        "current_price": None,
        "vwap": None,
        "vwap_dev": None,
        "rsi_14": None,
        "vol_surge": None,
        "momentum_score": None,
        "minutes_since_open": minutes_since_open(),
        "market_open": is_market_open(),
        "error": None,
    }

    try:
        df = build_intraday_features(ticker)
        if df.empty:
            result["error"] = "No intraday data"
            return result

        last = df.iloc[-1]
        mom  = float(last.get("momentum_score", 0) or 0)
        rsi  = float(last.get("rsi_14", 50) or 50)
        vdev = float(last.get("vwap_dev", 0) or 0)
        vsur = float(last.get("vol_surge", 1) or 1)
        mso  = minutes_since_open()

        result["current_price"]    = round(float(last["Close"]), 2)
        result["vwap"]             = round(float(last["vwap"]), 2)   if "vwap"           in last else None
        result["vwap_dev"]         = round(vdev * 100, 2)
        result["rsi_14"]           = round(rsi, 1)
        result["vol_surge"]        = round(vsur, 2)
        result["momentum_score"]   = round(mom, 4)

        # Convert momentum → probability (sigmoid-like)
        def mom_to_prob(m, scale=8):
            return round(1 / (1 + np.exp(-m * scale)), 3)

        # 1hr signal — most sensitive to recent momentum
        p1 = mom_to_prob(mom * 1.0)
        # 2hr signal — smoother, uses 2hr return
        ret_2hr = last.get("return_2hr", 0) or 0
        ret_2hr = 0 if ret_2hr != ret_2hr else float(ret_2hr)  # NaN check
        p2 = mom_to_prob((mom * 0.6 + ret_2hr * 0.4))
        # 4hr signal — rest of day, mean-revert toward VWAP
        # If late in day (>180min), mean reversion more likely
        late_day_factor = min(mso / 390, 1.0)  # 390 = full trading day mins
        p4 = mom_to_prob(mom * (1 - late_day_factor * 0.4) + vdev * late_day_factor * -2)

        def prob_to_signal(p):
            if p >= 0.60: return "UP"
            if p <= 0.40: return "DOWN"
            return "NEUTRAL"

        # Apply dark pool + skew multiplier — live API only during market hours
        try:
            if is_market_open():
                uw = _fetch_and_save_uw_today(ticker)
            else:
                # Read from DB pre/post market
                import sqlite3
                from pathlib import Path
                from datetime import date, timedelta
                _db = Path(__file__).parent.parent / "accuracy.db"
                _cutoff = str(date.today() - timedelta(days=1))
                with sqlite3.connect(_db, timeout=30) as _conn:
                    _dp = _conn.execute(
                        "SELECT dp_ratio FROM dark_pool_history WHERE ticker=? AND date>=? ORDER BY date DESC LIMIT 1",
                        (ticker, _cutoff)).fetchone()
                    _sk = _conn.execute(
                        "SELECT skew_25d FROM options_skew_history WHERE ticker=? AND date>=? ORDER BY date DESC LIMIT 1",
                        (ticker, _cutoff)).fetchone()
                uw = {"dp_ratio": _dp[0] if _dp else 0.0, "skew_25d": _sk[0] if _sk else 0.0}
            mult = _dp_skew_to_mult(uw["dp_ratio"], uw["skew_25d"])
            p1 = round(min(max(p1 * mult, 0.05), 0.95), 3)
            p2 = round(min(max(p2 * mult, 0.05), 0.95), 3)
            p4 = round(min(max(p4 * mult, 0.05), 0.95), 3)
            result["dp_ratio"]  = uw["dp_ratio"]
            result["skew_25d"]  = uw["skew_25d"]
            result["uw_mult"]   = mult
        except Exception:
            pass

        result["prob_1hr"]    = p1
        result["prob_2hr"]    = p2
        result["prob_4hr"]    = p4
        result["signal_1hr"]  = prob_to_signal(p1)
        result["signal_2hr"]  = prob_to_signal(p2)
        result["signal_4hr"]  = prob_to_signal(p4)

    except Exception as e:
        result["error"] = str(e)

    return result


def get_all_intraday_signals(tickers: list) -> list:
    """
    Batch download all tickers at once to avoid yfinance per-ticker caching issues.
    """
    import yfinance as yf
    import pandas as pd

    # Download all at once
    raw = yf.download(
        tickers, period="2d", interval="5m",
        progress=False, auto_adjust=True, group_by="ticker"
    )

    results = []
    for ticker in tickers:
        try:
            # Extract this ticker's data
            if isinstance(raw.columns, pd.MultiIndex):
                if ticker in raw.columns.get_level_values(0):
                    df = raw[ticker].copy()
                elif ticker in raw.columns.get_level_values(1):
                    df = raw.xs(ticker, axis=1, level=1).copy()
                else:
                    results.append(get_intraday_signal(ticker))
                    continue
            else:
                df = raw.copy()

            if df.empty or "Close" not in df.columns:
                results.append(get_intraday_signal(ticker))
                continue

            # Get last trading day
            df.index = pd.to_datetime(df.index)
            if df.index.tzinfo:
                df.index = df.index.tz_convert(ET)
            last_date = df.index[-1].date()
            df = df[df.index.date == last_date]

            if len(df) < 5:
                results.append(get_intraday_signal(ticker))
                continue

            # Compute features inline
            df = df.copy()
            df["return_5m"]  = df["Close"].pct_change()
            df["return_1hr"] = df["Close"].pct_change(12)
            df["return_2hr"] = df["Close"].pct_change(24)
            df["vol_surge"]  = df["Volume"] / df["Volume"].rolling(12).mean()
            df["vwap"]       = (df["Close"] * df["Volume"]).cumsum() / df["Volume"].cumsum()
            df["vwap_dev"]   = (df["Close"] - df["vwap"]) / df["vwap"]
            delta = df["Close"].diff()
            gain  = delta.clip(lower=0).rolling(14).mean()
            loss  = (-delta.clip(upper=0)).rolling(14).mean()
            rs    = gain / loss.replace(0, float("nan"))
            df["rsi_14"] = 100 - (100 / (1 + rs))
            df["momentum_score"] = (
                df["return_1hr"].fillna(0) * 0.4 +
                df["vwap_dev"].fillna(0)   * 0.3 +
                ((df["rsi_14"].fillna(50) - 50) / 50) * 0.3
            )

            last = df.iloc[-1]
            mom  = float(last.get("momentum_score", 0) or 0)
            rsi  = float(last.get("rsi_14", 50) or 50)
            vdev = float(last.get("vwap_dev", 0) or 0)
            vsur = float(last.get("vol_surge", 1) or 1)
            mso  = minutes_since_open()

            def mom_to_prob(m, scale=8):
                return round(1 / (1 + np.exp(-m * scale)), 3)

            p1 = mom_to_prob(mom)
            p2 = mom_to_prob(mom * 0.6 + float(last.get("return_2hr", 0) or 0) * 0.4)
            late = min(mso / 390, 1.0)
            p4 = mom_to_prob(mom * (1 - late * 0.4) + vdev * late * -2)

            def prob_to_signal(p):
                if p >= 0.60: return "UP"
                if p <= 0.40: return "DOWN"
                return "NEUTRAL"

            # Apply dark pool + skew multiplier — live API only during market hours
            dp_ratio_val = 0.0
            skew_25d_val = 0.0
            uw_mult_val  = 1.0
            try:
                if is_market_open():
                    uw = _fetch_and_save_uw_today(ticker)
                else:
                    import sqlite3
                    from pathlib import Path
                    from datetime import date, timedelta
                    _db = Path(__file__).parent.parent / "accuracy.db"
                    _cutoff = str(date.today() - timedelta(days=1))
                    with sqlite3.connect(_db, timeout=30) as _conn:
                        _dp = _conn.execute(
                            "SELECT dp_ratio FROM dark_pool_history WHERE ticker=? AND date>=? ORDER BY date DESC LIMIT 1",
                            (ticker, _cutoff)).fetchone()
                        _sk = _conn.execute(
                            "SELECT skew_25d FROM options_skew_history WHERE ticker=? AND date>=? ORDER BY date DESC LIMIT 1",
                            (ticker, _cutoff)).fetchone()
                    uw = {"dp_ratio": _dp[0] if _dp else 0.0, "skew_25d": _sk[0] if _sk else 0.0}
                uw_mult_val  = _dp_skew_to_mult(uw["dp_ratio"], uw["skew_25d"])
                dp_ratio_val = uw["dp_ratio"]
                skew_25d_val = uw["skew_25d"]
                p1 = round(min(max(p1 * uw_mult_val, 0.05), 0.95), 3)
                p2 = round(min(max(p2 * uw_mult_val, 0.05), 0.95), 3)
                p4 = round(min(max(p4 * uw_mult_val, 0.05), 0.95), 3)
            except Exception:
                pass

            results.append({
                "ticker":           ticker,
                "current_price":    round(float(last["Close"]), 2),
                "vwap":             round(float(last["vwap"]), 2),
                "vwap_dev":         round(vdev * 100, 2),
                "rsi_14":           round(rsi, 1),
                "vol_surge":        round(vsur, 2),
                "momentum_score":   round(mom, 4),
                "prob_1hr": p1, "prob_2hr": p2, "prob_4hr": p4,
                "signal_1hr": prob_to_signal(p1),
                "signal_2hr": prob_to_signal(p2),
                "signal_4hr": prob_to_signal(p4),
                "minutes_since_open": mso,
                "market_open": is_market_open(),
                "dp_ratio":  dp_ratio_val,
                "skew_25d":  skew_25d_val,
                "uw_mult":   uw_mult_val,
                "error": None,
            })
        except Exception as e:
            results.append(get_intraday_signal(ticker))
    return results
