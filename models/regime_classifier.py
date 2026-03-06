# models/regime_classifier.py
# ─────────────────────────────────────────────────────────────────────────────
# Macro Regime Classifier — labels current market as BULL / BEAR / VOLATILE
# and adjusts signal thresholds accordingly.
#
# How it works:
#   1. Downloads SPY, VIX, TLT (bonds), GLD (gold) daily data
#   2. Computes regime features: trend, volatility, breadth, risk-off signals
#   3. Rule-based + ML hybrid classification → 3 regimes
#   4. Each regime gets different confidence thresholds + signal multipliers
#
# Regimes:
#   BULL     — trending up, low volatility, risk-on
#              → lower threshold (0.52), amplify BUY signals
#   BEAR     — trending down, elevated volatility
#              → higher threshold (0.60), reduce position sizing
#   VOLATILE — high VIX, whipsaw, no clear trend
#              → highest threshold (0.65), very selective
#
# Usage:
#   from models.regime_classifier import get_current_regime, RegimeSignal
#   regime = get_current_regime()
#   print(regime.label, regime.confidence_threshold)
#
# Zero Streamlit imports. Backend only.
# ─────────────────────────────────────────────────────────────────────────────

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

# ── Cache ─────────────────────────────────────────────────────────────────────
CACHE_FILE = Path(os.getenv("MODEL_DIR", "models/saved")) / "regime_cache.json"
CACHE_TTL_HOURS = 4   # refresh every 4 hours


# ══════════════════════════════════════════════════════════════════════════════
#  DATA CLASSES
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class RegimeSignal:
    label:                str     # "BULL" | "BEAR" | "VOLATILE"
    confidence:           float   # 0-1 how confident we are in the regime
    confidence_threshold: float   # recommended min prob_up to fire BUY signal
    signal_multiplier:    float   # multiply prob_up by this before thresholding
    block_tau:            int     # recommended risk_next_3d block threshold
    description:          str     # human-readable explanation
    computed_at:          str = field(default_factory=lambda: datetime.utcnow().isoformat())

    # Regime indicators
    spy_trend_20d:    float = 0.0
    spy_trend_60d:    float = 0.0
    vix_level:        float = 20.0
    vix_percentile:   float = 50.0
    bond_signal:      str = "NEUTRAL"   # risk-on/off from TLT
    breadth_signal:   str = "NEUTRAL"   # % stocks above MA50

    @property
    def is_bull(self)     -> bool: return self.label == "BULL"
    @property
    def is_bear(self)     -> bool: return self.label == "BEAR"
    @property
    def is_volatile(self) -> bool: return self.label == "VOLATILE"

    def to_dict(self) -> dict:
        return {
            "label":                self.label,
            "confidence":           self.confidence,
            "confidence_threshold": self.confidence_threshold,
            "signal_multiplier":    self.signal_multiplier,
            "block_tau":            self.block_tau,
            "description":          self.description,
            "computed_at":          self.computed_at,
            "spy_trend_20d":        self.spy_trend_20d,
            "spy_trend_60d":        self.spy_trend_60d,
            "vix_level":            self.vix_level,
            "vix_percentile":       self.vix_percentile,
            "bond_signal":          self.bond_signal,
            "breadth_signal":       self.breadth_signal,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "RegimeSignal":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


# ── Regime config ─────────────────────────────────────────────────────────────
REGIME_CONFIG = {
    "BULL": {
        "confidence_threshold": 0.52,
        "signal_multiplier":    1.05,
        "block_tau":            4,
        "description":          "Bull market — trending up, low vol. Lower entry bar.",
    },
    "BEAR": {
        "confidence_threshold": 0.60,
        "signal_multiplier":    0.92,
        "block_tau":            2,
        "description":          "Bear market — downtrend, elevated risk. Higher entry bar.",
    },
    "VOLATILE": {
        "confidence_threshold": 0.65,
        "signal_multiplier":    0.85,
        "block_tau":            2,
        "description":          "Volatile/choppy market — high VIX, no clear trend. Very selective.",
    },
    "NEUTRAL": {
        "confidence_threshold": 0.55,
        "signal_multiplier":    1.00,
        "block_tau":            3,
        "description":          "Neutral market — no strong regime signal. Default thresholds.",
    },
}


# ══════════════════════════════════════════════════════════════════════════════
#  REGIME DETECTION
# ══════════════════════════════════════════════════════════════════════════════

def _fetch_regime_data(lookback_days: int = 252) -> Optional[pd.DataFrame]:
    """Fetch SPY, VIX, TLT, GLD for regime analysis."""
    import yfinance as yf

    end   = datetime.today().strftime("%Y-%m-%d")
    start = (datetime.today() - timedelta(days=lookback_days + 30)).strftime("%Y-%m-%d")

    tickers = ["SPY", "^VIX", "TLT", "GLD", "^GSPC"]
    try:
        raw = yf.download(tickers, start=start, end=end,
                           auto_adjust=True, progress=False)
        if isinstance(raw.columns, pd.MultiIndex):
            close = raw["Close"].copy()
        else:
            close = raw[["Close"]].copy()

        close.columns = [c.replace("^", "").replace(" ", "_") for c in close.columns]
        close = close.dropna(how="all")
        return close
    except Exception as e:
        print(f"[RegimeClassifier] Data fetch failed: {e}")
        return None


def _compute_regime_features(close: pd.DataFrame) -> dict:
    """Compute regime indicator features from price data."""
    features = {}

    # ── SPY trend ─────────────────────────────────────────────────────────────
    if "SPY" in close.columns:
        spy = close["SPY"].dropna()
        if len(spy) >= 60:
            features["spy_trend_20d"] = float(spy.iloc[-1] / spy.iloc[-20] - 1)
            features["spy_trend_60d"] = float(spy.iloc[-1] / spy.iloc[-60] - 1)
            features["spy_above_ma50"] = int(spy.iloc[-1] > spy.rolling(50).mean().iloc[-1])
            features["spy_above_ma200"] = int(spy.iloc[-1] > spy.rolling(200).mean().iloc[-1])
            # Trend consistency: % of last 20 days that closed up
            daily_ret = spy.pct_change().iloc[-20:]
            features["spy_up_days_pct"] = float((daily_ret > 0).mean())
        else:
            features.update({"spy_trend_20d": 0, "spy_trend_60d": 0,
                              "spy_above_ma50": 1, "spy_above_ma200": 1,
                              "spy_up_days_pct": 0.5})
    else:
        features.update({"spy_trend_20d": 0, "spy_trend_60d": 0,
                          "spy_above_ma50": 1, "spy_above_ma200": 1,
                          "spy_up_days_pct": 0.5})

    # ── VIX ───────────────────────────────────────────────────────────────────
    vix_col = next((c for c in close.columns if "VIX" in c.upper()), None)
    if vix_col and len(close[vix_col].dropna()) >= 20:
        vix = close[vix_col].dropna()
        features["vix_level"]      = float(vix.iloc[-1])
        features["vix_ma20"]       = float(vix.rolling(20).mean().iloc[-1])
        features["vix_trend"]      = float(vix.iloc[-1] / vix.iloc[-5] - 1)  # 1-week VIX change
        # VIX percentile over last year
        vix_252 = vix.iloc[-252:] if len(vix) >= 252 else vix
        features["vix_percentile"] = float((vix_252 < vix.iloc[-1]).mean() * 100)
    else:
        features.update({"vix_level": 20, "vix_ma20": 20,
                          "vix_trend": 0, "vix_percentile": 50})

    # ── Bond signal (TLT) — rising bonds = risk-off ───────────────────────────
    if "TLT" in close.columns:
        tlt = close["TLT"].dropna()
        if len(tlt) >= 20:
            tlt_trend = float(tlt.iloc[-1] / tlt.iloc[-20] - 1)
            features["tlt_trend_20d"] = tlt_trend
            # Risk-off: bonds rising + stocks falling = danger
            features["bond_signal"] = (
                "RISK_OFF" if tlt_trend > 0.02 else
                "RISK_ON"  if tlt_trend < -0.02 else
                "NEUTRAL"
            )
        else:
            features["tlt_trend_20d"] = 0
            features["bond_signal"] = "NEUTRAL"
    else:
        features["tlt_trend_20d"] = 0
        features["bond_signal"] = "NEUTRAL"

    # ── Gold signal (GLD) — rising gold often = risk-off ─────────────────────
    if "GLD" in close.columns:
        gld = close["GLD"].dropna()
        if len(gld) >= 20:
            features["gld_trend_20d"] = float(gld.iloc[-1] / gld.iloc[-20] - 1)
        else:
            features["gld_trend_20d"] = 0
    else:
        features["gld_trend_20d"] = 0

    return features


def _classify_regime(features: dict) -> RegimeSignal:
    """
    Rule-based regime classification.

    Priority order:
    1. VOLATILE — high VIX + erratic market
    2. BEAR     — downtrend + elevated VIX
    3. BULL     — clear uptrend + low VIX
    4. NEUTRAL  — everything else
    """
    vix        = features.get("vix_level", 20)
    vix_pct    = features.get("vix_percentile", 50)
    spy_20d    = features.get("spy_trend_20d", 0)
    spy_60d    = features.get("spy_trend_60d", 0)
    spy_ma50   = features.get("spy_above_ma50", 1)
    spy_ma200  = features.get("spy_above_ma200", 1)
    vix_trend  = features.get("vix_trend", 0)
    bond_sig   = features.get("bond_signal", "NEUTRAL")
    up_days    = features.get("spy_up_days_pct", 0.5)

    # ── VOLATILE: VIX >= 30 or extreme moves ──────────────────────────────────
    if vix >= 30 or vix_pct >= 85:
        confidence = max(0.30, min(1.0, (vix - 20) / 20))
        cfg = REGIME_CONFIG["VOLATILE"]
        return RegimeSignal(
            label="VOLATILE",
            confidence=round(confidence, 2),
            spy_trend_20d=spy_20d,
            spy_trend_60d=spy_60d,
            vix_level=vix,
            vix_percentile=vix_pct,
            bond_signal=bond_sig,
            breadth_signal="WEAK" if spy_20d < -0.05 else "MIXED",
            **cfg,
        )

    # ── BEAR: downtrend + elevated risk ───────────────────────────────────────
    bear_signals = sum([
        spy_20d < -0.05,           # SPY down >5% in 20 days
        spy_60d < -0.10,           # SPY down >10% in 60 days
        spy_ma50 == 0,             # below 50-day MA
        spy_ma200 == 0,            # below 200-day MA (death cross territory)
        vix >= 20,                  # elevated fear
        bond_sig == "RISK_OFF",    # bonds rallying (flight to safety)
        up_days < 0.40,            # less than 40% up days recently
    ])

    if bear_signals >= 4:
        confidence = min(1.0, bear_signals / 7)
        cfg = REGIME_CONFIG["BEAR"]
        return RegimeSignal(
            label="BEAR",
            confidence=round(confidence, 2),
            spy_trend_20d=spy_20d,
            spy_trend_60d=spy_60d,
            vix_level=vix,
            vix_percentile=vix_pct,
            bond_signal=bond_sig,
            breadth_signal="WEAK",
            **cfg,
        )

    # ── BULL: clear uptrend + low risk ────────────────────────────────────────
    bull_signals = sum([
        spy_20d > 0.02,            # SPY up >2% in 20 days
        spy_60d > 0.05,            # SPY up >5% in 60 days
        spy_ma50 == 1,             # above 50-day MA
        spy_ma200 == 1,            # above 200-day MA
        vix < 18,                  # low fear
        bond_sig != "RISK_OFF",   # no flight to safety
        up_days > 0.55,            # more than 55% up days
    ])

    if bull_signals >= 4:
        confidence = min(1.0, bull_signals / 7)
        cfg = REGIME_CONFIG["BULL"]
        return RegimeSignal(
            label="BULL",
            confidence=round(confidence, 2),
            spy_trend_20d=spy_20d,
            spy_trend_60d=spy_60d,
            vix_level=vix,
            vix_percentile=vix_pct,
            bond_signal=bond_sig,
            breadth_signal="STRONG",
            **cfg,
        )

    # ── NEUTRAL: mixed signals ────────────────────────────────────────────────
    cfg = REGIME_CONFIG["NEUTRAL"]
    return RegimeSignal(
        label="NEUTRAL",
        confidence=0.5,
        spy_trend_20d=spy_20d,
        spy_trend_60d=spy_60d,
        vix_level=vix,
        vix_percentile=vix_pct,
        bond_signal=bond_sig,
        breadth_signal="MIXED",
        **cfg,
    )


# ══════════════════════════════════════════════════════════════════════════════
#  PUBLIC API
# ══════════════════════════════════════════════════════════════════════════════

def get_current_regime(use_cache: bool = True) -> RegimeSignal:
    """
    Get the current macro regime. Uses cache to avoid repeated API calls.

    Parameters
    ----------
    use_cache : if True, use cached result if <4 hours old

    Returns
    -------
    RegimeSignal with label, thresholds, and indicators
    """
    # Check cache
    if use_cache and CACHE_FILE.exists():
        try:
            with open(CACHE_FILE) as f:
                cached = json.load(f)
            age_hours = (datetime.utcnow() -
                         datetime.fromisoformat(cached.get("computed_at", "2000-01-01"))
                         ).total_seconds() / 3600
            if age_hours < CACHE_TTL_HOURS:
                return RegimeSignal.from_dict(cached)
        except Exception:
            pass

    # Compute fresh
    close = _fetch_regime_data()
    if close is None or close.empty:
        # Default to NEUTRAL if data unavailable
        cfg = REGIME_CONFIG["NEUTRAL"]
        return RegimeSignal(label="NEUTRAL", confidence=0.0,
                            description="Data unavailable — using neutral defaults.", **cfg)

    features = _compute_regime_features(close)
    regime   = _classify_regime(features)

    # Save to cache
    try:
        CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(CACHE_FILE, "w") as f:
            json.dump(regime.to_dict(), f, indent=2)
    except Exception:
        pass

    return regime


def apply_regime_to_signal(
    prob_up:   float,
    regime:    RegimeSignal,
    base_threshold: float = 0.55,
) -> dict:
    """
    Apply regime multiplier to a raw probability signal.

    Returns dict with:
        prob_effective     : regime-adjusted probability
        threshold          : regime-adjusted threshold
        signal             : "BUY" or "HOLD"
        regime_boost       : how much the regime changed the signal
    """
    prob_eff  = prob_up * regime.signal_multiplier
    threshold = regime.confidence_threshold or base_threshold
    signal    = "BUY" if prob_eff >= threshold else "HOLD"

    return {
        "prob_effective": round(prob_eff, 4),
        "prob_raw":       round(prob_up, 4),
        "threshold":      threshold,
        "signal":         signal,
        "regime":         regime.label,
        "regime_boost":   round(prob_eff - prob_up, 4),
    }


def get_regime_history(lookback_days: int = 252) -> pd.DataFrame:
    """
    Compute rolling regime labels for the past N days.
    Useful for backtesting and the regime history chart.
    """
    import yfinance as yf

    end   = datetime.today().strftime("%Y-%m-%d")
    start = (datetime.today() - timedelta(days=lookback_days + 60)).strftime("%Y-%m-%d")

    try:
        tickers = ["SPY", "^VIX", "TLT"]
        raw  = yf.download(tickers, start=start, end=end,
                            auto_adjust=True, progress=False)
        if isinstance(raw.columns, pd.MultiIndex):
            close = raw["Close"].copy()
        else:
            return pd.DataFrame()

        close.columns = [c.replace("^", "") for c in close.columns]
        close = close.dropna(how="all").iloc[-lookback_days:]

        rows = []
        for i in range(min(30, len(close)-1), len(close)):
            window = close.iloc[:i+1]
            feats  = _compute_regime_features(window)
            regime = _classify_regime(feats)
            rows.append({
                "date":    close.index[i].date(),
                "regime":  regime.label,
                "vix":     feats.get("vix_level", 20),
                "spy_ret_20d": feats.get("spy_trend_20d", 0),
            })

        return pd.DataFrame(rows)
    except Exception as e:
        print(f"[RegimeClassifier] History failed: {e}")
        return pd.DataFrame()
