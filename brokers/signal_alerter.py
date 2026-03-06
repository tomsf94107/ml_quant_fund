"""
brokers/signal_alerter.py
Compares ML Quant Fund signals to broker holdings and generates alerts.
No order execution — read-only comparison only.

Alert types:
  BUY_SIGNAL_NOT_HELD   — model says BUY but you don't own it
  HOLD_SIGNAL_HELD      — model says HOLD but you own it (consider selling)
  HIGH_CONF_BUY         — BUY signal with prob >= high_conf_threshold
  RISK_BLOCK            — signal blocked by risk gate (risk_next_3d high)
  POSITION_NO_SIGNAL    — you hold it but it's not in the model's ticker list
"""

from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional, Set
import pandas as pd


# ── Alert data class ──────────────────────────────────────────────────────────

@dataclass
class BrokerAlert:
    alert_type:  str          # one of the types above
    ticker:      str
    signal:      str          # BUY / HOLD
    prob:        float        # model probability
    confidence:  str          # LOW / MEDIUM / HIGH
    held:        bool         # currently held in portfolio
    broker:      str          # "alpaca", "robinhood", "both"
    message:     str
    timestamp:   str = field(default_factory=lambda: datetime.utcnow().isoformat())
    market_value: Optional[float] = None
    unrealized_plpc: Optional[float] = None


# ── Alert generator ───────────────────────────────────────────────────────────

def generate_alerts(
    signals_df: pd.DataFrame,
    alpaca_held: Set[str] = None,
    robinhood_held: Set[str] = None,
    high_conf_threshold: float = 0.70,
    include_hold_alerts: bool = True,
    alpaca_positions: pd.DataFrame = None,
    robinhood_positions: pd.DataFrame = None,
) -> List[BrokerAlert]:
    """
    Compare ML signals to broker holdings and return a list of BrokerAlert.

    Parameters
    ----------
    signals_df           : DataFrame with columns [ticker, signal, prob_up, confidence, blocked]
    alpaca_held          : set of tickers currently held in Alpaca
    robinhood_held       : set of tickers currently held in Robinhood
    high_conf_threshold  : prob_up threshold for HIGH_CONF_BUY alerts
    include_hold_alerts  : if True, alert when model says HOLD but you own it
    alpaca_positions     : full Alpaca positions df (for enriching alerts)
    robinhood_positions  : full Robinhood positions df (for enriching alerts)
    """
    alpaca_held     = alpaca_held     or set()
    robinhood_held  = robinhood_held  or set()
    all_held        = alpaca_held | robinhood_held
    model_tickers   = set(signals_df["ticker"].str.upper()) if not signals_df.empty else set()

    alerts: List[BrokerAlert] = []

    # ── Signals → alerts ─────────────────────────────────────────────────────
    for _, row in signals_df.iterrows():
        ticker     = str(row["ticker"]).upper()
        signal     = str(row.get("signal", "HOLD"))
        prob       = float(row.get("prob_up", 0.5))
        confidence = str(row.get("confidence", "LOW"))
        blocked    = bool(row.get("blocked", False))
        held       = ticker in all_held

        # Determine which broker(s) hold this
        brokers = []
        if ticker in alpaca_held:    brokers.append("alpaca")
        if ticker in robinhood_held: brokers.append("robinhood")
        broker_str = "+".join(brokers) if brokers else "none"

        # Enrich with position data
        mv, plpc = None, None
        for pos_df, bkr_held in [(alpaca_positions, alpaca_held),
                                  (robinhood_positions, robinhood_held)]:
            if pos_df is not None and not pos_df.empty and ticker in bkr_held:
                row_pos = pos_df[pos_df["symbol"] == ticker]
                if not row_pos.empty:
                    mv   = float(row_pos["market_value"].iloc[0])
                    plpc = float(row_pos["unrealized_plpc"].iloc[0])

        if blocked:
            alerts.append(BrokerAlert(
                alert_type  = "RISK_BLOCK",
                ticker      = ticker,
                signal      = signal,
                prob        = prob,
                confidence  = confidence,
                held        = held,
                broker      = broker_str,
                message     = f"⛔ {ticker}: signal blocked by risk gate (risk_next_3d high)",
                market_value    = mv,
                unrealized_plpc = plpc,
            ))
            continue

        if signal == "BUY" and not held:
            alerts.append(BrokerAlert(
                alert_type  = "BUY_SIGNAL_NOT_HELD",
                ticker      = ticker,
                signal      = signal,
                prob        = prob,
                confidence  = confidence,
                held        = False,
                broker      = "none",
                message     = f"🟢 {ticker}: BUY signal (p={prob:.1%}) — not in portfolio",
            ))

        if signal == "BUY" and prob >= high_conf_threshold:
            alerts.append(BrokerAlert(
                alert_type  = "HIGH_CONF_BUY",
                ticker      = ticker,
                signal      = signal,
                prob        = prob,
                confidence  = confidence,
                held        = held,
                broker      = broker_str,
                message     = f"🔥 {ticker}: HIGH confidence BUY (p={prob:.1%})",
                market_value    = mv,
                unrealized_plpc = plpc,
            ))

        if signal == "HOLD" and held and include_hold_alerts:
            plpc_str = f", P&L {plpc:+.1f}%" if plpc is not None else ""
            alerts.append(BrokerAlert(
                alert_type  = "HOLD_SIGNAL_HELD",
                ticker      = ticker,
                signal      = signal,
                prob        = prob,
                confidence  = confidence,
                held        = True,
                broker      = broker_str,
                message     = f"🔴 {ticker}: HOLD signal — you own this{plpc_str}",
                market_value    = mv,
                unrealized_plpc = plpc,
            ))

    # ── Holdings not in model ─────────────────────────────────────────────────
    untracked = all_held - model_tickers
    for ticker in sorted(untracked):
        broker_str = "+".join(
            b for b, held_set in [("alpaca", alpaca_held), ("robinhood", robinhood_held)]
            if ticker in held_set
        )
        alerts.append(BrokerAlert(
            alert_type  = "POSITION_NO_SIGNAL",
            ticker      = ticker,
            signal      = "N/A",
            prob        = 0.0,
            confidence  = "N/A",
            held        = True,
            broker      = broker_str,
            message     = f"⚪ {ticker}: held in {broker_str} but not tracked by ML model",
        ))

    # Sort: HIGH_CONF_BUY first, then BUY_SIGNAL_NOT_HELD, then rest
    priority = {"HIGH_CONF_BUY": 0, "BUY_SIGNAL_NOT_HELD": 1,
                 "HOLD_SIGNAL_HELD": 2, "RISK_BLOCK": 3, "POSITION_NO_SIGNAL": 4}
    alerts.sort(key=lambda a: (priority.get(a.alert_type, 9), -a.prob))

    return alerts


def alerts_to_df(alerts: List[BrokerAlert]) -> pd.DataFrame:
    if not alerts:
        return pd.DataFrame()
    return pd.DataFrame([{
        "type":       a.alert_type,
        "ticker":     a.ticker,
        "signal":     a.signal,
        "prob":       f"{a.prob:.1%}",
        "confidence": a.confidence,
        "held":       "✓" if a.held else "—",
        "broker":     a.broker,
        "message":    a.message,
        "p&l":        f"{a.unrealized_plpc:+.1f}%" if a.unrealized_plpc is not None else "—",
    } for a in alerts])
