"""
brokers/alpaca_client.py
Alpaca Markets read-only client — fetches account, positions, and order history.
Requires ALPACA_API_KEY and ALPACA_SECRET_KEY in .streamlit/secrets.toml or env.

Install: pip install alpaca-py
"""

from __future__ import annotations
import os
from dataclasses import dataclass, field
from typing import List, Optional
import pandas as pd


# ── Config ────────────────────────────────────────────────────────────────────

PAPER_BASE_URL = "https://paper-api.alpaca.markets"
LIVE_BASE_URL  = "https://api.alpaca.markets"


def _get_creds() -> tuple[str, str, bool]:
    """Return (api_key, secret_key, is_paper) from secrets or env."""
    try:
        import streamlit as st
        key    = st.secrets.get("ALPACA_API_KEY",    os.getenv("ALPACA_API_KEY", ""))
        secret = st.secrets.get("ALPACA_SECRET_KEY", os.getenv("ALPACA_SECRET_KEY", ""))
        paper  = str(st.secrets.get("ALPACA_PAPER",  os.getenv("ALPACA_PAPER", "true"))).lower() == "true"
    except Exception:
        key    = os.getenv("ALPACA_API_KEY", "")
        secret = os.getenv("ALPACA_SECRET_KEY", "")
        paper  = os.getenv("ALPACA_PAPER", "true").lower() == "true"
    return key, secret, paper


# ── Data classes ─────────────────────────────────────────────────────────────

@dataclass
class AlpacaPosition:
    symbol:        str
    qty:           float
    avg_entry:     float
    current_price: float
    market_value:  float
    unrealized_pl: float
    unrealized_plpc: float
    side:          str   # "long" or "short"


@dataclass
class AlpacaAccount:
    equity:        float
    cash:          float
    buying_power:  float
    portfolio_value: float
    is_paper:      bool


# ── Client ────────────────────────────────────────────────────────────────────

class AlpacaClient:
    """
    Lightweight Alpaca read-only client.
    Falls back gracefully if alpaca-trade-api is not installed.
    """

    def __init__(self):
        self.api_key, self.secret_key, self.is_paper = _get_creds()
        self.base_url = PAPER_BASE_URL if self.is_paper else LIVE_BASE_URL
        self._api = None

    def _connect(self):
        if self._api is not None:
            return True
        if not self.api_key or not self.secret_key:
            return False
        try:
            from alpaca.trading.client import TradingClient
            self._api = TradingClient(
                self.api_key, self.secret_key,
                paper=self.is_paper
            )
            return True
        except ImportError:
            return False
        except Exception:
            return False

    def is_configured(self) -> bool:
        return bool(self.api_key and self.secret_key)

    def get_account(self) -> Optional[AlpacaAccount]:
        if not self._connect():
            return None
        try:
            acct = self._api.get_account()
            return AlpacaAccount(
                equity          = float(acct.equity),
                cash            = float(acct.cash),
                buying_power    = float(acct.buying_power),
                portfolio_value = float(acct.portfolio_value),
                is_paper        = self.is_paper,
            )
        except Exception:
            return None

    def get_positions(self) -> List[AlpacaPosition]:
        if not self._connect():
            return []
        try:
            from alpaca.trading.requests import GetAssetsRequest
            raw = self._api.get_all_positions()
            positions = []
            for p in raw:
                positions.append(AlpacaPosition(
                    symbol          = p.symbol,
                    qty             = float(p.qty),
                    avg_entry       = float(p.avg_entry_price),
                    current_price   = float(p.current_price),
                    market_value    = float(p.market_value),
                    unrealized_pl   = float(p.unrealized_pl),
                    unrealized_plpc = float(p.unrealized_plpc) * 100,
                    side            = p.side.value,
                ))
            return positions
        except Exception:
            return []

    def get_positions_df(self) -> pd.DataFrame:
        positions = self.get_positions()
        if not positions:
            return pd.DataFrame()
        return pd.DataFrame([{
            "symbol":        p.symbol,
            "qty":           p.qty,
            "avg_entry":     p.avg_entry,
            "current_price": p.current_price,
            "market_value":  p.market_value,
            "unrealized_pl": p.unrealized_pl,
            "unrealized_plpc": p.unrealized_plpc,
            "side":          p.side,
        } for p in positions])

    def get_held_tickers(self) -> set[str]:
        return {p.symbol for p in self.get_positions()}

    def get_recent_orders(self, limit: int = 20) -> pd.DataFrame:
        if not self._connect():
            return pd.DataFrame()
        try:
            from alpaca.trading.requests import GetOrdersRequest
            from alpaca.trading.enums import QueryOrderStatus
            req = GetOrdersRequest(status=QueryOrderStatus.ALL, limit=limit)
            orders = self._api.get_orders(filter=req)
            rows = []
            for o in orders:
                rows.append({
                    "symbol":     o.symbol,
                    "side":       o.side.value,
                    "qty":        o.qty,
                    "type":       o.type.value,
                    "status":     o.status.value,
                    "submitted":  o.submitted_at,
                    "filled_avg": o.filled_avg_price,
                })
            return pd.DataFrame(rows)
        except Exception:
            return pd.DataFrame()
