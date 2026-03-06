"""
brokers/robinhood_client.py
Robinhood read-only client via robin_stocks (unofficial API).
Uses username + password + MFA (TOTP) from secrets.

Install: pip install robin_stocks pyotp
"""

from __future__ import annotations
import os
from dataclasses import dataclass
from typing import List, Optional
import pandas as pd


# ── Config ────────────────────────────────────────────────────────────────────

def _get_creds() -> tuple[str, str, str]:
    """Return (username, password, mfa_secret) from secrets or env."""
    try:
        import streamlit as st
        user   = st.secrets.get("ROBINHOOD_USERNAME", os.getenv("ROBINHOOD_USERNAME", ""))
        passwd = st.secrets.get("ROBINHOOD_PASSWORD", os.getenv("ROBINHOOD_PASSWORD", ""))
        mfa    = st.secrets.get("ROBINHOOD_MFA_SECRET", os.getenv("ROBINHOOD_MFA_SECRET", ""))
    except Exception:
        user   = os.getenv("ROBINHOOD_USERNAME", "")
        passwd = os.getenv("ROBINHOOD_PASSWORD", "")
        mfa    = os.getenv("ROBINHOOD_MFA_SECRET", "")
    return user, passwd, mfa


# ── Data classes ─────────────────────────────────────────────────────────────

@dataclass
class RobinhoodPosition:
    symbol:        str
    qty:           float
    avg_entry:     float
    current_price: float
    market_value:  float
    unrealized_pl: float
    unrealized_plpc: float


@dataclass
class RobinhoodAccount:
    equity:        float
    cash:          float
    portfolio_value: float


# ── Client ────────────────────────────────────────────────────────────────────

class RobinhoodClient:
    """
    Lightweight Robinhood read-only client via robin_stocks.
    Handles TOTP MFA automatically if ROBINHOOD_MFA_SECRET is provided.
    """

    def __init__(self):
        self.username, self.password, self.mfa_secret = _get_creds()
        self._logged_in = False

    def is_configured(self) -> bool:
        return bool(self.username and self.password)

    def _login(self) -> bool:
        if self._logged_in:
            return True
        if not self.is_configured():
            return False
        try:
            import robin_stocks.robinhood as rh
            mfa_code = None
            if self.mfa_secret:
                try:
                    import pyotp
                    mfa_code = pyotp.TOTP(self.mfa_secret).now()
                except ImportError:
                    pass

            rh.login(
                self.username,
                self.password,
                mfa_code=mfa_code,
                store_session=False,
                pickle_name="ml_quant_rh",
            )
            self._logged_in = True
            return True
        except Exception as e:
            print(f"[RobinhoodClient] Login failed: {e}")
            self._logged_in = False
            return False

    def get_account(self) -> Optional[RobinhoodAccount]:
        if not self._login():
            return None
        try:
            import robin_stocks.robinhood as rh
            profile = rh.load_portfolio_profile()
            return RobinhoodAccount(
                equity          = float(profile.get("equity", 0)),
                cash            = float(profile.get("withdrawable_amount", 0)),
                portfolio_value = float(profile.get("market_value", 0)) +
                                  float(profile.get("withdrawable_amount", 0)),
            )
        except Exception:
            return None

    def get_positions(self) -> List[RobinhoodPosition]:
        if not self._login():
            return []
        try:
            import robin_stocks.robinhood as rh
            raw = rh.get_open_stock_positions()
            positions = []
            for p in (raw or []):
                try:
                    symbol = rh.get_symbol_by_url(p["instrument"])
                    qty    = float(p.get("quantity", 0))
                    entry  = float(p.get("average_buy_price", 0))
                    price_info = rh.get_latest_price(symbol)
                    price  = float(price_info[0]) if price_info else entry
                    mv     = qty * price
                    pl     = (price - entry) * qty
                    plpc   = ((price - entry) / entry * 100) if entry else 0.0
                    positions.append(RobinhoodPosition(
                        symbol          = symbol,
                        qty             = qty,
                        avg_entry       = entry,
                        current_price   = price,
                        market_value    = mv,
                        unrealized_pl   = pl,
                        unrealized_plpc = plpc,
                    ))
                except Exception:
                    continue
            return positions
        except Exception:
            return []

    def get_positions_df(self) -> pd.DataFrame:
        positions = self.get_positions()
        if not positions:
            return pd.DataFrame()
        return pd.DataFrame([{
            "symbol":           p.symbol,
            "qty":              p.qty,
            "avg_entry":        p.avg_entry,
            "current_price":    p.current_price,
            "market_value":     p.market_value,
            "unrealized_pl":    p.unrealized_pl,
            "unrealized_plpc":  p.unrealized_plpc,
        } for p in positions])

    def get_held_tickers(self) -> set[str]:
        return {p.symbol for p in self.get_positions()}

    def logout(self):
        if self._logged_in:
            try:
                import robin_stocks.robinhood as rh
                rh.logout()
            except Exception:
                pass
            self._logged_in = False
