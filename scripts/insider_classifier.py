# scripts/insider_classifier.py
# ─────────────────────────────────────────────────────────────────────────────
# Pure classification logic for insider Form 4 filings.
# No I/O, no DB, no notifications. Just: filing dict → (signal, rationale) | None.
#
# Used by scripts/insider_alert_check.py (Phase 2 — not built yet) and
# tested directly by tests/test_insider_classifier.py.
# ─────────────────────────────────────────────────────────────────────────────

from __future__ import annotations
from typing import Optional, TypedDict


class FilingDict(TypedDict, total=False):
    """Subset of insider_filings_raw row that the classifier reads.
    Keys match the column names in the raw table for direct dict-from-row use."""
    ticker:            str
    insider_name:      str
    insider_title:     str
    role_weight:       float
    transaction_code:  str          # P/S/A/G/F/M
    shares:            float
    price_per_share:   Optional[float]
    notional_usd:      Optional[float]
    acquired_disposed: str          # 'A' or 'D'
    is_csuite:         int          # 0 or 1


# ── Thresholds (USD notional). Tune over time based on logged alerts. ────────
# Buys are rare → lower thresholds (any meaningful C-suite buy matters).
# Sells are noisy (RSU vests, diversification) → higher thresholds.

GREEN_STRONG_CSUITE_BUY_USD     = 250_000
GREEN_WEAK_CSUITE_BUY_USD       = 50_000
GREEN_WEAK_ANY_BUY_USD          = 1_000_000     # large buy by any insider, even non-csuite

RED_STRONG_CSUITE_SELL_USD      = 1_000_000
RED_WEAK_CSUITE_SELL_USD        = 250_000

# Role-weight cutoff for "C-suite". CEO/CFO/COO/President = 3.0/2.5.
# VP/SVP/GC = 1.5. Director = 1.0.
CSUITE_ROLE_WEIGHT              = 2.5


# ── Signal types (string constants for safety vs. typos) ─────────────────────
GREEN_STRONG = "GREEN_STRONG"
GREEN_WEAK   = "GREEN_WEAK"
RED_STRONG   = "RED_STRONG"
RED_WEAK     = "RED_WEAK"

# Severity ordering for callers that want to take only the "strongest" signal
# in a batch from one insider in one filing.
SIGNAL_RANK = {GREEN_STRONG: 4, RED_STRONG: 3, GREEN_WEAK: 2, RED_WEAK: 1}


# ── Public classify function ─────────────────────────────────────────────────

def classify(filing: FilingDict, current_price: Optional[float] = None
             ) -> Optional[tuple[str, str]]:
    """
    Classify a single Form 4 transaction. Returns (signal, rationale) or None.

    Args:
        filing:        dict with keys from insider_filings_raw schema
        current_price: market price for the ticker (used as fallback when
                       filing's price_per_share is missing). If both are None,
                       notional is inferred only from filing data.

    Returns:
        (signal, rationale) tuple if filing crosses any alert threshold,
        else None.

    Logic:
        - Only open-market trades trigger alerts: code='P' (purchase), 'S' (sale).
          Grants (A), gifts (G), exercises (M), tax-withholding (F) are excluded.
        - Buys: lower threshold, especially for C-suite.
        - Sells: higher threshold (RSU vests are noisy).
    """
    code = (filing.get("transaction_code") or "").upper().strip()
    ad   = (filing.get("acquired_disposed") or "").upper().strip()

    # Only open-market P/S transactions are signal. Skip everything else early.
    if code not in ("P", "S"):
        return None
    if ad not in ("A", "D"):
        return None

    is_buy  = (code == "P" and ad == "A")
    is_sell = (code == "S" and ad == "D")
    if not (is_buy or is_sell):
        # Mismatched code/AD code (e.g. P with D) — log it elsewhere; don't classify.
        return None

    # Notional in USD: prefer filing's price, fall back to current market price.
    shares       = float(filing.get("shares") or 0)
    if shares <= 0:
        return None  # zero-share or malformed row

    notional = filing.get("notional_usd")
    if notional is None or notional == 0:
        price = filing.get("price_per_share") or current_price
        if price is None:
            # No notional at all. Without a dollar size we can't apply thresholds.
            return None
        notional = shares * float(price)
    notional = float(notional)

    is_csuite = bool(filing.get("is_csuite", 0)) or \
                (filing.get("role_weight") or 0) >= CSUITE_ROLE_WEIGHT
    title     = filing.get("insider_title") or "Insider"
    name      = filing.get("insider_name") or ""
    name_part = f" by {name}" if name else ""

    # ── BUY signals ──────────────────────────────────────────────────────────
    if is_buy:
        if is_csuite and notional >= GREEN_STRONG_CSUITE_BUY_USD:
            return (
                GREEN_STRONG,
                f"C-suite open-market BUY{name_part} ({title}) "
                f"~${notional:,.0f} on {filing.get('ticker','?')}"
            )
        if is_csuite and notional >= GREEN_WEAK_CSUITE_BUY_USD:
            return (
                GREEN_WEAK,
                f"C-suite open-market buy{name_part} ({title}) "
                f"~${notional:,.0f} on {filing.get('ticker','?')}"
            )
        if notional >= GREEN_WEAK_ANY_BUY_USD:
            return (
                GREEN_WEAK,
                f"Large open-market buy{name_part} ({title}) "
                f"~${notional:,.0f} on {filing.get('ticker','?')}"
            )
        return None

    # ── SELL signals ─────────────────────────────────────────────────────────
    if is_sell:
        if is_csuite and notional >= RED_STRONG_CSUITE_SELL_USD:
            return (
                RED_STRONG,
                f"C-suite SELL{name_part} ({title}) "
                f"~${notional:,.0f} on {filing.get('ticker','?')}"
            )
        if is_csuite and notional >= RED_WEAK_CSUITE_SELL_USD:
            return (
                RED_WEAK,
                f"C-suite sell{name_part} ({title}) "
                f"~${notional:,.0f} on {filing.get('ticker','?')}"
            )
        # Non-csuite sells are too noisy to alert on individually.
        return None

    return None


def classify_cluster(filings: list[FilingDict], window_days: int = 5
                     ) -> Optional[tuple[str, str]]:
    """
    Cluster signal: 3+ C-suite open-market BUYS on the same ticker within
    `window_days` is materially stronger than any individual filing.
    (Multiple C-suite sells happen routinely via RSU vests — clusters of
    sells are NOT auto-promoted.)

    Args:
        filings:     list of FilingDict, ALREADY filtered to a single ticker
                     and trade dates within window_days of each other.
        window_days: window already enforced by caller. Param here for clarity.

    Returns:
        ('GREEN_STRONG', rationale) if cluster of 3+ csuite buys,
        else None.
    """
    if not filings:
        return None
    ticker = filings[0].get("ticker", "?")

    # Count distinct C-suite insiders who did open-market buys
    csuite_buyers: set[str] = set()
    total_notional = 0.0
    for f in filings:
        if (f.get("transaction_code") or "").upper() != "P":
            continue
        if (f.get("acquired_disposed") or "").upper() != "A":
            continue
        is_csuite = bool(f.get("is_csuite", 0)) or \
                    (f.get("role_weight") or 0) >= CSUITE_ROLE_WEIGHT
        if not is_csuite:
            continue
        name = f.get("insider_name") or ""
        if name:
            csuite_buyers.add(name)
        if f.get("notional_usd"):
            total_notional += float(f["notional_usd"])

    if len(csuite_buyers) >= 3:
        return (
            GREEN_STRONG,
            f"Cluster: {len(csuite_buyers)} C-suite insiders bought {ticker} "
            f"in {window_days}d (~${total_notional:,.0f} total)"
        )
    return None


__all__ = [
    "classify",
    "classify_cluster",
    "GREEN_STRONG", "GREEN_WEAK", "RED_STRONG", "RED_WEAK",
    "SIGNAL_RANK",
    "FilingDict",
]
