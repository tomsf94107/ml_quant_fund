"""
config/keys.py — SINGLE SOURCE OF TRUTH for API keys.

ALL code in this project reads API keys from here.
Update keys ONLY in /Users/atomnguyen/Desktop/ML_Quant_Fund/.env

USAGE:
    from config.keys import MASSIVE_API_KEY, UW_API_KEY, ANTHROPIC_API_KEY

    # Or with required-not-empty enforcement:
    from config.keys import require
    key = require("MASSIVE_API_KEY")
"""
from __future__ import annotations
import os
from pathlib import Path

# Load .env from project root
try:
    from dotenv import load_dotenv
    _ENV_PATH = Path(__file__).resolve().parent.parent / ".env"
    load_dotenv(_ENV_PATH, override=False)
except ImportError:
    # If python-dotenv isn't installed, fall back to env vars only
    pass

# ─────────────────────────────────────────────────────────────────────────────
# Single export point — every key read by any code in this project
# ─────────────────────────────────────────────────────────────────────────────

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "").strip()
MASSIVE_API_KEY   = os.getenv("MASSIVE_API_KEY", "").strip()
UW_API_KEY        = os.getenv("UW_API_KEY", "").strip()

# Legacy alias — Polygon was rebranded to Massive. Same backend, same key.
# Old code uses POLYGON_API_KEY or POLYGON_KEY; new code uses MASSIVE_API_KEY.
# This alias prevents breakage during the rename transition.
POLYGON_API_KEY   = MASSIVE_API_KEY
POLYGON_KEY       = MASSIVE_API_KEY  # very old code

# Alpha Vantage (used as a 2nd-tier fallback in accuracy/sink.py)
ALPHA_VANTAGE_KEY = os.getenv("ALPHA_VANTAGE_KEY", "").strip()


def require(name: str) -> str:
    """
    Fetch a key by variable name, raise RuntimeError if empty.
    
    Use this when a key is REQUIRED for the operation to proceed.
    For optional keys (e.g. fallback paths), just import the variable
    and check `if MASSIVE_API_KEY:` yourself.
    """
    val = globals().get(name, "")
    if not val:
        raise RuntimeError(
            f"Missing API key: {name}. "
            f"Add it to .env in the project root."
        )
    return val


def status() -> dict:
    """
    Return a dict of which keys are present (without revealing values).
    Use for diagnostics: `python -c 'from config.keys import status; print(status())'`
    """
    return {
        "ANTHROPIC_API_KEY":  bool(ANTHROPIC_API_KEY),
        "MASSIVE_API_KEY":    bool(MASSIVE_API_KEY),
        "UW_API_KEY":         bool(UW_API_KEY),
        "ALPHA_VANTAGE_KEY":  bool(ALPHA_VANTAGE_KEY),
    }


if __name__ == "__main__":
    # Allow running this file directly to check key status
    print("API key status:")
    for name, present in status().items():
        marker = "OK " if present else "MISSING"
        print(f"  [{marker}] {name}")
