#!/usr/bin/env python3
"""
monitor_open.py — OPEN-only wrapper around monitor_earnings.py.

Reuses every line of logic in monitor_earnings.py but:
  - Locks the ticker to OPEN
  - Uses a separate DB (open_monitor.db) so it doesn't pollute the multi-ticker DB
  - Inherits all bug fixes from the parent script automatically

Usage:
    python scripts/monitor_open.py
    python scripts/monitor_open.py --since 2026-04-01
    python scripts/monitor_open.py --skip news macro
    python scripts/monitor_open.py --csv

Place this file in the same folder as monitor_earnings.py (scripts/).
Reuses UW_API_KEY, EDGAR_USER_AGENT, MASSIVE_API_KEY env vars.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Make the parent script importable regardless of where this is run from
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import monitor_earnings as me  # noqa: E402

# Separate DB so OPEN-only runs don't share state with the multi-ticker monitor
me.DB_PATH = Path("open_monitor.db")


def main_open() -> int:
    """Force --tickers OPEN; pass through everything else."""
    # Strip any user-supplied --tickers and its values, then append our own
    new_argv: list[str] = [sys.argv[0]]
    skip_next = False
    in_tickers = False
    for arg in sys.argv[1:]:
        if skip_next:
            skip_next = False
            continue
        if arg == "--tickers":
            in_tickers = True
            continue
        if in_tickers:
            # Consume tickers until next flag (starts with '-')
            if arg.startswith("-"):
                in_tickers = False
                new_argv.append(arg)
            # otherwise drop the user's ticker arg silently
            continue
        new_argv.append(arg)

    new_argv += ["--tickers", "OPEN"]
    sys.argv = new_argv
    return me.main()


if __name__ == "__main__":
    sys.exit(main_open())
