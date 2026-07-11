"""
mechanical_block_filter.py

Strip index-reconstitution / rebalance mega-prints from dark-pool signed-flow
skew before the monitor reports a directional lean.

WHY THIS EXISTS
---------------
On index reconstitution/rebalance days (e.g. Russell recon staging, S&P
rebalance), a single venue prints enormous crosses that are MECHANICAL, not
directional. The VWAP-sign heuristic then assigns the whole block a "buy" or
"sell" sign and the day's skew explodes to +95% / -80%, polluting the 7-day
aggregate. June 2026 had TWO such days back-to-back in the AI names:
  - 2026-06-18 (rebalance)
  - 2026-06-23 (Russell recon staging; recon effective 06-26)
Both must be excluded from skew, for every affected name, automatically.

WHAT IT DOES
------------
Flags a daily print row as mechanical if EITHER rule fires:
  RULE A (relative):  day's max single print >= MAX_PRINT_MULT x the trailing
                      median of daily max-prints (robust to the name's normal
                      block size). Catches names whose "normal" block is small.
  RULE B (absolute):  day's max single print >= ABS_PRINT_USD (a hard floor so
                      a $700M+ single cross is always caught even if the name
                      trades large blocks routinely).
  RULE C (date list): explicit known-mechanical dates (belt-and-suspenders for
                      recon/rebalance days you already know).

Then recomputes signed skew over the surviving days only, and returns an audit
trail of what was stripped and why.

This module is dependency-free (stdlib only) so it drops into the monitor with
no new installs. Wire `clean_skew()` into the dark-pool section and report
`result.clean_skew_pct` instead of the raw aggregate.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from statistics import median
from typing import Iterable, Sequence


# ---- Tunables -------------------------------------------------------------

# RULE A: a day is mechanical if its max single print is at least this multiple
# of the trailing median daily max-print. 8x is conservative — normal block
# clustering rarely exceeds ~4-5x; recon crosses are 15-300x.
MAX_PRINT_MULT = 8.0

# RULE B: hard dollar floor for a single print that is always mechanical.
# $250M single cross in one name is not a discretionary directional trade.
ABS_PRINT_USD = 250_000_000.0

# RULE A needs a minimum history to form a stable median; below this we lean on
# RULE B and RULE C only (avoid flagging on a 2-row window).
MIN_DAYS_FOR_MEDIAN = 5

# RULE C: explicit known recon/rebalance dates. Extend as you learn them.
# (June 2026 reconstitution staging + rebalance.)
KNOWN_MECHANICAL_DATES: set[str] = {
    "2026-06-18",
    "2026-06-23",
}


# ---- Data structures ------------------------------------------------------

@dataclass
class DayFlow:
    """One trading day of dark-pool signed-flow, as the monitor already has it."""
    day: str            # 'YYYY-MM-DD'
    buy_usd: float
    sell_usd: float
    max_print_usd: float

    @property
    def net_usd(self) -> float:
        return self.buy_usd - self.sell_usd

    @property
    def gross_usd(self) -> float:
        return self.buy_usd + self.sell_usd


@dataclass
class StripReason:
    day: str
    max_print_usd: float
    rules: list[str] = field(default_factory=list)   # which of A/B/C fired


@dataclass
class CleanSkewResult:
    raw_skew_pct: float
    clean_skew_pct: float
    raw_buy_usd: float
    raw_sell_usd: float
    clean_buy_usd: float
    clean_sell_usd: float
    days_total: int
    days_kept: int
    stripped: list[StripReason]

    @property
    def n_stripped(self) -> int:
        return len(self.stripped)

    def summary(self) -> str:
        if not self.stripped:
            return (f"skew {self.clean_skew_pct:+.1f}%  "
                    f"(no mechanical days; {self.days_kept} days)")
        days = ", ".join(f"{s.day}[{'+'.join(s.rules)}]" for s in self.stripped)
        return (f"skew {self.raw_skew_pct:+.1f}% raw -> "
                f"{self.clean_skew_pct:+.1f}% clean  "
                f"(stripped {self.n_stripped}: {days})")


# ---- Core -----------------------------------------------------------------

def _skew_pct(buy: float, sell: float) -> float:
    g = buy + sell
    return 0.0 if g == 0 else 100.0 * (buy - sell) / g


def flag_mechanical(
    days: Sequence[DayFlow],
    *,
    max_print_mult: float = MAX_PRINT_MULT,
    abs_print_usd: float = ABS_PRINT_USD,
    known_dates: Iterable[str] = KNOWN_MECHANICAL_DATES,
    min_days_for_median: int = MIN_DAYS_FOR_MEDIAN,
) -> list[StripReason]:
    """Return a StripReason for each day judged mechanical (rules A/B/C)."""
    known = set(known_dates)
    max_prints = [d.max_print_usd for d in days if d.max_print_usd > 0]
    med = median(max_prints) if len(max_prints) >= min_days_for_median else None

    out: list[StripReason] = []
    for d in days:
        fired: list[str] = []
        if med is not None and med > 0 and d.max_print_usd >= max_print_mult * med:
            fired.append("A")
        if d.max_print_usd >= abs_print_usd:
            fired.append("B")
        if d.day in known:
            fired.append("C")
        if fired:
            out.append(StripReason(day=d.day, max_print_usd=d.max_print_usd, rules=fired))
    return out


def clean_skew(
    days: Sequence[DayFlow],
    *,
    max_print_mult: float = MAX_PRINT_MULT,
    abs_print_usd: float = ABS_PRINT_USD,
    known_dates: Iterable[str] = KNOWN_MECHANICAL_DATES,
    min_days_for_median: int = MIN_DAYS_FOR_MEDIAN,
) -> CleanSkewResult:
    """
    Recompute 7-day (or N-day) signed skew with mechanical days removed.
    Pass in the same day rows the monitor already builds for the skew table.
    """
    stripped = flag_mechanical(
        days,
        max_print_mult=max_print_mult,
        abs_print_usd=abs_print_usd,
        known_dates=known_dates,
        min_days_for_median=min_days_for_median,
    )
    stripped_days = {s.day for s in stripped}

    raw_buy = sum(d.buy_usd for d in days)
    raw_sell = sum(d.sell_usd for d in days)
    kept = [d for d in days if d.day not in stripped_days]
    clean_buy = sum(d.buy_usd for d in kept)
    clean_sell = sum(d.sell_usd for d in kept)

    return CleanSkewResult(
        raw_skew_pct=_skew_pct(raw_buy, raw_sell),
        clean_skew_pct=_skew_pct(clean_buy, clean_sell),
        raw_buy_usd=raw_buy,
        raw_sell_usd=raw_sell,
        clean_buy_usd=clean_buy,
        clean_sell_usd=clean_sell,
        days_total=len(days),
        days_kept=len(kept),
        stripped=stripped,
    )


# ---- Self-test on the real numbers from today's dumps ---------------------

if __name__ == "__main__":
    # MRVL 7-day window from the 06-24 run. 06-18 had a $4.45B single print
    # that drove skew to +95.1%; 06-23 had a $106M cross.
    mrvl = [
        DayFlow("2026-06-24", 48_742_600, 53_451_963, 2_924_952),
        DayFlow("2026-06-23", 31_790_171, 46_726_343, 105_818_665),
        DayFlow("2026-06-22", 53_440_545, 47_426_180, 2_577_057),
        DayFlow("2026-06-18", 81_545_176, 2_032_245, 4_447_446_900),
        DayFlow("2026-06-17", 55_500_635, 57_709_650, 4_858_678),
        DayFlow("2026-06-16", 34_324_589, 49_070_888, 2_976_000),
        DayFlow("2026-06-15", 53_503_818, 66_463_853, 4_714_362),
    ]
    res = clean_skew(mrvl)
    print("MRVL :", res.summary())
    # Raw aggregate was +5.3% (buy 358.8M vs sell 322.9M) per the dump.
    # Stripping 06-18 (rules A+B+C) and 06-23 (rule A via $106M >> median, +C)
    # flips the read to genuinely negative — the real distribution signal.

    # CRWV: 06-18 printed $868M single (skew +76.2%); aggregate was +36.6% BUY.
    crwv = [
        DayFlow("2026-06-24", 25_086_277, 23_944_701, 3_058_262),
        DayFlow("2026-06-23", 47_449_591, 99_087_673, 20_064_916),
        DayFlow("2026-06-22", 46_624_295, 208_538_159, 112_301_181),
        DayFlow("2026-06-18", 1_764_522_931, 238_633_391, 868_748_930),
        DayFlow("2026-06-17", 66_987_505, 161_522_849, 36_361_428),
        DayFlow("2026-06-16", 45_973_364, 156_282_950, 27_963_967),
        DayFlow("2026-06-15", 78_788_527, 74_155_632, 9_581_598),
    ]
    res2 = clean_skew(crwv)
    print("CRWV :", res2.summary())
    # The headline "+36.6% BUY-side" kill-switch-relevant flag collapses once
    # the $1.76B mechanical buy-day is removed — exposing the real -50%-ish
    # distribution underneath, which matches the insider selling.
