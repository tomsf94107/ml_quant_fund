"""
recession/alert.py

The recession-probability alert system.

WHY THIS EXISTS
---------------
The NY Fed publishes its recession-probability chart monthly — but it is
a PASSIVE chart. Nobody is notified when the probability climbs; someone
has to remember to look. That passivity is a real reason warnings get
missed. This project had the same gap: refresh.py computed the weekly
probability and logged a number, but nothing flagged a meaningful change.

alert.py closes that gap. It turns the passive number into an active
signal: when the recession probability CROSSES UP through a threshold
since the last check, it raises an alert.

DESIGN — file-based, no credentials
-----------------------------------
The alert is written to a dedicated log (recession_alerts.log), and the
alert status is returned to the caller. It deliberately does NOT send
email or SMS: that would require storing credentials in or near the code,
which the project's safety rules forbid. A file-based alert is reliable,
credential-free, and a clean surface — email/SMS can be layered on top by
watching the alert file, without this module ever holding a secret.

WHAT COUNTS AS AN ALERT — a CROSSING, not a level
-------------------------------------------------
The alert fires on a threshold CROSSING: the probability is at or above a
threshold now, and was BELOW it at the previous check. A probability that
simply stays high does not re-alert every week — that would be noise. The
event of interest is the transition into a higher-risk band.

To detect a crossing, alert.py needs the PREVIOUS probability. It reads it
from the alert state file (recession_alert_state.json), which it also
updates. The first ever run has no previous value and cannot have crossed
anything — it records the baseline and does not alert.

THRESHOLDS — reused, not invented
---------------------------------
The threshold bands are the SAME ones the lead-time analysis already uses
(0.3 / 0.5 / 0.7). They are not new magic numbers — they are the
operating points the project already characterised for warning lead time
and false-alarm rate. Each band has a plain-language risk label.
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Optional


# threshold bands — the same operating points used by the lead-time
# analysis (recession/validation/lead_time.py). Ordered low to high.
ALERT_THRESHOLDS = [
    (0.30, "ELEVATED"),
    (0.50, "HIGH"),
    (0.70, "SEVERE"),
]


def _alert_log_path() -> Path:
    return Path(__file__).resolve().parent / "recession_alerts.log"


def _alert_state_path() -> Path:
    return Path(__file__).resolve().parent / "recession_alert_state.json"


def _highest_band_crossed(prev: Optional[float], curr: float) -> Optional[tuple]:
    """The highest threshold band the probability crossed UP through.

    A band (thr, label) is crossed up if curr >= thr and (prev is None is
    NOT a crossing — see below — so prev must be a real number) prev < thr.
    Returns the highest such (thr, label), or None if no upward crossing.
    """
    if prev is None:
        return None                      # no baseline => cannot have crossed
    crossed = [(thr, label) for thr, label in ALERT_THRESHOLDS
               if curr >= thr > prev]
    if not crossed:
        return None
    return max(crossed, key=lambda tl: tl[0])


def _current_band(curr: float) -> Optional[tuple]:
    """The highest band the current probability is in (for context),
    or None if below the lowest threshold."""
    in_band = [(thr, label) for thr, label in ALERT_THRESHOLDS
               if curr >= thr]
    if not in_band:
        return None
    return max(in_band, key=lambda tl: tl[0])


def _read_state(state_path: Path) -> Optional[float]:
    """The previous recorded probability, or None if no state yet."""
    if not state_path.exists():
        return None
    try:
        with open(state_path) as f:
            data = json.load(f)
        val = data.get("last_probability")
        return float(val) if val is not None else None
    except Exception:
        return None                      # corrupt state => treat as no baseline


def _write_state(state_path: Path, probability: float,
                 month: str) -> None:
    """Record the current probability as the new baseline."""
    try:
        with open(state_path, "w") as f:
            json.dump({"last_probability": probability,
                       "last_month": month,
                       "updated": datetime.now().isoformat(timespec="seconds")},
                      f, indent=2)
    except Exception:
        pass                             # state write failure must not crash refresh


def _append_alert(alert_log: Path, line: str) -> None:
    alert_log.parent.mkdir(parents=True, exist_ok=True)
    with open(alert_log, "a") as f:
        f.write(line.rstrip("\n") + "\n")


def check_alert(
    probability: float,
    month: str,
    *,
    alert_log_path: Optional[Path] = None,
    state_path: Optional[Path] = None,
) -> dict:
    """Check the latest recession probability for a threshold crossing.

    probability : the latest recession probability (0..1).
    month       : the month the probability is for, e.g. '2026-05'.

    Reads the previous probability from the state file, detects whether
    `probability` crossed UP through any threshold band, writes an alert
    line if so, and updates the state file.

    Returns:
        {'alerted': bool,
         'crossed_band': (threshold, label) or None,
         'current_band': (threshold, label) or None,
         'previous_probability': float or None,
         'message': str}
    """
    alert_log_path = alert_log_path or _alert_log_path()
    state_path = state_path or _alert_state_path()
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    prev = _read_state(state_path)
    crossed = _highest_band_crossed(prev, probability)
    current = _current_band(probability)

    # update the baseline for next time, regardless of whether we alerted
    _write_state(state_path, probability, month)

    if crossed is not None:
        thr, label = crossed
        prev_s = f"{prev:.4f}" if prev is not None else "n/a"
        line = (f"[{stamp}] ALERT  recession_prob({month})="
                f"{probability:.4f} CROSSED UP through {thr:.2f} "
                f"({label}) — previous {prev_s}")
        _append_alert(alert_log_path, line)
        return {
            "alerted": True,
            "crossed_band": crossed,
            "current_band": current,
            "previous_probability": prev,
            "message": line,
        }

    # no crossing — no alert written
    if prev is None:
        msg = (f"[{stamp}] baseline recorded recession_prob({month})="
               f"{probability:.4f} — no previous value, no crossing "
               f"possible")
    else:
        cur_label = current[1] if current else "below ELEVATED"
        msg = (f"recession_prob({month})={probability:.4f}, no upward "
               f"threshold crossing (previous {prev:.4f}; current band: "
               f"{cur_label})")
    return {
        "alerted": False,
        "crossed_band": None,
        "current_band": current,
        "previous_probability": prev,
        "message": msg,
    }
