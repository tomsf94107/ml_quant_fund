"""Single source of truth for timezone handling — always use ET."""
import pytz
from datetime import datetime

ET = pytz.timezone("America/New_York")

def now_et() -> datetime:
    return datetime.now(ET)

def today_et() -> str:
    return datetime.now(ET).strftime("%Y-%m-%d")

def ts_et() -> str:
    return datetime.now(ET).strftime("%Y-%m-%dT%H:%M:%S")

def fmt_et(fmt: str = "%Y-%m-%d %H:%M ET") -> str:
    return datetime.now(ET).strftime(fmt)
