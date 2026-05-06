# Walk-Forward Hang — Debug Plan May 6 2026

## Status as of May 5 end-of-day

5 cache fixes shipped today (FRED, macro, ticker OHLCV, UW skip, Session reuse).
All verified working in unit tests. Walk-forward STILL hangs in setup phase
across 4 attempts. Connection-pool state issue suspected, not API itself.

## Hypothesis (from ChatGPT consult)

`requests.Session` reuses TCP connections via urllib3 pooling. In long-running
walk-forward process, pooled connections can go stale. Each retry hits same
bad pooled connection → all retries time out. Fresh `curl` works because each
call gets fresh DNS/socket/TLS state.

yfinance uses `curl_cffi` backend with its own connection cache that has
similar staleness issues.

## Step 1 (ONLY do this first): Add faulthandler

Goal: stop guessing, get actual stack traces during hangs.

Add to TOP of `analysis/walk_forward.py` BEFORE any other imports:

```python
import faulthandler
import signal
from pathlib import Path

Path("logs").mkdir(exist_ok=True)
_STACK_LOG = open("logs/walk_forward_stacks.log", "a", buffering=1)
faulthandler.enable(file=_STACK_LOG, all_threads=True)
faulthandler.dump_traceback_later(60, repeat=True, file=_STACK_LOG)

if hasattr(signal, "SIGUSR1"):
    faulthandler.register(signal.SIGUSR1, file=_STACK_LOG, all_threads=True)
```

Launch with:
```bash
PYTHONUNBUFFERED=1 \
PYTHONFAULTHANDLER=1 \
ML_QUANT_SKIP_UW_CALENDAR=1 \
nohup caffeinate ~/.pyenv/versions/ml_quant_310/bin/python -u -m analysis.walk_forward --pit \
  > logs/walkforward/run_$(date +%Y%m%d_%H%M).log 2>&1 &
```

Wait 5 minutes. If hung, run:
```bash
PID=$(pgrep -f walk_forward)
kill -USR1 $PID            # forces faulthandler dump now
sample $PID 5              # 5-second profile
lsof -nP -p $PID | head -30
tail -50 logs/walk_forward_stacks.log
```

## Step 2: Branch based on stack trace

**If stack shows `urllib3/connectionpool.py`** → connection pool stale.
Apply Session-reset patch from ChatGPT response. Reduce max_retries 6→3.
File: features/massive_client.py.

**If stack shows `curl_cffi`** → yfinance backend issue.
Disable yfinance fallback during --pit. Add ML_QUANT_ALLOW_YFINANCE_FALLBACK=0
default. File: features/builder.py around _download_yfinance call.

**If stack shows elsewhere** (feature engineering, sqlite, etc.) → "network
hang" diagnosis is wrong. Reassess from new evidence.

## Step 3 (only if step 2 patch insufficient)

- Add circuit breaker: features/builder.py — track failed (provider, symbol)
  pairs, skip for 30 min after first failure
- Test `api.massive.com` vs `api.polygon.io` (rebrand)
- Add timing logs around every provider boundary

## What NOT to do

- Don't apply ChatGPT's full Session-reset patch without seeing the stack first.
- Don't disable yfinance fallback without confirming curl_cffi is in stack.
- Don't persist cache to disk yet — bigger architectural change, defer.

## State at start of May 6

All cache patches in place (commits bb9308a, d6b3ac6, 940b76e, 0d90302,
1979323, a802ec1, plus de5ce2e, 66226a8, e3ad223). Outcomes table has
8,651 corrected rows. Models retrained May 4 16:47 against corrected labels.
