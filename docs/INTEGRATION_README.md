# Closing the fundamentals/13F gap across ALL ticker reports — integration guide

## The root cause (why it was inconsistent)
Fundamentals + 13F lived in one chat's manual effort, not in the data pipeline. Chats
don't share memory, so other ticker chats never inherited it. Fix = put it in the MONITOR
(every chat pulls the monitor) + a shared report template (every chat reads the same spec).

## Why it must run in the monitor, not a Claude chat
The Claude bash sandbox BLOCKS data.sec.gov ('host_not_allowed' — only PyPI/npm/GitHub open).
Your LOCAL monitor has open internet and already resolves CIK+accession. So SEC pulls belong
in monitor_ticker.py, which runs locally. (Claude's web_fetch can spot-check one SEC URL after
a search, but can't run a pipeline.)

## Install (local box, ~15 min)
1. Drop `fundamentals_block.py` and `marquee_13f_block.py` next to monitor_ticker.py.
2. Set the User-Agent constant in BOTH to your real email (SEC blocks generic UAs).
3. In monitor_ticker.py, after the existing sections, add:
       from fundamentals_block import fundamentals_block
       from marquee_13f_block import format_snapshot, full_13f_block
       print(fundamentals_block(TICKER))                      # auto CIK
       # 13F — MODE A if you already store a snapshot:
       print(format_snapshot(TICKER, your_existing_13f_snapshot))
       # or MODE B (robust) if you have the ticker CUSIP:
       # print(full_13f_block(TICKER, cusip=CUSIP_FOR[TICKER]))
4. Cache company_tickers.json to disk (refresh weekly) to avoid re-downloading the CIK map.
5. Respect SEC limits: <=10 req/s (the modules sleep 150ms); descriptive User-Agent.

## Verify it works
    python3 fundamentals_block.py GOOG     # should print FCF/cash/debt table from XBRL
    python3 marquee_13f_block.py           # MODE A demo runs offline (no network)
If fundamentals_block prints '[warn] pull failed', check egress to data.sec.gov + the User-Agent.

## Then, in EACH ticker chat
Paste REPORT_TEMPLATE_v4.md first, then attach the monitor pull. Every report now carries
§6 fundamentals, §6.1 capex/FCF, §7.4 ownership, and the §8 rates line — automatically, GOOG or not.

## Tag discipline (keeps accuracy honest)
- XBRL numbers from the block  -> [fact]
- Conference / press metrics    -> [fact?] until in a filing
- 13F (quarter-lagged)          -> [fact], but label "ownership context, not timing"
- Anything not self-pulled      -> [fact?] + say so (never carry a peer report's number as your own)
