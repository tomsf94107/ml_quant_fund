"""
fundamentals_block.py  —  drop-in monitor module for SEC XBRL fundamentals.
Ticker-agnostic. Emits a =SEC FUNDAMENTALS= block for ANY US filer.

Closes the gap that made hand-built reports inconsistent: FCF / cash / debt /
capex / revenue now come from primary XBRL, tagged [fact], every pull.

Requires: internet to data.sec.gov + www.sec.gov (open on your local box; NOT
in the claude bash sandbox, which is why this must run in the monitor, not a chat).

Usage in monitor_ticker.py:
    from fundamentals_block import fundamentals_block
    print(fundamentals_block("GOOG"))          # auto-resolves CIK
    # or if you already resolved CIK elsewhere:
    print(fundamentals_block("GOOG", cik="0001652044"))
"""
import json, time, urllib.request, gzip, io, datetime as _dt

_UA = "ML Quant Fund research atom.v.nguyen@gmail.com"   # SEC REQUIRES a real contact
_HDR = {"User-Agent": _UA, "Accept-Encoding": "gzip, deflate"}
_CIK_CACHE = {}          # ticker -> 10-digit cik   (module-level; persist to disk if you like)
_RL_SLEEP = 0.15         # >100ms between calls keeps you under the 10 req/s SEC limit

# --- tag fallbacks: filers don't all use the same XBRL concept name ---------
_TAGS = {
    "revenue":  ["RevenueFromContractWithCustomerExcludingAssessedTax",
                 "RevenueFromContractWithCustomerIncludingAssessedTax", "Revenues"],
    "op_inc":   ["OperatingIncomeLoss"],
    "net_inc":  ["NetIncomeLoss"],
    "eps_dil":  ["EarningsPerShareDiluted"],
    "ocf":      ["NetCashProvidedByUsedInOperatingActivities",
                 "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations"],
    "capex":    ["PaymentsToAcquirePropertyPlantAndEquipment",
                 "PaymentsToAcquireProductiveAssets"],
    "cash":     ["CashAndCashEquivalentsAtCarryingValue"],
    "lt_debt":  ["LongTermDebtNoncurrent", "LongTermDebt"],
}

def _get(url):
    req = urllib.request.Request(url, headers=_HDR)
    r = urllib.request.urlopen(req, timeout=30)
    data = r.read()
    if r.headers.get("Content-Encoding") == "gzip":
        data = gzip.decompress(data)
    return data

def _cik_for(ticker, cik=None):
    if cik:
        return str(cik).zfill(10)
    t = ticker.upper().replace(".", "-")
    if not _CIK_CACHE:
        j = json.loads(_get("https://www.sec.gov/files/company_tickers.json"))
        for v in j.values():
            _CIK_CACHE[v["ticker"].upper()] = str(v["cik_str"]).zfill(10)
    # GOOG/GOOGL etc: try exact, then the base
    return _CIK_CACHE.get(t) or _CIK_CACHE.get(t.split("-")[0])

def _concept(cik, tag):
    url = f"https://data.sec.gov/api/xbrl/companyconcept/CIK{cik}/us-gaap/{tag}.json"
    try:
        time.sleep(_RL_SLEEP)
        return json.loads(_get(url))
    except Exception:
        return None

def _span_days(start, end):
    try:
        s = _dt.date.fromisoformat(start); e = _dt.date.fromisoformat(end)
        return (e - s).days
    except Exception:
        return None

def _rows(cik, keys):
    """
    Union 10-Q/10-K rows across ALL fallback tags (filers switch XBRL concepts over time,
    e.g. RevenueFromContractWithCustomerExcludingAssessedTax -> Revenues). De-dupe by
    (start,end), preferring the most-recently-filed value. Returns (rows, tag_label).
    """
    merged = {}          # (start,end) -> (start,end,val,form,filed)
    used = []
    for tag in keys:
        j = _concept(cik, tag)
        if not j:
            continue
        got = False
        for unit_rows in j.get("units", {}).values():
            for r in unit_rows:
                if r.get("end") and r.get("form") in ("10-Q", "10-K"):
                    k = (r.get("start"), r["end"])
                    cand = (r.get("start"), r["end"], r.get("val"), r["form"], r.get("filed",""))
                    p = merged.get(k)
                    if p is None or cand[4] > p[4]:
                        merged[k] = cand
                    got = True
        if got:
            used.append(tag)
    if not merged:
        return [], None
    return list(merged.values()), "+".join(used)

def _quarters_from_ytd(rows):
    """Derive discrete quarters from YTD/annual duration rows (handles Q4 = FY - 9M)."""
    ytd = {}
    for start, end, val, form, filed in rows:
        if not start or val is None:
            continue
        d = _span_days(start, end)
        if d is None or d < 80:
            continue
        p = ytd.get((start, end))
        if p is None or filed > p[1]:
            ytd[(start, end)] = (val, filed, form)
    by_start = {}
    for (start, end), (val, filed, form) in ytd.items():
        by_start.setdefault(start, []).append((end, val, form))
    out = {}
    for start, lst in by_start.items():
        lst.sort()
        prev = 0.0
        for end, val, form in lst:
            out[end] = (val - prev, form, "")
            prev = val
    return out

def _series(cik, keys, mode="q_best"):
    """
    mode="q_native": keep discrete-quarter rows (span 80-100d). For revenue/op-income.
    mode="q_ytd"   : cash-flow YTD-only -> derive quarter = YTD(end) - YTD(prev end same FY).
    mode="instant" : balance-sheet -> latest value per end date.
    Returns {end: (val, form, filed)} , tag
    """
    rows, tag = _rows(cik, keys)
    if not rows:
        return {}, None
    out = {}
    if mode == "q_best":
        native = {}
        for start, end, val, form, filed in rows:
            if not start: continue
            d = _span_days(start, end)
            if d is not None and 80 <= d <= 100:
                p = native.get(end)
                if p is None or filed > p[2]:
                    native[end] = (val, form, filed)
        derived = _quarters_from_ytd(rows)
        merged = dict(derived); merged.update(native)   # prefer native where present
        return merged, tag
    if mode == "instant":
        for start, end, val, form, filed in rows:
            p = out.get(end)
            if p is None or filed > p[2]:
                out[end] = (val, form, filed)
        return out, tag

    if mode == "q_native":
        for start, end, val, form, filed in rows:
            if not start:
                continue
            d = _span_days(start, end)
            if d is None or not (80 <= d <= 100):
                continue
            p = out.get(end)
            if p is None or filed > p[2]:
                out[end] = (val, form, filed)
        return out, tag

    # mode == "q_ytd": build best YTD value per (fiscal-year-start, end), then difference
    ytd = {}   # end -> (start, val, form, filed)
    for start, end, val, form, filed in rows:
        if not start or val is None:
            continue
        d = _span_days(start, end)
        if d is None or d < 80:          # keep 90/180/270/365 spans; drop odd stubs
            continue
        p = ytd.get(end)
        if p is None or filed > p[3]:
            ytd[end] = (start, val, form, filed)
    # index YTD rows by fiscal-year start; a quarter = this YTD minus the prior YTD in same FY
    by_start = {}
    for end,(start,val,form,filed) in ytd.items():
        by_start.setdefault(start, []).append((end,val,form,filed))
    for start, lst in by_start.items():
        lst.sort()                        # by end date ascending
        prev = 0.0
        for end,val,form,filed in lst:
            q = val - prev                # discrete quarter
            out[end] = (q, form, filed)
            prev = val
    return out, tag

def _fmt_b(v):
    return "n/a" if v is None else f"${v/1e9:,.1f}B"

def fundamentals_block(ticker, cik=None, quarters=6):
    """Return a printable =SEC FUNDAMENTALS= block, or a clear failure line."""
    L = ["=== SEC FUNDAMENTALS (XBRL, primary) — %s ===" % ticker.upper()]
    try:
        c = _cik_for(ticker, cik)
        if not c:
            return "\n".join(L + ["  [warn] CIK not found for %s — check ticker/CIK map." % ticker])
        L.append("  CIK: %s   source: data.sec.gov/api/xbrl (no key; [fact])" % c)
        rev, rtag = _series(c, _TAGS["revenue"], mode="q_best")
        ocf, _    = _series(c, _TAGS["ocf"], mode="q_best")
        capex, _  = _series(c, _TAGS["capex"], mode="q_best")
        cash, _   = _series(c, _TAGS["cash"], mode="instant")
        debt, _   = _series(c, _TAGS["lt_debt"], mode="instant")
        opinc, _  = _series(c, _TAGS["op_inc"], mode="q_best")
        eps, _    = _series(c, _TAGS["eps_dil"], mode="q_best")
        if not rev:
            return "\n".join(L + ["  [warn] no revenue concept resolved — inspect companyfacts JSON."])
        ends = sorted(rev.keys())[-quarters:]
        L.append("  Revenue tag: %s" % rtag)
        L.append("  %-12s %-10s %-10s %-10s %-10s %-9s" %
                 ("period_end","revenue","op_inc","OCF","capex","FCF"))
        for e in ends:
            rv = rev.get(e,(None,))[0]
            oi = opinc.get(e,(None,))[0]
            oc = ocf.get(e,(None,))[0]
            cx = capex.get(e,(None,))[0]
            fcf = (oc - cx) if (oc is not None and cx is not None) else None
            L.append("  %-12s %-10s %-10s %-10s %-10s %-9s" %
                     (e,_fmt_b(rv),_fmt_b(oi),_fmt_b(oc),_fmt_b(cx),_fmt_b(fcf)))
        # latest balance-sheet snapshot
        def _latest(d):
            if not d: return None,None
            k=sorted(d)[-1]; return k,d[k][0]
        ck,cv=_latest(cash); dk,dv=_latest(debt); ek,ev=_latest(eps)
        L.append("  ---")
        L.append("  Cash & equiv (%s): %s   LT debt (%s): %s   Dil-EPS (%s): %s" %
                 (ck,_fmt_b(cv),dk,_fmt_b(dv),ek, "n/a" if ev is None else f"${ev:.2f}"))
        L.append("  [note] Discrete-quarter values (80-100d spans); YTD/annual rows excluded. FCF = OCF - capex.")
        return "\n".join(L)
    except Exception as ex:
        return "\n".join(L + ["  [warn] fundamentals pull failed: %s: %s" %
                              (type(ex).__name__, str(ex)[:160]),
                              "  [note] check egress to data.sec.gov + User-Agent; sandbox blocks it, local box won't."])

if __name__ == "__main__":
    import sys
    print(fundamentals_block(sys.argv[1] if len(sys.argv)>1 else "GOOG"))
