"""
marquee_13f_block.py  —  drop-in monitor module for marquee-holder 13F deltas.
Ticker-agnostic on the HOLDER side (Berkshire, etc.); you pass the ticker's
CUSIP (or let it resolve from the ticker via the SEC map).

Answers "which big holders added/trimmed THIS name last quarter" from primary
13F-HR filings — the panel that's been a DEFECT_LEDGER research note all series.

Two modes:
  (A) LIGHT (recommended first): if your monitor already ingests a 13F snapshot
      (it does — GPT read it), just call format_snapshot() on that dict to PRINT
      it with QoQ deltas. No new network calls. Closes the Berkshire-precision gap today.
  (B) FULL: pull named managers' latest two 13F-HR filings from EDGAR and diff
      this ticker's holding. Robust, but one network call per manager.

Requires internet to data.sec.gov (local box only; not the claude sandbox).
"""
import json, time, urllib.request, gzip, datetime as _dt
import xml.etree.ElementTree as ET

_UA  = "ML Quant Fund research atom.v.nguyen@gmail.com"
_HDR = {"User-Agent": _UA, "Accept-Encoding": "gzip, deflate"}
_RL  = 0.15

# marquee holders you care about -> their SEC CIK (extend freely)
MARQUEE = {
    "Berkshire Hathaway": "0001067983",
    "JPMorgan Chase":     "0000019617",
    "FMR (Fidelity)":     "0000315066",
    "Invesco":            "0000914208",
    "BlackRock":          "0001364742",
    "Vanguard":           "0000102909",
    "State Street":       "0000093751",
}

def _get(url):
    r = urllib.request.urlopen(urllib.request.Request(url, headers=_HDR), timeout=30)
    d = r.read()
    if r.headers.get("Content-Encoding") == "gzip":
        d = gzip.decompress(d)
    return d

# ---------- MODE A: format a snapshot your monitor already has ----------
def format_snapshot(ticker, snapshot):
    """
    snapshot: list of dicts like
      [{"holder":"Berkshire Hathaway","shares":106_000_000,"value":37_800_000_000,
        "prev_shares":57_900_000_000... }, ...]  (whatever your monitor stores)
    Prints a =13F MARQUEE HOLDERS= block with QoQ share deltas.
    """
    L = ["=== 13F MARQUEE HOLDERS (delta) — %s ===" % ticker.upper(),
         "  source: monitor 13F snapshot (lagged to quarter-end; ownership context [fact])"]
    if not snapshot:
        return "\n".join(L + ["  [warn] no 13F snapshot in monitor for %s." % ticker])
    L.append("  %-22s %14s %12s %10s" % ("holder","shares","value","QoQ chg"))
    for h in snapshot:
        sh   = h.get("shares")
        val  = h.get("value")
        prev = h.get("prev_shares")
        dchg = ("%+.0f%%" % (100*(sh-prev)/prev)) if (sh and prev) else "n/a"
        L.append("  %-22s %14s %12s %10s" % (
            h.get("holder","?")[:22],
            "n/a" if sh is None else f"{sh/1e6:,.1f}M",
            "n/a" if val is None else f"${val/1e9:,.1f}B",
            dchg))
    L.append("  [note] 13F is lagged (quarter-end); use as ownership context, not timing.")
    return "\n".join(L)

# ---------- MODE B: pull latest 13F-HR per manager and diff this ticker ----------
def _latest_two_13f(cik):
    j = json.loads(_get(f"https://data.sec.gov/submissions/CIK{cik}.json"))
    rec = j["filings"]["recent"]
    accs = [(rec["accessionNumber"][i], rec["filingDate"][i])
            for i,f in enumerate(rec["form"]) if f == "13F-HR"]
    return accs[:2]   # most recent two

def _holding(cik, accession, cusip):
    acc = accession.replace("-","")
    base = f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{acc}"
    idx = json.loads(_get(base + "/index.json"))
    info = next((it["name"] for it in idx["directory"]["item"]
                 if it["name"].lower().endswith(".xml") and "info" in it["name"].lower()), None)
    if not info:
        return None
    time.sleep(_RL)
    root = ET.fromstring(_get(base + "/" + info))
    tot = 0
    for e in root.iter():
        if e.tag.endswith("infoTable"):
            cu = "".join(x.text or "" for x in e.iter() if x.tag.endswith("cusip"))
            if cusip and cusip.upper() in cu.upper():
                for x in e.iter():
                    if x.tag.endswith("sshPrnamt"):
                        try: tot += int(x.text)
                        except: pass
    return tot

def full_13f_block(ticker, cusip, holders=None):
    L = ["=== 13F MARQUEE HOLDERS (delta, from EDGAR) — %s ===" % ticker.upper(),
         "  source: 13F-HR primary filings, diffed QoQ ([fact], lagged)"]
    if not cusip:
        return "\n".join(L + ["  [warn] pass the ticker CUSIP to diff holdings."])
    hold = holders or MARQUEE
    L.append("  %-22s %12s %12s %9s" % ("holder","prev qtr","latest","QoQ"))
    for name, cik in hold.items():
        try:
            accs = _latest_two_13f(cik); time.sleep(_RL)
            if len(accs) < 1:
                continue
            latest = _holding(cik, accs[0][0], cusip)
            prev   = _holding(cik, accs[1][0], cusip) if len(accs) > 1 else None
            if not latest and not prev:
                continue
            d = ("%+.0f%%" % (100*(latest-prev)/prev)) if (latest and prev) else "NEW" if latest else "exit"
            L.append("  %-22s %12s %12s %9s" % (
                name[:22],
                "n/a" if prev is None else f"{prev/1e6:,.1f}M",
                "n/a" if latest is None else f"{latest/1e6:,.1f}M", d))
        except Exception as ex:
            L.append("  %-22s  [warn] %s" % (name[:22], type(ex).__name__))
    L.append("  [note] holdings summed by CUSIP; share classes may split GOOG/GOOGL.")
    return "\n".join(L)

if __name__ == "__main__":
    # demo of MODE A with a hand-built snapshot (no network needed):
    demo = [{"holder":"Berkshire Hathaway","shares":106_000_000,"value":37_800_000_000,"prev_shares":57_900_000},
            {"holder":"Invesco","shares":40_000_000,"value":14_000_000_000,"prev_shares":15_300_000}]
    print(format_snapshot("GOOG", demo))
