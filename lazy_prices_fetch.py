#!/usr/bin/env python3
"""
================================================================================
ML QUANT FUND — LAZY PRICES: EDGAR FETCHER (brick #3, step a)
================================================================================
"Lazy Prices" (Cohen-Malloy-Nguyen 2020): firms that CHANGE the wording of their
10-K/10-Q vs the prior comparable filing UNDERPERFORM. Signal = YoY text similarity;
LOW similarity (big change) -> LOW forward return. A different information axis
(disclosure language) -> plausibly uncorrelated with price/positioning bricks.

This builds filing_similarity.db that validate_lazy_prices.py consumes.

PIPELINE (all FREE, no API key; SEC requires a User-Agent with your email):
  1. ticker -> CIK: https://www.sec.gov/files/company_tickers.json (cached once).
  2. per CIK filings: https://data.sec.gov/submissions/CIK##########.json
     -> form, accessionNumber, filingDate (PUBLIC date = PIT), reportDate, primaryDocument.
     Filtered to 10-K / 10-Q. (Handles the 'files' overflow for >1000-filing filers.)
  3. per filing primary doc:
     https://www.sec.gov/Archives/edgar/data/<cik_int>/<accession_nodashes>/<primaryDoc>
     -> strip HTML -> text (cached to disk).
  4. YoY cosine similarity:
       10-K vs the firm's PRIOR 10-K
       10-Q vs the firm's prior-YEAR SAME-QUARTER 10-Q (reportDate ~ -365d; fallback 3-back)
     cosine of term-frequency vectors over alphabetic tokens (numbers dropped: they
     change every period and would swamp the language signal).
  5. store (ticker, filing_date, form, similarity, prior_date).

PIT: filing_date = the SEC filingDate (when the document became public). Forward
returns in the validator start strictly after that date. No look-ahead.

RATE LIMITS (SEC fair-access): User-Agent with your email REQUIRED; <=10 req/sec;
this script sleeps --sleep (default 0.15s) between requests and backs off on 429/403.
~400 tickers x ~40-50 filings ~ 8-20k docs -> a few hours. CACHED + RESUMABLE: the
ticker map, submissions JSON, and stripped filing text are cached to --cache-dir, and
already-scored (ticker, filing, form) rows are skipped, so you can stop/restart.

SMOKE TEST FIRST (verify against real EDGAR on a handful before the full multi-hour run):
  python lazy_prices_fetch.py --root . --email "you@example.com" --max-tickers 5
Then the full pull:
  python lazy_prices_fetch.py --root . --email "you@example.com"
Then validate:
  python validate_lazy_prices.py --root . --hold 40

IMPORTANT: pass a REAL email in --email (SEC blocks generic/no User-Agent). Not optional.
================================================================================
"""
import argparse, os, sys, json, re, time, sqlite3, math, datetime, urllib.request, urllib.error
from collections import defaultdict, Counter
from html.parser import HTMLParser

UA_TMPL="ML-Quant-Research/1.0 ({email})"
TICKERS_URL="https://www.sec.gov/files/company_tickers.json"
SUBM_URL="https://data.sec.gov/submissions/CIK{cik10}.json"
SUBM_OVERFLOW="https://data.sec.gov/submissions/{name}"
ARCHIVE_URL="https://www.sec.gov/Archives/edgar/data/{cikint}/{accn}/{doc}"
LINE="="*78

# ---------------- network (monkeypatchable for offline audit) ----------------
def _http_get(url, email, want_json=False, sleep=0.15, timeout=60, tries=4):
    headers={"User-Agent":UA_TMPL.format(email=email),
             "Accept-Encoding":"gzip, deflate","Host":None}
    # urllib sets Host automatically; remove our None
    headers={k:v for k,v in headers.items() if v is not None}
    last=None
    for attempt in range(tries):
        try:
            req=urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=timeout) as r:
                raw=r.read()
                enc=r.headers.get("Content-Encoding","")
                if "gzip" in enc:
                    import gzip; raw=gzip.decompress(raw)
                time.sleep(sleep)
                txt=raw.decode("utf-8","replace")
                return json.loads(txt) if want_json else txt
        except urllib.error.HTTPError as e:
            last=e
            if e.code in (429,403,503):
                time.sleep(1.5*(attempt+1)); continue
            if e.code==404: return None
            time.sleep(0.5*(attempt+1))
        except Exception as e:
            last=e; time.sleep(0.5*(attempt+1))
    sys.stderr.write("  [fetch fail] %s (%s)\n"%(url, repr(last)[:120]))
    return None

# ---------------- HTML -> text ----------------
class _Strip(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True); self.buf=[]; self.skip=0
    def handle_starttag(self,t,a):
        if t in ("script","style"): self.skip+=1
    def handle_endtag(self,t):
        if t in ("script","style") and self.skip>0: self.skip-=1
    def handle_data(self,d):
        if self.skip==0: self.buf.append(d)
def strip_html(html):
    if "<" not in html: return html
    try:
        p=_Strip(); p.feed(html); return " ".join(p.buf)
    except Exception:
        return re.sub(r"<[^>]+>"," ",html)

_word=re.compile(r"[a-z]{2,}")
def tokenize(text):
    return Counter(_word.findall(text.lower()))

def cosine(a,b):
    if not a or not b: return None
    # dot over shared keys
    if len(a)>len(b): a,b=b,a
    dot=sum(cnt*b.get(w,0) for w,cnt in a.items())
    if dot==0: return 0.0
    na=math.sqrt(sum(c*c for c in a.values())); nb=math.sqrt(sum(c*c for c in b.values()))
    if na<=0 or nb<=0: return None
    return max(0.0, min(1.0, dot/(na*nb)))  # clamp FP overshoot to [0,1]

# ---------------- EDGAR helpers ----------------
def load_cik_map(email, cachedir, sleep):
    cf=os.path.join(cachedir,"company_tickers.json")
    data=None
    if os.path.isfile(cf):
        try: data=json.load(open(cf))
        except Exception: data=None
    if data is None:
        data=_http_get(TICKERS_URL, email, want_json=True, sleep=sleep)
        if data is None: return {}
        try: json.dump(data, open(cf,"w"))
        except Exception: pass
    out={}
    # file is {idx: {cik_str, ticker, title}}
    itr=data.values() if isinstance(data,dict) else data
    for row in itr:
        try:
            tk=str(row["ticker"]).upper(); cik=int(row["cik_str"])
            out[tk]="%010d"%cik
        except Exception: continue
    return out

def get_submissions(cik10, email, cachedir, sleep):
    cf=os.path.join(cachedir,"subm_%s.json"%cik10)
    data=None
    if os.path.isfile(cf):
        try: data=json.load(open(cf))
        except Exception: data=None
    if data is None:
        data=_http_get(SUBM_URL.format(cik10=cik10), email, want_json=True, sleep=sleep)
        if data is None: return []
        try: json.dump(data, open(cf,"w"))
        except Exception: pass
    filings=[]
    def add_block(b):
        forms=b.get("form",[]); accn=b.get("accessionNumber",[]); fdate=b.get("filingDate",[])
        rdate=b.get("reportDate",[]); pdoc=b.get("primaryDocument",[])
        for i in range(len(forms)):
            f=forms[i]
            if f not in ("10-K","10-Q"): continue
            filings.append(dict(form=f, accession=accn[i] if i<len(accn) else "",
                                filingDate=fdate[i] if i<len(fdate) else "",
                                reportDate=rdate[i] if i<len(rdate) else "",
                                primaryDocument=pdoc[i] if i<len(pdoc) else ""))
    fblock=data.get("filings",{})
    add_block(fblock.get("recent",{}))
    for extra in fblock.get("files",[]):
        nm=extra.get("name")
        if not nm: continue
        ocf=os.path.join(cachedir,"subm_%s"%nm)
        od=None
        if os.path.isfile(ocf):
            try: od=json.load(open(ocf))
            except Exception: od=None
        if od is None:
            od=_http_get(SUBM_OVERFLOW.format(name=nm), email, want_json=True, sleep=sleep)
            if od is not None:
                try: json.dump(od, open(ocf,"w"))
                except Exception: pass
        if od is not None: add_block(od)
    return filings

def get_filing_text(cik10, accession, primaryDocument, email, cachedir, sleep):
    if not accession or not primaryDocument: return None
    accn=accession.replace("-","")
    tdir=os.path.join(cachedir,"text"); os.makedirs(tdir,exist_ok=True)
    tf=os.path.join(tdir,"%s.txt"%accn)
    if os.path.isfile(tf):
        try: return open(tf,encoding="utf-8").read()
        except Exception: pass
    cikint=str(int(cik10))
    url=ARCHIVE_URL.format(cikint=cikint, accn=accn, doc=primaryDocument)
    html=_http_get(url, email, want_json=False, sleep=sleep)
    if html is None: return None
    text=strip_html(html)
    try: open(tf,"w",encoding="utf-8").write(text)
    except Exception: pass
    return text

def nd(s):
    try: return datetime.date.fromisoformat(str(s)[:10])
    except Exception: return None

def pair_filings(filings):
    """filings: list of dicts with form, filingDate, reportDate, and a 'tok' Counter.
    Returns list of (filing_date, form, similarity, prior_date)."""
    out=[]
    for form in ("10-K","10-Q"):
        fs=[f for f in filings if f["form"]==form and f.get("tok") and nd(f["filingDate"])]
        fs.sort(key=lambda x: nd(x["filingDate"]))
        for i,cur in enumerate(fs):
            prior=None
            cur_rd=nd(cur.get("reportDate"))
            if form=="10-K":
                if i>=1: prior=fs[i-1]
            else:
                # prior-year same quarter: reportDate closest to -365d within [300,430]
                if cur_rd:
                    best=None; bestgap=None
                    for j in range(i):
                        rd=nd(fs[j].get("reportDate"))
                        if not rd: continue
                        gap=(cur_rd-rd).days
                        if 300<=gap<=430:
                            if bestgap is None or abs(gap-365)<abs(bestgap-365):
                                best=fs[j]; bestgap=gap
                    prior=best
                if prior is None and i>=3:   # fallback: 3 10-Qs back ~ 1 year
                    prior=fs[i-3]
            if prior is None or not prior.get("tok"): continue
            sim=cosine(cur["tok"], prior["tok"])
            if sim is None: continue
            out.append((cur["filingDate"][:10], form, float(sim), prior["filingDate"][:10]))
    return out

def ensure_db(path):
    c=sqlite3.connect(path,timeout=30)
    c.execute("""CREATE TABLE IF NOT EXISTS filing_similarity(
        ticker TEXT, filing_date TEXT, form TEXT, similarity REAL, prior_date TEXT,
        PRIMARY KEY(ticker, filing_date, form))""")
    c.commit(); return c

def load_universe(root):
    for name,table,col in (("short_interest.db","short_interest","ticker"),
                           ("prices.db","daily_prices","ticker"),
                           ("earnings.db","earnings_surprises","ticker")):
        p=os.path.join(root,name)
        if os.path.isfile(p):
            try:
                c=sqlite3.connect("file:"+os.path.abspath(p)+"?mode=ro",uri=True)
                u=set(r[0].upper() for r in c.execute('SELECT DISTINCT %s FROM %s'%(col,table)) if r[0])
                c.close()
                if u: return sorted(u), name
            except Exception: continue
    return [], None

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--root",default=".")
    ap.add_argument("--email",default=None,help="REQUIRED: your email for the SEC User-Agent")
    ap.add_argument("--out-db",default=None)
    ap.add_argument("--cache-dir",default=None)
    ap.add_argument("--sleep",type=float,default=0.15)
    ap.add_argument("--max-tickers",type=int,default=None,help="smoke-test on first N tickers")
    ap.add_argument("--since-year",type=int,default=2014,help="skip filings before this year")
    a=ap.parse_args(); a.root=os.path.expanduser(a.root)
    out_db=a.out_db or os.path.join(a.root,"filing_similarity.db")
    cachedir=a.cache_dir or os.path.join(a.root,"cache","edgar"); os.makedirs(cachedir,exist_ok=True)

    print("\n"+LINE+"\nLAZY PRICES — EDGAR FETCHER\n"+LINE)
    if not a.email or "@" not in a.email:
        print("  [STOP] --email is REQUIRED (SEC blocks requests without a real User-Agent email).")
        print("         e.g.  python lazy_prices_fetch.py --root . --email you@example.com --max-tickers 5")
        return
    uni,src=load_universe(a.root)
    if not uni: print("  [STOP] no universe found (short_interest.db/prices.db/earnings.db)."); return
    if a.max_tickers: uni=uni[:a.max_tickers]
    print("  universe: %d tickers (from %s)%s"%(len(uni),src," [SMOKE TEST]" if a.max_tickers else ""))
    print("  cache: %s | sleep %.2fs/req | since %d"%(cachedir,a.sleep,a.since_year))

    cikmap=load_cik_map(a.email, cachedir, a.sleep)
    if not cikmap: print("  [STOP] could not load ticker->CIK map from SEC."); return
    print("  ticker->CIK map: %d names\n"%len(cikmap))

    db=ensure_db(out_db)
    done=set((r[0],r[1],r[2]) for r in db.execute("SELECT ticker,filing_date,form FROM filing_similarity"))
    n_tk=0; n_sim=0; n_miss=0
    for tk in uni:
        cik10=cikmap.get(tk)
        if not cik10:
            n_miss+=1; continue
        subs=get_submissions(cik10, a.email, cachedir, a.sleep)
        subs=[s for s in subs if (nd(s["filingDate"]) and nd(s["filingDate"]).year>=a.since_year)]
        if not subs: 
            print("  %-6s CIK %s: no 10-K/10-Q since %d"%(tk,cik10,a.since_year)); continue
        # fetch + tokenize each filing's text (cached)
        for s in subs:
            txt=get_filing_text(cik10, s["accession"], s["primaryDocument"], a.email, cachedir, a.sleep)
            s["tok"]=tokenize(txt) if txt else None
        sims=pair_filings(subs)
        wrote=0
        for fdate,form,sim,pdate in sims:
            if (tk,fdate,form) in done: continue
            db.execute("""INSERT INTO filing_similarity(ticker,filing_date,form,similarity,prior_date)
                          VALUES(?,?,?,?,?) ON CONFLICT(ticker,filing_date,form) DO UPDATE SET
                          similarity=excluded.similarity, prior_date=excluded.prior_date""",
                       (tk,fdate,form,sim,pdate))
            wrote+=1; n_sim+=1
        db.commit(); n_tk+=1
        print("  %-6s CIK %s: %d filings -> %d similarities (%d new)"%(tk,cik10,len(subs),len(sims),wrote))

    print("\n"+LINE)
    print("  tickers processed: %d | similarities stored: %d | tickers w/o CIK: %d"%(n_tk,n_sim,n_miss))
    tot=db.execute("SELECT COUNT(*), COUNT(DISTINCT ticker), MIN(filing_date), MAX(filing_date) FROM filing_similarity").fetchone()
    print("  filing_similarity.db now: %d rows, %d tickers, %s..%s"%(tot[0],tot[1],tot[2],tot[3]))
    db.close()
    print("\n  Next: python validate_lazy_prices.py --root . --hold 40")
    print("  (filings scatter across dates; the validator buckets to monthly cross-sections.)")

if __name__=="__main__":
    try: main()
    except KeyboardInterrupt: print("\ninterrupted (progress cached -- safe to re-run).")
    except Exception:
        import traceback; print("\n[UNEXPECTED ERROR] paste back:"); traceback.print_exc()
