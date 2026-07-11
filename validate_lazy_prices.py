#!/usr/bin/env python3
"""
================================================================================
ML QUANT FUND — LAZY PRICES VALIDATOR (skeleton; step (b) of brick #3 hunt)
================================================================================
"Lazy Prices" (Cohen-Malloy-Nguyen 2020): when a company CHANGES the wording of its
10-K/10-Q vs the prior comparable filing, the stock tends to UNDERPERFORM -- firms
quietly bury bad news in language changes that few investors read closely. The signal
is YEAR-OVER-YEAR TEXT SIMILARITY: LOW similarity (big change) -> LOW forward return.
So direction is +1 on SIMILARITY (high similarity = high return = "no news is good
news"), equivalently the change/dissimilarity predicts negative returns.

This is the VALIDATOR (step b). It assumes the similarity data already exists and runs
the SAME audited machinery that confirmed short interest: per-date cross-sectional IC,
Newey-West t (overlap-corrected), %-years sign, null control. So if Lazy Prices is real
on your universe, it gets confirmed to the identical standard as brick #2.

The FETCHER (step a, built next) populates the similarity DB from free SEC EDGAR.

>>> EXPECTED SCHEMA (filing_similarity.db, table filing_similarity):
      ticker TEXT, filing_date TEXT, form TEXT, similarity REAL
    where similarity in [0,1] is cosine similarity of this filing's text vs the prior
    SAME-form filing (10-K vs prior 10-K, 10-Q vs prior 10-Q -- same fiscal quarter to
    avoid seasonal language). filing_date = EDGAR acceptance date (when it became public
    -> PIT-correct, the date the market could first see it).

>>> THE EDGAR DATA-PULL PLAN (for the step-(a) fetcher -- all FREE, no API key):
    1. Ticker->CIK: download https://www.sec.gov/files/company_tickers.json once
       (maps ticker -> 10-digit zero-padded CIK). Cache locally.
    2. Per CIK, get filing history: https://data.sec.gov/submissions/CIK##########.json
       -> lists every filing with form type, accession number, acceptance date (PIT).
       Filter to form in ('10-K','10-Q'). One request per ticker (~400 total).
    3. Per filing, fetch the primary document text:
       https://www.sec.gov/Archives/edgar/data/<cik>/<accession_nodashes>/<primary_doc>
       Strip HTML -> plain text. (Primary doc filename is in the submissions JSON.)
    4. Compute YoY similarity: for each 10-K, cosine-similarity its text vs the PRIOR
       10-K (same firm); for each 10-Q, vs the prior-YEAR same-quarter 10-Q. Use TF-IDF
       cosine or simple bag-of-words cosine (the paper uses cosine + Jaccard; cosine is
       the workhorse and is robust).
    5. Store (ticker, filing_date=acceptance_date, form, similarity).
    RATE LIMITS: SEC requires a User-Agent header (your email) and <=10 req/sec; add a
    100ms sleep between requests. ~400 tickers x ~20 filings each ~ 8000 docs -> a few
    hours with polite throttling. Full filing text is large; cache aggressively, never
    re-fetch. Survivorship: EDGAR has delisted filers too, but prices.db is survivor-
    tilted, so coverage will be limited to your existing universe (flagged at runtime).

RULE 1: per-date IC + Newey-West (machinery copied verbatim from validate_si_v2);
filing_date is acceptance date (PIT -- no look-ahead); forward returns strictly after;
null control; READ-ONLY; no network in the validation step (data pre-loaded, same
discipline as FINRA/short interest).

USAGE (once filing_similarity.db exists):
  python validate_lazy_prices.py --root . --hold 40 --direction +1
  python validate_lazy_prices.py --root . --form 10-K --hold 60
  python validate_lazy_prices.py --status      # checks for data, prints the pull plan
================================================================================
"""
import argparse, os, sqlite3, math, datetime
from collections import defaultdict
import numpy as np

def ro(p): return sqlite3.connect("file:"+os.path.abspath(p)+"?mode=ro&immutable=1",uri=True,timeout=30)
def Q(c,s,p=()): return c.execute(s,p).fetchall()
def cols_of(c,t): return [r[1] for r in Q(c,'PRAGMA table_info("'+t+'")')]
def nd(s):
    if s is None: return None
    try: return datetime.date.fromisoformat(str(s)[:10])
    except Exception: return None
def spearman(x,y):
    n=len(x)
    if n<5: return None
    rx=np.argsort(np.argsort(x)).astype(float); ry=np.argsort(np.argsort(y)).astype(float)
    if rx.std()==0 or ry.std()==0: return None
    return float(np.corrcoef(rx,ry)[0,1])
def nw_se_mean(x,lag):
    x=np.asarray(x,float); n=len(x)
    if n<2: return None
    e=x-x.mean(); g0=float(e@e)/n; s=g0
    for k in range(1,min(lag,n-1)+1):
        gk=float(e[k:]@e[:-k])/n; w=1.0-k/(lag+1.0); s+=2.0*w*gk
    return math.sqrt(s/n) if s>0 else None
LINE="="*78

PULL_PLAN = """
  THE EDGAR PULL PLAN (free, no API key -- this is what the step-(a) fetcher will do):
   1. Ticker->CIK: https://www.sec.gov/files/company_tickers.json (download once).
   2. Per CIK filing list: https://data.sec.gov/submissions/CIK<10-digit>.json
      -> form types + accession numbers + ACCEPTANCE DATES (PIT).
   3. Per filing primary doc: https://www.sec.gov/Archives/edgar/data/<cik>/<accn>/<doc>
      -> strip HTML to text.
   4. YoY cosine similarity: 10-K vs prior 10-K; 10-Q vs prior-year same-quarter 10-Q.
   5. Store (ticker, filing_date=acceptance_date, form, similarity) in filing_similarity.db.
   RATE LIMIT: User-Agent header required, <=10 req/sec, 100ms sleep between calls.
   ~400 tickers -> a few hours polite. Cache; never re-fetch.
"""

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--root",default=".")
    ap.add_argument("--db",default=None)
    ap.add_argument("--prices-db",default=None)
    ap.add_argument("--form",default=None,help="restrict to 10-K or 10-Q (default: both)")
    ap.add_argument("--direction",type=int,default=1,
                    help="+1 if HIGH similarity predicts HIGH return (Lazy Prices: low change=good)")
    ap.add_argument("--hold",type=int,default=40)
    ap.add_argument("--min-names",type=int,default=20)
    ap.add_argument("--status",action="store_true")
    a=ap.parse_args(); a.root=os.path.expanduser(a.root)
    sig_db=a.db or os.path.join(a.root,"filing_similarity.db")
    prices_db=a.prices_db or os.path.join(a.root,"prices.db")

    print("\n"+LINE+"\nLAZY PRICES VALIDATOR (10-K/10-Q text-change signal)\n"+LINE)

    if not os.path.isfile(sig_db):
        print("  filing_similarity.db NOT FOUND at %s"%sig_db)
        print("  Expected state for step (b): the validator exists; the data comes from the")
        print("  step-(a) EDGAR fetcher (built next).")
        print(PULL_PLAN)
        print("  Once filing_similarity.db exists (table filing_similarity:")
        print("  ticker, filing_date, form, similarity), re-run WITHOUT --status to validate.")
        return
    if a.status:
        c=ro(sig_db)
        try:
            cols=cols_of(c,"filing_similarity")
            n=Q(c,"SELECT COUNT(*) FROM filing_similarity")[0][0]
            dr=Q(c,"SELECT MIN(filing_date),MAX(filing_date) FROM filing_similarity")[0]
            ndt=Q(c,"SELECT COUNT(DISTINCT filing_date) FROM filing_similarity")[0][0]
        finally: c.close()
        print("  filing_similarity.db present: %d rows, %d distinct filing dates, %s to %s"%(n,ndt,dr[0],dr[1]))
        print("  columns: %s"%", ".join(cols))
        if ndt<20: print("  >> NOTE: only %d distinct dates -- likely too thin to validate yet (monitor)."%ndt)
        print(PULL_PLAN)
        return

    if not os.path.isfile(prices_db): print("  [STOP] prices.db not found"); return

    # prices
    cp=ro(prices_db)
    try: prows=Q(cp,"SELECT ticker,date,adj_close FROM daily_prices WHERE adj_close IS NOT NULL")
    finally: cp.close()
    px=defaultdict(list)
    for tk,d,p in prows:
        do=nd(d)
        if do is None: continue
        try: pf=float(p)
        except Exception: continue
        if pf>0: px[tk].append((do,pf))
    for tk in px: px[tk].sort()
    pos_of={tk:{d:i for i,(d,_) in enumerate(lst)} for tk,lst in px.items()}
    def fwd(tk,d):
        lst=px.get(tk); idx=pos_of.get(tk)
        if not lst or not idx: return None
        i=None
        for off in range(0,6):
            cc=d+datetime.timedelta(days=off)
            if cc in idx: i=idx[cc]; break
        if i is None: return None
        x=i+a.hold
        if x>=len(lst): return None
        p0=lst[i][1]; return (lst[x][1]/p0-1.0) if p0>0 else None

    # similarity signal
    c=ro(sig_db)
    try:
        if a.form:
            rows=Q(c,'SELECT ticker,filing_date,similarity FROM filing_similarity WHERE form=? AND similarity IS NOT NULL',(a.form,))
        else:
            rows=Q(c,'SELECT ticker,filing_date,similarity FROM filing_similarity WHERE similarity IS NOT NULL')
    finally: c.close()
    by_date=defaultdict(list)
    for tk,d,v in rows:
        do=nd(d)
        if do is None or v is None: continue
        try: fv=float(v)
        except Exception: continue
        by_date[do].append((tk.upper(),fv))

    # NOTE: filings don't cluster on common dates the way bi-monthly SI does -- each firm
    # files on its own schedule. So a strict "per filing_date cross-section" will be sparse.
    # The honest approach (matches the literature): bucket filings into the period they
    # were filed (e.g. monthly), and within each bucket rank firms by similarity vs that
    # bucket's forward returns. This validator does per-filing-date IC AND will warn if the
    # cross-sections are too thin -- in which case the fetcher should aggregate to monthly
    # filing buckets (handled in step a).
    def compute(shuffle=False,rng=None):
        ics=[]; dates=[]
        for d in sorted(by_date):
            recs=[(tk,v,fwd(tk,d)) for tk,v in by_date[d]]
            recs=[(tk,v,r) for tk,v,r in recs if r is not None]
            if len(recs)<a.min_names: continue
            sig=np.array([v for _,v,_ in recs])*a.direction
            ret=np.array([r for _,_,r in recs])
            if shuffle: ret=rng.permutation(ret)
            ic=spearman(sig,ret)
            if ic is not None: ics.append(ic); dates.append(d)
        return np.array(ics),dates

    ics,dates=compute()
    N=len(dates)
    if N<6:
        print("  [STOP] only %d cross-sections with >=%d filings on the same date."%(N,a.min_names))
        print("  Filings don't cluster on common dates (each firm files on its own schedule),")
        print("  so per-date cross-sections are naturally sparse. The step-(a) fetcher will")
        print("  aggregate filings into MONTHLY buckets so each cross-section has enough names.")
        print("  This is expected for the skeleton -- not a failure.")
        return
    lag=max(1,int(math.ceil(a.hold/15.0)))
    mean_ic=ics.mean(); se=nw_se_mean(ics,lag); t=mean_ic/se if se else 0
    print("\n  form=%s  direction=%+d  hold=%dd  cross-sections=%d"%(a.form or "10-K+10-Q",a.direction,a.hold,N))
    print("  mean per-date IC = %+.4f | Newey-West t = %+.2f | %%right-sign = %.0f%%"%(mean_ic,t,100*np.mean(ics>0)))

    rng=np.random.default_rng(11); nulls=[]
    for _ in range(200):
        nc,_=compute(shuffle=True,rng=rng)
        if len(nc): nulls.append(nc.mean())
    nulls=np.array(nulls); z=(mean_ic-nulls.mean())/nulls.std() if nulls.std()>0 else 0
    print("  null control: real IC %.1f std's from shuffled-null (need >=3)"%z)

    print("\n"+LINE+"\nVERDICT\n"+LINE)
    if abs(z)<3:
        print("  >> NOT A BRICK / weak: real IC within the null. No edge on this evidence.")
    elif abs(t)>=3 and abs(z)>=3:
        print("  >> CANDIDATE BRICK #3: IC %+.4f, NW t %+.2f, %.1f std's from null. Then run the same"%(mean_ic,t,z))
        print("     follow-ups brick #2 got: sector-neutral, year-by-year sign, decorrelation vs")
        print("     momentum/PEAD/short-interest, and combination tests.")
    else:
        print("  >> SUGGESTIVE: IC %+.4f (t %+.2f) but below the t>=3 bar. More history needed."%(mean_ic,t))
    print("\n  Honest n=%d cross-sections. Same verified machinery as the short-interest brick."%N)
    print("  In-sample, survivor-tilted (prices.db). A confirmed result still needs OOS.")

if __name__=="__main__":
    try: main()
    except KeyboardInterrupt: print("\ninterrupted.")
    except Exception:
        import traceback; print("\n[UNEXPECTED ERROR] paste back:"); traceback.print_exc()
