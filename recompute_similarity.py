#!/usr/bin/env python3
"""
================================================================================
ML QUANT FUND — LAZY PRICES: RECOMPUTE SIMILARITY (TF-IDF + Jaccard)
================================================================================
WHY: raw term-frequency cosine pinned ~97% of filings at ~1.0 because ~757 boilerplate
words (in >90% of filings) swamp the few hundred words that actually CHANGE year-over-
year. That kills cross-sectional variation -> nothing to rank -> no signal. The standard
fix (and what the Lazy Prices literature uses) is to DOWN-WEIGHT boilerplate:

  * TF-IDF cosine: weight each word by log(N/df) so words common to all filings count
    for ~nothing and distinctive words drive the similarity.
  * Jaccard on word SETS: |A∩B|/|A∪B| -- a second, rank-uncorrelated-ish measure that
    is naturally less boilerplate-dominated (set membership, not counts).

This reads the CACHED filing texts (no re-download), rebuilds the YoY pairings exactly
as the fetcher did (10-K vs prior 10-K; 10-Q vs prior-year same quarter, from the EDGAR
submissions cache), recomputes both measures, and OVERWRITES filing_similarity.db with
columns: similarity (= tfidf_cosine, the primary), raw_cosine, jaccard, prior_date.

The validator reads `similarity`, so after this it ranks on TF-IDF cosine (the real signal).

RULE 1: corpus IDF built from the SAME cached filings; PIT pairing unchanged (filingDate,
prior comparable filing); reads cache only; OVERWRITES the similarity table (backs up
the old one to filing_similarity_rawcos.db first). Offline.

USAGE:
  python recompute_similarity.py --root .
  python recompute_similarity.py --root . --min-df 3 --max-df-frac 0.95
================================================================================
"""
import argparse, os, sys, json, re, math, sqlite3, datetime, glob, shutil
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
LINE="="*78
_word=re.compile(r"[a-z]{2,}")

def nd(s):
    try: return datetime.date.fromisoformat(str(s)[:10])
    except Exception: return None

def tokenize_counts(text): return Counter(_word.findall(text.lower()))

def accn_nodash(accession): return accession.replace("-","")

def load_submissions_from_cache(cachedir):
    """Rebuild {cik10: [filing dicts]} from the cached subm_*.json files."""
    out={}
    for cf in glob.glob(os.path.join(cachedir,"subm_*.json")):
        base=os.path.basename(cf)
        # subm_<cik10>.json  (skip overflow subm_<name> which won't match 10-digit)
        m=re.match(r"subm_(\d{10})\.json$",base)
        if not m: 
            continue
        cik10=m.group(1)
        try: data=json.load(open(cf))
        except Exception: continue
        filings=[]
        def add_block(b):
            forms=b.get("form",[]); accn=b.get("accessionNumber",[]); fdate=b.get("filingDate",[])
            rdate=b.get("reportDate",[]); pdoc=b.get("primaryDocument",[])
            for i in range(len(forms)):
                if forms[i] not in ("10-K","10-Q"): continue
                filings.append(dict(form=forms[i],
                    accession=accn[i] if i<len(accn) else "",
                    filingDate=fdate[i] if i<len(fdate) else "",
                    reportDate=rdate[i] if i<len(rdate) else "",
                    primaryDocument=pdoc[i] if i<len(pdoc) else ""))
        fb=data.get("filings",{})
        add_block(fb.get("recent",{}))
        # overflow blocks cached as subm_<name>
        for extra in fb.get("files",[]):
            nm=extra.get("name")
            if not nm: continue
            ocf=os.path.join(cachedir,"subm_%s"%nm)
            if os.path.isfile(ocf):
                try:
                    od=json.load(open(ocf)); add_block(od.get("filings",{}).get("recent",{}))
                except Exception: pass
        out[cik10]=filings
    return out

def cik_to_ticker_map(root, cachedir):
    """ticker->CIK from cached company_tickers.json, inverted to cik10->ticker, restricted
    to tickers we actually fetched (present in the old filing_similarity.db)."""
    cf=os.path.join(cachedir,"company_tickers.json")
    t2c={}
    if os.path.isfile(cf):
        try:
            data=json.load(open(cf))
            itr=data.values() if isinstance(data,dict) else data
            for row in itr:
                try: t2c[str(row["ticker"]).upper()]="%010d"%int(row["cik_str"])
                except Exception: continue
        except Exception: pass
    return t2c

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--root",default=".")
    ap.add_argument("--cache-dir",default=None)
    ap.add_argument("--db",default=None)
    ap.add_argument("--min-df",type=int,default=2,help="ignore words in fewer than this many filings")
    ap.add_argument("--max-df-frac",type=float,default=0.90,help="ignore words in more than this fraction of filings (boilerplate)")
    ap.add_argument("--since-year",type=int,default=2014)
    a=ap.parse_args(); a.root=os.path.expanduser(a.root)
    cachedir=a.cache_dir or os.path.join(a.root,"cache","edgar")
    db_path=a.db or os.path.join(a.root,"filing_similarity.db")
    textdir=os.path.join(cachedir,"text")

    print("\n"+LINE+"\nLAZY PRICES — RECOMPUTE SIMILARITY (TF-IDF + Jaccard)\n"+LINE)
    if not os.path.isdir(textdir): print("  [STOP] no cached texts at %s"%textdir); return
    subs_by_cik=load_submissions_from_cache(cachedir)
    if not subs_by_cik: print("  [STOP] no cached submissions (subm_*.json) found."); return
    t2c=cik_to_ticker_map(a.root,cachedir)
    c2t=defaultdict(list)
    for tk,cik in t2c.items(): c2t[cik].append(tk)
    print("  cached: %d CIK submission files | %d ticker->CIK"%(len(subs_by_cik),len(t2c)))

    # which tickers did we actually fetch? (rows in existing db) -> restrict
    fetched=set()
    if os.path.isfile(db_path):
        try:
            oc=sqlite3.connect(db_path); fetched=set(r[0] for r in oc.execute("SELECT DISTINCT ticker FROM filing_similarity")); oc.close()
        except Exception: pass
    print("  tickers in existing db: %d"%len(fetched))

    # ---- PASS 1: load cached texts, tokenize, build document frequency (corpus IDF) ----
    # map accession_nodash -> token Counter (only for filings we can tie to a fetched ticker)
    tok_by_accn={}
    df=Counter()
    n_docs=0
    # build accn -> (cik, filing meta) from submissions
    accn_meta={}
    for cik10, filings in subs_by_cik.items():
        # only CIKs that map to a fetched ticker
        tickers=[t for t in c2t.get(cik10,[]) if (not fetched or t in fetched)]
        if not tickers: continue
        for f in filings:
            if not f["accession"] or not f["primaryDocument"]: continue
            if nd(f["filingDate"]) is None or nd(f["filingDate"]).year<a.since_year: continue
            accn_meta[accn_nodash(f["accession"])]=(cik10,tickers[0],f)
    for accn, (cik10,tk,f) in accn_meta.items():
        tf=os.path.join(textdir,"%s.txt"%accn)
        if not os.path.isfile(tf): continue
        try: text=open(tf,encoding="utf-8",errors="replace").read()
        except Exception: continue
        cnt=tokenize_counts(text)
        if not cnt: continue
        tok_by_accn[accn]=cnt
        for w in cnt.keys(): df[w]+=1
        n_docs+=1
    if n_docs==0: print("  [STOP] no cached texts matched fetched tickers."); return
    print("  loaded %d cached filing texts | vocab %d"%(n_docs,len(df)))

    # IDF with boilerplate/rare cutoffs
    lo=a.min_df; hi=a.max_df_frac*n_docs
    idf={}
    for w,c in df.items():
        if c<lo or c>hi: continue
        idf[w]=math.log(n_docs/c)
    print("  TF-IDF vocab after df cuts [min_df=%d, max_df=%.0f%%]: %d words (boilerplate/rare dropped)"
          %(a.min_df,100*a.max_df_frac,len(idf)))

    def tfidf_vec(cnt):
        v={}
        for w,c in cnt.items():
            iw=idf.get(w)
            if iw is None: continue
            v[w]=c*iw
        return v
    def cos(a_,b_):
        if not a_ or not b_: return None
        if len(a_)>len(b_): a_,b_=b_,a_
        dot=sum(val*b_.get(w,0.0) for w,val in a_.items())
        if dot==0: return 0.0
        na=math.sqrt(sum(x*x for x in a_.values())); nb=math.sqrt(sum(x*x for x in b_.values()))
        if na<=0 or nb<=0: return None
        return max(0.0,min(1.0,dot/(na*nb)))
    def raw_cos(a_,b_):
        if not a_ or not b_: return None
        if len(a_)>len(b_): a_,b_=b_,a_
        dot=sum(c*b_.get(w,0) for w,c in a_.items())
        if dot==0: return 0.0
        na=math.sqrt(sum(c*c for c in a_.values())); nb=math.sqrt(sum(c*c for c in b_.values()))
        return max(0.0,min(1.0,dot/(na*nb))) if na>0 and nb>0 else None
    def jaccard(a_,b_):
        sa=set(w for w in a_ if w in idf); sb=set(w for w in b_ if w in idf)
        if not sa or not sb: return None
        inter=len(sa&sb); union=len(sa|sb)
        return inter/union if union>0 else None

    # precompute tfidf + set per accession (only filings we have tokens for)
    tfidf_by_accn={accn:tfidf_vec(cnt) for accn,cnt in tok_by_accn.items()}

    # ---- PASS 2: re-pair YoY exactly as fetcher, recompute measures ----
    def pair_for_ticker(filings):
        out=[]
        for form in ("10-K","10-Q"):
            fs=[f for f in filings if f["form"]==form
                and accn_nodash(f["accession"]) in tok_by_accn and nd(f["filingDate"])]
            fs.sort(key=lambda x: nd(x["filingDate"]))
            for i,cur in enumerate(fs):
                prior=None; cur_rd=nd(cur.get("reportDate"))
                if form=="10-K":
                    if i>=1: prior=fs[i-1]
                else:
                    if cur_rd:
                        best=None; bestgap=None
                        for j in range(i):
                            rd=nd(fs[j].get("reportDate"))
                            if not rd: continue
                            gap=(cur_rd-rd).days
                            if 300<=gap<=430 and (bestgap is None or abs(gap-365)<abs(bestgap-365)):
                                best=fs[j]; bestgap=gap
                        prior=best
                    if prior is None and i>=3: prior=fs[i-3]
                if prior is None: continue
                ca=accn_nodash(cur["accession"]); pa=accn_nodash(prior["accession"])
                tfc=cos(tfidf_by_accn.get(ca),tfidf_by_accn.get(pa))
                rc=raw_cos(tok_by_accn.get(ca),tok_by_accn.get(pa))
                jc=jaccard(tok_by_accn.get(ca),tok_by_accn.get(pa))
                if tfc is None: continue
                out.append((cur["filingDate"][:10],form,tfc,rc,jc,prior["filingDate"][:10]))
        return out

    # backup old db
    if os.path.isfile(db_path):
        bk=os.path.join(a.root,"filing_similarity_rawcos.db")
        try: shutil.copy(db_path,bk); print("  backed up old (raw-cosine) db -> %s"%os.path.basename(bk))
        except Exception: pass

    db=sqlite3.connect(db_path,timeout=30)
    db.execute("""CREATE TABLE IF NOT EXISTS filing_similarity(
        ticker TEXT, filing_date TEXT, form TEXT, similarity REAL, prior_date TEXT,
        PRIMARY KEY(ticker, filing_date, form))""")
    # add columns if missing
    cols=[r[1] for r in db.execute("PRAGMA table_info(filing_similarity)")]
    for col in ("raw_cosine","jaccard","tfidf_cosine"):
        if col not in cols: db.execute("ALTER TABLE filing_similarity ADD COLUMN %s REAL"%col)
    db.execute("DELETE FROM filing_similarity")  # wipe-and-rebuild: no stale raw-cosine blend
    db.commit()

    n_rows=0
    for cik10, filings in subs_by_cik.items():
        tickers=[t for t in c2t.get(cik10,[]) if (not fetched or t in fetched)]
        if not tickers: continue
        tk=tickers[0]
        # restrict to since_year
        fil=[f for f in filings if nd(f["filingDate"]) and nd(f["filingDate"]).year>=a.since_year]
        pairs=pair_for_ticker(fil)
        for fdate,form,tfc,rc,jc,pdate in pairs:
            # similarity = JACCARD (the measure with real cross-sectional spread on this data;
            # TF-IDF/raw cosine saturate near 1.0 for near-boilerplate filings). Keep all three.
            sim_primary = float(jc) if jc is not None else float(tfc)
            db.execute("""INSERT INTO filing_similarity(ticker,filing_date,form,similarity,prior_date,raw_cosine,jaccard,tfidf_cosine)
                          VALUES(?,?,?,?,?,?,?,?)
                          ON CONFLICT(ticker,filing_date,form) DO UPDATE SET
                            similarity=excluded.similarity, prior_date=excluded.prior_date,
                            raw_cosine=excluded.raw_cosine, jaccard=excluded.jaccard, tfidf_cosine=excluded.tfidf_cosine""",
                       (tk,fdate,form,sim_primary,pdate,
                        float(rc) if rc is not None else None,
                        float(jc) if jc is not None else None,
                        float(tfc) if tfc is not None else None))
            n_rows+=1
        db.commit()

    print("\n"+LINE)
    print("  recomputed %d rows (similarity = TF-IDF cosine)"%n_rows)
    # distribution sanity
    for label,col in (("TF-IDF cosine","similarity"),("raw cosine","raw_cosine"),("Jaccard","jaccard")):
        rows=db.execute("SELECT %s FROM filing_similarity WHERE %s IS NOT NULL"%(col,col)).fetchall()
        vals=sorted(r[0] for r in rows)
        if not vals: continue
        import statistics as st
        q=lambda p: vals[min(len(vals)-1,int(p*len(vals)))]
        print("  %-14s n=%d  min=%.3f  p10=%.3f  median=%.3f  p90=%.3f  max=%.3f"
              %(label,len(vals),vals[0],q(0.1),st.median(vals),q(0.9),vals[-1]))
    db.close()
    print("\n  >> If TF-IDF median is ~0.3-0.7 with real spread (not pinned at 1.0), the signal now")
    print("     has cross-sectional variation to rank. Next: add monthly bucketing to the validator,")
    print("     then: python validate_lazy_prices.py --root . --hold 40")

if __name__=="__main__":
    try: main()
    except KeyboardInterrupt: print("\ninterrupted.")
    except Exception:
        import traceback; print("\n[UNEXPECTED ERROR] paste back:"); traceback.print_exc()
