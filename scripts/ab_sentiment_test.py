#!/usr/bin/env python3
"""
Standalone A/B sentiment impact test.
Run BEFORE sentiment: python scripts/ab_sentiment_test.py --save A
Run AFTER sentiment + runfund: python scripts/ab_sentiment_test.py --save B
Compare next day after close: python scripts/ab_sentiment_test.py --compare
"""
import json, sys, sqlite3, shutil
from pathlib import Path

ROOT   = Path("/Users/atomnguyen/Desktop/ML_Quant_Fund")
CACHE  = ROOT / "data" / "signals_cache.json"
CACHE_A = ROOT / "data" / "ab_cache_A.json"
CACHE_B = ROOT / "data" / "ab_cache_B.json"

def save_snapshot(label):
    if not CACHE.exists():
        print("No cache found — run runfund first")
        return
    dest = CACHE_A if label == "A" else CACHE_B
    shutil.copy(CACHE, dest)
    with open(dest) as f:
        d = json.load(f)
    print(f"Saved Run {label}: {d.get('generated_at')} — {len(d.get('signals',[]))} signals")

def compare():
    if not CACHE_A.exists() or not CACHE_B.exists():
        print("Missing snapshots — save both A and B first")
        return

    with open(CACHE_A) as f: da = json.load(f)
    with open(CACHE_B) as f: db = json.load(f)

    sigs_a = {s['ticker']: s for s in da.get('signals',[]) if s.get('horizon')==1}
    sigs_b = {s['ticker']: s for s in db.get('signals',[]) if s.get('horizon')==1}

    changed = []
    for t in sigs_a:
        if t in sigs_b:
            pa = sigs_a[t].get('prob', sigs_a[t].get('prob_up', 0.5))
            pb = sigs_b[t].get('prob', sigs_b[t].get('prob_up', 0.5))
            if abs(pa - pb) > 0.01:
                changed.append((t, pa, pb))

    print(f"\n{'='*50}")
    print(f"  A/B Sentiment Impact Test")
    print(f"  Run A: {da.get('generated_at')} (pre-sentiment)")
    print(f"  Run B: {db.get('generated_at')} (post-sentiment)")
    print(f"{'='*50}")
    print(f"  Signals changed: {len(changed)}/{len(sigs_a)}")
    if changed:
        print(f"\n  Top 10 changes:")
        for t, pa, pb in sorted(changed, key=lambda x: abs(x[2]-x[1]), reverse=True)[:10]:
            print(f"    {t:6s}: {pa:.3f} → {pb:.3f} ({'↑' if pb > pa else '↓'}) ({pb-pa:+.3f})")

    date_str = da.get('date')
    conn = sqlite3.connect(ROOT / "accuracy.db")

    def score(sigs):
        correct = total = 0
        for t, s in sigs.items():
            row = conn.execute(
                "SELECT actual_return FROM outcomes WHERE ticker=? AND prediction_date=? AND horizon=1",
                (t, date_str)
            ).fetchone()
            if row:
                pred_up = s.get('prob', s.get('prob_up', 0.5)) > 0.5
                actual_up = row[0] > 0
                correct += int(pred_up == actual_up)
                total += 1
        return correct, total

    ca, ta = score(sigs_a)
    cb, tb = score(sigs_b)
    conn.close()

    if ta > 0 and tb > 0:
        print(f"\n  Accuracy:")
        print(f"  Run A (pre-sentiment):  {ca}/{ta} = {100*ca/ta:.1f}%")
        print(f"  Run B (post-sentiment): {cb}/{tb} = {100*cb/tb:.1f}%")
        diff = (cb/tb - ca/ta) * 100
        winner = "B better ✅" if diff > 0 else "A better" if diff < 0 else "No difference"
        print(f"  Sentiment impact: {diff:+.1f}% — {winner}")
    else:
        print(f"\n  Outcomes not available yet for {date_str} — run after market close")
    print(f"{'='*50}\n")

if "--save" in sys.argv:
    save_snapshot(sys.argv[sys.argv.index("--save") + 1].upper())
elif "--compare" in sys.argv:
    compare()
else:
    print("Usage:")
    print("  abtest --save A    (morning after pipeline runs)")
    print("  abtest --save B    (after runfund post-sentiment)")
    print("  abtest --compare   (next day after market close)")
