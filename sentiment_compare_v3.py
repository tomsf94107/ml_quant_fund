import sqlite3, json, re, time, sys
import numpy as np, pandas as pd
import anthropic

client = anthropic.Anthropic()
MODELS = ["claude-haiku-4-5-20251001", "claude-sonnet-4-6", "claude-opus-4-8"]
SHORT = {"claude-haiku-4-5-20251001":"haiku","claude-sonnet-4-6":"sonnet","claude-opus-4-8":"opus"}
N = 120

con = sqlite3.connect("accuracy.db"); con.execute("ATTACH 'data/sentiment.db' AS s")
df = pd.read_sql("""SELECT m.ticker, m.score_date, m.headlines, o.actual_return
FROM s.monday_sentiment m JOIN outcomes o
  ON m.ticker=o.ticker AND m.score_date=o.prediction_date AND o.horizon=5
WHERE m.headlines NOT IN ('[]','') AND length(m.headlines)>30
ORDER BY m.score_date DESC LIMIT 600""", con); con.close()
df = df.iloc[::max(1,len(df)//N)].head(N).reset_index(drop=True)
print(f"scoring {len(df)} rows x 3 models = {len(df)*3} calls", flush=True)

def score(model, hj, tk):
    try: hl = json.loads(hj)
    except Exception: hl = [hj]
    text = " | ".join(hl[:8])[:1500]
    r = client.messages.create(model=model, max_tokens=10, timeout=30,
        messages=[{"role":"user","content":
        f"Headlines for {tk}:\n{text}\n\nRate net sentiment for the next 5 trading days "
        "as one integer -100 (very bearish) to +100 (very bullish). Reply ONLY the integer."}])
    mm = re.search(r"-?\d+", r.content[0].text.strip())
    return float(mm.group()) if mm else np.nan

for model in MODELS:
    s, sh = [], SHORT[model]
    for i, row in enumerate(df.itertuples()):
        try: s.append(score(model, row.headlines, row.ticker))
        except Exception as e: s.append(np.nan)
        if (i+1) % 20 == 0: print(f"  {sh}: {i+1}/{len(df)}", flush=True)
    df[sh] = s
    print(f"{sh}: DONE", flush=True)

print("\n=== DIVERGENCE (points on -100..100) ===", flush=True)
for a,b in [("haiku","sonnet"),("haiku","opus"),("sonnet","opus")]:
    d=(df[a]-df[b]).abs(); print(f"  {a:6s} vs {b:6s}: mean|diff|={d.mean():5.1f} corr={df[[a,b]].corr().iloc[0,1]:+.2f}", flush=True)
print("\n=== PREDICTION (rank-IC vs actual h5 return) ===", flush=True)
for m in ["haiku","sonnet","opus"]:
    sub=df[[m,"actual_return"]].dropna(); ic=sub[m].corr(sub["actual_return"],method="spearman")
    t=ic*np.sqrt(len(sub)-2)/np.sqrt(1-ic**2) if abs(ic)<1 else 0
    print(f"  {m:6s}: rank-IC={ic:+.3f} t={t:+.2f} n={len(sub)}", flush=True)
# time-split robustness: early vs late dates for opus
df_sorted = df.sort_values("score_date")
half = len(df_sorted)//2
print("\n=== OPUS time-split (is it stable across periods?) ===", flush=True)
for lbl, part in [("early", df_sorted.iloc[:half]), ("late", df_sorted.iloc[half:])]:
    sub=part[["opus","actual_return"]].dropna(); ic=sub["opus"].corr(sub["actual_return"],method="spearman")
    print(f"  {lbl}: opus rank-IC={ic:+.3f} n={len(sub)}", flush=True)
df.to_csv("sentiment_compare_v3.csv", index=False)
print("\nsaved", flush=True)
