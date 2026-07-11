import sqlite3, json, re, time
import numpy as np, pandas as pd
import anthropic

client = anthropic.Anthropic()
MODELS = ["claude-haiku-4-5-20251001", "claude-sonnet-4-6", "claude-opus-4-8"]
SHORT = {"claude-haiku-4-5-20251001":"haiku", "claude-sonnet-4-6":"sonnet", "claude-opus-4-8":"opus"}
N = 150  # larger sample

# pull historical rows: real headlines + resolved h5 outcome
con = sqlite3.connect("accuracy.db")
con.execute("ATTACH 'data/sentiment.db' AS s")
q = """
SELECT m.ticker, m.score_date, m.headlines, o.actual_return
FROM s.monday_sentiment m
JOIN outcomes o ON m.ticker=o.ticker AND m.score_date=o.prediction_date AND o.horizon=5
WHERE m.headlines NOT IN ('[]','') AND length(m.headlines) > 30
ORDER BY m.score_date DESC LIMIT 400
"""
df = pd.read_sql(q, con); con.close()
# sample N spread across the pool (every kth row) for date diversity
df = df.iloc[::max(1,len(df)//N)].head(N).reset_index(drop=True)
print(f"scoring {len(df)} ticker-dates x {len(MODELS)} models = {len(df)*len(MODELS)} calls\n")

def score(model, headlines_json, ticker):
    try:
        hl = json.loads(headlines_json)
    except Exception:
        hl = [headlines_json]
    text = " | ".join(hl[:8])[:1500]
    prompt = (f"Headlines for {ticker}:\n{text}\n\n"
              "Rate the net sentiment for the stock's next 5 trading days as a single "
              "integer from -100 (very bearish) to +100 (very bullish). Reply ONLY the integer.")
    r = client.messages.create(model=model, max_tokens=10,
                               messages=[{"role":"user","content":prompt}])
    t = r.content[0].text.strip()
    mm = re.search(r"-?\d+", t)
    return float(mm.group()) if mm else np.nan

for model in MODELS:
    scores = []
    for row in df.itertuples():
        try:
            scores.append(score(model, row.headlines, row.ticker))
        except Exception as e:
            scores.append(np.nan)
        time.sleep(0.3)
    df[SHORT[model]] = scores
    print(f"{SHORT[model]}: done")

df["fwd_ret"] = df["actual_return"]
print("\n=== DIVERGENCE (how far apart, in points on -100..100) ===")
for a,b in [("haiku","sonnet"),("haiku","opus"),("sonnet","opus")]:
    d = (df[a]-df[b]).abs()
    corr = df[[a,b]].corr().iloc[0,1]
    print(f"  {a:6s} vs {b:6s}: mean|diff|={d.mean():5.1f}  max|diff|={d.max():5.0f}  corr={corr:+.2f}")

print("\n=== PREDICTION (rank-IC: does the score predict the actual h5 return?) ===")
for m in ["haiku","sonnet","opus"]:
    sub = df[[m,"fwd_ret"]].dropna()
    ic = sub[m].corr(sub["fwd_ret"], method="spearman")
    t = ic*np.sqrt(len(sub)-2)/np.sqrt(1-ic**2) if abs(ic)<1 else 0
    print(f"  {m:6s}: rank-IC={ic:+.3f}  t={t:+.2f}  n={len(sub)}")

df.to_csv("sentiment_model_compare.csv", index=False)
print("\nsaved sentiment_model_compare.csv")
