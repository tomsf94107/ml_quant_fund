#!/usr/bin/env python3
"""
Retire a filled SI cohort out of si_live_ledger.csv.

WHY: si_positions_live.py's LEVERAGE GUARD sums EVERY row in the ledger:
    for _r in csv.DictReader(_f): _open_usd += float(_r.get("usd") or 0)
No date filter, no hold_days check, no closed flag. hold_days is WRITTEN but never
READ BACK. _open_usd is monotonic -- once logged, a cohort counts against capital
forever and every later --log-ledger is blocked. No close operation exists in the repo.

Rows are MOVED, not deleted, to si_ledger_closed.csv, so P&L history survives:
    python si_track.py --ledger si_ledger_closed.csv

Trading days come from daily_prices -- the same calendar the rest of the system uses.
"""
import argparse, csv, os, shutil, sqlite3, sys

ap = argparse.ArgumentParser()
ap.add_argument("--root", default=".")
ap.add_argument("--gen", default=None, help="generated_on of the cohort to close")
ap.add_argument("--all-due", action="store_true", help="close every cohort past its hold")
ap.add_argument("--force", action="store_true", help="close even if the hold has NOT elapsed")
ap.add_argument("--dry-run", action="store_true")
a = ap.parse_args()

led = os.path.join(a.root, "si_live_ledger.csv")
closed = os.path.join(a.root, "si_ledger_closed.csv")
pdb = os.path.join(a.root, "prices.db")
if not os.path.isfile(led): print("[STOP] %s not found" % led); sys.exit(1)
if not (a.gen or a.all_due): print("[STOP] pass --gen YYYY-MM-DD or --all-due"); sys.exit(1)

with open(led, newline="") as f:
    rd = csv.DictReader(f); fields = rd.fieldnames; rows = list(rd)
if not rows: print("  ledger empty; nothing to close"); sys.exit(0)

def _load_calendar(p):
    ap_ = os.path.abspath(p)
    attempts = [("file:%s?mode=ro" % ap_, True),
                ("file:%s?mode=ro&immutable=1" % ap_, True),
                (p, False)]
    for target, as_uri in attempts:
        try:
            c = sqlite3.connect(target, uri=True, timeout=30) if as_uri else sqlite3.connect(target, timeout=30)
            try:
                return [r[0] for r in c.execute("SELECT DISTINCT date FROM daily_prices ORDER BY date")]
            finally:
                c.close()
        except sqlite3.OperationalError:
            continue
    return []

cal = _load_calendar(pdb) if os.path.isfile(pdb) else []
if not cal:
    print("[STOP] could not read a trading calendar from %s" % pdb)
    print("  Without it, elapsed trading days cannot be verified and a cohort")
    print("  could be closed before its hold completes. Refusing.")
    sys.exit(1)
cal_idx = {d: i for i, d in enumerate(cal)}

def elapsed_td(gen):
    if not cal: return None
    if gen in cal_idx: return len(cal) - cal_idx[gen]
    nxt = [d for d in cal if d >= gen]
    return (len(cal) - cal_idx[nxt[0]]) if nxt else None

cohorts = {}
for r in rows: cohorts.setdefault((r.get("generated_on"), r.get("settlement")), []).append(r)

print("  ledger: {} rows | {} cohort(s) | newest close: {}".format(
    len(rows), len(cohorts), cal[-1] if cal else "unknown"))
targets = []
for (gen, settle), rs in sorted(cohorts.items()):
    usd = 0.0
    for r in rs:
        try: usd += float(r.get("usd") or 0)
        except Exception: pass
    try: hold = int(float(rs[0].get("hold_days") or 40))
    except Exception: hold = 40
    el = elapsed_td(gen); due = (el is not None and el >= hold)
    print("    gen={} settlement={} positions={} usd=${:,.0f} elapsed={}/{}td {}".format(
        gen, settle, len(rs), usd, el, hold, "DUE" if due else "open"))
    if (a.all_due and due) or (a.gen and gen == a.gen):
        if not due and not a.force:
            print("      [REFUSED] hold has NOT elapsed. Re-run with --force to close early.")
            continue
        targets.append((gen, settle))

if not targets: print("\n  nothing to close."); sys.exit(0)
print("\n  closing: %s" % ", ".join("gen=%s/settle=%s" % t for t in targets))
if a.dry_run: print("  DRY-RUN: no writes"); sys.exit(0)

shutil.copy2(led, led + ".bak")
move = [r for r in rows if (r.get("generated_on"), r.get("settlement")) in targets]
keep = [r for r in rows if (r.get("generated_on"), r.get("settlement")) not in targets]
new_closed = not os.path.isfile(closed)
with open(closed, "a", newline="") as f:
    w = csv.DictWriter(f, fieldnames=fields)
    if new_closed: w.writeheader()
    w.writerows(move)
with open(led, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(keep)
print("  moved %d row(s) to %s | %d row(s) remain open | backup: %s.bak"
      % (len(move), closed, len(keep), led))
