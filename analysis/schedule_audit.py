#!/usr/bin/env python3
"""
schedule_audit.py -- every scheduled job (cron + launchd) on one clock, and
every pair that runs at the same time AND touches the same SQLite file.

READ-ONLY. Reads `crontab -l`, ~/Library/LaunchAgents/com.atom.*.plist, the
scripts they run plus the local modules those import (two levels deep), and
PRAGMA journal_mode on each database found. Writes nothing.

USAGE
    python analysis/schedule_audit.py
    python analysis/schedule_audit.py --days 120 --default-min 15

WHAT IT CAN AND CANNOT SEE
    Schedules  exact, from crontab and the plists.
    Durations  MEASURED where a real run was timed (KNOWN below); otherwise
               ESTIMATED as the log's last-modified time minus the last
               scheduled start before it; otherwise DEFAULTED and flagged.
               A job that writes its log early and keeps working looks
               shorter than it is.
    Databases  STATIC -- which .db names each script and its local imports
               mention, and whether a write statement sits near a mention.
               A table named at runtime is invisible, and root sentiment.db
               and data/sentiment.db are reported under one name.
    So every conflict is a CANDIDATE to confirm from logs, not a proof.
"""
import argparse
import plistlib
import re
import sqlite3
import subprocess
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# (regex on the command, minutes, source). First match wins, so the specific
# patterns sit above the general ones.
KNOWN = [
    (r"pipeline_chain_ADB\.sh", 354, "measured 04:00-09:54, 2026-09-19"),
    (r"pipeline_C_preopen\.sh", 138, "measured 17:00-19:18, 2026-09-18"),
    (r"etl_insider_raw", 81, "measured 05:00-06:21, 2026-09-21"),
    (r"etl_finbert_filings", 9, "measured 05:30-05:39, 2026-09-21"),
    (r"h40_shadow\.py.*--status", 1, "status query only"),
    (r"h40_shadow\.py", 25, "measured ~25 min per 415-ticker build"),
    (r"rebuild_earnings_from_uw\.py", 10, "estimate: 422 tickers x 0.6s + API"),
]

# Invocations that only READ even though the script file can write.
READONLY = [r"--status\b"]

# The A -> D -> B chain runs as one launchd job but writes different databases
# in different stages. Treating it as one 354-minute writer flags anything in
# 04:00-09:54 that touches accuracy.db, though B only writes predictions near
# the end. Stage offsets and lengths measured on 2026-09-19 (A marker 05:24,
# "D exited -> starting B" 06:48, CHAIN END 09:54).
CHAIN_RE = r"pipeline_chain_ADB\.sh"
CHAIN_STAGES = [("scripts/pipeline_A_ingest.sh", 0, 84),
                ("scripts/pipeline_D_alpha_panel.sh", 84, 84),
                ("scripts/pipeline_B_train_predict.sh", 168, 186)]

WRITE_RE = re.compile(
    r"\b(INSERT\s+(OR\s+\w+\s+)?INTO|UPDATE\s+\w+\s+SET|DELETE\s+FROM|"
    r"REPLACE\s+INTO|CREATE\s+TABLE|DROP\s+TABLE|ALTER\s+TABLE|\.to_sql\()",
    re.I)
PY_DB_RE = re.compile(r"""["'/]([A-Za-z0-9_\-]+\.db)\b""")
SH_DB_RE = re.compile(r"""(?:^|[\s"'/=])([A-Za-z0-9_\-]+\.db)\b""")
STOP = {"self", "args", "a", "cfg", "config", "con", "conn", "c", "db",
        "opts", "options", "os", "sys", "settings", "p"}
IMPORT_RE = re.compile(r"^\s*(?:from\s+([\w.]+)\s+import\s+([\w, ]+)|import\s+([\w.]+))",
                       re.M)


# ─────────────────────────────────────────────────────────── cron fields
def expand(field, lo, hi, is_dow=False):
    out = set()
    for part in field.split(","):
        step = 1
        if "/" in part:
            part, s = part.split("/", 1)
            step = int(s)
        if part == "*":
            a, b = lo, hi
        elif "-" in part:
            a, b = (int(x) for x in part.split("-", 1))
        else:
            a = int(part)
            b = hi if step > 1 else a
        out.update(range(a, b + 1, step))
    if is_dow and 7 in out:
        out.discard(7)
        out.add(0)
    return out


MACROS = {"@hourly": "0 * * * *", "@daily": "0 0 * * *", "@midnight": "0 0 * * *",
          "@weekly": "0 0 * * 0", "@monthly": "0 0 1 * *", "@yearly": "0 0 1 1 *",
          "@annually": "0 0 1 1 *"}


def parse_crontab(text):
    jobs, skipped = [], []
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        first = line.split()[0]
        if "=" in first and not first.startswith("@"):
            continue                      # environment assignment
        if first.startswith("@"):
            if first not in MACROS:
                skipped.append(line)       # @reboot etc.
                continue
            line = MACROS[first] + " " + line[len(first):].strip()
        f = line.split(None, 5)
        if len(f) < 6:
            skipped.append(line)
            continue
        m, h, dom, mon, dow, cmd = f
        try:
            sched = dict(minutes=expand(m, 0, 59), hours=expand(h, 0, 23),
                         doms=expand(dom, 1, 31), months=expand(mon, 1, 12),
                         dows=expand(dow, 0, 7, True),
                         dom_star=(dom == "*"), dow_star=(dow == "*"))
        except ValueError:
            skipped.append(line)
            continue
        jobs.append(dict(src="cron", spec=" ".join(f[:5]), cmd=cmd, scheds=[sched]))
    return jobs, skipped


def parse_agents(agent_dir):
    jobs, other = [], []
    for p in sorted(Path(agent_dir).glob("com.atom.*.plist")):
        try:
            d = plistlib.load(open(p, "rb"))
        except Exception as e:
            other.append((p.name, f"unreadable: {e}"))
            continue
        cmd = " ".join(d.get("ProgramArguments", []) or [d.get("Program", "")])
        log = d.get("StandardOutPath") or d.get("StandardErrorPath")
        sci = d.get("StartCalendarInterval")
        if sci is None:
            if "StartInterval" in d:
                other.append((p.name, f"every {d['StartInterval']}s"))
            elif d.get("KeepAlive"):
                other.append((p.name, "KeepAlive (continuous)"))
            else:
                other.append((p.name, "no schedule (on demand)"))
            continue
        scheds, specs = [], []
        for e in (sci if isinstance(sci, list) else [sci]):
            sched = dict(
                minutes={e["Minute"]} if "Minute" in e else set(range(60)),
                hours={e["Hour"]} if "Hour" in e else set(range(24)),
                doms={e["Day"]} if "Day" in e else set(range(1, 32)),
                months={e["Month"]} if "Month" in e else set(range(1, 13)),
                dows={e["Weekday"] % 7} if "Weekday" in e else set(range(7)),
                dom_star="Day" not in e, dow_star="Weekday" not in e)
            scheds.append(sched)
            specs.append(e)
        hm = {(e.get("Hour"), e.get("Minute")) for e in specs}
        wd = sorted({e["Weekday"] % 7 for e in specs if "Weekday" in e})
        if len(hm) == 1:
            h, m = next(iter(hm))
            spec = (f"{'*' if m is None else m} {'*' if h is None else h} * * "
                    f"{','.join(map(str, wd)) if wd else '*'}")
        else:
            spec = f"{len(specs)} entries"
        jobs.append(dict(src="launchd:" + p.stem.replace("com.atom.", ""),
                         spec=spec + " (L)", cmd=cmd, scheds=scheds, logpath=log))
    return jobs, other


def day_matches(s, d):
    cron_dow = (d.weekday() + 1) % 7
    if d.month not in s["months"]:
        return False
    dm, wm = d.day in s["doms"], cron_dow in s["dows"]
    if not s["dom_star"] and not s["dow_star"]:
        return dm or wm
    return dm and wm


def starts(scheds, d0, days):
    out = set()
    for i in range(days):
        d = d0 + timedelta(days=i)
        for s in scheds:
            if day_matches(s, d):
                for h in s["hours"]:
                    for m in s["minutes"]:
                        out.add(datetime(d.year, d.month, d.day, h, m))
    return sorted(out)


# ─────────────────────────────────────────────────────── script analysis
def localize(path_str, base):
    s = re.sub(r"^.*?/ML_Quant_Fund/", "", path_str)
    for cand in (ROOT / s, base / s):
        if cand.exists():
            return cand.resolve()
    return None


def scripts_in(cmd, base):
    found = []
    for mod in re.findall(r"-m\s+([A-Za-z_][\w.]*)", cmd):
        for cand in (ROOT / (mod.replace(".", "/") + ".py"),
                     ROOT / mod.replace(".", "/") / "__init__.py"):
            if cand.exists():
                found.append(cand.resolve())
                break
    for pat in (r"([\w./~-]+\.py)\b", r"([\w./~-]+\.sh)\b"):
        for s in re.findall(pat, cmd):
            p = localize(s, base)
            if p:
                found.append(p)
    return list(dict.fromkeys(found))


def local_imports(text, here):
    out = []
    for frm, names, imp in IMPORT_RE.findall(text):
        mods = []
        if frm and not frm.startswith("."):
            mods.append(frm)
            mods += [f"{frm}.{n.strip()}" for n in names.split(",") if n.strip()]
        if imp:
            mods.append(imp)
        for mod in mods:
            rel = mod.replace(".", "/")
            for cand in (ROOT / (rel + ".py"), ROOT / rel / "__init__.py",
                         here.parent / (rel + ".py")):
                if cand.exists():
                    out.append(cand.resolve())
                    break
    return out


def db_access(path):
    """Return {db: 'W' | 'W?' | 'R'} for one file."""
    try:
        text = path.read_text(errors="ignore")
    except Exception:
        return {}
    rx = SH_DB_RE if path.suffix == ".sh" else PY_DB_RE
    lines = text.splitlines()
    mentions = defaultdict(list)
    for i, l in enumerate(lines):
        if "mode=ro" in l:
            continue
        for db in rx.findall(l):
            if db.rsplit(".", 1)[0] not in STOP:
                mentions[db].append(i)
    ro_only = {db for l in lines if "mode=ro" in l for db in rx.findall(l)}
    acc = {db: "R" for db in list(mentions) + list(ro_only)
           if db.rsplit(".", 1)[0] not in STOP}
    wlines = [j for j, l in enumerate(lines)
              if WRITE_RE.search(l) and not l.lstrip().startswith("#")]
    if not wlines or not mentions:
        return acc
    if len(mentions) == 1:
        acc[next(iter(mentions))] = "W"
        return acc
    unattributed = False
    for j in wlines:
        best = None
        for db, idx in mentions.items():
            near = [i for i in idx if j - 80 <= i <= j]
            if near and (best is None or max(near) > best[1]):
                best = (db, max(near))
        if best:
            acc[best[0]] = "W"
        else:
            unattributed = True
    if unattributed:
        for db in mentions:
            if acc.get(db) == "R":
                acc[db] = "W?"
    return acc


RANK = {"R": 0, "W?": 1, "W": 2}


def job_dbs(cmd, depth_sh=3, depth_py=2):
    access, seen, files = {}, set(), []

    def merge(a):
        for db, mode in a.items():
            if RANK[mode] > RANK.get(access.get(db, "R"), -1) or db not in access:
                access[db] = mode

    def walk_py(p, d):
        if p in seen:
            return
        seen.add(p)
        files.append(p)
        merge(db_access(p))
        if d > 0:
            try:
                text = p.read_text(errors="ignore")
            except Exception:
                return
            for q in local_imports(text, p):
                walk_py(q, d - 1)

    def walk_sh(p, d):
        if p in seen:
            return
        seen.add(p)
        files.append(p)
        merge(db_access(p))
        if d <= 0:
            return
        try:
            text = p.read_text(errors="ignore")
        except Exception:
            return
        for q in scripts_in(text, p.parent):
            (walk_sh if q.suffix == ".sh" else walk_py)(q, d - 1 if q.suffix == ".sh" else depth_py)

    for s in scripts_in(cmd, ROOT):
        (walk_sh if s.suffix == ".sh" else walk_py)(s, depth_sh if s.suffix == ".sh" else depth_py)
    return access, files


# ───────────────────────────────────────────────────────────── durations
def logfile(cmd):
    m = re.search(r">>?\s*([^\s;&|]+\.log)", cmd)
    if not m:
        return None
    p = localize(m.group(1), ROOT)
    return p if p else (ROOT / m.group(1) if not m.group(1).startswith("/") else Path(m.group(1)))


def duration(job, default_min):
    for pat, mins, note in KNOWN:
        if re.search(pat, job["cmd"]):
            return mins, "measured" if note.startswith("measured") else "est", note
    lp = job.get("logpath")
    lp = Path(lp) if lp else logfile(job["cmd"])
    if lp and lp.exists():
        mt = datetime.fromtimestamp(lp.stat().st_mtime)
        prior = [s for s in starts(job["scheds"], (mt - timedelta(days=8)).date(), 9) if s <= mt]
        if prior:
            gap = (mt - prior[-1]).total_seconds() / 60
            if 0 <= gap <= 12 * 60:
                return max(1, round(gap)), "log", f"{lp.name} mtime {mt:%m-%d %H:%M}"
    return default_min, "DEFAULT", "no timing available"


def journal_mode(db):
    for cand in (ROOT / db, ROOT / "data" / db):
        if cand.exists():
            try:
                c = sqlite3.connect(f"file:{cand}?mode=ro", uri=True, timeout=5)
                m = c.execute("PRAGMA journal_mode").fetchone()[0]
                c.close()
                return m
            except Exception as e:
                return f"? ({type(e).__name__})"
    return "not found"


# ───────────────────────────────────────────────────────────────── main
def label(job):
    names = [p.name for p in job["files"][:1]] or [job["cmd"][:40]]
    tail = ""
    for flag in ("--book", "--status", "--also-book", "--incremental", "--no-cursor"):
        m = re.search(rf"{flag}(\s+[\w.-]+)?", job["cmd"])
        if m:
            tail += " " + m.group(0).strip()
    return (names[0] + tail)[:58]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=120,
                    help="window to simulate; 120 covers monthly and quarterly jobs")
    ap.add_argument("--default-min", type=int, default=15)
    ap.add_argument("--crontab", help="read crontab text from a file (testing)")
    ap.add_argument("--agents", default=str(Path.home() / "Library" / "LaunchAgents"))
    ap.add_argument("--start", help="YYYY-MM-DD; default today")
    a = ap.parse_args()

    text = (open(a.crontab).read() if a.crontab else
            subprocess.run(["crontab", "-l"], capture_output=True, text=True).stdout)
    cron, skipped = parse_crontab(text)
    agents, other = parse_agents(a.agents)
    jobs = []
    for j in cron + agents:
        if re.search(CHAIN_RE, j["cmd"]):
            for rel, off, mins in CHAIN_STAGES:
                if (ROOT / rel).exists():
                    jobs.append(dict(j, cmd=str(ROOT / rel), offset=off, fixed=mins,
                                     spec=j["spec"] + f" +{off}m"))
        else:
            jobs.append(dict(j, offset=0))
    d0 = (datetime.strptime(a.start, "%Y-%m-%d") if a.start else datetime.now()).date()

    for j in jobs:
        j["dbs"], j["files"] = job_dbs(j["cmd"])
        if any(re.search(r, j["cmd"]) for r in READONLY):
            j["dbs"] = {db: "R" for db in j["dbs"]}
        if j.get("fixed"):
            j["dur"], j["dsrc"], j["dnote"] = j["fixed"], "measured", "chain stage, 2026-09-19"
        else:
            j["dur"], j["dsrc"], j["dnote"] = duration(j, a.default_min)
        j["label"] = label(j)

    # ── job table
    print(f"{len(jobs)} scheduled jobs ({len(cron)} cron, {len(agents)} launchd entries)"
          f"  window {d0} + {a.days} days\n")
    print(f"  {'schedule':<30}{'min':>5} {'src':<9}{'job':<58} databases (W write, W? possible, R read)")
    for j in sorted(jobs, key=lambda x: (min(min(s["hours"]) for s in x["scheds"]) * 60
                                         + min(min(s["minutes"]) for s in x["scheds"])
                                         + x.get("offset", 0))):
        dbs = ", ".join(f"{k}:{v}" for k, v in sorted(j["dbs"].items())) or "-"
        print(f"  {j['spec']:<30}{j['dur']:>5} {j['dsrc']:<9}{j['label']:<58} {dbs}")
    if skipped:
        print(f"\n  not simulated: {len(skipped)} line(s)")
        for s in skipped:
            print(f"    {s[:100]}")
    for name, why in other:
        print(f"  launchd {name}: {why}")

    # ── journal modes
    all_dbs = sorted({db for j in jobs for db in j["dbs"]})
    modes = {db: journal_mode(db) for db in all_dbs}
    print("\njournal modes (wal = readers are not blocked by a writer):")
    for db in all_dbs:
        print(f"  {db:<32}{modes[db]}")

    # ── intervals and self-overlap
    ivs = []
    self_overlap = []
    for k, j in enumerate(jobs):
        st = [t + timedelta(minutes=j.get("offset", 0)) for t in starts(j["scheds"], d0, a.days)]
        for s1, s2 in zip(st, st[1:]):
            if (s2 - s1).total_seconds() / 60 < j["dur"]:
                self_overlap.append((j, s1, s2))
                break
        for s in st:
            ivs.append((s, s + timedelta(minutes=j["dur"]), k))

    # ── conflicts: same db, overlapping time, at least one writer
    hits = defaultdict(lambda: [0, None])
    by_db = defaultdict(list)
    for s, e, k in ivs:
        for db, mode in jobs[k]["dbs"].items():
            by_db[db].append((s, e, k, mode))
    for db, lst in by_db.items():
        lst.sort()
        active = []
        for s, e, k, mode in lst:
            active = [x for x in active if x[1] > s]
            for s2, e2, k2, mode2 in active:
                if k2 == k or (mode == "R" and mode2 == "R"):
                    continue
                if jobs[k]["cmd"] == jobs[k2]["cmd"]:
                    continue
                pair = tuple(sorted((k, k2)))
                ww = mode.startswith("W") and mode2.startswith("W")
                certain = "?" not in mode + mode2
                wal = str(modes.get(db, "")).lower() == "wal"
                sev = ("HIGH" if ww and certain else
                       "MED" if ww or (not wal and certain) else "LOW")
                key = (sev, db, pair)
                hits[key][0] += 1
                if hits[key][1] is None:
                    hits[key][1] = max(s, s2)
            active.append((s, e, k, mode))

    order = {"HIGH": 0, "MED": 1, "LOW": 2}
    print(f"\nconflicts: {len(hits)} job-pair/database combinations")
    print("  HIGH  both write, both certain")
    print("  MED   both write but one uncertain, or write vs read on a rollback-journal db")
    print("  LOW   write vs read on a WAL db, or involves only possible writes\n")
    for (sev, db, (k1, k2)), (n, first) in sorted(
            hits.items(), key=lambda x: (order[x[0][0]], -x[1][0])):
        j1, j2 = jobs[k1], jobs[k2]
        print(f"  {sev:<5}{db:<24}x{n:<4} first {first:%a %m-%d %H:%M}")
        print(f"         {j1['spec']:<30}{j1['label']}  [{j1['dbs'][db]}, {j1['dur']}m {j1['dsrc']}]")
        print(f"         {j2['spec']:<30}{j2['label']}  [{j2['dbs'][db]}, {j2['dur']}m {j2['dsrc']}]")

    if self_overlap:
        print("\njobs that can overlap their own next run:")
        for j, s1, s2 in self_overlap:
            print(f"  {j['spec']:<22}{j['label']}  runs {j['dur']}m, next start {s2:%a %H:%M} after {s1:%a %H:%M}")

    dflt = [j for j in jobs if j["dsrc"] == "DEFAULT"]
    if dflt:
        print(f"\n{len(dflt)} job(s) have NO timing -- assumed {a.default_min} min; "
              f"their conflicts are the least certain:")
        for j in dflt:
            print(f"  {j['spec']:<22}{j['label']}")


if __name__ == "__main__":
    main()
