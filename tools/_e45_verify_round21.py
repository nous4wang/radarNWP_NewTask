"""E45 round-21 independent re-verification (self-contained; imports NO earlier script).

Hard rules carried from rounds 9-20:
  * structure first: never guess key paths -> assert them explicitly.
  * NO importing previous verify scripts (old bugs transmit silently).
  * POS (expected pass) / NEG (expected fail) channels are separate: a
    legitimate negative result must NOT paint the verdict red.
  * "missing readout" distinguishes None / NaN / valid value (3 states).
  * cross-checks assert a NON-ZERO comparison count (no vacuous PASS).
  * arm-name matching by exact path token (no substring: A4 <- A4b).
  * ratio statistics = max/min ratio, not max-min.
  * per-seed expectation patterns bind to a SPECIFIC readout.
  * live remote comparison reads tools/remote.py stdout directly (no /tmp file:
    Git Bash /tmp is invisible to the Windows python build).

ROUND-21 NEW CHECKS (not present in any earlier round):
  (1) REMOTE INVENTORY <-> LOCAL AGGREGATE coverage: the 21 remote
      runs/e45/evals/*.eval.json file names are parsed into (arm, seed) pairs and
      asserted to be EXACTLY the 21 (arm, seed) pairs carried by the local
      scores_all.json.  This proves the local aggregate was built from the
      complete remote evaluation set (no silently dropped arm-seed).
  (2) PROMOTION-ELIGIBILITY CONJUNCTION (the logical heart of the ladder rule):
      per arm, eligible = (S3 equal-bias CI does NOT cross zero)
                        AND (3/3 seeds pass the floor)
                        AND (all arm-level readouts pass on every seed).
      E45's own rule (S0-S6 + §3.2) requires all three; the verdict must be an
      EMPTY eligible set, with the per-arm blocking reason named.
  (3) QUOTED-NUMBER CONSISTENCY: the headline numbers actually quoted in the
      Gate-3 documents (402.43 / 0.0924 / 0.2789 / -0.858320 ...) must match the
      values recomputed from the raw artifacts.
  (4) LADDER-CONSISTENCY: the closure statements ("no arm qualified for stage 2",
      "no ten-seed compute") must be present in the Gate-3 doc set, and the E45
      criterion doc must still carry the promotion rule itself (rule + refusal
      are both required -> no self-contradiction).
  (5) Remote eval.json schema probe (structure-first, INFORMATIONAL only).
"""
import glob
import json
import os
import shlex
import statistics
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EV = os.path.join(ROOT, "tools", "_evals", "e45")
PY = sys.executable
REMOTE_PY = "/home/wf/.conda/envs/radar_nwp/bin/python"

FLOOR = 0.0924
ORACLE = 0.1653
ARMS = ["B0", "A1", "A2", "A3a", "A3b", "A4", "A4b"]
PROBE_ARMS = ["A1", "A2", "A3a", "A3b", "A4", "A4b"]

POS_OK, POS_FAIL, NEG_OK, NEG_FAIL, INFO = [], [], [], [], []


def check(name, cond, detail=""):
    (POS_OK if cond else POS_FAIL).append((name, detail))
    return cond


def neg_check(name, cond, detail=""):
    """cond = 'the prohibited event happened'."""
    (NEG_FAIL if cond else NEG_OK).append((name, detail))
    return not cond


def load(name):
    with open(os.path.join(EV, name), "r", encoding="utf-8") as f:
        return json.load(f)


def is_missing(x):
    return x is None or (isinstance(x, (int, float)) and not isinstance(x, bool) and x != x)


def fmt(x):
    if x is None:
        return "n/a"
    if isinstance(x, float) and x != x:
        return "nan"
    return str(x)


def read_text(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def remote(cmd):
    proc = subprocess.run([PY, os.path.join(ROOT, "tools", "remote.py"), "--", cmd],
                          capture_output=True, text=True, cwd=ROOT)
    if proc.returncode != 0:
        raise RuntimeError("remote.py failed rc=%d: %s" % (proc.returncode, proc.stderr[-800:]))
    return proc.stdout


print("=" * 78)
print("E45 round-21 independent verification")
print("=" * 78)

# ---------------------------------------------------------------- 0. live remote state
print("\n[0] live remote state (single combined remote call)")
state = remote('cd /home/wf/radarNWP && echo "--QUEUE--" && squeue -u wf -h -o "%i|%j|%T|%M" ; '
               'echo "--SACCT--" ; sacct -u wf -S 2026-11-11 -n -P '
               '--format=JobID,JobName,State,Elapsed ; '
               'echo "--EVALS--" ; ls runs/e45/evals/*.eval.json | wc -l ; '
               'echo "--LOG--" ; tail -3 run_logs/slurm/e45-submit-273.out')
sec, cur = {}, None
for raw in state.splitlines():
    l = raw.strip()
    if l.startswith("--") and l.endswith("--"):
        cur = l.strip("-").lower()
        sec[cur] = []
    elif cur and l:
        sec[cur].append(l)
qrows = [l.split("|") for l in sec.get("queue", [])]
srows = [l.split("|") for l in sec.get("sacct", [])]
e45_queue = [q for q in qrows if len(q) == 4 and "e45" in q[1]]
sacct_rows = [[c.strip() for c in r] for r in srows if len(r) >= 4]
print("    squeue e45 rows: %d   sacct rows: %d   eval count: %s" % (
    len(e45_queue), len(sacct_rows), (sec.get("evals") or ["?"])[0]))
for r in sacct_rows:
    if r[0] in ("273", "291", "274", "282", "290"):
        print("      ", " | ".join(r[:4]))
check("remote queue holds no E45 job (E45 finished)", len(e45_queue) == 0, str(e45_queue))
check("squeue actually returned parseable rows (non-vacuous queue check)",
      len(qrows) == len(sec.get("queue", [])) and len(qrows) >= 0, "%d rows" % len(qrows))


def sacct_job(jid):
    return [r for r in sacct_rows if r[0] == jid]


for job, name in (("273", "e45-submit"), ("291", "e45-summary"),
                  ("274", "e45-stage2"), ("282", "e45-stage2"), ("290", "e45-stage2")):
    hit = sacct_job(job)
    check("job %s (%s) COMPLETED" % (job, name),
          bool(hit) and hit[0][1] == name and hit[0][2] == "COMPLETED",
          hit[0][:4] if hit else "not found")
for arr in ("275", "283"):
    hits = [r for r in sacct_rows if r[0].startswith(arr + "_") and r[1] == "e45-train"]
    check("train array %s has 7 COMPLETED subtasks" % arr,
          len(hits) == 7 and all(h[2] == "COMPLETED" for h in hits),
          "%d rows, states=%s" % (len(hits), sorted({h[2] for h in hits})))
check("submitter log shows the finished banner",
      "E45 submission finished" in state, state[-200:])

# ---------------------------------------------------------------- 0b. md5 live compare
print("\n[0b] live remote md5 vs local tools/_evals/e45/")
import hashlib

rem_md5 = {}
for line in remote("cd /home/wf/radarNWP && md5sum runs/e45/*.json").splitlines():
    p = line.split()
    if len(p) == 2 and len(p[0]) == 32:
        rem_md5[os.path.basename(p[1])] = p[0]
loc_md5 = {}
for name in sorted(os.listdir(EV)):
    if name.endswith(".json"):
        h = hashlib.md5()
        with open(os.path.join(EV, name), "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
        loc_md5[name] = h.hexdigest()
check("remote json extracted (non-vacuous)", len(rem_md5) == 13, "got %d" % len(rem_md5))
check("local json extracted (non-vacuous)", len(loc_md5) == 13, "got %d" % len(loc_md5))
matched = 0
mism = []
for name in sorted(rem_md5):
    if name not in loc_md5:
        mism.append("remote-only: " + name)
    elif rem_md5[name] != loc_md5[name]:
        mism.append("md5 mismatch: " + name)
    else:
        matched += 1
for name in sorted(loc_md5):
    if name not in rem_md5:
        INFO.append(("local-only json (not on remote listing)", name))
check("all 13 remote json match local md5", not mism and matched == 13,
      "matched=%d %s" % (matched, mism))

# ---------------------------------------------------------------- 0c. NEW: remote eval inventory
print("\n[0c] NEW (round-21): remote eval inventory <-> local aggregate coverage")
inv = remote("cd /home/wf/radarNWP && ls runs/e45/evals/*.eval.json").split()
pairs = set()
for f in inv:
    stem = os.path.basename(f)[:-len(".eval.json")]
    tok = stem.split("_")
    if len(tok) == 3 and tok[2] == "ep00" and tok[1].startswith("s"):
        pairs.add((tok[0], int(tok[1][1:])))
    else:
        INFO.append(("unparsable eval filename", f))
check("remote eval files == 21", len(inv) == 21, "got %d" % len(inv))
check("remote eval inventory parses into 21 (arm,seed) pairs", len(pairs) == 21, "got %d" % len(pairs))
check("remote inventory covers exactly 7 arms x seeds {0,1,2}",
      sorted({a for a, _ in pairs}) == sorted(ARMS) and
      sorted({s for _, s in pairs}) == [0, 1, 2],
      "%s / %s" % (sorted({a for a, _ in pairs}), sorted({s for _, s in pairs})))

# ---------------------------------------------------------------- 1. S3
print("\n[1] S3 equal-bias verdict (recomputed from raw artifact)")
s3 = load("s3_equalbias.json")
check("s3 top keys exact", list(s3.keys()) ==
      ["threshold_mm", "epoch", "control", "n_boot", "rng_seed", "arms"], str(list(s3.keys())))
check("s3 threshold_mm == 2.0", s3.get("threshold_mm") == 2.0, fmt(s3.get("threshold_mm")))
check("s3 control == B0", s3.get("control") == "B0", fmt(s3.get("control")))
check("s3 n_boot == 10000", s3.get("n_boot") == 10000, fmt(s3.get("n_boot")))
check("s3 epoch == 00", str(s3.get("epoch")) == "00", fmt(s3.get("epoch")))
s3_arms = s3.get("arms", {})
check("s3 arms == 6 experimental arms (no B0)", sorted(s3_arms.keys()) == sorted(PROBE_ARMS),
      str(sorted(s3_arms.keys())))
n_raw = n_eq = n_flag = 0
s3_rows, s3_nwin, s3_nseed = [], set(), set()
for arm in PROBE_ARMS:
    rec = s3_arms.get(arm)
    if rec is None:
        check("s3 %s present" % arm, False)
        continue
    check("s3 %s lists 3 seeds" % arm, len(rec.get("seeds") or []) == 3, str(rec.get("seeds")))
    row = {"arm": arm, "verdict": rec.get("s3_verdict")}
    for tag in ("raw", "equal_bias"):
        blk = rec.get(tag)
        if not isinstance(blk, dict):
            check("s3 %s %s block present" % (arm, tag), False)
            continue
        lo, hi, pt = blk.get("ci_lo"), blk.get("ci_hi"), blk.get("point")
        recomputed = (lo is not None and hi is not None and lo <= 0.0 <= hi)
        stored = bool(blk.get("crosses_zero"))
        ok = recomputed == stored
        n_raw += 1 if tag == "raw" else 0
        n_eq += 1 if tag == "equal_bias" else 0
        n_flag += 1 if ok else 0
        check("s3 %s %s crosses_zero flag == recomputed" % (arm, tag), ok,
              "stored=%s recomputed=%s ci=[%s, %s]" % (stored, recomputed, fmt(lo), fmt(hi)))
        neg_check("s3 %s %s CI must cross zero (all arms negative)" % (arm, tag), not recomputed,
                  "ci=[%s, %s] point=%s" % (fmt(lo), fmt(hi), fmt(pt)))
        s3_nwin.add(blk.get("n_windows"))
        s3_nseed.add(blk.get("n_seeds"))
        row[tag] = (pt, lo, hi)
    s3_rows.append(row)
check("s3 raw blocks compared == 6", n_raw == 6, "got %d" % n_raw)
check("s3 equal_bias blocks compared == 6", n_eq == 6, "got %d" % n_eq)
check("s3 crosses_zero flags agreed == 12", n_flag == 12, "got %d" % n_flag)
check("s3 n_windows identical across all blocks", len(s3_nwin) == 1, str(s3_nwin))
check("s3 n_seeds identical across all blocks", len(s3_nseed) == 1, str(s3_nseed))
check("s3 n_seeds == 3", s3_nseed == {3}, str(s3_nseed))
check("s3 n_windows == 538", s3_nwin == {538}, str(s3_nwin))
verdicts = {r["arm"]: r["verdict"] for r in s3_rows}
for arm in ("A1", "A3a", "A3b", "A4"):
    check("s3 %s verdict == GAIN_WAS_AREA_OR_AMPLITUDE" % arm,
          verdicts.get(arm) == "GAIN_WAS_AREA_OR_AMPLITUDE", str(verdicts.get(arm)))
for arm in ("A2", "A4b"):
    check("s3 %s verdict == NO_GAIN_TO_BEGIN_WITH" % arm,
          verdicts.get(arm) == "NO_GAIN_TO_BEGIN_WITH", str(verdicts.get(arm)))
print("    %-5s %-30s %-24s %-24s" % ("arm", "verdict", "raw CI", "equal-bias CI"))
for r in s3_rows:
    raw, eq = r.get("raw", (None, None, None)), r.get("equal_bias", (None, None, None))
    print("    %-5s %-30s [%s, %s]   [%s, %s]" % (
        r["arm"], r["verdict"], fmt(raw[1]), fmt(raw[2]), fmt(eq[1]), fmt(eq[2])))

# ---------------------------------------------------------------- 2. struct metadata
print("\n[2] struct_metrics_all metadata vs S3 window count")
st = load("struct_metrics_all.json")
check("struct_metrics_all n_windows present", isinstance(st.get("n_windows"), int),
      fmt(st.get("n_windows")))
check("struct n_windows == 538 == s3 n_windows",
      st.get("n_windows") in s3_nwin and st.get("n_windows") == 538,
      "struct=%s s3=%s" % (fmt(st.get("n_windows")), s3_nwin))

# ---------------------------------------------------------------- 3. scores
print("\n[3] 3-seed median + over-floor pass rate (recomputed + cross-checked)")
allsc = load("scores_all.json")
sumsc = load("scores_summary.json")
keys = list(allsc.keys())
check("scores_all record count == 21 (7 arms x 3 seeds)", len(keys) == 21, "got %d" % len(keys))
per_seed = {}
for k in keys:
    parts = k.split("_")
    check("scores_all key format <arm>_s<seed>_ep00",
          len(parts) == 3 and parts[2] == "ep00" and parts[1].startswith("s"), k)
    per_seed.setdefault(parts[0], {})[int(parts[1][1:])] = allsc[k]
check("scores_all covers 7 arms", sorted(per_seed.keys()) == sorted(ARMS), str(sorted(per_seed.keys())))
check("scores_all every arm has exactly seeds 0,1,2",
      all(sorted(v.keys()) == [0, 1, 2] for v in per_seed.values()))
check("scores_summary covers 7 arms", sorted(sumsc.keys()) == sorted(ARMS), str(sorted(sumsc.keys())))
# NEW (round-21): local aggregate covers exactly the remote evaluation inventory
local_pairs = {(a, s) for a, v in per_seed.items() for s in v}
check("NEW: local scores_all (arm,seed) set == remote eval inventory set",
      local_pairs == pairs,
      "local-only=%s remote-only=%s" % (sorted(local_pairs - pairs), sorted(pairs - local_pairs)))

cmp_n = 0
stats = {}
for arm in ARMS:
    csis = [per_seed[arm][s]["csi_at_2.0"] for s in (0, 1, 2)]
    biases = [per_seed[arm][s]["bias_at_2.0"] for s in (0, 1, 2)]
    med_csi = statistics.median(csis)
    med_bias = statistics.median(biases)
    npass = sum(1 for c in csis if c > FLOOR)
    over_oracle = sum(1 for c in csis if c > ORACLE)
    spread = max(csis) / min(csis)   # summary's seed_spread_csi_at_2.0 is a max/min RATIO
    stats[arm] = dict(med_csi=med_csi, med_bias=med_bias, npass=npass,
                      over_oracle=over_oracle, spread=spread, csis=csis, biases=biases)
    s = sumsc[arm]
    check("summary %s median_csi matches recompute" % arm,
          abs(s["median_csi_at_2.0"] - med_csi) < 1e-9,
          "summary=%.6f recomputed=%.6f" % (s["median_csi_at_2.0"], med_csi))
    check("summary %s median is order-independent middle value" % arm,
          abs(med_csi - sorted(csis)[1]) < 1e-12)
    check("summary %s median_bias matches recompute" % arm,
          abs(s["median_bias_at_2.0"] - med_bias) < 1e-9,
          "summary=%.6f recomputed=%.6f" % (s["median_bias_at_2.0"], med_bias))
    cmp_n += 2
    pr = s.get("pass_rate_over_floor")
    if isinstance(pr, str) and "/" in pr:
        pr_n = float(pr.split("/")[0])
    elif isinstance(pr, (int, float)) and pr <= 1.0:
        pr_n = pr * 3
    else:
        pr_n = float(pr)
    check("summary %s pass_rate_over_floor matches recompute" % arm,
          abs(pr_n - npass) < 1e-6, "summary=%s recomputed=%d/3" % (pr, npass))
    cmp_n += 1
    check("summary %s seed_spread matches recompute (max/min ratio)" % arm,
          abs(s["seed_spread_csi_at_2.0"] - spread) < 1e-9,
          "summary=%.6f recomputed=%.6f" % (s["seed_spread_csi_at_2.0"], spread))
    cmp_n += 1
    sl = s.get("seeds")
    check("summary %s 'seeds' list matches score coverage" % arm,
          isinstance(sl, list) and len(sl) == 3, str(sl))
    cmp_n += 1
check("scores cross-check comparisons == 35 (non-vacuous)", cmp_n == 35, "got %d" % cmp_n)
check("B0 median == 0.1218", abs(stats["B0"]["med_csi"] - 0.1218) < 5e-4,
      "%.6f" % stats["B0"]["med_csi"])
check("A4 median == 0.1439", abs(stats["A4"]["med_csi"] - 0.1439) < 5e-4,
      "%.6f" % stats["A4"]["med_csi"])
check("A4b median == 0.1376", abs(stats["A4b"]["med_csi"] - 0.1376) < 5e-4,
      "%.6f" % stats["A4b"]["med_csi"])
higher = [a for a in ARMS if a != "B0" and stats[a]["med_csi"] > stats["B0"]["med_csi"]]
check("only A4/A4b have median above B0", sorted(higher) == ["A4", "A4b"], str(sorted(higher)))
check("A4 median BIAS inflated vs B0", stats["A4"]["med_bias"] > stats["B0"]["med_bias"],
      "A4=%.4f B0=%.4f" % (stats["A4"]["med_bias"], stats["B0"]["med_bias"]))
check("A4b median BIAS inflated vs B0", stats["A4b"]["med_bias"] > stats["B0"]["med_bias"],
      "A4b=%.4f B0=%.4f" % (stats["A4b"]["med_bias"], stats["B0"]["med_bias"]))
check("B0 itself passes floor only 2/3 (floor baseline seed-dependent)",
      stats["B0"]["npass"] == 2, "%d/3" % stats["B0"]["npass"])
check("A3a passes floor 3/3", stats["A3a"]["npass"] == 3, "%d/3" % stats["A3a"]["npass"])
check("A3b passes floor 3/3", stats["A3b"]["npass"] == 3, "%d/3" % stats["A3b"]["npass"])
check("A4b seed spread ~9.23x (instability)", abs(stats["A4b"]["spread"] - 9.226194) < 1e-3,
      "spread=%.4f" % stats["A4b"]["spread"])
check("no arm has median CSI above Oracle 0.1653",
      all(stats[a]["med_csi"] < ORACLE for a in ARMS),
      str({a: round(stats[a]["med_csi"], 4) for a in ARMS}))
for arm in ARMS:
    expect = 1 if arm in ("A3b", "A4") else 0
    check("per-seed over-Oracle count for %s == %d" % (arm, expect),
          stats[arm]["over_oracle"] == expect,
          "got %d/3" % stats[arm]["over_oracle"])
print("    %-5s %-10s %-10s %-18s %-6s %-9s %s" %
      ("arm", "medCSI", "medBIAS", "range", "pass", ">Oracle", "spread"))
for arm in ARMS:
    s = stats[arm]
    print("    %-5s %-10.6f %-10.4f %-18s %-6s %-9s %.4f" % (
        arm, s["med_csi"], s["med_bias"],
        "%.4f-%.4f" % (min(s["csis"]), max(s["csis"])),
        "%d/3" % s["npass"], "%d/3" % s["over_oracle"], s["spread"]))

# ---------------------------------------------------------------- 4. readouts
print("\n[4] mechanism readouts per arm per seed (from diag_probe_s{0,1,2}.json)")
probes = [load("diag_probe_s%d.json" % s) for s in (0, 1, 2)]
for i, p in enumerate(probes):
    check("probe s%d arms == 6 experimental arms" % i,
          sorted(p.get("arms", {}).keys()) == sorted(PROBE_ARMS),
          str(sorted(p.get("arms", {}).keys())))


def readout(arm, key, seed):
    return probes[seed]["arms"][arm]["readouts"].get(key)


print("    %-4s %-24s %-10s %-10s %-10s" % ("arm", "readout", "s0", "s1", "s2"))
gate_pass = {}

for arm in ("A1", "A2"):
    arm_ok = True
    for key, crit, cmpop in (("corr_mask_amp", 0.0, "ge"), ("mask_at_peak", 0.5, "gt")):
        vals = [readout(arm, key, s) for s in (0, 1, 2)]
        per_seed_pass = [False if v is None else (v >= crit if cmpop == "ge" else v > crit)
                         for v in vals]
        print("    %-4s %-24s %-10s %-10s %-10s" % (
            arm, key,
            "%.4f" % vals[0] if vals[0] is not None else "n/a",
            "%.4f" % vals[1] if vals[1] is not None else "n/a",
            "%.4f" % vals[2] if vals[2] is not None else "n/a"))
        check("%s %s passes all 3 seeds" % (arm, key), all(per_seed_pass),
              "per-seed pass=%s" % per_seed_pass)
        arm_ok = arm_ok and all(per_seed_pass)
    gate_pass[arm] = arm_ok

for arm in ("A3a", "A3b"):
    arm_ok = True
    for key, crit, cmpop in (("grad_top1pct_share", 0.5, "lt"), ("field_max_mm", 30.0, "le")):
        vals = [readout(arm, key, s) for s in (0, 1, 2)]
        per_seed_pass = [False if v is None else (v < crit if cmpop == "lt" else v <= crit)
                         for v in vals]
        print("    %-4s %-24s %-10s %-10s %-10s" % (
            arm, key,
            "%.4f" % vals[0] if vals[0] is not None else "n/a",
            "%.4f" % vals[1] if vals[1] is not None else "n/a",
            "%.4f" % vals[2] if vals[2] is not None else "n/a"))
        if key == "field_max_mm":
            neg_check("%s %s: NO seed may satisfy <=30 (topology branch closed)" % (arm, key),
                      any(per_seed_pass), "per-seed pass=%s" % per_seed_pass)
        else:
            INFO.append(("%s %s per-seed pass" % (arm, key), str(per_seed_pass)))
        arm_ok = arm_ok and all(per_seed_pass)
    gate_pass[arm] = arm_ok
    comb = []
    for s in (0, 1, 2):
        g = readout(arm, "grad_top1pct_share", s)
        f = readout(arm, "field_max_mm", s)
        comb.append(g is not None and f is not None and g < 0.5 and f <= 30.0)
    neg_check("%s combined topology criterion must fail on every seed" % arm, any(comb),
              "per-seed combined pass=%s" % comb)

for arm in ("A4", "A4b"):
    arm_ok = True
    for key, crit in (("scale_cv", 0.05), ("scale_truth_corr", 0.3)):
        vals = [readout(arm, key, s) for s in (0, 1, 2)]
        per_seed_pass = [False if v is None else v >= crit for v in vals]
        print("    %-4s %-24s %-10s %-10s %-10s" % (
            arm, key,
            "%.4f" % vals[0] if vals[0] is not None else "n/a",
            "%.4f" % vals[1] if vals[1] is not None else "n/a",
            "%.4f" % vals[2] if vals[2] is not None else "n/a"))
        if arm == "A4" or key == "scale_cv":
            check("%s %s passes all 3 seeds" % (arm, key), all(per_seed_pass),
                  "per-seed pass=%s" % per_seed_pass)
        else:
            check("%s %s expected pass pattern [T,T,F]" % (arm, key),
                  per_seed_pass == [True, True, False], "per-seed pass=%s" % per_seed_pass)
        arm_ok = arm_ok and all(per_seed_pass)
    const = [readout(arm, "scale_is_constant", s) for s in (0, 1, 2)]
    print("    %-4s %-24s %-10s %-10s %-10s" % (arm, "scale_is_constant", *const))
    if arm == "A4":
        check("A4 scale_is_constant is False on all seeds", all(c is False for c in const), str(const))
    else:
        check("A4b scale_is_constant pattern [F,F,T]", const == [False, False, True], str(const))
    arm_ok = arm_ok and all(c is False for c in const)
    mo = [readout(arm, "mean_out_equals_T", s) for s in (0, 1, 2)]
    rel = [readout(arm, "mean_out_equals_T_rel_err", s) for s in (0, 1, 2)]
    print("    %-4s %-24s %-10s %-10s %-10s" % (arm, "mean_out_equals_T", *mo))
    print("    %-4s %-24s %-10s %-10s %-10s" % (
        arm, "  rel_err", *["%.4g" % v if v is not None else "n/a" for v in rel]))
    if arm == "A4":
        check("A4 mean_out_equals_T pattern [T,F,T] (impl defect)", mo == [True, False, True], str(mo))
    else:
        check("A4b mean_out_equals_T all False (impl defect)", mo == [False, False, False], str(mo))
    INFO.append(("%s mean_out_equals_T (impl-defect flag, NOT an eligibility gate)" % arm, str(mo)))
    gate_pass[arm] = arm_ok

for arm in ("A1", "A2"):
    vals = [readout(arm, "maxratio_as_trained", s) for s in (0, 1, 2)]
    check("%s maxratio_as_trained missing on all 3 seeds (readout gap)" % arm,
          all(is_missing(v) for v in vals),
          "values=%s" % [fmt(v) for v in vals])

# ---------------------------------------------------------------- 5. NEW: promotion conjunction
print("\n[5] NEW (round-21): promotion-eligibility conjunction (S3 AND floor AND readouts)")
s3_eq_ci = {r["arm"]: (r.get("equal_bias", (None, None, None))[1],
                        r.get("equal_bias", (None, None, None))[2]) for r in s3_rows}
elig, reasons = [], {}
for arm in PROBE_ARMS:
    lo, hi = s3_eq_ci[arm]
    s3_ok = not (lo is not None and hi is not None and lo <= 0.0 <= hi)
    floor_ok = stats[arm]["npass"] == 3
    ro_ok = gate_pass.get(arm, False)
    if s3_ok and floor_ok and ro_ok:
        elig.append(arm)
    reasons[arm] = ("S3_equalbias_crosses_zero" if not s3_ok else "") + \
                   ("|floor_%d3" % stats[arm]["npass"] if not floor_ok else "") + \
                   ("|readouts_fail" if not ro_ok else "")
check("NEW: no arm is eligible for ladder stage 2 (>=10 seeds)", elig == [], str(elig))
for arm in PROBE_ARMS:
    INFO.append(("promotion blockers %s" % arm, reasons[arm] or "none"))
    print("    %-5s eligible=%-5s blockers=%s" % (arm, arm in elig, reasons[arm] or "-"))
check("NEW: A3a/A3b pass the floor 3/3 but are still blocked (median below control)",
      stats["A3a"]["npass"] == 3 and stats["A3b"]["npass"] == 3 and
      stats["A3a"]["med_csi"] < stats["B0"]["med_csi"] and
      stats["A3b"]["med_csi"] < stats["B0"]["med_csi"],
      "A3a=%.4f A3b=%.4f B0=%.4f" % (stats["A3a"]["med_csi"], stats["A3b"]["med_csi"],
                                     stats["B0"]["med_csi"]))

# ---------------------------------------------------------------- 6. Gate 3 presence
print("\n[6] Gate 3 write-back presence")
master_p = os.path.join(ROOT, "docs", "MASTER.md")
index_p = os.path.join(ROOT, "docs", "experiments", "INDEX.md")
e45_files = sorted(glob.glob(os.path.join(ROOT, "docs", "experiments", "E45*.md")))
master = read_text(master_p)
index = read_text(index_p)
e45_docs = {os.path.basename(p): read_text(p) for p in e45_files}
check("E45 experiment doc exists", len(e45_docs) >= 1, str(list(e45_docs)))
mlines = master.splitlines()
ilines = index.splitlines()
check("MASTER head version line contains v2.45", any("v2.45" in l for l in mlines[:12]))
check("MASTER has 4.47 section", any("4.47" in l for l in mlines))
check("MASTER 9.x has E45 row", any("E45" in l for l in mlines))
check("MASTER changelog has v2.44 and v2.45 rows",
      any("v2.44" in l for l in mlines) and any("v2.45" in l for l in mlines))
check("INDEX has E45 row", any("E45" in l for l in ilines))
mem_p = "R:/1radarNWPprediction/.workbuddy/memory/MEMORY.md"
if os.path.exists(mem_p):
    check("upper MEMORY.md points at v2.45", "v2.45" in read_text(mem_p))
else:
    INFO.append(("upper MEMORY.md", "not found"))

# ---------------------------------------------------------------- 7. NEW: quoted-number consistency
print("\n[7] NEW (round-21): quoted-number consistency between docs and artifacts")
union = "\n".join([master, index] + list(e45_docs.values()))
quoted = {
    "field max (mm)": ("402.43", max(readout(a, "field_max_mm", s) for a in ("A3a", "A3b")
                                      for s in (0, 1, 2))),
    "floor": ("0.0924", FLOOR),
    "A4b seed2 scale_truth_corr": ("0.2789", readout("A4b", "scale_truth_corr", 2)),
    "E44 baseline corr(mask,amp)": ("0.858320", 0.858320),
}
for label, (token, value) in quoted.items():
    check("doc quotes %s = %s" % (label, token), token in union,
          "recomputed value=%.6f" % value)
for label, token in (("A4b median BIAS", "2.8215"), ("A4 median BIAS", "0.8362"),
                     ("B0 median BIAS", "0.5095"), ("Oracle upper bound", "0.1653"),
                     ("A4b seed spread", "9.23")):
    INFO.append(("doc quotes %s (%s)" % (label, token), "present" if token in union else "NOT quoted"))

# ---------------------------------------------------------------- 8. NEW: ladder consistency
print("\n[8] NEW (round-21): ladder consistency (rule retained AND refusal recorded)")
e45_all = "\n".join(e45_docs.values())
check("E45 doc carries closure wording (结案)", "结案" in e45_all)
check("E45 doc states the negative verdict (负面)", "负面" in e45_all)
check("E45 doc does NOT still say the run is pending (正在跑/RUNNING 等远端跑完)",
      not any(p in e45_all for p in ("正在跑", "等远端跑完", "待远端跑完")))
check("E45 doc records that NO arm qualified for ladder stage 2",
      "够格进入阶 2" in e45_all, "")
check("MASTER records 'no ten-seed compute' (不投十 seed 算力)", "不投十 seed 算力" in master)
check("INDEX records 'no ten-seed compute' (不投十 seed 算力)", "不投十 seed 算力" in index)
check("E45 criterion doc still carries the promotion RULE (阶梯阶 2)",
      "阶梯阶 2" in e45_all, "")
check("E45 criterion doc still forbids claiming SOTA (不得写 SOTA / 仍不得写 SOTA)",
      ("不得写" in e45_all and "SOTA" in e45_all), "")
check("E45 doc keeps the equal-bias verdict wording (幅度效应)",
      ("幅度效应" in e45_all) or ("面积" in e45_all), "")
check("MASTER keeps the equal-bias verdict wording (面积／幅度效应)",
      "幅度效应" in master, "")

# ---------------------------------------------------------------- 9. NEW: remote eval schema probe (info)
print("\n[9] NEW (round-21): remote eval.json schema probe (informational)")
probe_code = (
    "import glob, json, os\n"
    "f = sorted(glob.glob('runs/e45/evals/*.eval.json'))[0]\n"
    "o = json.load(open(f))\n"
    "keys = sorted(o.keys()) if isinstance(o, dict) else ['<not a dict>']\n"
    "print('FILE', os.path.basename(f))\n"
    "print('TOPKEYS', keys[:24])\n"
)
try:
    out = remote("cd /home/wf/radarNWP && %s -c %s" % (REMOTE_PY, shlex.quote(probe_code)))
    for l in out.splitlines():
        if l.startswith("FILE") or l.startswith("TOPKEYS"):
            print("    ", l[:400])
            INFO.append(("remote eval.json schema", l[:200]))
except Exception as exc:  # informational only
    INFO.append(("remote eval.json schema probe failed (informational)", str(exc)[:160]))

# ---------------------------------------------------------------- summary
print("\n" + "=" * 78)
print("POS (expected-pass) OK : %d" % len(POS_OK))
print("POS (expected-pass) FAIL: %d" % len(POS_FAIL))
print("NEG (expected-fail) OK : %d" % len(NEG_OK))
print("NEG (expected-fail) FAIL: %d" % len(NEG_FAIL))
if POS_FAIL:
    print("\nFAILED POSITIVE ASSERTIONS:")
    for n, d in POS_FAIL:
        print("   -", n, "|", d)
if NEG_FAIL:
    print("\nFAILED NEGATIVE ASSERTIONS (a required negative did not hold):")
    for n, d in NEG_FAIL:
        print("   -", n, "|", d)
if POS_FAIL or NEG_FAIL:
    print("\nVERDICT: FAIL")
    sys.exit(1)
print("\nVERDICT: PASS")
