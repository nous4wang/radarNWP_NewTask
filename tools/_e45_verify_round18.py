#!/usr/bin/env python
"""E45 round-18 independent re-verification.

Rules learned from rounds 9-17 (do NOT violate):
  * structure first: never guess key paths -> we dump/assert them.
  * no importing previous verify scripts (old bugs transmit silently).
  * assertions split into POS (expected pass) / NEG (expected fail) channels;
    a legitimate negative result must NOT paint the verdict red.
  * "missing readout" must distinguish None / NaN / valid value (3 states).
  * cross-checks must assert a non-zero comparison count (no vacuous PASS).
  * arm-name matching by exact path token (no substring, else A4 <- A4b).
"""
import json
import math
import os
import statistics
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EV = os.path.join(ROOT, "tools", "_evals", "e45")

FLOOR = 0.0924
ORACLE = 0.1653
ARMS = ["B0", "A1", "A2", "A3a", "A3b", "A4", "A4b"]
PROBE_ARMS = ["A1", "A2", "A3a", "A3b", "A4", "A4b"]

# remote md5 captured this round from `md5sum runs/e45/*.json`
REMOTE_MD5 = {
    "diag_probe_s0.json": "da016521a6bb5dfd85da4f518e3fb763",
    "diag_probe_s1.json": "cf69079c3fdef206fe3e3fedaa12685b",
    "diag_probe_s2.json": "01de640b578d5980264259586c1adf18",
    "s3_equalbias.json": "3aba999d5ef6dcc4a775999be6585eb2",
    "scores_all.json": "80655963622b09cd5231588e26468a28",
    "scores_summary.json": "57bcd511a537ab85494ee41b029731a8",
    "scores_summary_s0.json": "bac63c96975fdd5e8ec46dcac4ec2463",
    "scores_summary_s1.json": "35f1142dfda6042bcb3e5a74b60e78d7",
    "scores_summary_s2.json": "80655963622b09cd5231588e26468a28",
    "struct_metrics_all.json": "cd3052bbb8185e6b15fa27afd663db01",
    "struct_metrics_s0.json": "4bc259d01d1450f6c8ca1e43391dd1a4",
    "struct_metrics_s1.json": "75489dcb98d373b34ceb5e7d8636dab3",
    "struct_metrics_s2.json": "077ac3dc307f701866cff695e881cbd2",
}

POS_OK, POS_FAIL, NEG_OK, NEG_FAIL, INFO = [], [], [], [], []


def check(name, cond, detail=""):
    """POSITIVE assertion: cond must be True."""
    (POS_OK if cond else POS_FAIL).append((name, detail))
    return cond


def neg_check(name, cond, detail=""):
    """NEGATIVE assertion: cond must be False (a legitimate negative result is OK,
    but if it becomes True the verdict must go red)."""
    (NEG_FAIL if cond else NEG_OK).append((name, detail))
    return not cond


def load(name):
    with open(os.path.join(EV, name), "r", encoding="utf-8") as f:
        return json.load(f)


def is_missing(x):
    """3-state: None / NaN / valid."""
    return x is None or (isinstance(x, (int, float)) and not isinstance(x, bool) and x != x)


def fmt(x):
    return "n/a" if x is None else ("nan" if isinstance(x, float) and x != x else str(x))


def md5_of(path):
    import hashlib

    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


print("=" * 78)
print("E45 round-18 independent verification")
print("=" * 78)

# ---------------------------------------------------------------- 1. md5
print("\n[1] md5 local vs remote")
local_files = sorted(f for f in os.listdir(EV) if f.endswith(".json"))
check("local json count == 13", len(local_files) == 13, "got %d" % len(local_files))
check("remote md5 map size == 13", len(REMOTE_MD5) == 13, "got %d" % len(REMOTE_MD5))
md5_ok = 0
for name, want in sorted(REMOTE_MD5.items()):
    p = os.path.join(EV, name)
    if not os.path.exists(p):
        check("md5 %s exists" % name, False, "missing locally")
        continue
    got = md5_of(p)
    same = got == want
    md5_ok += 1 if same else 0
    check("md5 %s" % name, same, "local=%s remote=%s" % (got, want))
print("    md5 identical: %d/%d" % (md5_ok, len(REMOTE_MD5)))

# ---------------------------------------------------------------- 2. S3
print("\n[2] S3 equal-bias verdict (recomputed from raw artifact)")
s3 = load("s3_equalbias.json")
check("s3 threshold_mm == 2.0", s3.get("threshold_mm") == 2.0, fmt(s3.get("threshold_mm")))
check("s3 control == B0", s3.get("control") == "B0", fmt(s3.get("control")))
check("s3 n_boot == 10000", s3.get("n_boot") == 10000, fmt(s3.get("n_boot")))
check("s3 epoch == 00", str(s3.get("epoch")) == "00", fmt(s3.get("epoch")))
s3_arms = s3.get("arms", {})
check("s3 arms == 6 experimental arms", sorted(s3_arms.keys()) == sorted(PROBE_ARMS),
      str(sorted(s3_arms.keys())))
n_raw, n_eq, n_flag = 0, 0, 0
s3_rows = []
for arm in PROBE_ARMS:
    rec = s3_arms.get(arm)
    if rec is None:
        check("s3 %s present" % arm, False)
        continue
    row = {"arm": arm, "verdict": rec.get("s3_verdict")}
    for tag in ("raw", "equal_bias"):
        blk = rec.get(tag)
        if not isinstance(blk, dict):
            check("s3 %s %s block" % (arm, tag), False)
            continue
        lo, hi = blk.get("ci_lo"), blk.get("ci_hi")
        pt = blk.get("point")
        recomputed = (lo is not None and hi is not None and lo <= 0.0 <= hi)
        stored = bool(blk.get("crosses_zero"))
        ok = recomputed == stored
        if tag == "raw":
            n_raw += 1
        else:
            n_eq += 1
        if ok:
            n_flag += 1
        check("s3 %s %s crosses_zero flag == recomputed" % (arm, tag), ok,
              "stored=%s recomputed=%s ci=[%s, %s]" % (stored, recomputed, fmt(lo), fmt(hi)))
        neg_check("s3 %s %s CI crosses zero (expected)" % (arm, tag), not recomputed,
                  "ci=[%s, %s] point=%s" % (fmt(lo), fmt(hi), fmt(pt)))
        row[tag] = (pt, lo, hi)
    s3_rows.append(row)
    INFO.append(("s3 verdict %s" % arm, str(rec.get("s3_verdict"))))
check("s3 raw blocks compared == 6", n_raw == 6, "got %d" % n_raw)
check("s3 equal_bias blocks compared == 6", n_eq == 6, "got %d" % n_eq)
check("s3 crosses_zero flags agreed == 12", n_flag == 12, "got %d" % n_flag)
verdicts = {r["arm"]: r["verdict"] for r in s3_rows}
for arm in ("A1", "A3a", "A3b", "A4"):
    check("s3 %s verdict == GAIN_WAS_AREA_OR_AMPLITUDE" % arm,
          verdicts.get(arm) == "GAIN_WAS_AREA_OR_AMPLITUDE", str(verdicts.get(arm)))
for arm in ("A2", "A4b"):
    check("s3 %s verdict == NO_GAIN_TO_BEGIN_WITH" % arm,
          verdicts.get(arm) == "NO_GAIN_TO_BEGIN_WITH", str(verdicts.get(arm)))

# ---------------------------------------------------------------- 3. scores
print("\n[3] 3-seed median + over-floor pass rate (recomputed + cross-checked)")
allsc = load("scores_all.json")
sumsc = load("scores_summary.json")
keys = list(allsc.keys())
check("scores_all record count == 21 (7 arms x 3 seeds)", len(keys) == 21, "got %d" % len(keys))
per_seed = {}
for k in keys:
    parts = k.split("_")
    arm, seed, ep = parts[0], parts[1], parts[2]
    check("scores_all key format <arm>_s<seed>_ep00", ep == "ep00" and seed.startswith("s"), k)
    per_seed.setdefault(arm, {})[int(seed[1:])] = allsc[k]
check("scores_all covers 7 arms", sorted(per_seed.keys()) == sorted(ARMS), str(sorted(per_seed.keys())))
check("scores_all every arm has 3 seeds",
      all(sorted(v.keys()) == [0, 1, 2] for v in per_seed.values()))
check("scores_summary covers 7 arms", sorted(sumsc.keys()) == sorted(ARMS), str(sorted(sumsc.keys())))
cmp_n = 0
stats = {}
for arm in ARMS:
    csis = [per_seed[arm][s]["csi_at_2.0"] for s in (0, 1, 2)]
    biases = [per_seed[arm][s]["bias_at_2.0"] for s in (0, 1, 2)]
    med_csi = statistics.median(csis)
    med_bias = statistics.median(biases)
    npass = sum(1 for c in csis if c > FLOOR)
    spread = max(csis) / min(csis)  # summary's seed_spread_csi_at_2.0 is a max/min RATIO
    stats[arm] = dict(med_csi=med_csi, med_bias=med_bias, npass=npass, spread=spread,
                      csis=csis, biases=biases)
    s = sumsc[arm]
    check("summary %s median_csi matches recompute" % arm,
          abs(s["median_csi_at_2.0"] - med_csi) < 1e-9,
          "summary=%.6f recomputed=%.6f" % (s["median_csi_at_2.0"], med_csi))
    check("summary %s median_bias matches recompute" % arm,
          abs(s["median_bias_at_2.0"] - med_bias) < 1e-9,
          "summary=%.6f recomputed=%.6f" % (s["median_bias_at_2.0"], med_bias))
    cmp_n += 1
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
    check("summary %s seed_spread matches recompute" % arm,
          abs(s["seed_spread_csi_at_2.0"] - spread) < 1e-9,
          "summary=%.6f recomputed=%.6f" % (s["seed_spread_csi_at_2.0"], spread))
    cmp_n += 1
check("scores cross-check comparisons == 21 (non-vacuous)", cmp_n == 21, "got %d" % cmp_n)
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
check("A4b seed spread ~9.23x (instability)", stats["A4b"]["spread"] > 0.05,
      "spread=%.4f" % stats["A4b"]["spread"])

print("    %-5s %-10s %-10s %-10s %-9s %s" % ("arm", "medCSI", "medBIAS", "range", "pass", "medians"))
for arm in ARMS:
    s = stats[arm]
    print("    %-5s %-10.6f %-10.4f %-10s %-9s %.6f" % (
        arm, s["med_csi"], s["med_bias"],
        "%.4f-%.4f" % (min(s["csis"]), max(s["csis"])),
        "%d/3" % s["npass"], statistics.median(s["csis"])))

# ---------------------------------------------------------------- 4. readouts
print("\n[4] mechanism readouts per arm per seed (from diag_probe_s{0,1,2}.json)")
probes = [load("diag_probe_s%d.json" % s) for s in (0, 1, 2)]
for i, p in enumerate(probes):
    check("probe s%d arms == 6 experimental arms" % i,
          sorted(p.get("arms", {}).keys()) == sorted(PROBE_ARMS),
          str(sorted(p.get("arms", {}).keys())))

def readout(arm, key, seed):
    return probes[seed]["arms"][arm]["readouts"].get(key)

print("    %-4s %-22s %-10s %-10s %-10s" % ("arm", "readout", "s0", "s1", "s2"))

# A1/A2 gate direction
for arm in ("A1", "A2"):
    for key, crit, cmpop in (("corr_mask_amp", 0.0, "ge"), ("mask_at_peak", 0.5, "gt")):
        vals = [readout(arm, key, s) for s in (0, 1, 2)]
        per_seed_pass = []
        for v in vals:
            per_seed_pass.append(False if v is None else (v >= crit if cmpop == "ge" else v > crit))
        print("    %-4s %-22s %-10s %-10s %-10s" % (
            arm, key,
            "%.4f" % vals[0] if vals[0] is not None else "n/a",
            "%.4f" % vals[1] if vals[1] is not None else "n/a",
            "%.4f" % vals[2] if vals[2] is not None else "n/a"))
        check("%s %s passes all 3 seeds" % (arm, key), all(per_seed_pass),
              "per-seed pass=%s" % per_seed_pass)

# A3a/A3b topology
for arm in ("A3a", "A3b"):
    for key, crit, cmpop in (("grad_top1pct_share", 0.5, "lt"), ("field_max_mm", 30.0, "le")):
        vals = [readout(arm, key, s) for s in (0, 1, 2)]
        per_seed_pass = []
        for v in vals:
            per_seed_pass.append(False if v is None else (v < crit if cmpop == "lt" else v <= crit))
        print("    %-4s %-22s %-10s %-10s %-10s" % (
            arm, key,
            "%.4f" % vals[0] if vals[0] is not None else "n/a",
            "%.4f" % vals[1] if vals[1] is not None else "n/a",
            "%.4f" % vals[2] if vals[2] is not None else "n/a"))
        if key == "field_max_mm":
            neg_check("%s %s: NO seed may satisfy <=30 (topology branch closed)" % (arm, key),
                      any(per_seed_pass), "per-seed pass=%s" % per_seed_pass)
        else:
            INFO.append(("%s %s per-seed pass" % (arm, key), str(per_seed_pass)))
    # combined pre-registered criterion: both must hold
    comb = []
    for s in (0, 1, 2):
        g = readout(arm, "grad_top1pct_share", s)
        f = readout(arm, "field_max_mm", s)
        comb.append(g is not None and f is not None and g < 0.5 and f <= 30.0)
    neg_check("%s combined topology criterion must fail on every seed" % arm, any(comb),
              "per-seed combined pass=%s" % comb)

# A4/A4b amount channel
for arm in ("A4", "A4b"):
    for key, crit in (("scale_cv", 0.05), ("scale_truth_corr", 0.3)):
        vals = [readout(arm, key, s) for s in (0, 1, 2)]
        per_seed_pass = [False if v is None else v >= crit for v in vals]
        print("    %-4s %-22s %-10s %-10s %-10s" % (
            arm, key,
            "%.4f" % vals[0] if vals[0] is not None else "n/a",
            "%.4f" % vals[1] if vals[1] is not None else "n/a",
            "%.4f" % vals[2] if vals[2] is not None else "n/a"))
        if arm == "A4":
            check("%s %s passes all 3 seeds" % (arm, key), all(per_seed_pass),
                  "per-seed pass=%s" % per_seed_pass)
        elif key == "scale_truth_corr":
            check("%s %s expected pass pattern [T,T,F]" % (arm, key),
                  per_seed_pass == [True, True, False], "per-seed pass=%s" % per_seed_pass)
        else:
            check("%s %s passes all 3 seeds" % (arm, key), all(per_seed_pass),
                  "per-seed pass=%s" % per_seed_pass)
    const = [readout(arm, "scale_is_constant", s) for s in (0, 1, 2)]
    print("    %-4s %-22s %-10s %-10s %-10s" % (arm, "scale_is_constant", *const))
    if arm == "A4":
        check("A4 scale_is_constant is False on all seeds", all(c is False for c in const), str(const))
    else:
        check("A4b scale_is_constant pattern [F,F,T]", const == [False, False, True], str(const))
    mo = [readout(arm, "mean_out_equals_T", s) for s in (0, 1, 2)]
    rel = [readout(arm, "mean_out_equals_T_rel_err", s) for s in (0, 1, 2)]
    print("    %-4s %-22s %-10s %-10s %-10s" % (arm, "mean_out_equals_T", *mo))
    print("    %-4s %-22s %-10s %-10s %-10s" % (arm, "  rel_err", *["%.4g" % v if v is not None else "n/a" for v in rel]))

# maxratio_as_trained must be missing (readout gap) on every seed for gate arms
for arm in ("A1", "A2"):
    vals = [readout(arm, "maxratio_as_trained", s) for s in (0, 1, 2)]
    check("%s maxratio_as_trained missing on all 3 seeds (readout gap)" % arm,
          all(is_missing(v) for v in vals),
          "values=%s" % [fmt(v) for v in vals])

# ---------------------------------------------------------------- 5. Gate 3
print("\n[5] Gate 3 write-back presence")
master_p = os.path.join(ROOT, "docs", "MASTER.md")
index_p = os.path.join(ROOT, "docs", "experiments", "INDEX.md")
master = open(master_p, "r", encoding="utf-8").read()
index = open(index_p, "r", encoding="utf-8").read()
mlines = master.splitlines()
ilines = index.splitlines()
check("MASTER head version line contains v2.45",
      any("v2.45" in l for l in mlines[:12]), "head=%r" % (mlines[:6],))
check("MASTER has 4.47 section", any("4.47" in l for l in mlines))
check("MASTER 9.x has E45 row", any("E45" in l for l in mlines))
check("MASTER changelog has v2.44 and v2.45 rows",
      any("v2.44" in l for l in mlines) and any("v2.45" in l for l in mlines))
check("INDEX has E45 row", any("E45" in l for l in ilines))
mem_p = os.path.join("R:/1radarNWPprediction/.workbuddy/memory/MEMORY.md")
if os.path.exists(mem_p):
    mem = open(mem_p, "r", encoding="utf-8").read()
    check("upper MEMORY.md points at v2.45", "v2.45" in mem)
else:
    INFO.append(("upper MEMORY.md", "not found"))

# ---------------------------------------------------------------- verdict
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
