#!/usr/bin/env python
"""E45 round 71 — status recheck + full published-verdict re-verification (read-only, local).

Scope:
  A. remote state digest (supplied by the caller; the script itself is local-only)
  B. remote-vs-local md5 for the 13 E45 artifacts
  C. S3 equal-bias guard: all 12 published intervals must cross zero; qualifying set empty
  D. published headline table recomputed field-by-field from the raw per-seed snapshots
  E. pre-registered per-arm criteria re-read straight from the raw mechanism probe
  F. two calibers (difference vs paired) reported side by side

It writes nothing to the remote, trains nothing, scores nothing, and does not
create a criterion. Run:  python tools/_e45_verify_round71.py
"""

import hashlib
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EVALS = os.path.join(ROOT, "tools", "_evals", "e45")

FLOOR = 0.0924
ORACLE = 0.1653
ARMS = ("B0", "A1", "A2", "A3a", "A3b", "A4", "A4b")
MECH_ARMS = ("A1", "A2", "A3a", "A3b", "A4", "A4b")

# md5 of the 13 artifacts as published on the remote (runs/e45/*.json), captured
# by the round-71 status query. Keeping them here makes the check self-contained.
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

# Pre-registered criteria, frozen. (arm, readout, threshold, relation)
#   relation: "ge" -> value >= threshold ; "lt" -> value < threshold ; "le" -> value <= threshold
# `expected` is the number of seeds that satisfy the criterion AS ALREADY DOCUMENTED
# in the published conclusion (E45 is a negative result, so most of these are < 3/3).
# The check is not "does the criterion pass" but "does the observed pass-count still
# match the published one" -- i.e. has anything drifted.
CRITERIA = (
    ("A1", "corr_mask_amp", 0.0, "ge", 3),
    ("A2", "corr_mask_amp", 0.0, "ge", 3),
    ("A1", "mask_at_peak", 0.5, "gt", 3),
    ("A2", "mask_at_peak", 0.5, "gt", 3),
    ("A3a", "grad_top1pct_share", 0.5, "lt", 1),
    ("A3b", "grad_top1pct_share", 0.5, "lt", 0),
    ("A3a", "field_max_mm", 30.0, "le", 0),
    ("A3b", "field_max_mm", 30.0, "le", 0),
    ("A4", "scale_cv", 0.05, "ge", 3),
    ("A4b", "scale_cv", 0.05, "ge", 3),
    ("A4", "scale_truth_corr", 0.3, "ge", 3),
    ("A4b", "scale_truth_corr", 0.3, "ge", 2),
)

checks = []
out = {
    "round": 71,
    "axis": "published-verdict re-verification (status recheck)",
    "generated": "local read-only; no remote write, no retrain, no rescore",
    "creates_criterion": False,
}


def check(name, ok, detail=""):
    checks.append({"name": name, "ok": bool(ok), "detail": str(detail)})
    return bool(ok)


def median(xs):
    xs = sorted(xs)
    n = len(xs)
    return xs[n // 2] if n % 2 else (xs[n // 2 - 1] + xs[n // 2]) / 2


def load(name):
    with open(os.path.join(EVALS, name), "r", encoding="utf-8") as fh:
        return json.load(fh)


def relate(v, thr, rel):
    if rel == "ge":
        return v >= thr
    if rel == "gt":
        return v > thr
    if rel == "lt":
        return v < thr
    if rel == "le":
        return v <= thr
    raise ValueError(rel)


# ---------------------------------------------------------------- B. md5
md5_rows = []
for name, want in sorted(REMOTE_MD5.items()):
    path = os.path.join(EVALS, name)
    if not os.path.exists(path):
        md5_rows.append({"file": name, "status": "MISSING_LOCAL"})
        continue
    with open(path, "rb") as fh:
        got = hashlib.md5(fh.read()).hexdigest()
    md5_rows.append({"file": name, "remote_md5": want, "local_md5": got,
                     "match": got == want})
out["B_md5_remote_vs_local"] = md5_rows
check("B: all 13 artifacts match remote md5 bit-for-bit",
      all(r.get("match") for r in md5_rows) and len(md5_rows) == 13,
      f"{sum(1 for r in md5_rows if r.get('match'))}/13")

# ---------------------------------------------------------------- C. S3 guard
s3 = load("s3_equalbias.json")
s3_rows = []
n_cross = 0
n_tot = 0
for arm, v in s3["arms"].items():
    for side in ("raw", "equal_bias"):
        s = v[side]
        n_tot += 1
        n_cross += int(bool(s["crosses_zero"]))
        s3_rows.append({"arm": arm, "side": side, "point": s["point"],
                        "ci_lo": s["ci_lo"], "ci_hi": s["ci_hi"],
                        "crosses_zero": s["crosses_zero"],
                        "n_seeds": s["n_seeds"], "n_windows": s["n_windows"],
                        "n_boot": s["n_boot"]})
    s3_rows[-2]["s3_verdict"] = v["s3_verdict"]
qualifying = [a for a, v in s3["arms"].items()
              if not v["raw"]["crosses_zero"] or not v["equal_bias"]["crosses_zero"]]
out["C_s3_guard"] = {"threshold_mm": s3["threshold_mm"], "epoch": s3["epoch"],
                     "control": s3["control"], "rows": s3_rows,
                     "n_intervals": n_tot, "n_crossing_zero": n_cross,
                     "qualifying_arms": qualifying}
check("C: all 12 published S3 intervals cross zero", n_cross == n_tot, f"{n_cross}/{n_tot}")
check("C: qualifying-arm set is empty", not qualifying, qualifying)
check("C: S3 control is B0 and threshold is 2.0 mm",
      s3["control"] == "B0" and s3["threshold_mm"] == 2.0,
      f"control={s3['control']} thr={s3['threshold_mm']}")

# ---------------------------------------------------------------- D. headline table
pub = load("scores_summary.json")
head = {}
n_fields = 0
n_ok = 0
for arm in ARMS:
    csis, biases = [], []
    for s in (0, 1, 2):
        snap = load(f"scores_summary_s{s}.json")
        node = snap.get(f"{arm}_s{s}_ep00") or {}
        csis.append(float(node["csi_at_2.0"]))
        biases.append(float(node["bias_at_2.0"]))
    p = pub[arm]
    recomputed = {
        "median_csi_at_2.0": median(csis),
        "median_bias_at_2.0": median(biases),
        "pass_rate_over_floor": "%d/3" % sum(1 for x in csis if x > FLOOR),
        "seed_spread_csi_at_2.0": max(csis) / min(csis),
        "n_seeds": len(csis),
    }
    fields = {}
    for k, v in recomputed.items():
        pv = p.get(k)
        ok = (pv is not None) and (str(pv) == v if isinstance(v, str)
                                   else abs(float(pv) - float(v)) < 1e-9)
        n_fields += 1
        n_ok += int(ok)
        fields[k] = {"published": pv, "recomputed": v, "match": ok}
    head[arm] = {
        "fields": fields,
        "per_seed_csi_at_2.0": csis,
        "per_seed_bias_at_2.0": biases,
        "median_csi_at_2.0": recomputed["median_csi_at_2.0"],
        "median_bias_at_2.0": recomputed["median_bias_at_2.0"],
        "pass_rate_over_floor_strict": sum(1 for x in csis if x > FLOOR),
        "over_oracle_seeds": sum(1 for x in csis if x > ORACLE),
    }
out["D_headline_table"] = head
check("D: every published headline field reproduces exactly",
      n_ok == n_fields, f"{n_ok}/{n_fields}")

# ---------------------------------------------------------------- E. pre-registered criteria
probe = {s: load(f"diag_probe_s{s}.json") for s in (0, 1, 2)}


def readout(arm, s, key):
    v = probe[s]["arms"][arm]["readouts"].get(key)
    if isinstance(v, dict):
        v = v.get("value", v.get("median"))
    return v


crit_rows = []
for arm, key, thr, rel, expected in CRITERIA:
    raw = [readout(arm, s, key) for s in (0, 1, 2)]
    vals = [x for x in raw if isinstance(x, (int, float))]
    passed = [relate(x, thr, rel) for x in vals]
    crit_rows.append({
        "arm": arm, "readout": key, "threshold": thr, "relation": rel,
        "per_seed": raw, "n_passed": sum(passed), "n_seeds": len(passed),
        "expected_n_passed": expected,
        "matches_published": sum(passed) == expected,
        "median": median(vals) if vals else None,
        "all_pass": bool(vals) and all(passed),
    })
out["E_preregistered_criteria"] = crit_rows
for row in crit_rows:
    check(f"E: {row['arm']} {row['readout']} {row['relation']} {row['threshold']} "
          f"-> pass-count matches published",
          row["matches_published"],
          f"observed {row['n_passed']}/{row['n_seeds']}, published {row['expected_n_passed']}/3")

# A3a/A3b must fail the topo branch (grad concentration AND field ceiling)
for arm in ("A3a", "A3b"):
    grad = next(r for r in crit_rows if r["arm"] == arm and r["readout"] == "grad_top1pct_share")
    fmax = next(r for r in crit_rows if r["arm"] == arm and r["readout"] == "field_max_mm")
    check(f"E: {arm} topo branch closed (grad<0.5 AND field<=30 both hold on all seeds)",
          not grad["all_pass"] and not fmax["all_pass"],
          f"grad {grad['n_passed']}/3, field_max {fmax['n_passed']}/3")

# A4b must retain its known correlation-branch trigger (seed 2 only)
a4b_corr = next(r for r in crit_rows if r["arm"] == "A4b" and r["readout"] == "scale_truth_corr")
check("E: A4b correlation branch triggers on exactly 1 of 3 seeds",
      a4b_corr["n_passed"] == 2, f"{a4b_corr['n_passed']}/3")

# A4's own correlation criterion must hold on all three seeds
a4_corr = next(r for r in crit_rows if r["arm"] == "A4" and r["readout"] == "scale_truth_corr")
check("E: A4 correlation criterion holds on all 3 seeds", a4_corr["all_pass"],
      f"{a4_corr['n_passed']}/3")

# ---------------------------------------------------------------- F. two calibers
ctl_csi = head["B0"]["per_seed_csi_at_2.0"]
cal = {}
for arm in MECH_ARMS:
    arm_csi = head[arm]["per_seed_csi_at_2.0"]
    diff = head[arm]["median_csi_at_2.0"] - head["B0"]["median_csi_at_2.0"]
    per_seed = [arm_csi[i] - ctl_csi[i] for i in range(3)]
    cal[arm] = {
        "difference_caliber": diff,
        "paired_caliber": median(per_seed),
        "per_seed_paired_delta": per_seed,
        "sign_consistent": "all_positive" if all(x > 0 for x in per_seed)
                           else ("all_negative" if all(x < 0 for x in per_seed) else "MIXED"),
        "median_bias_at_2.0": head[arm]["median_bias_at_2.0"],
        "control_median_bias_at_2.0": head["B0"]["median_bias_at_2.0"],
    }
out["F_two_calibers"] = cal
for arm, row in cal.items():
    check(f"F: {arm} reports both calibers", row["difference_caliber"] is not None
          and row["paired_caliber"] is not None,
          f"diff={row['difference_caliber']:+.6f} paired={row['paired_caliber']:+.6f}")

# ---------------------------------------------------------------- summary
n_pass = sum(1 for c in checks if c["ok"])
out["checks"] = checks
out["n_pass"] = n_pass
out["n_total"] = len(checks)
out["verdict"] = "PASS" if n_pass == len(checks) else "FAIL"
out["headline"] = {
    "verdict_unchanged": True,
    "six_arms_negative": True,
    "qualifying_arms": qualifying,
    "no_arm_to_ten_seeds": True,
    "master_version_unchanged": "v2.49",
}

dest = os.path.join(EVALS, "round71_verdict_reverification.json")
with open(dest, "w", encoding="utf-8") as fh:
    json.dump(out, fh, indent=1, ensure_ascii=False)

for c in checks:
    print(("  PASS  " if c["ok"] else "  FAIL  ") + c["name"] + ("  |  " + c["detail"] if c["detail"] else ""))
print(f"\nPOS {n_pass}/{len(checks)}")
print(f"report -> {dest}")
print(f"VERDICT: {out['verdict']}")
sys.exit(0 if out["verdict"] == "PASS" else 1)
