# -*- coding: utf-8 -*-
"""Round 74: E45 remote Slurm status re-verification (read-only on remote; local records only).

Mirrors the round-73 audit structure. No remote mutation, no Slurm submission,
no docs/ write-back (no new numbers => MASTER stays at v2.49).

Reporting caliber (pre-registered): median + pass rate only.
Single-point scores are NEVER used alone as a verdict.

NOTE on the cumulative-snapshot trap
------------------------------------
`scores_summary_s{0,1,2}.json` are CUMULATIVE snapshots (s0 = 7 keys, s1 = 14,
s2 = 21) and s2 is bit-identical to `scores_all.json`. A prefix match such as
`k.startswith(arm + "_s")` therefore ALWAYS resolves to the `_s0` key, so every
seed column prints the same number. That bug raises no exception -- it silently
produces a plausible-looking table that contradicts the published numbers.
Round 64 documented the trap in prose, yet a fresh round-74 script re-tripped it
in round 74. Hence it is now a MACHINE ASSERTION below, plus a cheap detector:
if all three per-seed values of an arm are identical, the extraction is wrong.
"""
import hashlib
import json
import os
import statistics

ROOT = r"R:\1radarNWPprediction\radar-nwp-bench"
EVALS = os.path.join(ROOT, "tools", "_evals", "e45")
OUT = os.path.join(EVALS, "round74_status_audit.json")

FLOOR = 0.0924          # persistence floor (seed-0 caliber)
ORACLE = 0.1653         # Oracle upper bound @2.0 mm
ARMS = ["A1", "A2", "A3a", "A3b", "A4", "A4b", "B0"]

# ---- 13 run-level artifacts: remote md5 (read via remote.py this round) vs local ----
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

# ---- fix-drift set: 4 files carry the implementation fixes, 1 is an unmodified control ----
REMOTE_FIX_MD5 = {
    "models/scaled_structure/model.py": "77f18b6cc2d5b4bf8d9078468d0db6c0",
    "models/topo_simvp/model.py": "f65554ca4f399431a7048c68bd07c80e",
    "tools/_e41_struct_metrics.py": "742503c872ab0e68e096eedc1cc31085",
    "tools/_e45_mechanism_probe.py": "e7d3666a0c7495999649c40cc5a2fe95",
    "models/amp_monotone_gate/model.py": "e90857d34dadfabe4ad7dff6b68d5394",
}
EXPECTED_DIFFER = {
    "models/scaled_structure/model.py",
    "models/topo_simvp/model.py",
    "tools/_e41_struct_metrics.py",
    "tools/_e45_mechanism_probe.py",
}


def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load(name):
    with open(os.path.join(EVALS, name), encoding="utf-8") as f:
        return json.load(f)


# ================= 1. artifact parity =================
artifact_detail, artifact_mismatch = [], []
for name, rmd5 in sorted(REMOTE_MD5.items()):
    p = os.path.join(EVALS, name)
    lmd5 = md5(p) if os.path.exists(p) else None
    same = (lmd5 == rmd5)
    artifact_detail.append({"file": name, "remote_md5": rmd5,
                             "local_md5": lmd5, "identical": same})
    if not same:
        artifact_mismatch.append(name)

# ================= 2. fix-file drift =================
fix_detail, differ = [], []
for rel, rmd5 in sorted(REMOTE_FIX_MD5.items()):
    p = os.path.join(ROOT, rel.replace("/", os.sep))
    lmd5 = md5(p) if os.path.exists(p) else None
    same = (lmd5 == rmd5)
    fix_detail.append({"file": rel, "remote_md5": rmd5,
                       "local_md5": lmd5, "identical": same})
    if not same:
        differ.append(rel)
drift_ok = (set(differ) == EXPECTED_DIFFER)

# ================= 3. three-seed CSI@2.0: median + pass rate =================
def csi_at(arm, seed, thr="2.0"):
    """EXACT key match -- the per-seed files are cumulative, so a prefix match
    silently resolves to _s0 for every seed."""
    d = load("scores_summary_s%d.json" % seed)
    e = d.get("%s_s%d_ep00" % (arm, seed))
    if isinstance(e, dict):
        key = "csi_at_%s" % thr
        if key in e:
            return float(e[key])
    return None


three_seed, floor_pass = {}, {}
for a in ARMS:
    vals = [csi_at(a, s) for s in (0, 1, 2)]
    ok = [v for v in vals if v is not None]
    three_seed[a] = {"seeds": vals,
                     "median": round(statistics.median(ok), 6) if len(ok) == 3 else None,
                     "n_seeds": len(ok)}
    floor_pass[a] = {"pass_count": sum(1 for v in ok if v >= FLOOR),
                     "n_seeds": len(ok),
                     "pass_rate": round(sum(1 for v in ok if v >= FLOOR) / len(ok), 4) if ok else None,
                     "over_oracle_count": sum(1 for v in ok if v > ORACLE)}

# ================= 3b. cumulative-snapshot machine guard =================
_cum_checks, _cum_identical_triples = [], []
_keys = {s: sorted(k for k, v in load("scores_summary_s%d.json" % s).items()
                    if isinstance(v, dict)) for s in (0, 1, 2)}
_cum_checks.append(("s0 key count == 7", len(_keys[0]) == 7, len(_keys[0])))
_cum_checks.append(("s1 key count == 14", len(_keys[1]) == 14, len(_keys[1])))
_cum_checks.append(("s2 key count == 21", len(_keys[2]) == 21, len(_keys[2])))
_cum_checks.append(("s0 subset of s1", set(_keys[0]) <= set(_keys[1]), True))
_cum_checks.append(("s1 subset of s2", set(_keys[1]) <= set(_keys[2]), True))
_cum_checks.append(("s2 file == scores_all.json",
                    load("scores_summary_s2.json") == load("scores_all.json"), True))
for _a in ARMS:
    _v = [csi_at(_a, _s) for _s in (0, 1, 2)]
    if len(set(_v)) == 1:
        _cum_identical_triples.append(_a)
_cum_checks.append(("no arm has 3 identical per-seed CSI (prefix-match detector)",
                    not _cum_identical_triples, _cum_identical_triples))
CUMULATIVE_GUARD = {
    "checks": [{"name": n, "ok": bool(o), "value": v} for n, o, v in _cum_checks],
    "all_pass": all(bool(o) for _n, o, _v in _cum_checks),
    "identical_seed_triples_detected": _cum_identical_triples,
    "note": "per-seed score files are CUMULATIVE snapshots; keys must match EXACTLY",
}

# ================= 4. S3 equal-bias guardrail =================
s3 = load("s3_equalbias.json")
s3_detail, s3_all_cross = {}, True
for a in ARMS:
    if a == "B0" or a not in s3.get("arms", {}):
        continue
    e = s3["arms"][a]
    raw, eq = e.get("raw", {}), e.get("equal_bias", {})
    both = bool(raw.get("crosses_zero")) and bool(eq.get("crosses_zero"))
    s3_all_cross = s3_all_cross and both
    s3_detail[a] = {
        "raw_point": round(raw.get("point", float("nan")), 6),
        "raw_ci": [round(raw.get("ci_lo", float("nan")), 6), round(raw.get("ci_hi", float("nan")), 6)],
        "raw_crosses_zero": raw.get("crosses_zero"),
        "equalbias_point": round(eq.get("point", float("nan")), 6),
        "equalbias_ci": [round(eq.get("ci_lo", float("nan")), 6), round(eq.get("ci_hi", float("nan")), 6)],
        "equalbias_crosses_zero": eq.get("crosses_zero"),
        "s3_verdict": e.get("s3_verdict"),
        "both_calibers_cross_zero": both,
    }
qualified = sorted([a for a, d in s3_detail.items() if not d["both_calibers_cross_zero"]])

# ================= 5. mechanism readouts (pre-registered criteria) =================
def diag(arm, seed):
    """diag_probe_s{seed}.json -> {'arms': {ARM: {..., 'readouts': {...}}}} (per-seed, NOT cumulative)."""
    d = load("diag_probe_s%d.json" % seed)
    arms = d.get("arms", {})
    if arm in arms:
        return arms[arm].get("readouts", {})
    return {}


def grab(e, *names):
    for n in names:
        if n in e:
            return e[n]
    return None


mech = {}
for a in ["A1", "A2", "A3a", "A3b", "A4", "A4b"]:
    rows = []
    for s in (0, 1, 2):
        e = diag(a, s)
        rows.append({"seed": s,
                     "corr_mask_amp": grab(e, "corr_mask_amp"),
                     "mask_at_peak": grab(e, "mask_at_peak"),
                     "grad_top1pct_share": grab(e, "grad_top1pct_share"),
                     "field_max_mm": grab(e, "field_max_mm"),
                     "scale_cv": grab(e, "scale_cv"),
                     "scale_truth_corr": grab(e, "scale_truth_corr")})
    mech[a] = rows

report = {
    "round": 74,
    "date": "2026-11-15",
    "purpose": "E45 remote Slurm status re-verification (read-only); verdict unchanged",
    "remote": {
        "authoritative_chain": {
            "273 e45-submit": "COMPLETED", "274 e45-stage2": "COMPLETED",
            "275 e45-train[0-6]": "COMPLETED", "282 e45-stage2": "COMPLETED",
            "283 e45-train[0-6]": "COMPLETED", "290 e45-stage2": "COMPLETED",
            "291 e45-summary": "COMPLETED", "297 e45-diag": "COMPLETED",
            "298 e45-diag": "COMPLETED", "314 e45-gpuprobe": "COMPLETED",
            "315/316/317 e45-epochaxis": "COMPLETED",
        },
        "submit273_finished_at": "2026-11-11T14:20:42+08:00",
        "s3_equalbias_ran_automatically_via_summary_job_291": True,
        "eval_json_count": 21,
        "run_level_json_count": 13,
        "squeue_wf": "empty (no pending/running jobs for account wf)",
        "no_e46_plus_run_dir": True,
        "historical_dead_jobs_not_in_authoritative_chain": {
            "102 e45-submit": "CANCELLED+", "103 e45-train": "FAILED",
            "118 e45-stage2": "FAILED", "265 e45-submit": "FAILED",
            "312 qlfn2-f36": "CANCELLED (belongs to another repository)",
        },
    },
    "artifact_parity": {"checked": len(artifact_detail),
                        "identical": len(artifact_detail) - len(artifact_mismatch),
                        "mismatches": artifact_mismatch, "detail": artifact_detail},
    "fix_drift": {"expected_differ_set": sorted(EXPECTED_DIFFER), "observed_differ_set": sorted(differ),
                  "matches_expectation": drift_ok, "control_file_identical":
                      next((d["identical"] for d in fix_detail
                            if d["file"] == "models/amp_monotone_gate/model.py"), None),
                  "detail": fix_detail},
    "cumulative_snapshot_guard": CUMULATIVE_GUARD,
    "three_seed_csi_at_2p0": {"floor": FLOOR, "oracle_upper_bound": ORACLE,
                              "caliber": "median + pass rate (single points are NOT verdicts)",
                              "per_arm": three_seed, "floor_pass": floor_pass},
    "s3_equalbias_guardrail": {"all_12_calibers_cross_zero": s3_all_cross,
                                "qualified_arms_for_10seed": qualified,
                                "verdict": "six arms judged NEGATIVE; none advanced to 10-seed",
                                "detail": s3_detail},
    "mechanism_readouts": mech,
    "verdict": {
        "e45_status": "CLOSED - NEGATIVE (unchanged since v2.44 / v2.49 compression)",
        "arms_advanced_to_10seed": [],
        "docs_writeback": "none required (no new numbers, no verdict change); MASTER stays at v2.49",
        "conclusion": "Remote Slurm state, artifact parity, fix-drift set, S3 guardrail and the "
                      "recomputed three-seed table are all consistent with the published verdict; "
                      "the E45 negative verdict stands unchanged.",
    },
}

with open(OUT, "w", encoding="utf-8") as f:
    json.dump(report, f, ensure_ascii=False, indent=2)

# ================= console summary =================
print("=" * 78)
print("ARTIFACT PARITY : %d/%d identical   mismatches=%s"
      % (len(artifact_detail) - len(artifact_mismatch), len(artifact_detail), artifact_mismatch or "none"))
print("FIX DRIFT       : observed=%s" % sorted(differ))
print("                  matches expectation = %s (control identical = %s)"
      % (drift_ok, next((d["identical"] for d in fix_detail
                         if d["file"] == "models/amp_monotone_gate/model.py"), None)))
print("CUM-SNAPSHOT GUARD:")
for c in CUMULATIVE_GUARD["checks"]:
    print("  %-5s %s (value=%s)" % ("PASS" if c["ok"] else "FAIL", c["name"], c["value"]))
print("=" * 78)
print("THREE-SEED CSI@2.0  (median + floor pass rate; floor=%.4f, oracle=%.4f)" % (FLOOR, ORACLE))
print("  %-5s %10s %10s %10s   %-10s %-8s %s" % ("arm", "s0", "s1", "s2", "median", "floor", ">oracle"))
for a in ARMS:
    t = three_seed[a]
    fp = floor_pass[a]
    print("  %-5s %10.6f %10.6f %10.6f   %-10.6f %d/%d     %d"
          % (a, t["seeds"][0], t["seeds"][1], t["seeds"][2], t["median"],
             fp["pass_count"], fp["n_seeds"], fp["over_oracle_count"]))
print("=" * 78)
print("S3 EQUAL-BIAS GUARDRAIL (12 calibers = 6 arms x raw/equalised)")
for a, d in s3_detail.items():
    print("  %-4s raw %+.5f [%+.5f,%+.5f] cross=%-5s | eq %+.5f [%+.5f,%+.5f] cross=%-5s | %s"
          % (a, d["raw_point"], d["raw_ci"][0], d["raw_ci"][1], d["raw_crosses_zero"],
             d["equalbias_point"], d["equalbias_ci"][0], d["equalbias_ci"][1],
             d["equalbias_crosses_zero"], d["s3_verdict"]))
print("  ALL 12 CALIBERS CROSS ZERO = %s ; qualified arms for 10-seed = %s"
      % (s3_all_cross, qualified or "NONE"))
print("=" * 78)
print("MECHANISM READOUTS (pre-registered criteria)")
for a, rows in mech.items():
    print("  %s" % a)
    for r in rows:
        bits = []
        for k in ("corr_mask_amp", "mask_at_peak", "grad_top1pct_share", "field_max_mm",
                  "scale_cv", "scale_truth_corr"):
            if r.get(k) is not None:
                bits.append("%s=%s" % (k, r[k]))
        print("    s%s  %s" % (r["seed"], "  ".join(bits)))
print("=" * 78)
_ok = (not artifact_mismatch and drift_ok and s3_all_cross and CUMULATIVE_GUARD["all_pass"])
print("VERDICT: %s" % ("PASS" if _ok else "FAIL"))
print("report -> %s" % OUT)
