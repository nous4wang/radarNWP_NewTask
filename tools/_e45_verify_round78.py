#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""E45 round 78 status audit (read-only, local only; no remote calls).

Scope
-----
A. Remote-vs-local artifact consistency (md5) for the 13 run-level JSONs.
B. Sync-drift set of the three locally-repaired implementation defects
   (expected drift set must be exactly those 4 files; the untouched control
   file must be byte-identical on both sides).
C. Document gate regression suite (4 gates).
D. Classification of the frozen v2.48 gate's residual failures into
   CONTENT-class (an E45 number / claim missing from one of the 5 docs ->
   a real regression that must be zero) vs SNAPSHOT-class (version-pointer
   assertions that are expired by design once the project advances past
   v2.48 -> documented, tolerated, but must be enumerated).
E. Independent re-derivation of the S3 equal-bias verdict: 12 intervals
   (6 arms x {raw, equal_bias}) must all cross zero; eligible set empty.
F. Re-derivation of the published three-seed headline table
   (median + pass rate only; single-point scores are never a verdict),
   including the cumulative-snapshot guard.

Discipline (inherited)
----------------------
* Read-only: touches no remote host, no model code, no remote state.
* "Criterion passed" != "review passed": assertions are anchored on
  OBSERVED counts matching PUBLISHED counts, not on criteria passing.
* Cumulative-snapshot guard: scores_summary_s{0,1,2}.json are cumulative
  snapshots (7/14/21 keys), so per-seed values must be extracted by EXACT
  key match, plus a "no arm has three identical seed columns" detector.
* Every assertion must be able to fail.

Usage:  python tools/_e45_verify_round78.py
"""
from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
EVALS = REPO / "tools" / "_evals" / "e45"
OUT = EVALS / "round78_status_audit.json"

FLOOR_CSI = 0.0924          # persistence floor, thr=2.0 mm
ORACLE_CSI = 0.1653         # oracle upper bound, thr=2.0 mm
ARMS = ["B0", "A1", "A2", "A3a", "A3b", "A4", "A4b"]

# Published three-seed headline table (median CSI@2.0, floor pass rate).
PUBLISHED = {
    "B0":  (0.12181703042952949, "2/3"),
    "A1":  (0.1183974362784095,  "2/3"),
    "A2":  (0.11200272630175023, "2/3"),
    "A3a": (0.11001479850492543, "3/3"),
    "A3b": (0.1160553613879921,  "3/3"),
    "A4":  (0.14389625096252312, "2/3"),
    "A4b": (0.13756654739888727, "2/3"),
}

# --------------------------------------------------------------------------
# Remote md5 measured this run (2026-11-15, read-only remote query:
# md5sum runs/e45/*.json ; md5sum <5 code files>).  Locked here so the audit
# is reproducible without another remote round-trip.
# --------------------------------------------------------------------------
REMOTE_RUN_MD5 = {
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

REMOTE_CODE_MD5 = {
    "models/scaled_structure/model.py": "77f18b6cc2d5b4bf8d9078468d0db6c0",
    "models/topo_simvp/model.py": "f65554ca4f399431a7048c68bd07c80e",
    "models/amp_monotone_gate/model.py": "e90857d34dadfabe4ad7dff6b68d5394",
    "tools/_e41_struct_metrics.py": "742503c872ab0e68e096eedc1cc31085",
    "tools/_e45_mechanism_probe.py": "e7d3666a0c7495999649c40cc5a2fe95",
}
EXPECTED_DRIFT = {
    "models/scaled_structure/model.py",
    "models/topo_simvp/model.py",
    "tools/_e41_struct_metrics.py",
    "tools/_e45_mechanism_probe.py",
}
CONTROL_FILE = "models/amp_monotone_gate/model.py"

# v2.48 gate assertions that are about VERSION POINTERS / changelog layout,
# i.e. expired by design once the project advances past v2.48.  Documented
# since round 70; must be enumerated, never silently ignored.
SNAPSHOT_CLASS = {
    "all five version pointers agree",
    "the version pointer is v2.48",
    "the v2.48 changelog row is followed by a blank line",
    "the v2.48 changelog row has the same cell count as the row it follows",
    "the v2.48 row follows another v2.4x row",
    "MASTER section 12 has exactly one v2.48 changelog row",
    "MASTER header version line carries v2.48",
    "MASTER section 4.47 card carries v2.48",
    "MASTER section 9 row carries v2.48",
}

report: dict = {"round": 78, "date": "2026-11-15"}
pos = neg = 0


def check(ok: bool, name: str, detail: str = "") -> bool:
    global pos, neg
    ok = bool(ok)
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"  | {detail}" if detail else ""))
    if ok:
        pos += 1
    else:
        neg += 1
    report.setdefault("checks", []).append({"name": name, "ok": ok, "detail": detail})
    return ok


def md5(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def run_gate(script: str) -> dict:
    p = subprocess.run([sys.executable, str(REPO / "tools" / script)],
                       cwd=str(REPO), capture_output=True, text=True)
    out = p.stdout + p.stderr
    m = re.search(r"POS (\d+)/(\d+)\s+NEG (\d+)/(\d+)", out)
    v = re.search(r"VERDICT:\s*(\w+)", out)
    fails = []
    for line in out.splitlines():
        s = line.strip()
        if not s.startswith("FAIL"):
            continue
        body = s[len("FAIL"):].strip()
        # format: "<label>"  or  "<label>  [detail]"
        if "  [" in body:
            label, detail = body.split("  [", 1)
            detail = detail.rstrip("]")
        else:
            label, detail = body, ""
        fails.append({"label": label.strip(), "detail": detail.strip()})
    return {"script": script, "rc": p.returncode,
            "pos": int(m.group(1)) if m else None,
            "neg": int(m.group(3)) if m else None,
            "verdict": v.group(1) if v else None, "fails": fails}


def find_ci_intervals(obj, path=""):
    if isinstance(obj, dict):
        low = {k.lower(): v for k, v in obj.items()}
        lo = hi = None
        for a, b in (("ci_low", "ci_high"), ("ci_lo", "ci_hi"),
                     ("low", "high"), ("lo", "hi")):
            if a in low and b in low and isinstance(low[a], (int, float)) \
                    and isinstance(low[b], (int, float)):
                lo, hi = float(low[a]), float(low[b])
                break
        if lo is None and isinstance(low.get("ci"), (list, tuple)) \
                and len(low["ci"]) == 2 and all(isinstance(x, (int, float)) for x in low["ci"]):
            lo, hi = float(low["ci"][0]), float(low["ci"][1])
        if lo is not None and hi is not None:
            yield path, lo, hi
        for k, v in obj.items():
            yield from find_ci_intervals(v, f"{path}.{k}" if path else str(k))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            yield from find_ci_intervals(v, f"{path}[{i}]")


# ==========================================================================
print("=== A. remote-vs-local artifact consistency (md5) ===")
mismatch, missing = [], []
for name, remote in REMOTE_RUN_MD5.items():
    lp = EVALS / name
    if not lp.exists():
        missing.append(name)
        continue
    local = md5(lp)
    if local != remote:
        mismatch.append({"file": name, "remote": remote, "local": local})
check(not missing, "all 13 run-level JSONs present locally", f"missing={missing}")
check(not mismatch, "13/13 run-level JSONs byte-identical to remote", f"mismatch={mismatch}")
report["artifact_md5"] = {"n_remote": len(REMOTE_RUN_MD5),
                          "n_match": len(REMOTE_RUN_MD5) - len(missing) - len(mismatch),
                          "mismatch": mismatch, "missing": missing}

# ==========================================================================
print("\n=== B. sync-drift set of the repaired implementation defects ===")
drift, same = [], []
for rel, remote in REMOTE_CODE_MD5.items():
    lp = REPO / rel
    local = md5(lp) if lp.exists() else None
    (drift if local != remote else same).append(rel)
check(set(drift) == EXPECTED_DRIFT, "drift set is exactly the 4 repaired files",
      f"drift={sorted(drift)}")
check(CONTROL_FILE in same, "untouched control file identical on both sides",
      f"control={CONTROL_FILE}")
report["code_drift"] = {"drift": sorted(drift), "identical": sorted(same)}

# ==========================================================================
print("\n=== C. document gate regression suite ===")
gates = {}
for s in ("_check_tables.py", "_check_v245_row.py",
          "_e45_docs_numbers.py", "_check_v248_row.py"):
    g = run_gate(s)
    gates[s] = g
    print(f"  {s:24s} rc={g['rc']} POS {g['pos']} NEG {g['neg']} VERDICT {g['verdict']}")
check(gates["_check_tables.py"]["verdict"] == "PASS", "_check_tables PASS")
check(gates["_check_v245_row.py"]["verdict"] == "PASS", "_check_v245_row PASS")
check(gates["_e45_docs_numbers.py"]["verdict"] == "PASS", "_e45_docs_numbers PASS")
report["gates"] = {k: {kk: vv for kk, vv in v.items() if kk != "fails"}
                   for k, v in gates.items()}

# ==========================================================================
print("\n=== D. v2.48 gate: CONTENT-class vs SNAPSHOT-class residual failures ===")
v248 = gates["_check_v248_row.py"]
content_fails, snapshot_fails, other_fails = [], [], []
for f in v248["fails"]:
    lbl = f["label"]
    if lbl in SNAPSHOT_CLASS:
        snapshot_fails.append(lbl)
    elif any(t in lbl for t in ("3.0188", "24.7", "magnitude", "collapse",
                                "correlation branch", "退化", "塌缩")):
        content_fails.append(lbl)
    else:
        other_fails.append(lbl)
check(not content_fails,
      "zero CONTENT-class failures (no E45 number/claim missing from the 5 docs)",
      f"content_fails={content_fails}")
check(not other_fails, "zero unclassified failures in the v2.48 gate",
      f"other={other_fails}")
print(f"  snapshot-class (expired by design, enumerated): {snapshot_fails}")
report["v248_classification"] = {
    "content_class_fails": content_fails, "snapshot_class_fails": snapshot_fails,
    "unclassified_fails": other_fails, "pos": v248["pos"], "neg": v248["neg"],
    "rc": v248["rc"],
}

# ==========================================================================
print("\n=== E. S3 equal-bias verdict re-derived independently ===")
s3 = json.loads((EVALS / "s3_equalbias.json").read_text(encoding="utf-8"))
ivs = list(find_ci_intervals(s3.get("arms", {}), "arms"))
not_crossing = [p for p, lo, hi in ivs if not (lo <= 0.0 <= hi)]
print(f"  intervals found: {len(ivs)}   crossing zero: {len(ivs) - len(not_crossing)}")
check(len(ivs) == 12, "12 S3 intervals found (6 arms x {raw, equal_bias})", f"n={len(ivs)}")
check(not not_crossing, "all 12 S3 intervals cross zero", f"not_crossing={not_crossing}")
verdicts = {}
def harvest(node, arm):
    if isinstance(node, dict):
        for k, v in node.items():
            if k in ("verdict", "s3_verdict") and isinstance(v, str):
                verdicts.setdefault(arm, v)
            harvest(v, arm)
    elif isinstance(node, list):
        for v in node:
            harvest(v, arm)
for arm, blob in s3.get("arms", {}).items():
    harvest(blob, arm)
eligible = [a for a, v in verdicts.items()
            if any(t in v.upper() for t in ("PASS", "ELIGIBLE", "ADVANCE", "WINNER"))]
check(len(verdicts) == 6, "all 6 arms carry an S3 verdict string", f"n={len(verdicts)}")
check(not eligible, "eligible-arm set is empty", f"eligible={eligible}")
for a in sorted(verdicts):
    print(f"  {a:4s} {verdicts[a]}")
report["s3"] = {"n_intervals": len(ivs), "n_crossing_zero": len(ivs) - len(not_crossing),
                "not_crossing": not_crossing, "verdicts": verdicts, "eligible": eligible}

# ==========================================================================
print("\n=== F. three-seed headline table re-derived (median + pass rate) ===")
merged = json.loads((EVALS / "scores_summary.json").read_text(encoding="utf-8"))
seed_files = {s: json.loads((EVALS / f"scores_summary_s{s}.json").read_text(encoding="utf-8"))
              for s in (0, 1, 2)}
headline, dev_pub, dev_merge = {}, {}, []
for arm in ARMS:
    vals = []
    for s in (0, 1, 2):
        blob = seed_files[s].get(f"{arm}_s{s}_ep00")     # EXACT key match
        if isinstance(blob, dict):
            v = blob.get("csi_at_2.0")
            if isinstance(v, (int, float)):
                vals.append(float(v))
    if len(vals) != 3:
        continue
    med = sorted(vals)[1]
    rate = f"{sum(v >= FLOOR_CSI for v in vals)}/3"
    headline[arm] = {"median_csi_at_2.0": med, "pass_rate_over_floor": rate,
                     "seeds_over_oracle": sum(v > ORACLE_CSI for v in vals),
                     "seed_values": vals}
    pm, pr = PUBLISHED[arm]
    dev_pub[arm] = max(abs(med - pm), 0.0)
    if rate != pr:
        dev_merge.append((arm, rate, pr))
    m = merged.get(arm, {})
    mv = m.get("csi_at_2.0", {})
    for s in (0, 1, 2):
        if f"s{s}" in mv and abs(mv[f"s{s}"] - vals[s]) > 1e-12:
            dev_merge.append((arm, f"s{s}", mv[f"s{s}"], vals[s]))
check(len(headline) == 7, "all 7 arms re-derived from per-seed snapshots",
      f"n={len(headline)}")
for arm in ARMS:
    h = headline.get(arm, {})
    print(f"  {arm:4s} median CSI@2.0 = {h.get('median_csi_at_2.0', float('nan')):.6f}"
          f"   floor-pass {h.get('pass_rate_over_floor')}"
          f"   seeds>oracle {h.get('seeds_over_oracle')}")
check(not dev_merge, "re-derived values match the merged summary and the published table",
      f"diff={dev_merge}")
same_cols = [a for a, h in headline.items() if len(set(h["seed_values"])) == 1]
check(not same_cols, "no arm has three identical seed columns (cumulative-snapshot guard)",
      f"suspect={same_cols}")
check(max(dev_pub.values()) <= 1e-12, "published medians reproduced exactly",
      f"maxdev={max(dev_pub.values())}")
report["headline"] = headline

# ==========================================================================
print("\n=== G. verdict guard (authoritative, pre-registered) ===")
r71 = subprocess.run([sys.executable, str(REPO / "tools" / "_e45_verify_round71.py")],
                     cwd=str(REPO), capture_output=True, text=True)
o = r71.stdout + r71.stderr
m = re.search(r"POS (\d+)/(\d+)", o)
check("VERDICT: PASS" in o, "_e45_verify_round71 VERDICT: PASS",
      f"rc={r71.returncode} pos={m.group(0) if m else 'n/a'}")
report["verdict_guard"] = {"rc": r71.returncode,
                           "pos": int(m.group(1)) if m else None,
                           "total": int(m.group(2)) if m else None}

report["pos"], report["neg"] = pos, neg
report["verdict"] = "PASS" if neg == 0 else "FAIL"
OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"\nreport -> {OUT}")
print(f"POS {pos}/{pos + neg}")
print(f"VERDICT: {report['verdict']}")
sys.exit(0 if neg == 0 else 1)
