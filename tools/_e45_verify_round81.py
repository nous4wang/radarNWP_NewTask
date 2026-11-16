#!/usr/bin/env python3
"""Round 81: E45 remote status re-check + artifact consistency / fix-file drift /
doc gates + a NEW audit surface: does the epoch-axis re-evaluation reproduce the
main E45 run's ep00 grids?

Read-only: touches nothing remote, writes only its own report JSON.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
E45 = os.path.join(ROOT, "tools", "_evals", "e45")
REPORT = os.path.join(E45, "round81_status_audit.json")

# Remote md5 captured this round from /home/wf/radarNWP (runs/e45/*.json).
REMOTE_RUN_JSON_MD5 = {
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

# Remote md5 of the 4 locally-repaired-but-unsynced files + 1 untouched control.
REMOTE_CODE_MD5 = {
    "models/scaled_structure/model.py": "77f18b6cc2d5b4bf8d9078468d0db6c0",
    "models/topo_simvp/model.py": "f65554ca4f399431a7048c68bd07c80e",
    "tools/_e41_struct_metrics.py": "742503c872ab0e68e096eedc1cc31085",
    "tools/_e45_mechanism_probe.py": "e7d3666a0c7495999649c40cc5a2fe95",
    "models/amp_monotone_gate/model.py": "e90857d34dadfabe4ad7dff6b68d5394",
}
CONTROL_FILE = "models/amp_monotone_gate/model.py"
EXPECTED_DRIFT = sorted(
    p for p in REMOTE_CODE_MD5 if p != CONTROL_FILE
)

ARMS = ["B0", "A1", "A2", "A3a", "A3b", "A4", "A4b"]
FLOOR = 0.0924
ORACLE = 0.1653

# Published headline table (from MASTER v2.44-v2.50 / HANDOVER): median CSI@2.0
# and pass-rate-over-floor for the seven arms.
PUBLISHED_MEDIAN_CSI = {
    "B0": 0.121817, "A1": 0.118397, "A2": 0.112003,
    "A3a": 0.110015, "A3b": 0.116055, "A4": 0.143896, "A4b": 0.137567,
}
PUBLISHED_PASS_RATE = {
    "B0": 2, "A1": 2, "A2": 2, "A3a": 3, "A3b": 3, "A4": 2, "A4b": 2,
}

checks: list[dict] = []


def check(name: str, ok: bool, detail: str = "") -> bool:
    checks.append({"name": name, "ok": bool(ok), "detail": detail})
    return bool(ok)


def md5(path: str) -> str:
    with open(path, "rb") as fh:
        return hashlib.md5(fh.read()).hexdigest()


def run(cmd: list[str]) -> tuple[int, str]:
    try:
        p = subprocess.run(
            cmd, cwd=ROOT, capture_output=True, text=True, timeout=1800
        )
        return p.returncode, (p.stdout or "") + (p.stderr or "")
    except Exception as exc:  # pragma: no cover
        return -1, f"{type(exc).__name__}: {exc}"


# ---------------------------------------------------------------- audit 1: artifacts
missing = [n for n in REMOTE_RUN_JSON_MD5 if not os.path.isfile(os.path.join(E45, n))]
check("all 13 run-level JSONs present locally", not missing, f"missing={missing}")

mismatch = []
for name, remote_md5 in sorted(REMOTE_RUN_JSON_MD5.items()):
    p = os.path.join(E45, name)
    if os.path.isfile(p) and md5(p) != remote_md5:
        mismatch.append(name)
check("13/13 run-level JSONs byte-identical to remote", not mismatch,
      f"mismatch={mismatch}")

# ---------------------------------------------------------------- audit 2: code drift
drift = sorted(
    rel for rel, remote_md5 in REMOTE_CODE_MD5.items()
    if os.path.isfile(os.path.join(ROOT, rel.replace("/", os.sep)))
    and md5(os.path.join(ROOT, rel.replace("/", os.sep))) != remote_md5
)
check("drift set is exactly the 4 repaired files", drift == EXPECTED_DRIFT,
      f"drift={drift}")
check("untouched control file identical on both sides",
      md5(os.path.join(ROOT, CONTROL_FILE.replace("/", os.sep)))
      == REMOTE_CODE_MD5[CONTROL_FILE],
      f"control={CONTROL_FILE}")

# ---------------------------------------------------------------- audit 3: doc gates
gate_rc = {}
for script in ("_check_tables", "_check_v245_row", "_e45_docs_numbers"):
    rc, out = run([sys.executable, os.path.join("tools", f"{script}.py")])
    gate_rc[script] = rc
    check(f"{script} PASS", rc == 0 and "VERDICT: PASS" in out,
          f"rc={rc}")

rc248, out248 = run([sys.executable, os.path.join("tools", "_check_v248_row.py")])
m = re.search(r"POS\s+(\d+)/(\d+)\s+NEG\s+(\d+)/(\d+)", out248)
pos, tot, neg, _ = (int(m.group(1)), int(m.group(2)), int(m.group(3)),
                    int(m.group(4))) if m else (0, 0, 0, 0)
check("v2.48 gate reaches registered baseline 59/61", (pos, tot) == (59, 61),
      f"pos={pos}/{tot} neg={neg}")
# CONTENT-class failures = E45 numbers/claims missing from the five docs.
content_fails = [ln for ln in out248.splitlines()
                 if ln.strip().startswith("FAIL")
                 and re.search(r"content|number|claim|csi|bias|median", ln, re.I)]
check("zero CONTENT-class failures (E45 numbers still present in the 5 docs)",
      not content_fails, f"content_fails={content_fails[:3]}")
gate_rc["_check_v248_row"] = rc248

# ---------------------------------------------------------------- audit 4: verdict
rc71, out71 = run([sys.executable, os.path.join("tools", "_e45_verify_round71.py")])
check("_e45_verify_round71 VERDICT: PASS",
      rc71 == 0 and "VERDICT: PASS" in out71, f"rc={rc71}")
rc78, out78 = run([sys.executable, os.path.join("tools", "_e45_verify_round78.py")])
check("_e45_verify_round78 VERDICT: PASS",
      rc78 == 0 and "VERDICT: PASS" in out78, f"rc={rc78}")

s3_path = os.path.join(E45, "s3_equalbias.json")
s3 = json.load(open(s3_path, encoding="utf-8"))
# Structure (verified this round): arms/<ARM>/{raw,equal_bias}/{point,ci_lo,ci_hi,
# crosses_zero} + arms/<ARM>/s3_verdict. Intervals are NOT 2-element lists.
intervals = []
for arm, arm_node in sorted(s3.get("arms", {}).items()):
    for variant in ("raw", "equal_bias"):
        node = arm_node.get(variant)
        if not isinstance(node, dict):
            continue
        lo, hi = node.get("ci_lo"), node.get("ci_hi")
        if isinstance(lo, (int, float)) and isinstance(hi, (int, float)):
            intervals.append((arm, variant, lo, hi, node.get("crosses_zero")))
check("S3 guard: 12 intervals located (6 arms x {raw, equal_bias})",
      len(intervals) == 12, f"n={len(intervals)}")
not_crossing = [
    f"{a}/{v}({lo:.6g},{hi:.6g})" for a, v, lo, hi, _ in intervals
    if not (lo <= 0.0 <= hi)
]
check("S3 guard: all 12 intervals cross zero", not not_crossing,
      f"not_crossing={not_crossing}")
flag_mismatch = [
    f"{a}/{v}" for a, v, lo, hi, flag in intervals
    if flag is not None and bool(flag) != bool(lo <= 0.0 <= hi)
]
check("S3 guard: stored crosses_zero flag agrees with the interval itself",
      not flag_mismatch, f"mismatch={flag_mismatch}")
verdicts = {a: n.get("s3_verdict") for a, n in sorted(s3.get("arms", {}).items())}
check("all 6 arms carry an S3 verdict string",
      len(verdicts) == 6 and all(isinstance(v, str) and v for v in verdicts.values()),
      f"n={len(verdicts)}")
check("eligible-arm set is empty (no arm survives the S3 guard)",
      not [a for a, v in verdicts.items()
           if v not in ("GAIN_WAS_AREA_OR_AMPLITUDE", "NO_GAIN_TO_BEGIN_WITH")],
      f"verdicts={sorted(set(verdicts.values()))}")

# ---------------------------------------------------------------- audit 5 (NEW): headline table
snap = {}
for seed in (0, 1, 2):
    p = os.path.join(E45, f"scores_summary_s{seed}.json")
    snap[seed] = json.load(open(p, encoding="utf-8"))
check("cumulative snapshot key counts are 7/14/21",
      [len(snap[s]) for s in (0, 1, 2)] == [7, 14, 21],
      f"counts={[len(snap[s]) for s in (0, 1, 2)]}")

median_csi = {}
pass_rate = {}
seed_csi = {}
for arm in ARMS:
    vals = []
    for seed in (0, 1, 2):
        key = f"{arm}_s{seed}_ep00"          # EXACT key match (no prefix match)
        if key not in snap[seed]:
            check(f"exact key present for {key}", False, "missing")
            continue
        vals.append(snap[seed][key]["csi_at_2.0"])
    seed_csi[arm] = vals
    median_csi[arm] = sorted(vals)[len(vals) // 2]
    pass_rate[arm] = sum(1 for v in vals if v >= FLOOR)

dev_med = max(abs(median_csi[a] - PUBLISHED_MEDIAN_CSI[a]) for a in ARMS)
check("published medians reproduced exactly", dev_med < 5e-7, f"maxdev={dev_med}")
check("published pass rates reproduced exactly",
      all(pass_rate[a] == PUBLISHED_PASS_RATE[a] for a in ARMS),
      f"rates={ {a: pass_rate[a] for a in ARMS} }")
check("no arm has three identical seed columns (cumulative-snapshot guard)",
      all(len(set(seed_csi[a])) > 1 for a in ARMS),
      f"suspect={[a for a in ARMS if len(set(seed_csi[a])) == 1]}")

over_oracle = sorted(a for a in ARMS
                     if any(v > ORACLE for v in seed_csi[a]))
check("arms exceeding the Oracle upper bound are A3b and A4 only",
      over_oracle == ["A3b", "A4"], f"over_oracle={over_oracle}")

# ------------------------------------------- audit 6 (NEW): epoch-axis vs main-run grids
def _eval_index(dirpath: str) -> dict:
    out = {}
    if not os.path.isdir(dirpath):
        return out
    for fn in os.listdir(dirpath):
        if not fn.endswith(".eval.json"):
            continue
        try:
            d = json.load(open(os.path.join(dirpath, fn), encoding="utf-8"))
        except Exception:
            continue
        m = re.search(r"(B0|A1|A2|A3a|A3b|A4|A4b)[._-]s(\d+)[._-]ep(\d+)", fn)
        if not m:
            continue
        arm, seed, epoch = m.group(1), int(m.group(2)), m.group(3)
        csi = None
        for key in ("overall", "metrics", "scores"):
            node = d.get(key)
            if isinstance(node, dict):
                for thr in ("2.0", "2"):
                    sub = node.get(thr)
                    if isinstance(sub, dict) and "csi" in sub:
                        csi = sub["csi"]
                        break
                    if isinstance(sub, dict) and "csi_at_2.0" in sub:
                        csi = sub["csi_at_2.0"]
                        break
            if csi is not None:
                break
        if csi is None:
            thr = d.get("thresholds")
            if isinstance(thr, dict):
                sub = thr.get("2.0") or thr.get("2") or {}
                if isinstance(sub, dict):
                    csi = sub.get("csi", sub.get("csi_at_2.0"))
        out[(arm, seed, epoch)] = (csi, fn)
    return out

main_dir = os.path.join(E45, "evals_raw", "evals")
if not os.path.isdir(main_dir):
    main_dir = os.path.join(E45, "runs", "e45", "evals")
epoch_dir = os.path.join(E45, "epoch_axis", "evals")
main_idx = _eval_index(main_dir)
epoch_idx = _eval_index(epoch_dir)
common = sorted(set(main_idx) & set(epoch_idx))
check("epoch-axis and main-run eval grids are both locally available",
      len(main_idx) == 21 and len(epoch_idx) == 63,
      f"main={len(main_idx)} epoch={len(epoch_idx)} common={len(common)}")
check("main-run grid holds exactly 7 arms x 3 seeds at ep00",
      len(main_idx) == 21
      and sorted({k[2] for k in main_idx}) == ["00"]
      and len({(k[0], k[1]) for k in main_idx}) == 21,
      f"epochs={sorted({k[2] for k in main_idx})}")
check("epoch-axis grid holds exactly 7 arms x 3 seeds x 3 epochs",
      len(epoch_idx) == 63
      and sorted({k[2] for k in epoch_idx}) == ["00", "01", "02"],
      f"epochs={sorted({k[2] for k in epoch_idx})}")

# The epoch-axis job re-evaluated the SAME checkpoints; its ep00 grids must
# reproduce the main run's ep00 grids in metric value.
ep00_common = sorted(k for k in common if k[2] == "00")
deltas = []
for key in ep00_common:
    a, b = main_idx[key][0], epoch_idx[key][0]
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        deltas.append(abs(a - b))
check("all 21 (arm, seed, ep00) cells are comparable across both grids",
      len(ep00_common) == 21 and len(deltas) == 21,
      f"common_ep00={len(ep00_common)} deltas={len(deltas)}")
if deltas:
    max_delta = max(deltas)
    check("epoch-axis ep00 grids reproduce the main-run ep00 grids within 1e-9",
          max_delta < 1e-9, f"maxdelta={max_delta} n={len(deltas)}")
else:
    check("epoch-axis ep00 grid comparison executed", False,
          "no comparable CSI pairs found")

# And ep01 must differ from ep00 (otherwise the epoch axis would be a copy).
ep_changes = []
for arm in ARMS:
    for seed in (0, 1, 2):
        v00 = epoch_idx.get((arm, seed, "00"), (None,))[0]
        v01 = epoch_idx.get((arm, seed, "01"), (None,))[0]
        if isinstance(v00, (int, float)) and isinstance(v01, (int, float)):
            ep_changes.append(abs(v01 - v00))
check("epoch-axis ep01 differs from ep00 in every cell (axis is not a copy)",
      len(ep_changes) == 21 and min(ep_changes) > 0,
      f"n={len(ep_changes)} min_delta={min(ep_changes) if ep_changes else None}")

# ---------------------------------------------------------------- report
n_pass = sum(1 for c in checks if c["ok"])
n_fail = len(checks) - n_pass
report = {
    "round": 81,
    "date": "2026-11-15",
    "remote": {
        "authoritative_chain_all_completed": True,
        "submitter_273": "COMPLETED 05:04:04",
        "summary_291_ran_s3_automatically": True,
        "eval_json_count": 21,
        "evals_dir_total": 42,
        "run_level_json_count": 13,
        "squeue_other_project_line": "8 x qlfn2-f37 (jobs 327-334, other repo)",
        "slurmctld": "active",
        "no_e46_plus_dirs": True,
        "historical_waste_jobs_not_in_authority_chain": [102, 103, 118, 265],
    },
    "headline_table": {
        arm: {
            "median_csi_at_2.0": median_csi[arm],
            "median_bias_at_2.0": None,
            "pass_rate_over_floor": f"{pass_rate[arm]}/3",
            "seeds": [round(v, 6) for v in seed_csi[arm]],
        }
        for arm in ARMS
    },
    "s3_guard": {"intervals_checked": len(intervals),
                 "all_cross_zero": not not_crossing,
                 "verdicts": verdicts},
    "checks": checks,
    "gates": gate_rc,
    "pos": n_pass,
    "neg": n_fail,
    "verdict": "PASS" if n_fail == 0 else "FAIL",
    "writeback": {
        "docs_changed": False,
        "reason": "no new numbers, no verdict change -> MASTER stays v2.50",
    },
}
with open(REPORT, "w", encoding="utf-8") as fh:
    json.dump(report, fh, ensure_ascii=False, indent=2)

for c in checks:
    print(("  PASS  " if c["ok"] else "  FAIL  ") + c["name"]
          + (f"  |  {c['detail']}" if c["detail"] else ""))
print()
print(f"POS {n_pass}/{len(checks)}   NEG {n_fail}/{len(checks)}")
print("report ->", REPORT)
print("VERDICT:", report["verdict"])
sys.exit(0 if n_fail == 0 else 1)
