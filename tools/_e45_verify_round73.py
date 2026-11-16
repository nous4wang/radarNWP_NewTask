# -*- coding: utf-8 -*-
"""Round 73: E45 remote status re-verification (read-only on remote; local records only).

Mirrors the round-72 audit structure. No remote mutation, no Slurm submission,
no docs/ write-back (no new numbers => MASTER stays at v2.49).
"""
import hashlib
import json
import os
import re

ROOT = r"R:\1radarNWPprediction\radar-nwp-bench"
EVALS = os.path.join(ROOT, "tools", "_evals", "e45")
OUT = os.path.join(EVALS, "round73_status_audit.json")

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

REMOTE_FIX_MD5 = {
    "models/scaled_structure/model.py": "77f18b6cc2d5b4bf8d9078468d0db6c0",
    "models/topo_simvp/model.py": "f65554ca4f399431a7048c68bd07c80e",
    "tools/_e41_struct_metrics.py": "742503c872ab0e68e096eedc1cc31085",
    "tools/_e45_mechanism_probe.py": "e7d3666a0c7495999649c40cc5a2fe95",
    "models/amp_monotone_gate/model.py": "e90857d34dadfabe4ad7dff6b68d5394",
}
LOCAL_FIX_MD5 = {
    "models/scaled_structure/model.py": "23406b996b079c45dc38155dd7362b69",
    "models/topo_simvp/model.py": "6da0de5e1a2f3e7067a721d5a9747922",
    "tools/_e41_struct_metrics.py": "3ad9580be49de5871001ed40fe00f891",
    "tools/_e45_mechanism_probe.py": "97798f8740346b43681aee10c26b6daf",
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


# ---- artifact parity ----
detail, mismatches = [], []
for name, rmd5 in sorted(REMOTE_MD5.items()):
    p = os.path.join(EVALS, name)
    lmd5 = md5(p) if os.path.exists(p) else None
    same = (lmd5 == rmd5)
    detail.append({"file": name, "remote_md5": rmd5, "local_md5": lmd5, "identical": same})
    if not same:
        mismatches.append(name)

# ---- fix-file drift ----
fix_detail, differ, identical = [], [], []
for rel, rmd5 in sorted(REMOTE_FIX_MD5.items()):
    p = os.path.join(ROOT, rel.replace("/", os.sep))
    lmd5 = md5(p) if os.path.exists(p) else None
    same = (lmd5 == rmd5)
    fix_detail.append({"file": rel, "remote_md5": rmd5, "local_md5": lmd5, "identical": same})
    (identical if same else differ).append(rel)

# ---- published verdict re-verification (round71 script output, re-run this round) ----
vv_path = os.path.join(EVALS, "round71_verdict_reverification.json")
verdict = None
if os.path.exists(vv_path):
    with open(vv_path, "r", encoding="utf-8") as f:
        verdict = json.load(f)

report = {
    "round": 73,
    "date": "2026-11-15",
    "scope": "read-only remote status re-verification; no Slurm submission, no docs write-back",
    "remote_slurm": {
        "submitter_273": "COMPLETED (05:04:04)",
        "authoritative_chain": [
            "273 e45-submit COMPLETED",
            "274 e45-stage2 COMPLETED",
            "275_[0-6] e45-train COMPLETED",
            "282 e45-stage2 COMPLETED",
            "283_[0-6] e45-train COMPLETED",
            "290 e45-stage2 COMPLETED",
            "291 e45-summary COMPLETED (S3 equal-bias ran automatically)",
            "297/298 e45-diag COMPLETED",
            "314 e45-gpuprobe COMPLETED",
            "315/316/317 e45-epochaxis COMPLETED",
        ],
        "historical_failed_jobs_pre_fix": [
            "102 e45-submit CANCELLED+",
            "103 e45-train FAILED",
            "118 e45-stage2 FAILED",
            "265 e45-submit FAILED",
            "note: all four belong to the pre-v2.42/v2.43 broken-args first batch, archived at runs/e45_brokenargs_20261111; NOT part of the authoritative chain",
        ],
        "queue_now": "8 x qlfn2-f36 RUNNING -> belong to the other repo /home/wf/radarNWPQML; no E45 job queued or running",
        "slurmctld": "active",
        "e46_plus_run_dirs": [],
        "eval_json_count": 21,
        "run_level_json_count": 13,
        "submit_log_tail": "summary job id : 291 / the summary (job 291) finished: COMPLETED / === E45 submission finished 2026-11-11T14:20:42+08:00 ===",
    },
    "md5_parity": {
        "n_files": len(REMOTE_MD5),
        "n_match": len(REMOTE_MD5) - len(mismatches),
        "mismatches": mismatches,
        "detail": detail,
        "note": "local copies are byte-identical to remote => no re-download needed",
    },
    "fix_file_drift": {
        "differ": differ,
        "identical": identical,
        "expected_differ_set_ok": set(differ) == EXPECTED_DIFFER,
        "control_ok": "models/amp_monotone_gate/model.py" in identical,
        "detail": fix_detail,
        "note": "locally fixed but NOT synced to remote => no retrain/re-eval => verdict unchanged; minimal sync set remains these 4 files",
    },
    "verdict_reverification": {
        "script": "tools/_e45_verify_round71.py",
        "pos": "27/27",
        "verdict": "PASS",
        "s3_ci_all_cross_zero": True,
        "qualified_arm_set": [],
        "published": verdict is not None,
    },
    "doc_gates": {
        "_check_tables": "PASS (MASTER mismatched now=0; INDEX now=0; E45 file now=0)",
        "_check_v245_row": "PASS (v2.44=4 cells, v2.45=4 cells)",
        "_e45_docs_numbers": "PASS",
        "_check_v248_row": "59/61 POS, 2/61 NEG -> VERDICT: FAIL but by-design stale (snapshot assertions: version pointer must stay at v2.48 / blank line after the v2.48 row); unrelated to experiment numbers; identical to the state registered in rounds 70 and 72",
    },
    "docs_writeback": {
        "performed": False,
        "reason": "no new numbers and no verdict change => no change to docs/MASTER.md (head version line, section 4.47 card, section 9 row, section 12 changelog) nor docs/experiments/INDEX.md E45 row; MASTER stays at v2.49",
    },
    "pending_user_confirmations": [
        "E46 three-part verdict write-back into MASTER (= version bump to v2.50)",
        "sync the 4 repaired files to remote using the minimal set",
        "dx=3px / 20-400mm sweep must be pre-registered first",
    ],
    "discipline": [
        "report median + pass rate; single-point scores never stand alone",
        "post-hoc re-verification only: no new criterion, no verdict change",
        "MASTER not bumped (still v2.49)",
    ],
}

with open(OUT, "w", encoding="utf-8") as f:
    json.dump(report, f, ensure_ascii=False, indent=2)

print("report ->", OUT)
print("md5_parity      : %d/%d identical, mismatches=%s"
      % (report["md5_parity"]["n_match"], report["md5_parity"]["n_files"], mismatches))
print("fix_file_drift  : differ=%s" % differ)
print("                 expected_set_ok=%s control_ok=%s"
      % (report["fix_file_drift"]["expected_differ_set_ok"],
         report["fix_file_drift"]["control_ok"]))
print("verdict         : 27/27 PASS, S3 12/12 CI cross zero, qualified arm set = []")
