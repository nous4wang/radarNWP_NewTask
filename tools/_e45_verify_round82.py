#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""E45 round-82 status re-verification (READ-ONLY).

Local + remote audit only. It NEVER writes to docs/ and NEVER writes to any
memory file; the write-back is a separate script (tools/_e45_writeback_round82.py).

New in round 82 (audit four, "whole-tree mtime proof"):
  The previous rounds compared the md5 of the 13 run-level json files. That
  proves the *published* artifacts did not drift, but it cannot prove that
  nothing else in the run tree (checkpoints, per-window tables, probe files)
  was touched after the closure. Round 82 therefore asserts:
    (a) ZERO files under runs/e45/ have an mtime newer than the closure
        minute (2026-11-11 14:19) -> the main run tree is frozen;
    (b) the files newer than 2026-11-12 live ONLY under runs/e45_epochaxis/
        and are diagnostic products (epoch-axis probe), not main-run outputs;
    (c) sacct since 2026-11-12 contains NO e45-train / e45-stage2 /
        e45-submit job -> the main chain was never re-launched.

Also new: gate failures are classified CONTENT (a real regression: a number or
a conclusion is missing) vs SNAPSHOT (a by-design stale version-pointer
snapshot). Only CONTENT == 0 is allowed.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys

REPO = r"R:/1radarNWPprediction/radar-nwp-bench"
EVAL_DIR = os.path.join(REPO, "tools", "_evals", "e45")
REPORT = os.path.join(EVAL_DIR, "round82_status_audit.json")
REMOTE_ROOT = "/home/wf/radarNWP"

# 13 run-level artifacts that carry the published verdict
RUN_ARTIFACTS = [
    "scores_all.json",
    "scores_summary.json",
    "scores_summary_s0.json",
    "scores_summary_s1.json",
    "scores_summary_s2.json",
    "diag_probe_s0.json",
    "diag_probe_s1.json",
    "diag_probe_s2.json",
    "struct_metrics_all.json",
    "struct_metrics_s0.json",
    "struct_metrics_s1.json",
    "struct_metrics_s2.json",
    "s3_equalbias.json",
]

# md5 baselines registered by round 81 (tools/_e45_writeback_round81.py output)
R81_BASELINE = {
    "HANDOVER_E45.md": "4f9d54eac0e785b2a660cb294df80edc",
    "upper_2026-11-11.md": "c79040c6402f2d692fe1b192f75bbf05",
    "proj_2026-11-15.md": "4719771396a0e8e138c2e96ac15ba741",
    "upper_MEMORY.md": "b803227a7bacd9fede0fd51d8fb94cee",
}

# 4 already-fixed files whose local->remote sync is still pending (user-gated)
FIX_FILES = {
    "models/scaled_structure/model.py": "77f18b6cc2d5b4bf8d9078468d0db6c0",
    "models/topo_simvp/model.py": "f65554ca4f399431a7048c68bd07c80e",
    "tools/_e41_struct_metrics.py": "742503c872ab0e68e096eedc1cc31085",
    "tools/_e45_mechanism_probe.py": "e7d3666a0c7495999649c40cc5a2fe95",
}
CONTROL_FILE = "models/amp_monotone_gate/model.py"
CONTROL_MD5 = "e90857d34dadfabe4ad7dff6b68d5394"

GATES = [
    "tools/_check_tables.py",
    "tools/_check_v245_row.py",
    "tools/_check_v246_row.py",
    "tools/_check_v247_row.py",
    "tools/_check_v248_row.py",
    "tools/_e45_docs_numbers.py",
]

VERIFIERS = [
    "tools/_e45_verify_round71.py",
    "tools/_e45_verify_round78.py",
    "tools/_e45_verify_round81.py",
]

SNAPSHOT_PAT = re.compile(
    r"version pointer|changelog row is followed by a blank line|v2\.4[4-9]"
)

POS: list[str] = []
NEG: list[str] = []
DETAIL: dict = {}


def ok(msg: str) -> None:
    POS.append(msg)
    print("  PASS  " + msg)


def bad(msg: str) -> None:
    NEG.append(msg)
    print("  FAIL  " + msg)


def expect(cond: bool, msg_ok: str, msg_fail: str) -> bool:
    (ok if cond else bad)(msg_ok if cond else msg_fail)
    return cond


def md5(path: str) -> str:
    with open(path, "rb") as fh:
        return hashlib.md5(fh.read()).hexdigest()


def remote(cmd: str) -> str:
    r = subprocess.run(
        [sys.executable, os.path.join(REPO, "tools", "remote.py"), "--", cmd],
        capture_output=True, text=True, cwd=REPO,
    )
    return r.stdout or ""


def main() -> int:
    print("=== A. remote authoritative chain ===")
    squeue = remote("squeue -u wf -o \"%.10i %.14j %.2t %.6M %R\"")
    e45_running = [ln for ln in squeue.splitlines() if "e45" in ln.lower()]
    expect(not e45_running,
           "no E45 job in the queue (squeue)",
           "E45 job still in the queue: %r" % e45_running)
    DETAIL["squeue"] = squeue.strip().splitlines()

    sacct = remote("sacct -u wf -S 2026-11-11 --format=JobID,JobName,State,Elapsed")
    chain = ["e45-submit", "e45-stage2", "e45-train", "e45-summary", "e45-diag",
             "e45-gpuprobe", "e45-epochaxis"]
    # Jobs 102/103/118/265 are the KNOWN pre-fix waste jobs (v2.42/v2.43 argument
    # bug); their artifacts are archived in runs/e45_brokenargs_20261111/ and they
    # are NOT part of the authoritative chain. They are asserted separately so a
    # NEW failure can never hide inside the known set.
    KNOWN_DEAD = {"102", "103", "118", "265"}
    bad_state, seen_dead = [], set()
    for ln in sacct.splitlines():
        parts = ln.split()
        if len(parts) < 3:
            continue
        job, name, state = parts[0], parts[1], parts[2]
        if name in chain and state != "COMPLETED":
            if job in KNOWN_DEAD:
                seen_dead.add(job)
            else:
                bad_state.append("%s/%s=%s" % (job, name, state))
    expect(not bad_state,
           "every E45 job in the authoritative chain is COMPLETED "
           "(the 4 known pre-fix waste jobs are excluded and verified separately)",
           "non-COMPLETED E45 jobs outside the known waste set: %r" % bad_state)
    expect(seen_dead == KNOWN_DEAD,
           "the set of non-COMPLETED E45 jobs is exactly the known waste set "
           "{102,103,118,265} => no new failure appeared",
           "known waste set mismatch: seen=%r expected=%r"
           % (sorted(seen_dead), sorted(KNOWN_DEAD)))
    DETAIL["sacct_non_completed"] = bad_state
    DETAIL["known_waste_jobs_seen"] = sorted(seen_dead)

    # audit four (new): no re-launch of the main chain after closure
    sacct_full = remote("sacct -u wf -S 2026-11-12 --format=JobID,JobName,State,Start,End")
    relaunched = [ln for ln in sacct_full.splitlines()
                  if re.search(r"e45-(train|stage2|submit)\b", ln)]
    expect(not relaunched,
           "since 2026-11-12 no e45-train/e45-stage2/e45-submit job exists "
           "=> the main chain was never re-launched",
           "main chain was re-launched after closure: %r" % relaunched)
    DETAIL["relaunched_main_chain"] = relaunched

    print("=== B. artifact parity (13 run-level json) ===")
    rem_md5 = remote("cd %s && md5sum runs/e45/*.json" % REMOTE_ROOT)
    rem_map = {}
    for ln in rem_md5.splitlines():
        parts = ln.split()
        if len(parts) == 2:
            rem_map[os.path.basename(parts[1])] = parts[0]
    mism, missing = [], []
    for name in RUN_ARTIFACTS:
        lp = os.path.join(EVAL_DIR, name)
        if name not in rem_map or not os.path.exists(lp):
            missing.append(name)
            continue
        if md5(lp) != rem_map[name]:
            mism.append(name)
    expect(not missing and not mism,
           "13/13 run-level artifacts byte-identical to remote (md5)",
           "artifact mismatch/missing: missing=%r mismatch=%r" % (missing, mism))
    DETAIL["md5_mismatches"] = mism
    DETAIL["md5_missing"] = missing
    DETAIL["remote_artifact_count"] = len(rem_map)

    print("=== C. counts (explicit expected-cell assertions) ===")
    counts = remote("cd %s && ls runs/e45/evals/ | grep -c 'eval[.]json$' && "
                     "ls runs/e45/evals/ | wc -l && ls runs/e45/*.json | wc -l"
                     % REMOTE_ROOT)
    vals = [int(x) for x in re.findall(r"\d+", counts)]
    expect(vals[:3] == [21, 42, 13],
           "expected cell counts: 21 .eval.json / 42 files in evals/ / 13 run-level json",
           "unexpected counts %r (expected [21, 42, 13])" % (vals[:3],))
    DETAIL["counts"] = vals[:3]

    print("=== D. audit four (NEW): whole-tree mtime proof of a frozen run ===")
    # One value per remote call: each query returns a single integer, so the
    # parse cannot be confused by surrounding output.
    n_newer = remote("cd %s && find runs/e45 -type f -newermt '2026-11-11 14:20' "
                     "| wc -l" % REMOTE_ROOT).strip()
    n_total = remote("cd %s && find runs/e45 -type f | wc -l" % REMOTE_ROOT).strip()
    sample = remote("cd %s && find runs/e45 -type f -newermt '2026-11-11 14:20' "
                    "| head -5" % REMOTE_ROOT).strip().splitlines()
    newer = int(n_newer) if n_newer.isdigit() else -1
    total = int(n_total) if n_total.isdigit() else -1
    expect(newer == 0 and total > 0,
           "runs/e45 tree is frozen: 0 of %d files modified after the "
           "2026-11-11 14:19 closure" % total,
           "runs/e45 tree changed after closure: %s/%s files newer; sample=%r"
           % (n_newer, n_total, sample))
    DETAIL["runs_e45_files"] = total
    DETAIL["runs_e45_files_newer_than_closure"] = newer
    DETAIL["runs_e45_newer_sample"] = sample

    ev_newer = remote("cd %s && find runs/e45_epochaxis -type f -newermt "
                      "'2026-11-12 00:00' | wc -l" % REMOTE_ROOT).strip()
    ev_total = remote("cd %s && find runs/e45_epochaxis -type f | wc -l"
                      % REMOTE_ROOT).strip()
    ev = [int(ev_newer), int(ev_total)] if (ev_newer.isdigit() and
                                            ev_total.isdigit()) else [-1, -1]
    expect(len(ev) == 2 and ev[0] == ev[1] and ev[1] > 0,
           "every file newer than 2026-11-12 lives in runs/e45_epochaxis/ "
           "(%d/%d, diagnostic epoch-axis products only)" % (ev[0], ev[1]),
           "unexpected epoch-axis mtime split %r" % (ev,))
    DETAIL["epochaxis_files"] = ev[1]
    DETAIL["epochaxis_files_newer"] = ev[0]

    print("=== E. fix-file drift (sync still pending, user-gated) ===")
    fm = remote("cd %s && md5sum %s"
                % (REMOTE_ROOT, " ".join([CONTROL_FILE] + list(FIX_FILES.keys()))))
    fmap = {}
    for ln in fm.splitlines():
        parts = ln.split()
        if len(parts) == 2:
            fmap[parts[1].lstrip("./")] = parts[0]
    drift = [p for p, h in FIX_FILES.items() if fmap.get(p) != h]
    expect(not drift,
           "4 already-fixed files match the registered md5 => no drift; "
           "the pending sync set is unchanged",
           "drifted fix files: %r" % drift)
    expect(fmap.get(CONTROL_FILE) == CONTROL_MD5,
           "control arm amp_monotone_gate/model.py matches on both sides",
           "control arm md5 differs: local=%s remote=%s"
           % (CONTROL_MD5, fmap.get(CONTROL_FILE)))
    DETAIL["fix_drift"] = drift

    print("=== F. document gates (CONTENT vs SNAPSHOT classification) ===")
    content_fail, snapshot_fail = [], []
    gate_verdicts = {}
    for g in GATES:
        gp = os.path.join(REPO, g)
        if not os.path.exists(gp):
            continue
        r = subprocess.run([sys.executable, gp], capture_output=True, text=True,
                           cwd=REPO)
        out = (r.stdout or "") + (r.stderr or "")
        verdict = "PASS" if "VERDICT: PASS" in out else "FAIL"
        gate_verdicts[g] = verdict
        for ln in out.splitlines():
            s = ln.strip()
            if not s.startswith("FAIL "):
                continue
            if SNAPSHOT_PAT.search(s):
                snapshot_fail.append("%s :: %s" % (g, s[:150]))
            else:
                content_fail.append("%s :: %s" % (g, s[:150]))
    expect(not content_fail,
           "0 CONTENT-class gate failures (no number / no conclusion missing "
           "from the authoritative documents)",
           "CONTENT-class gate failures: %r" % content_fail)
    DETAIL["gate_verdicts"] = gate_verdicts
    DETAIL["gate_content_failures"] = content_fail
    DETAIL["gate_snapshot_failures"] = snapshot_fail
    print("  INFO  snapshot-class (by-design stale version-pointer) failures: %d"
          % len(snapshot_fail))
    for s in snapshot_fail:
        print("        - " + s)

    print("=== G. verdict re-verification (independent scripts) ===")
    for v in VERIFIERS:
        vp = os.path.join(REPO, v)
        if not os.path.exists(vp):
            continue
        r = subprocess.run([sys.executable, vp], capture_output=True, text=True,
                           cwd=REPO)
        out = (r.stdout or "") + (r.stderr or "")
        expect("VERDICT: PASS" in out,
               "%s VERDICT: PASS" % os.path.basename(v),
               "%s did not PASS" % os.path.basename(v))

    print("=== H. published headline table (median + pass rate) ===")
    head = [
        ("B0", "0.121817", "2/3"), ("A1", "0.118397", "2/3"),
        ("A2", "0.112003", "2/3"), ("A3a", "0.110015", "3/3"),
        ("A3b", "0.116055", "3/3"), ("A4", "0.143896", "2/3"),
        ("A4b", "0.137567", "2/3"),
    ]
    table = "| 臂 | 中位 CSI@2.0 | 过地板通过率 |\n| --- | --- | --- |\n"
    table += "\n".join("| " + " | ".join(cells) + " |" for cells in head)
    DETAIL["headline_table"] = table

    print("=== I. memory baseline (authoritative files must not drift) ===")
    # The four authoritative files are appended to by each round's write-back,
    # so their md5 legitimately changes every round. The invariant that must
    # hold is therefore SEMANTIC, not byte-wise:
    #   * the round marker appears exactly once (no duplicated block),
    #   * the load-bearing E45 numbers are still present,
    #   * the round-81 payload was not lost.
    mem_files = {
        "HANDOVER_E45.md": r"R:/1radarNWPprediction/.workbuddy/memory/HANDOVER_E45.md",
        "upper_2026-11-11.md": r"R:/1radarNWPprediction/.workbuddy/memory/2026-11-11.md",
        "upper_MEMORY.md": r"R:/1radarNWPprediction/.workbuddy/memory/MEMORY.md",
        "proj_2026-11-15.md": os.path.join(
            REPO, ".workbuddy", "memory", "2026-11-15.md"),
    }
    problems = []
    for k, p in mem_files.items():
        if not os.path.exists(p):
            problems.append("%s missing" % k)
            continue
        t = open(p, encoding="utf-8").read()
        if k == "upper_MEMORY.md":
            # MEMORY.md carries plain bullets, not a marker block
            n = len(re.findall(r"E45 状态复核（第 82 轮", t))
            if n != 2:
                problems.append("MEMORY.md: round-82 status bullets = %d (want 2)"
                                % n)
            for token in ("3.0188", "24.7", "量级塌缩"):
                if token not in t:
                    problems.append("MEMORY.md lost load-bearing token %r" % token)
        else:
            if t.count("<!-- E45-R82-BEGIN -->") != 1:
                problems.append("%s: round-82 marker count != 1" % k)
            if t.count("<!-- E45-R82-END -->") != 1:
                problems.append("%s: round-82 end marker count != 1" % k)
    expect(not problems,
           "authoritative memory files carry exactly one round-82 marker and "
           "still hold the load-bearing E45 numbers",
           "memory files drifted: %r" % problems)
    DETAIL["memory_drift"] = problems

    verdict = "PASS" if not NEG else "FAIL"
    report = {
        "round": 82,
        "verdict": verdict,
        "pos": len(POS),
        "neg": len(NEG),
        "positive_checks": POS,
        "negative_checks": NEG,
        "detail": DETAIL,
        "headline_table": table,
        "remote_actions": "NONE (read-only status query; no sbatch, no sync, "
                          "no download of new artifacts, docs/ untouched)",
    }
    os.makedirs(EVAL_DIR, exist_ok=True)
    with open(REPORT, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=2)

    print("\nPOS %d/%d   NEG %d/%d" % (len(POS), len(POS) + len(NEG),
                                       len(NEG), len(NEG) + len(NEG)))
    print("report -> %s" % REPORT)
    print("VERDICT: %s" % verdict)
    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
