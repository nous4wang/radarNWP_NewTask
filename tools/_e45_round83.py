#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""E45 round-83: remote status re-verification (audit) + idempotent write-back.

Two phases, one file:

  python tools/_e45_round83.py audit      # READ-ONLY on remote and on docs/
  python tools/_e45_round83.py writeback  # touches ONLY memory/handover files

NEW IN ROUND 83 — three changes, all forced by evidence:

(1) TRANSPORT DISCIPLINE (new audit zero, "sentinel + count + digest").
    tools/remote.py::run() allocates a PTY and leaves its read loop as soon as
    channel.exit_status_ready() fires. A fast-finishing command can therefore
    lose its trailing stdout SILENTLY. Round 83 measured exactly that: one
    `md5sum runs/e45/evals/*` read returned 37 of 42 lines, while the same
    command read in isolation returned all 42. Every remote read in this
    script therefore
      * appends a sentinel token and treats a missing sentinel as a truncated
        payload (retry),
      * asserts the EXPECTED CELL COUNT of the payload (never "the script ran"),
      * cross-checks a multi-line manifest against an aggregate digest
        computed on the remote side,
      * and moves filtering/counting of the big listings (squeue / sacct) to
        the remote side so that only small, deterministic payloads are parsed.

(2) AUDIT THREE (new): main-run eval-tree parity, file by file.
    Rounds 76-82 compared the md5 of the 13 run-level json files only. Round
    83 compares the REMOTE md5 manifest of runs/e45/evals/* against the LOCAL
    copy (e45_evals.tar.gz members read in-memory, plus any extracted trees),
    BASENAME-NORMALISED, in BOTH directions (missing AND extra).

(3) AUDIT FOUR (new): the S3 equal-bias guard is recomputed WITHOUT assuming
    a key layout: the JSON is walked recursively, every dict carrying a
    "crosses_zero" field is collected with its path, and the expected number
    of such cells (12) and arms (6) is asserted.

Idempotency contract (same as rounds 81/82):
  * marker blocks (E45-R83-BEGIN/END) are stripped before re-append;
  * MEMORY.md bullets are matched by regex, not by a hard-coded round number;
  * table rows are built from cell lists; the separator is inserted by code.

`docs/` is intentionally NOT touched: no new numbers, no verdict change
-> MASTER stays v2.50 (v2.50 belongs to E46).
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
import tarfile

REPO = r"R:/1radarNWPprediction/radar-nwp-bench"
EVAL_DIR = os.path.join(REPO, "tools", "_evals", "e45")
REPORT = os.path.join(EVAL_DIR, "round83_status_audit.json")
REMOTE_ROOT = "/home/wf/radarNWP"
SENTINEL = "__WB_END__"

UPPER = r"R:/1radarNWPprediction/.workbuddy/memory"
PROJ = os.path.join(REPO, ".workbuddy", "memory")
HANDOVER = os.path.join(UPPER, "HANDOVER_E45.md")
UPPER_LOG = os.path.join(UPPER, "2026-11-11.md")
UPPER_MEM = os.path.join(UPPER, "MEMORY.md")
PROJ_LOG = os.path.join(PROJ, "2026-11-15.md")
AUTO_MEM = os.path.join(PROJ, "automations",
                        "e3cc6636-3e60-4a58-8f7e-67d4c896b0b6", "memory.md")

BEGIN = "<!-- E45-R83-BEGIN -->"
END = "<!-- E45-R83-END -->"

RUN_ARTIFACTS = [
    "scores_all.json", "scores_summary.json",
    "scores_summary_s0.json", "scores_summary_s1.json", "scores_summary_s2.json",
    "diag_probe_s0.json", "diag_probe_s1.json", "diag_probe_s2.json",
    "struct_metrics_all.json", "struct_metrics_s0.json",
    "struct_metrics_s1.json", "struct_metrics_s2.json", "s3_equalbias.json",
]

FIX_FILES = {
    "models/scaled_structure/model.py": "77f18b6cc2d5b4bf8d9078468d0db6c0",
    "models/topo_simvp/model.py": "f65554ca4f399431a7048c68bd07c80e",
    "tools/_e41_struct_metrics.py": "742503c872ab0e68e096eedc1cc31085",
    "tools/_e45_mechanism_probe.py": "e7d3666a0c7495999649c40cc5a2fe95",
}
CONTROL_FILE = "models/amp_monotone_gate/model.py"
CONTROL_MD5 = "e90857d34dadfabe4ad7dff6b68d5394"

TARBALL = os.path.join(EVAL_DIR, "e45_evals.tar.gz")
TARBALL_MD5_R75 = "2d859447fce615230d96b0c55536640f"

GATES = [
    "tools/_check_tables.py",
    "tools/_check_v245_row.py",
    "tools/_check_v246_row.py",
    "tools/_check_v247_row.py",
    "tools/_check_v248_row.py",
    "tools/_e45_docs_numbers.py",
]
# NAMED EXPLICITLY - never batch-scan and execute (round-82 discipline).
VERIFIERS = [
    "tools/_e45_verify_round71.py",
    "tools/_e45_verify_round78.py",
    "tools/_e45_verify_round81.py",
]

SNAPSHOT_PAT = re.compile(
    r"version pointer|changelog row is followed by a blank line|v2\.4[4-9]")

HEAD = [
    ("B0", "0.121817", "2/3"), ("A1", "0.118397", "2/3"), ("A2", "0.112003", "2/3"),
    ("A3a", "0.110015", "3/3"), ("A3b", "0.116055", "3/3"),
    ("A4", "0.143896", "2/3"), ("A4b", "0.137567", "2/3"),
]

POS: list[str] = []
NEG: list[str] = []
DETAIL: dict = {}
TRUNCATIONS: list[str] = []


def ok(msg: str) -> None:
    POS.append(msg)
    print("  PASS  " + msg)


def bad(msg: str) -> None:
    NEG.append(msg)
    print("  FAIL  " + msg)


def expect(cond: bool, msg_ok: str, msg_fail: str) -> bool:
    (ok if cond else bad)(msg_ok if cond else msg_fail)
    return cond


def md5_bytes(b: bytes) -> str:
    return hashlib.md5(b).hexdigest()


def md5(path: str) -> str:
    with open(path, "rb") as fh:
        return md5_bytes(fh.read())


def read(path: str) -> str:
    with open(path, encoding="utf-8") as fh:
        return fh.read()


def write(path: str, text: str) -> None:
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(text)


def row(cells: list) -> str:
    """Build a markdown table row from a cell list; the separator is added here."""
    return "| " + " | ".join(str(c) for c in cells) + " |"


def table(header: list, rows: list) -> str:
    out = [row(header), row(["---"] * len(header))]
    out.extend(row(r) for r in rows)
    return "\n".join(out)


HEAD_TABLE = table(["臂", "中位 CSI@2.0", "过地板通过率"],
                   [[a, m, p] for a, m, p in HEAD])


# --------------------------------------------------------------------------
# transport-safe remote reads
# --------------------------------------------------------------------------
def _remote(cmd: str) -> str:
    r = subprocess.run(
        [sys.executable, os.path.join(REPO, "tools", "remote.py"), "--", cmd],
        capture_output=True, text=True, cwd=REPO)
    return (r.stdout or "").replace("\r", "")


def remote_raw(cmd: str, tries: int = 4) -> tuple[str, bool]:
    """Fetch a remote payload with a sentinel; a missing sentinel == truncation."""
    payload, ok = "", False
    for i in range(tries):
        out = _remote("( %s ) ; echo %s" % (cmd, SENTINEL))
        if SENTINEL in out:
            payload = out.split(SENTINEL)[0]
            ok = True
            if i:
                TRUNCATIONS.append("retry %d/%d: %s" % (i, tries, cmd[:60]))
            break
        TRUNCATIONS.append("truncated payload (attempt %d/%d): %s"
                           % (i + 1, tries, cmd[:60]))
        print("  WARN  truncated remote payload (attempt %d/%d) -> retry"
              % (i + 1, tries))
    return payload, ok


def remote_one_line(cmd: str, tries: int = 4) -> tuple[str, bool]:
    """Run a command whose output is exactly one line; return that line."""
    payload, ok = remote_raw("cd %s && %s" % (REMOTE_ROOT, cmd), tries)
    lines = [ln.strip() for ln in payload.splitlines() if ln.strip()]
    if len(lines) != 1:
        return (lines[0] if lines else ""), False
    return lines[0], ok


def remote_int(cmd: str, tries: int = 4) -> tuple[int, bool]:
    line, ok = remote_one_line(cmd, tries)
    m = re.findall(r"-?\d+", line)
    return (int(m[0]) if m else -1), ok


def remote_pairs(cmd: str, n_expected: int, tries: int = 5) -> tuple[list, bool]:
    """Fetch an md5sum-style manifest as ONE line; assert the pair count."""
    pairs, ok = [], False
    for i in range(tries):
        line, sok = remote_one_line(
            "printf '%%s' \"$( %s | tr '\\n' '|' )\" | tr -d '\\r'" % cmd, tries=1)
        segs = [s.strip() for s in line.split("|") if s.strip()]
        cand = []
        for s in segs:
            p = s.split()
            if len(p) == 2 and len(p[0]) == 32 and re.fullmatch(r"[0-9a-f]{32}", p[0]):
                cand.append((p[0], p[1]))
        pairs = cand
        ok = sok and len(cand) == n_expected
        if ok:
            if i:
                TRUNCATIONS.append("manifest retry %d/%d recovered full %d pairs"
                                   % (i, tries, n_expected))
            break
        TRUNCATIONS.append("manifest short read (attempt %d/%d): %d/%d pairs"
                           % (i + 1, tries, len(cand), n_expected))
        print("  WARN  manifest attempt %d/%d: %d/%d pairs -> retry"
              % (i + 1, tries, len(cand), n_expected))
    return pairs, ok


# --------------------------------------------------------------------------
# AUDIT (read-only)
# --------------------------------------------------------------------------
def audit() -> int:
    print("=== A. remote authoritative chain ===")
    n_e45_q, ok_q = remote_int(
        "squeue -u wf -h -o '%%i %%j' | grep -ic e45 || true")
    expect(n_e45_q == 0 and ok_q,
           "no E45 job in the queue (remote-side count = 0, sentinel-verified)",
           "E45 job in the queue (count=%s, ok=%s)" % (n_e45_q, ok_q))
    DETAIL["squeue_e45_count"] = n_e45_q

    squeue, ok_sq = remote_one_line(
        "squeue -u wf -o '%%.10i %%.14j %%.2t %%.6M %%R' | tr '\\n' '|' | tr -d '\\r'")
    DETAIL["squeue"] = [x for x in squeue.split("|") if x.strip()]
    DETAIL["squeue_other_jobs"] = [x.split()[0] for x in DETAIL["squeue"][1:] if x.split()]
    DETAIL["squeue_other_names"] = sorted({x.split()[1] for x in DETAIL["squeue"][1:]
                                           if len(x.split()) > 1})
    print("  INFO  other project line in queue: %s"
          % ", ".join(DETAIL["squeue_other_jobs"]) or "none")

    # remote-side filtering: only the non-COMPLETED E45 jobs (small payload)
    dead_line, ok_d = remote_one_line(
        "sacct -u wf -S 2026-11-11 --format=JobID,JobName,State --noheader "
        "| awk '$2 ~ /^e45/ && $3 !~ /^COMPLETED/ {print $1}' | tr '\\n' ' '")
    dead = [x for x in dead_line.split() if x]
    KNOWN_DEAD = {"102", "103", "118", "265"}
    expect(set(dead) == KNOWN_DEAD and ok_d,
           "the only non-COMPLETED E45 jobs are exactly the known pre-fix waste "
           "set {102,103,118,265} => no new failure appeared",
           "non-COMPLETED E45 job set mismatch: %r (expected %r)"
           % (sorted(dead), sorted(KNOWN_DEAD)))
    DETAIL["known_waste_jobs_seen"] = sorted(dead)

    n_chain, ok_c = remote_int(
        "sacct -u wf -S 2026-11-11 --format=JobName,State --noheader "
        "| awk '$1 ~ /^e45/ && $2 ~ /^COMPLETED/' | wc -l")
    expect(n_chain > 0 and ok_c,
           "the E45 chain reports %d COMPLETED job records" % n_chain,
           "could not count COMPLETED E45 job records (%s, ok=%s)"
           % (n_chain, ok_c))
    DETAIL["completed_chain_records"] = n_chain

    n_rel, ok_r = remote_int(
        "sacct -u wf -S 2026-11-12 --format=JobName --noheader "
        "| grep -cE 'e45-(train|stage2|submit)' || true")
    expect(n_rel == 0 and ok_r,
           "since 2026-11-12 there is no e45-train/e45-stage2/e45-submit job "
           "=> the main chain was never re-launched",
           "the main chain was re-launched after closure (%s records)" % n_rel)
    DETAIL["relaunched_main_chain_records"] = n_rel

    print("=== B. artifact parity (13 run-level json) ===")
    pairs, ok13 = remote_pairs("md5sum runs/e45/*.json", len(RUN_ARTIFACTS))
    rem_map = {os.path.basename(p): h for h, p in pairs}
    mism, missing = [], []
    for name in RUN_ARTIFACTS:
        lp = os.path.join(EVAL_DIR, name)
        if name not in rem_map or not os.path.exists(lp):
            missing.append(name)
            continue
        if md5(lp) != rem_map[name]:
            mism.append(name)
    expect(ok13 and not missing and not mism,
           "13/13 run-level artifacts byte-identical to remote (md5, "
           "count-verified manifest)",
           "artifact mismatch/missing: missing=%r mismatch=%r (manifest_ok=%s)"
           % (missing, mism, ok13))
    DETAIL["md5_mismatches"] = mism
    DETAIL["md5_missing"] = missing
    DETAIL["remote_artifact_count"] = len(rem_map)

    print("=== C. counts (explicit expected-cell assertions) ===")
    cnt_line, ok_cnt = remote_one_line(
        "echo \"$(ls runs/e45/evals/ | grep -c 'eval[.]json$') "
        "$(ls runs/e45/evals/ | wc -l) $(ls runs/e45/*.json | wc -l)\"")
    vals = [int(x) for x in re.findall(r"\d+", cnt_line)]
    expect(ok_cnt and vals[:3] == [21, 42, 13],
           "expected cell counts: 21 .eval.json / 42 files in evals/ / 13 run-level json",
           "unexpected counts %r (expected [21, 42, 13], ok=%s)" % (vals[:3], ok_cnt))
    DETAIL["counts"] = vals[:3]

    print("=== D. AUDIT THREE (NEW): main-run eval-tree parity, 42 files ===")
    comp_line, ok_comp = remote_one_line(
        "echo \"$(find runs/e45/evals -type f | wc -l) "
        "$(find runs/e45/evals -mindepth 1 -type d | wc -l) "
        "$(ls -1A runs/e45/evals | wc -l)\"")
    comp = [int(x) for x in re.findall(r"\d+", comp_line)]
    expect(ok_comp and comp[:3] == [42, 0, 42],
           "remote evals/ = 42 regular files, 0 subdirectories, 42 entries",
           "remote evals/ composition unexpected: %r (ok=%s)" % (comp[:3], ok_comp))
    DETAIL["evtree_remote_composition"] = comp[:3]

    ev_pairs, ok_ev = remote_pairs("md5sum runs/e45/evals/*", 42)
    ev_map = {os.path.basename(p): h for h, p in ev_pairs}
    # aggregate digest cross-check (independent of the pair count)
    agg, ok_agg = remote_one_line("md5sum runs/e45/evals/* | md5sum")
    local_agg = md5_bytes("".join("%s  %s\n" % (h, p) for h, p in ev_pairs)
                          .encode("utf-8"))
    expect(ok_agg and agg.split()[0] == local_agg,
           "the received 42-line manifest reproduces the remote aggregate "
           "digest (%s) => the manifest is complete, not truncated"
           % (agg.split()[0][:12] if agg else "?"),
           "aggregate digest mismatch: remote=%r local=%r (ok=%s)"
           % (agg[:40], local_agg, ok_agg))
    DETAIL["evtree_aggregate_digest"] = local_agg
    DETAIL["evtree_aggregate_match"] = bool(ok_agg and agg.split()
                                            and agg.split()[0] == local_agg)

    local_map = {}
    tar_files = tar_dirs = 0
    if os.path.exists(TARBALL):
        with tarfile.open(TARBALL, "r:gz") as tf:
            for m in tf.getmembers():
                if m.isdir():
                    tar_dirs += 1
                    continue
                if not m.isfile():
                    continue
                fh = tf.extractfile(m)
                if fh is None:
                    continue
                tar_files += 1
                local_map[os.path.basename(m.name)] = md5_bytes(fh.read())
    for sub in ("evals_raw", "_raw"):
        root = os.path.join(EVAL_DIR, sub)
        if not os.path.isdir(root):
            continue
        for dirpath, _d, files in os.walk(root):
            for f in files:
                if f.endswith(".json"):
                    local_map.setdefault(f, md5(os.path.join(dirpath, f)))
    ev_missing = sorted(set(ev_map) - set(local_map))
    ev_extra = sorted(set(local_map) - set(ev_map))
    ev_mism = sorted(n for n in set(ev_map) & set(local_map)
                      if ev_map[n] != local_map[n])
    expect(ok_ev and len(ev_map) == 42 and tar_files == 42 and
           not ev_missing and not ev_mism and not ev_extra,
           "42/42 main-run eval files (21 .eval.json + 21 .windows.json) "
           "byte-identical to remote, both directions (missing 0 / extra 0 / "
           "mismatch 0)",
           "eval-tree parity broken: manifest_ok=%s remote=%d tar_files=%d "
           "missing=%r extra=%r mismatch=%r"
           % (ok_ev, len(ev_map), tar_files, ev_missing[:8], ev_extra[:8],
              ev_mism[:8]))
    DETAIL["evtree_remote"] = len(ev_map)
    DETAIL["evtree_tar_files"] = tar_files
    DETAIL["evtree_tar_dirs"] = tar_dirs
    DETAIL["evtree_missing"] = ev_missing
    DETAIL["evtree_extra_local"] = ev_extra
    DETAIL["evtree_mismatch"] = ev_mism
    tb_md5 = md5(TARBALL) if os.path.exists(TARBALL) else ""
    DETAIL["tarball_md5_now"] = tb_md5
    DETAIL["tarball_md5_r75"] = TARBALL_MD5_R75
    print("  INFO  tarball md5 now=%s (round-75 registered=%s)"
          % (tb_md5, TARBALL_MD5_R75))

    print("=== E. fix-file drift (sync still pending, user-gated) ===")
    fix_cmd = "md5sum %s" % " ".join([CONTROL_FILE] + list(FIX_FILES.keys()))
    fix_pairs, ok_fix = remote_pairs(fix_cmd, len(FIX_FILES) + 1)
    fmap = {p.lstrip("./"): h for h, p in fix_pairs}
    drift = [p for p, h in FIX_FILES.items() if fmap.get(p) != h]
    expect(ok_fix and not drift,
           "4 already-fixed files match the registered md5 => no drift; "
           "the pending sync set is unchanged",
           "fix-file drift: %r (manifest_ok=%s)" % (drift, ok_fix))
    expect(fmap.get(CONTROL_FILE) == CONTROL_MD5,
           "control arm amp_monotone_gate/model.py matches on both sides",
           "control arm md5 differs: local=%s remote=%s"
           % (CONTROL_MD5, fmap.get(CONTROL_FILE)))
    DETAIL["fix_drift"] = drift

    print("=== F. run-tree mtime freeze + submit-log semantics ===")
    newer, ok_n = remote_int(
        "find runs/e45 -type f -newermt '2026-11-11 14:20' | wc -l")
    total, ok_t = remote_int("find runs/e45 -type f | wc -l")
    expect(ok_n and ok_t and newer == 0 and total > 0,
           "runs/e45 tree is frozen: 0 of %d files modified after the "
           "2026-11-11 14:19 closure" % total,
           "runs/e45 tree changed after closure: %s/%s (ok=%s/%s)"
           % (newer, total, ok_n, ok_t))
    DETAIL["runs_e45_files"] = total
    DETAIL["runs_e45_files_newer_than_closure"] = newer

    sub_line, ok_sub = remote_one_line(
        "tail -8 run_logs/slurm/e45-submit-273.out | tr '\\n' '|' | tr -d '\\r'")
    expect(ok_sub and "summary job id : 291" in sub_line,
           "submitter 273 log names the summary job 291",
           "submitter 273 log does not name summary job 291: %r" % sub_line)
    expect("the summary (job 291) finished: COMPLETED" in sub_line,
           "submitter 273 log confirms summary 291 COMPLETED => the S3 "
           "equal-bias control was run automatically (no manual sbatch)",
           "submitter 273 log does not confirm summary 291 completion")
    expect("E45 submission finished 2026-11-11T14:20:42+08:00" in sub_line,
           "submitter 273 log carries the canonical closure timestamp",
           "submitter 273 log misses the canonical closure timestamp")
    DETAIL["submit273_tail"] = [x for x in sub_line.split("|") if x.strip()]

    dirs_line, ok_dirs = remote_one_line("ls -d runs/* | tr '\\n' '|' | tr -d '\\r'")
    e45_dirs = [x.strip() for x in dirs_line.split("|") if "e45" in x.lower()]
    DETAIL["runs_e45_entries"] = e45_dirs
    no_e46 = not any(re.search(r"e4[6-9]|e5\d", d) for d in e45_dirs)
    expect(no_e46 and ok_dirs,
           "no E46+ run directory exists (E45 stays the latest E-series run)",
           "an E46+ run directory exists: %r" % e45_dirs)
    ping, _ = remote_one_line("scontrol ping")
    DETAIL["controller_ping"] = ping[:120]
    print("  INFO  controller: %s" % (ping[:90] or "n/a"))

    print("=== G. document gates (CONTENT vs SNAPSHOT classification) ===")
    content_fail, snapshot_fail, gate_verdicts = [], [], {}
    for g in GATES:
        gp = os.path.join(REPO, g)
        if not os.path.exists(gp):
            continue
        r = subprocess.run([sys.executable, gp], capture_output=True, text=True,
                           cwd=REPO)
        out = (r.stdout or "") + (r.stderr or "")
        gate_verdicts[g] = "PASS" if "VERDICT: PASS" in out else "FAIL"
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

    print("=== H. verdict re-verification (named scripts only) ===")
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

    print("=== I. AUDIT FOUR (NEW): S3 guard recomputed by structural discovery ===")
    s3_path = os.path.join(EVAL_DIR, "s3_equalbias.json")
    cells = []

    def walk(node, path):
        if isinstance(node, dict):
            if "crosses_zero" in node:
                cells.append((tuple(path), node))
            for k, v in node.items():
                walk(v, path + [str(k)])
        elif isinstance(node, list):
            for i, v in enumerate(node):
                walk(v, path + [str(i)])

    if os.path.exists(s3_path):
        with open(s3_path, encoding="utf-8") as fh:
            walk(json.load(fh), [])
    expect(len(cells) == 12,
           "S3 equal-bias guard exposes exactly 12 CI cells (6 arms x 2 variants)",
           "S3 guard exposes %d CI cells (expected 12)" % len(cells))
    arms = set()
    for path, _n in cells:
        arms.add(path[1] if len(path) >= 2 and path[0] in ("arms", "arm")
                 else path[0] if path else "?")
    all_cross = all(bool(n.get("crosses_zero")) for _p, n in cells)
    expect(all_cross,
           "all 12 S3 CI cells cross zero => no arm is qualified",
           "at least one S3 CI cell does NOT cross zero")
    expect(len(arms) == 6,
           "the 12 S3 CI cells belong to 6 distinct arms",
           "S3 CI cells resolve to %d arms (expected 6): %r"
           % (len(arms), sorted(arms)))
    DETAIL["s3_cells"] = len(cells)
    DETAIL["s3_arms"] = sorted(arms)
    DETAIL["s3_all_cross_zero"] = all_cross
    DETAIL["s3_sample_paths"] = [" > ".join(p) for p, _n in cells[:4]]

    print("=== J. memory baseline (previous round's payload must survive) ===")
    mem_files = {"HANDOVER_E45.md": HANDOVER, "upper_2026-11-11.md": UPPER_LOG,
                 "proj_2026-11-15.md": PROJ_LOG, "auto_memory.md": AUTO_MEM}
    problems = []
    for k, p in mem_files.items():
        if not os.path.exists(p):
            problems.append("%s missing" % k)
            continue
        t = read(p)
        if t.count("<!-- E45-R82-BEGIN -->") != 1:
            problems.append("%s: round-82 marker count != 1" % k)
        if t.count("<!-- E45-R82-END -->") != 1:
            problems.append("%s: round-82 end marker count != 1" % k)
    if not os.path.exists(UPPER_MEM):
        problems.append("upper_MEMORY.md missing")
    else:
        mt = read(UPPER_MEM)
        n82 = len(re.findall(r"E45 状态复核（第 82 轮", mt))
        n_all = len(re.findall(r"E45 状态复核（第 \d+ 轮", mt))
        if n82 != 2 or n_all != 2:
            problems.append("MEMORY.md: round-82 bullets=%d total=%d (want 2/2)"
                            % (n82, n_all))
        for token in ("3.0188", "24.7", "量级塌缩"):
            if token not in mt:
                problems.append("MEMORY.md lost load-bearing token %r" % token)
    expect(not problems,
           "authoritative memory files carry exactly one round-82 marker and "
           "still hold the load-bearing E45 numbers",
           "memory files drifted: %r" % problems)
    DETAIL["memory_drift"] = problems

    verdict = "PASS" if not NEG else "FAIL"
    report = {
        "round": 83,
        "verdict": verdict,
        "pos": len(POS), "neg": len(NEG),
        "positive_checks": POS, "negative_checks": NEG,
        "detail": DETAIL,
        "headline_table": HEAD_TABLE,
        "transport_truncations": TRUNCATIONS,
        "remote_actions": "NONE (read-only status query; no sbatch, no sync, "
                          "no download of new artifacts, docs/ untouched)",
    }
    os.makedirs(EVAL_DIR, exist_ok=True)
    with open(REPORT, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=2)

    print("\nPOS %d/%d   NEG %d/%d" % (len(POS), len(POS) + len(NEG),
                                       len(NEG), len(NEG) + len(NEG)))
    print("transport truncations observed: %d" % len(TRUNCATIONS))
    print("report -> %s" % REPORT)
    print("VERDICT: %s" % verdict)
    return 0 if verdict == "PASS" else 1


# --------------------------------------------------------------------------
# WRITE-BACK (memory files only; docs/ deliberately untouched)
# --------------------------------------------------------------------------
def strip_marker(text: str) -> str:
    pat = re.compile(re.escape(BEGIN) + r".*?" + re.escape(END) + r"\r?\n?", re.S)
    return pat.sub("", text)


def append_block(path: str, block: str) -> None:
    text = strip_marker(read(path))
    text = text.rstrip() + "\n\n" + block.rstrip() + "\n"
    write(path, text)


def build_blocks() -> tuple[str, str, str, str]:
    with open(REPORT, encoding="utf-8") as fh:
        rep = json.load(fh)
    if rep.get("verdict") != "PASS":
        raise SystemExit("audit verdict is %r - refusing to write back"
                         % rep.get("verdict"))
    d = rep["detail"]
    other_jobs = ", ".join(d.get("squeue_other_jobs", [])) or "none"
    other_names = ", ".join(d.get("squeue_other_names", [])) or "none"
    waste = ", ".join(d.get("known_waste_jobs_seen", [])) or "none"
    ntr = len(rep.get("transport_truncations", []))

    handover = f"""{BEGIN}
### 2026-11-15｜第 83 轮（E45 远端运行状态复核：权威链终态 ＋ **主跑评测树 42 份逐文件 md5 一致性** ＋ **S3 护栏结构化重算** ＋ **远端读取传输纪律**）

- **判定**：远端 E45 **全部完成、无新活动、裁决不变**（六臂判负、无一臂进十 seed、不投十 seed 算力）。**远端动作：零**（只读状态查询；未 sbatch、未同步、未下载、未改远端状态）。
- **远端**：权威链 **273（提交器）／274·282·290（stage2）／275·283（训练数组）／291（summary）／297·298（诊断）／314（gpuprobe）／315·316·317（epoch 轴）全部 `COMPLETED`**（远端侧计数 **{d.get('completed_chain_records')}** 条 COMPLETED 记录）；提交器 273 日志尾部仍写明 `summary job id : 291` ＋ `the summary (job 291) finished: COMPLETED` ＋ `=== E45 submission finished 2026-11-11T14:20:42+08:00 ===` ⇒ **S3 等偏置对照当年即由汇总作业自动跑完（未手动 sbatch）**；`runs/e45/evals/` **42 份**（**21 份 `.eval.json` ＋ 21 份 `.windows.json`，0 个子目录**）、`runs/e45/` **13 份** run 级 json 齐；队列里 E45 作业数 = **{d.get('squeue_e45_count')}**，在跑的是 {other_jobs} 个 **`{other_names}`**，属同一 `wf` 账号下的**另一条项目线**（与 E45 无关，不干预）；`runs/` 下**无 E46+ 目录**。**易误读点（沿用第 73 轮）**：`sacct` 里的 **{waste}** 是修传参 bug 前的**废作业**，**不属权威链**（本轮继续把它单独断言为「已知的废作业集合且集合未变」，避免新失败藏在旧噪声里）。
- **审计一（产物一致性）**：13 份 run 级 json 的远端 md5 与本地 `tools/_evals/e45/` **13/13 逐位相同、0 处漂移**；显式断言期望格数 **21 份评测 / 42 份 evals 目录 / 13 份 run 级 json**。
- **审计二（已修实现缺陷的同步漂移）**：远端 4 个已修文件 md5 与第 71–82 轮登记值**逐位相同**（`scaled_structure/model.py` `77f18b6c…`、`topo_simvp/model.py` `f65554ca…`、`tools/_e41_struct_metrics.py` `742503c8…`、`tools/_e45_mechanism_probe.py` `e7d3666a…`），**对照组 `amp_monotone_gate/model.py` 双方同为 `e90857d3…`** ⇒ **待同步最小集仍是这 4 个文件** ⇒ **未同步即未重训重测 ⇒ 裁决不变**。
- **审计三（本轮新增：主跑评测树逐文件一致性）**：前 82 轮只核 **13 份 run 级 json**——那只证明**已发布裁决产物**没漂移，**没有逐文件核过 42 份主跑评测树**。本轮取远端 `md5sum runs/e45/evals/*` 清单（远端侧实测：**普通文件 42、��目录 0、目录条目 42**），与本地 `e45_evals.tar.gz` 的成员（内存直读，不落盘；**{d.get('evtree_tar_files')} 个文件成员、{d.get('evtree_tar_dirs')} 个目录成员**）**按 basename 归一**后**双向**逐份比对：**缺失 0、多余 0、逐位不一致 0** ⇒ **21 份 `.eval.json` ＋ 21 份 `.windows.json` 全部与远端逐位相同**。
- **审计四（本轮新增：S3 护栏结构化重算）**：不假设键布局，直接对 `s3_equalbias.json` **递归发现**所有带 `crosses_zero` 的字典节点：共 **{d.get('s3_cells')} 个 CI 格**（断言 = 12），归属于 **{len(d.get('s3_arms', []))} 条臂**（断言 = 6），**全部跨零** ⇒ **合格臂集合为空**。取样路径形如 `{d.get('s3_sample_paths', ['n/a'])[0]}`。
- **审计五（运行树冻结）**：`runs/e45/` 共 **{d.get('runs_e45_files')}** 份文件，**结案（2026-11-11 14:19）之后被修改的 = {d.get('runs_e45_files_newer_than_closure')} 份**；`sacct -S 2026-11-12` **无任何 e45-train／e45-stage2／e45-submit** ⇒ **主链从未重启**。
- **审计六（本轮新增且是本轮真正的技术收获：远端读取的传输纪律）**：本轮首次**实测到远端读取会静默截断**——`tools/remote.py::run()` 分配了 PTY，且其读循环在 `channel.exit_status_ready()` 成立时立刻 break；命令结束得够快时，**stdout 的尾部还没到就被丢弃**。证据：同一条 `md5sum runs/e45/evals/*` 在审计进程里只收到 **37/42 行**，而单独读取时是完整的 **42 行** ⇒ **旧式「直接读多行 md5sum 输出再解析」会把截断当成内容缺失**（首版审计就误报「远端 37 份、期望 42 份」）。⇒ 新纪律落地为四件事：① **哨兵**（每条远端命令后接 `echo __WB_END__`，缺哨兵即判定截断并重试）；② **计数校验**（断言期望格数，不是「脚本跑通」）；③ **聚合摘要交叉验证**（远端 `md5sum … | md5sum` 与本地按收到顺序重建的清单摘要比对，本轮摘要一致＝清单完整）；④ **把过滤/计数放到远端做**（`squeue`/`sacct` 的判定改成远端 `awk|grep -c` 后只回传小而确定的载荷）。⇒ **加固后的读者本轮重试 {ntr} 次即取全**（加固前的那次直读曾静默截断到 37/42 行）。
- **文档闸门**：CONTENT／SNAPSHOT 分类后 **CONTENT 类失败 = {len(d.get('gate_content_failures', []))}**（余项全是「版本指针须停在旧版」类快照断言，与实验数字无关）。
- **裁决复验**：`tools/_e45_verify_round71.py`／`_e45_verify_round78.py`／`_e45_verify_round81.py` **均 PASS**；本轮 `tools/_e45_round83.py` **POS {rep['pos']}/{rep['pos'] + rep['neg']} VERDICT: PASS** ⇒ **六臂判负、无一臂进十 seed、不投十 seed 算力**。

{HEAD_TABLE}

- **写回**：**未改 `docs/`**（无新数字、无裁决变化 ⇒ **不升版**，MASTER 仍 **v2.50**，v2.50 属 E46）；仅按惯例把本轮登记进 HANDOVER／逐日日志／MEMORY 状态段。
- **本轮产物**：`tools/_e45_round83.py`、`tools/_evals/e45/round83_status_audit.json`。
- **三项待用户确认仍挂着**（未擅自执行）：① E46 第二轮探针结果是否写回 MASTER（＝升 v2.51）；② 4 个修复文件按最小集同步远端；③ dx=3px／20–400mm 扫描须先预注册。
- **纪律**：① 仍按「**中位数 + 通过率**」报，单点分数不作裁决；② 事后复核、**不构成新判据、不改变任何裁决**；③ **MASTER 不升版**（仍 v2.50）。
{END}
"""

    proj = f"""{BEGIN}
### E45 第 83 轮（2026-11-15）｜远端运行状态复核 ＋ 主跑评测树 42 份逐文件 md5 一致性 ＋ S3 护栏结构化重算 ＋ 远端读取传输纪律

- 远端权威链 273/274·282·290/275·283/291/297·298/314/315·316·317 全 `COMPLETED`；21 份 `.eval.json` 齐；13 份 run 级 json 远端↔本地 md5 13/13 全同；S3 由 summary 291 当年自动跑完；无 E46+ 目录；队列里 {other_jobs} 个 `{other_names}` 属另一项目线、不干预。**远端动作：零**。
- 审计三（本轮新增）：远端 `runs/e45/evals/` 实测 **42 普通文件 / 0 子目录 / 42 条目**；与本地 tar 成员（{d.get('evtree_tar_files')} 文件成员）按 basename 归一**双向**逐份比对 ⇒ **缺失 0、多余 0、逐位不一致 0**。
- 审计四（本轮新增）：S3 护栏**不假设键布局**、递归发现 `crosses_zero` 节点 ⇒ **{d.get('s3_cells')} 个 CI 格**（断言 = 12）、**{len(d.get('s3_arms', []))} 条臂**（断言 = 6）、**全部跨零** ⇒ 合格臂集合为空。
- 审计六（本轮新增的技术收获）：**实测到 `remote.py` 的 PTY 读循环会在 stdout 尾部到达前提前 break**，同一条 `md5sum` 在审计进程里只收到 37/42 行、单独读则是 42/42 ⇒ **远端读取必须哨兵 + 计数校验 + 聚合摘要交叉验证**（加固后读者重试 {ntr} 次即取全；加固前的直读曾静默截断到 37/42 行）。
- 审计二/五：同步漂移仍恰为 4 个已修文件（对照组 md5 相同）⇒ 裁决不变；`runs/e45/` {d.get('runs_e45_files')} 份文件结案后被修改 = {d.get('runs_e45_files_newer_than_closure')} 份；`sacct -S 2026-11-12` 无任何 e45-train／e45-stage2／e45-submit ⇒ 主链从未重启。
- 文档闸门 CONTENT 类失败 = {len(d.get('gate_content_failures', []))}；`_e45_verify_round71/78/81` 均 PASS；本轮 POS {rep['pos']}/{rep['pos'] + rep['neg']} ⇒ **六臂判负、不投十 seed 算力**。
- **未改 `docs/`**（无新数字、无裁决变化 ⇒ 不升版，MASTER 仍 v2.50）。产物：`tools/_e45_round83.py`、`tools/_evals/e45/round83_status_audit.json`。
{END}
"""

    mem_concl = (
        "- **E45 状态复核（第 83 轮，2026-11-15）**：远端权威链全 `COMPLETED`、21 份评测齐、"
        "13 份 run 级 json 与本地 md5 **13/13 逐位相同**、E45 队列空（他项目线在跑）、"
        "无 E46+ 目录 ⇒ **E45 无新活动**。文档闸门 CONTENT 类失败 = "
        f"{len(d.get('gate_content_failures', []))}。"
        "**本轮新增三项审计**：① **主跑评测树 42 份逐文件 md5 一致性**（远端 42 普通文件/0 子目录 "
        "vs 本地 tar 42 文件成员，按 basename 归一双向比对，缺失 0、多余 0、逐位不一致 0）；"
        "② **S3 护栏结构化重算**（递归发现 `crosses_zero` 节点 "
        f"{d.get('s3_cells')} 格 / {len(d.get('s3_arms', []))} 臂，全部跨零 ⇒ 合格臂集合为空）；"
        "③ **远端读取传输纪律**——实测 `remote.py` 的 PTY 读循环会在 stdout 尾部到达前提前 break，"
        f"同一条 `md5sum` 在审计进程里只收到 37/42 行、单独读则是 42/42 ⇒ **远端读取必须"
        "哨兵 + 计数校验 + 聚合摘要交叉验证**，否则会把传输截断误读成内容缺失。"
        "**裁决不变**（六臂判负、不投十 seed 算力）。审计脚本 `tools/_e45_round83.py`"
        f"（POS {rep['pos']}/{rep['pos'] + rep['neg']} VERDICT: PASS）。"
    )

    mem_status = (
        "- **E45 状态复核（第 83 轮，2026-11-15，远端终态）**：远端权威链全 `COMPLETED`"
        "（273／274·282·290／275·283／291／297·298／314／315·316·317）、`runs/e45/evals/` 42 份"
        "（21 `.eval.json` ＋ 21 `.windows.json`）、13 份 run 级 json 远端↔本地 md5 **13/13 全同**；"
        "S3 由 summary 291 当年自动跑完（未手动 sbatch）；队列里 " + other_jobs + " 个 `"
        + other_names + "` 属**另一条项目线**、不干预；无 E46+ 目录 ⇒ **E45 无新活动**。"
        "**本轮新增三项审计面**：① **主跑评测树逐文件一致性**——远端实测 **42 普通文件 / 0 子目录 / "
        f"42 条目**，与本地 `e45_evals.tar.gz` 的 {d.get('evtree_tar_files')} 个文件成员**按 basename 归一**"
        "**双向**逐份比对，**缺失 0、多余 0、逐位不一致 0**；② **S3 护栏结构化重算**——不假设键布局、"
        f"递归发现所有带 `crosses_zero` 的节点 ⇒ **{d.get('s3_cells')} 个 CI 格**（断言 = 12）、"
        f"**{len(d.get('s3_arms', []))} 条臂**（断言 = 6）、**全部跨零** ⇒ **合格臂集合为空**；"
        "③ **远端读取传输纪律**——本轮**实测到 `tools/remote.py::run()` 的 PTY 读循环会在 stdout 尾部"
        f"到达前提前 break**（同一条 `md5sum runs/e45/evals/*` 在审计进程里只收到 37/42 行、单独读是 42/42）"
        "⇒ 新纪律：**每条远端读取必须带哨兵（`echo __WB_END__`）、必须断言期望格数、必须用聚合摘要"
        f"交叉验证、过滤与计数尽量放到远端做**（加固后读者重试 {ntr} 次即取全；加固前的直读曾静默截断到 37/42 行）。"
        f"同步漂移仍恰为 4 个已修文件（对照组 md5 相同）⇒ 未同步即未重训重测 ⇒ **裁决不变**；"
        f"`runs/e45/` {d.get('runs_e45_files')} 份文件结案后被修改 = "
        f"{d.get('runs_e45_files_newer_than_closure')} 份，且 `sacct -S 2026-11-12` **无任何 "
        "e45-train／e45-stage2／e45-submit** ⇒ 主链从未重启。文档闸门 CONTENT 类失败 = "
        f"{len(d.get('gate_content_failures', []))}；`_e45_verify_round71`／`_e45_verify_round78`／"
        "`_e45_verify_round81` 均 PASS；本轮 POS "
        f"{rep['pos']}/{rep['pos'] + rep['neg']} ⇒ **六臂判负、无一臂进十 seed、不投十 seed 算力**。"
        "**未改 `docs/`**（无新数字、无裁决变化 ⇒ 不升版，MASTER 仍 v2.50）。"
    )
    return handover, proj, mem_concl, mem_status


def writeback() -> int:
    handover, proj, mem_concl, mem_status = build_blocks()
    for t in (HANDOVER, UPPER_LOG, PROJ_LOG, UPPER_MEM, AUTO_MEM):
        if not os.path.isfile(t):
            raise SystemExit("missing write-back target: %s" % t)

    append_block(HANDOVER, handover)
    append_block(UPPER_LOG, handover)
    append_block(PROJ_LOG, proj)
    append_block(AUTO_MEM, proj)

    text = read(UPPER_MEM)
    pat = re.compile(r"^- \*\*E45 状态复核（第 \d+ 轮[^\n]*\n", re.M)
    matches = list(pat.finditer(text))
    if len(matches) < 2:
        raise SystemExit("EXPECTED >=2 E45 status bullets in MEMORY.md, found %d "
                         "- refusing to write silently" % len(matches))
    for idx, repl in ((1, mem_status), (0, mem_concl)):
        m = matches[idx]
        text = text[:m.start()] + repl + "\n" + text[m.end():]
    write(UPPER_MEM, text)

    for t in (HANDOVER, UPPER_LOG, PROJ_LOG, AUTO_MEM):
        txt = read(t)
        n = txt.count(BEGIN)
        if n != 1:
            raise SystemExit("%s: marker count %d != 1" % (t, n))
        print("%-22s md5=%s  markers=%d" % (os.path.basename(t), md5(t), n))
    mem_txt = read(UPPER_MEM)
    n83 = len(re.findall(r"E45 状态复核（第 83 轮", mem_txt))
    n_all = len(re.findall(r"E45 状态复核（第 \d+ 轮", mem_txt))
    if n83 != 2 or n_all != 2:
        raise SystemExit("MEMORY.md: expected 2 round-83 status bullets, got %d "
                         "(total %d)" % (n83, n_all))
    print("%-22s md5=%s  bullets=%d" % ("MEMORY.md", md5(UPPER_MEM), n83))
    print("WRITEBACK: OK (docs/ intentionally untouched)")
    return 0


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "audit"
    if mode == "audit":
        sys.exit(audit())
    elif mode == "writeback":
        sys.exit(writeback())
    else:
        raise SystemExit("usage: _e45_round83.py [audit|writeback]")
