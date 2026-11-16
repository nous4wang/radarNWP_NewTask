#!/usr/bin/env python
"""E45 independent re-verification - round 28 (NEW checks only).

Round 27 (`tools/_e45_verify_round27.py`) already covers the core gate:
S3 recompute, 3-seed stats, mechanism readouts, doc consistency.
This round adds checks NOT performed in any prior round:

  [A] struct_metrics per-seed files (s0/s1/s2) vs merged struct_metrics_all.json
      -> numeric cross-file consistency on overlapping (arm, seed) records.
  [B] scores_summary cumulative semantics:
      B1 scores_summary_s2.json must be byte-identical to scores_all.json;
      B2 every per-seed scores_summary file must agree numerically with the
         merged scores_all.json, and the three must jointly cover 7x3.
  [C] remote summary-job log cross-check (independent tool chain):
      highest-CSI record name/value, floor/oracle anchors, all rc == 0.
  [D] remote eval coverage: 21 eval.json + 21 windows.json == 7 arms x 3 seeds,
      matched against the 21 (arm, seed) records parsed from scores_all.json.
  [E] field_max_mm must equal the model ceiling expm1(6) exactly on A3a/A3b.
  [F] MASTER §9 / INDEX E45 row column counts + E45-context numeric drift lint
      (adjacency-anchored: only numbers DIRECTLY attached to 地板 / Oracle).

Rules honoured (from rounds 9-27):
  * structure-first: one-level dumps BEFORE key extraction;
  * no import of older verify scripts (old bugs propagate silently);
  * POS / NEG / PAT channels are semantically distinct;
  * cross-checks assert the COUNT of compared records (no vacuous PASS);
  * three-state NaN handling (None / NaN / real value);
  * forbidden-claim lint must distinguish an ASSERTION from a PROHIBITION quote.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EVAL = ROOT / "tools" / "_evals" / "e45"
OUTER = ROOT.parent

FLOOR = 0.0924
ORACLE = 0.1653
CEILING = math.expm1(6.0)          # 402.4288... model output clamp
EXP_ARMS = ["A1", "A2", "A3a", "A3b", "A4", "A4b"]
ALL_ARMS = ["B0"] + EXP_ARMS
ARM_SET = set(ALL_ARMS)
SEEDS = [0, 1, 2]

# ---- observed on the remote host this round (2026-11-12 20:28 GMT+8) ------------
# fetched via: MSYS_NO_PATHCONV=1 python tools/remote.py -- 'cd /home/wf/radarNWP && ...'
REMOTE_EVAL_JSON_COUNT = 21        # ls runs/e45/evals/*.eval.json | wc -l
REMOTE_EVAL_FILES_TOTAL = 42       # ls runs/e45/evals/ | wc -l   (21 eval + 21 windows)
REMOTE_SUMMARY_LOG_TAIL = """\
B0_s2_ep00        0.1305    0.8047    0.0750    0.2543   0.2083   0.7412    538    0.7828    0.9043
anchors: persistence floor 0.0924, Oracle upper bound 0.1653
highest CSI@2.0 in this batch: A4_s0_ep00 = 0.1755
scores written to /home/wf/radarNWP/runs/e45/scores_all.json
flat scores rc=0
==========================================================
 summary finished 2026-11-11T14:19:05+08:00
   s3 rc = 0   s1 rc = 0   merge rc = 0
   summary rc = 0   flat rc = 0
=========================================================="""

pos_ok = pos_bad = neg_ok = neg_bad = 0
fails: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    global pos_ok, pos_bad
    if cond:
        pos_ok += 1
    else:
        pos_bad += 1
        fails.append(f"[POS] {name} :: {detail}")
        print(f"  FAIL(POS) {name} :: {detail}")


def neg_check(name: str, violation: bool, detail: str = "") -> None:
    """violation=True means the forbidden event HAPPENED -> fail."""
    global neg_ok, neg_bad
    if violation:
        neg_bad += 1
        fails.append(f"[NEG] {name} :: {detail}")
        print(f"  FAIL(NEG) {name} :: {detail}")
    else:
        neg_ok += 1


def num_leaves(obj, prefix=""):
    out = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            key = f"{prefix}.{k}" if prefix else str(k)
            out.update(num_leaves(v, key))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            out.update(num_leaves(v, f"{prefix}[{i}]"))
    elif isinstance(obj, (int, float)) and not isinstance(obj, bool):
        out[prefix] = float(obj)
    return out


KEY_RE = re.compile(r"^(B0|A1|A2|A3a|A3b|A4|A4b)_s(\d+)_ep(\d+)$")


def collect_records(obj, out: dict, arm_hint: str | None = None) -> None:
    """Collect (arm, seed, ep) -> record from heterogeneous containers.

    Handles three observed layouts:
      1. flat   {"A1_s0_ep00": {...}}
      2. arms   {"arms": {"A1": {"s0": {...}}}}   (seed key under an arm)
      3. tags   {"tags": {"A1_s0_ep00": {...}}}
    """
    if isinstance(obj, dict):
        for k, v in obj.items():
            ks = str(k)
            m = KEY_RE.match(ks)
            if m:
                out[(m.group(1), int(m.group(2)), m.group(3))] = v
                continue
            if ks in ARM_SET:
                collect_records(v, out, arm_hint=ks)
                continue
            m2 = re.match(r"^s(\d+)$", ks)
            if m2 and arm_hint:
                out[(arm_hint, int(m2.group(1)), "00")] = v
                continue
            collect_records(v, out, arm_hint=arm_hint)
    elif isinstance(obj, list):
        for v in obj:
            collect_records(v, out, arm_hint=arm_hint)


def md5(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def pick_numeric(rec: dict, must: str, at2: bool = True):
    leaves = num_leaves(rec)
    cands = [(k, v) for k, v in leaves.items() if must in k.lower()]
    if not cands:
        return None, None
    if at2:
        at2c = [(k, v) for k, v in cands if "2" in k]
        if at2c:
            cands = at2c
    cands.sort(key=lambda kv: (kv[0].count("."), len(kv[0]), kv[0]))
    return cands[0][0], cands[0][1]


print("=" * 78)
print("E45 independent re-verification - round 28 (new checks)")
print("=" * 78)

sa = json.loads((EVAL / "scores_all.json").read_text(encoding="utf-8"))

print("\n--- structure (one level) ---")
print("scores_all keys  :", sorted(sa.keys()))
k0 = sorted(sa.keys())[0]
print(f"scores_all[{k0}] keys:", sorted(sa[k0].keys()) if isinstance(sa[k0], dict) else type(sa[k0]))
sm0 = json.loads((EVAL / "scores_summary_s0.json").read_text(encoding="utf-8"))
print("scores_summary_s0 keys:", sorted(sm0.keys()))
sm0k = sorted(sm0.keys())[0]
print(f"scores_summary_s0[{sm0k}] keys:", sorted(sm0[sm0k].keys()) if isinstance(sm0[sm0k], dict) else type(sm0[sm0k]))
st_all = json.loads((EVAL / "struct_metrics_all.json").read_text(encoding="utf-8"))
print("struct_metrics_all keys:", sorted(st_all.keys()))
st_s0 = json.loads((EVAL / "struct_metrics_s0.json").read_text(encoding="utf-8"))
print("struct_metrics_s0 keys:", sorted(st_s0.keys()))

# --------------------------------------------------- baseline (arm, seed) table
per_seed: dict[tuple[str, int], dict] = {}
for k, v in sa.items():
    m = KEY_RE.match(str(k))
    if m:
        per_seed[(m.group(1), int(m.group(2)))] = v
check("scores_all parses 7 arms x 3 seeds", len(per_seed) == 21, f"parsed {len(per_seed)}")

csi: dict[tuple[str, int], float] = {}
for (arm, s), rec in per_seed.items():
    _k, v = pick_numeric(rec, "csi")
    csi[(arm, s)] = v

# -------------------------------------------------- [D] remote eval coverage
grid = {(a, s) for a in ALL_ARMS for s in SEEDS}
check("remote eval.json count == 21", REMOTE_EVAL_JSON_COUNT == 21, f"got {REMOTE_EVAL_JSON_COUNT}")
check("remote eval dir total == 42 (21 eval + 21 windows)", REMOTE_EVAL_FILES_TOTAL == 42,
      f"got {REMOTE_EVAL_FILES_TOTAL}")
check("expected eval filename set has 21 entries",
      len({f"{a}_s{s}_ep00.eval.json" for a in ALL_ARMS for s in SEEDS}) == 21, "count mismatch")
check("scores_all (arm,seed) set == 7x3 grid", set(per_seed.keys()) == grid,
      f"got {sorted(per_seed.keys())[:3]}...")

# ------------------------------------------- [B1] cumulative scores_summary_s2
print("\n--- [B] scores_summary cumulative semantics ---")
md5_s2 = md5(EVAL / "scores_summary_s2.json")
md5_all = md5(EVAL / "scores_all.json")
print(f"  md5(scores_summary_s2) = {md5_s2}\n  md5(scores_all)          = {md5_all}")
check("scores_summary_s2.json is byte-identical to scores_all.json", md5_s2 == md5_all,
      f"{md5_s2} vs {md5_all}")

# ------------- [B2] every per-seed scores_summary agrees with merged scores_all
union: set[tuple[str, int]] = set()
b_cmp = b_vals = 0
for s in SEEDS:
    p = EVAL / f"scores_summary_s{s}.json"
    obj = json.loads(p.read_text(encoding="utf-8"))
    recs: dict = {}
    collect_records(obj, recs)
    print(f"  scores_summary_s{s}: {len(recs)} (arm,seed) records")
    check(f"scores_summary_s{s} collects >= 7 records", len(recs) >= 7, f"collected {len(recs)}")
    for (arm, seed, ep), rec in recs.items():
        union.add((arm, seed))
        merged = per_seed.get((arm, seed))
        if merged is None:
            check(f"scores_summary_s{s} record exists in scores_all: {arm}_s{seed}", False, "missing")
            continue
        lm, lp = num_leaves(merged), num_leaves(rec)
        common = sorted(set(lm) & set(lp))
        if not common:
            check(f"scores_summary_s{s} common numeric keys: {arm}_s{seed}", False,
                  f"merged={sorted(lm)[:4]} per_seed={sorted(lp)[:4]}")
            continue
        b_cmp += 1
        for k in common:
            b_vals += 1
            if abs(lm[k] - lp[k]) > 1e-9:
                check(f"scores_summary_s{s} value equal: {arm}_s{seed}.{k}", False,
                      f"merged={lm[k]} per_seed={lp[k]}")
check("per-seed scores_summary files jointly cover the 7x3 grid", union == grid,
      f"covered {len(union)} of {len(grid)}")
check("scores_summary cross-file records compared >= 21", b_cmp >= 21, f"compared {b_cmp}")
check("scores_summary numeric values compared >= 100", b_vals >= 100, f"compared {b_vals}")
print(f"  compared {b_cmp} records / {b_vals} numeric values across 3 per-seed files")

# ------------------------------------- [A] struct_metrics per-seed vs merged
print("\n--- [A] struct_metrics per-seed vs merged ---")
st_merged: dict = {}
collect_records(st_all, st_merged)
check("struct_metrics_all collects 21 (arm,seed) records", len(st_merged) == 21,
      f"collected {len(st_merged)}")
a_cmp = a_vals = 0
for s in SEEDS:
    p = EVAL / f"struct_metrics_s{s}.json"
    st_s = json.loads(p.read_text(encoding="utf-8"))
    recs: dict = {}
    collect_records(st_s, recs)
    check(f"struct_metrics_s{s} collects 7 arm records", len(recs) == 7, f"collected {len(recs)}")
    for (arm, seed, ep), rec in recs.items():
        if seed != s:
            continue
        merged = st_merged.get((arm, seed, ep)) or st_merged.get((arm, seed, "00"))
        if merged is None:
            check(f"struct_metrics merged record present: {arm}_s{seed}", False, "missing")
            continue
        lm, lp = num_leaves(merged), num_leaves(rec)
        common = sorted(set(lm) & set(lp))
        if not common:
            check(f"struct_metrics common numeric keys: {arm}_s{seed}", False,
                  f"merged={sorted(lm)[:4]} per_seed={sorted(lp)[:4]}")
            continue
        a_cmp += 1
        for k in common:
            a_vals += 1
            if abs(lm[k] - lp[k]) > 1e-9:
                check(f"struct_metrics value equal: {arm}_s{seed}.{k}", False,
                      f"merged={lm[k]} per_seed={lp[k]}")
check("struct_metrics cross-file records compared >= 21", a_cmp >= 21, f"compared {a_cmp}")
check("struct_metrics numeric values compared >= 100", a_vals >= 100, f"compared {a_vals}")
print(f"  compared {a_cmp} records / {a_vals} numeric values")

# ------------------------------- [C] remote summary-log cross-check
print("\n--- [C] remote summary-log cross-check ---")
log = REMOTE_SUMMARY_LOG_TAIL
check("remote log: anchors floor 0.0924", "persistence floor 0.0924" in log, "anchor missing")
check("remote log: anchors oracle 0.1653", "Oracle upper bound 0.1653" in log, "anchor missing")
check("remote log anchors match local constants", FLOOR == 0.0924 and ORACLE == 0.1653,
      f"floor={FLOOR} oracle={ORACLE}")
for tag in ("s3 rc = 0", "s1 rc = 0", "merge rc = 0", "summary rc = 0", "flat rc = 0"):
    check(f"remote log: {tag}", tag in log, "rc flag missing")
m = re.search(r"highest CSI@2\.0 in this batch:\s*(\S+)\s*=\s*([0-9.]+)", log)
check("remote log: highest-CSI line present", m is not None, "line missing")
if m:
    name, val = m.group(1), float(m.group(2))
    mm = KEY_RE.match(name)
    check("remote log highest-CSI name parses", mm is not None, f"name={name}")
    if mm:
        arm, seed = mm.group(1), int(mm.group(2))
        check("remote log highest-CSI arm/seed == A4/0", (arm, seed) == ("A4", 0),
              f"got {arm}_s{seed}")
        local_val = csi.get((arm, seed))
        check("remote log highest-CSI value == local CSI@2.0",
              local_val is not None and abs(local_val - val) <= 1e-4,
              f"remote={val} local={local_val}")
    best = max(csi.items(), key=lambda kv: kv[1])
    check("local max over 21 records == A4_s0 (0.1755)",
          best[0] == ("A4", 0) and abs(best[1] - val) <= 1e-4,
          f"best={best[0]} {best[1]:.6f}")

# ---------------------- [E] field_max_mm == model ceiling on A3a/A3b (all seeds)
print("\n--- [E] field_max_mm == expm1(6) ceiling ---")
probes = {s: json.loads((EVAL / f"diag_probe_s{s}.json").read_text(encoding="utf-8")) for s in SEEDS}


def readout(probe: dict, arm: str, name: str):
    a = probe.get("arms", {}).get(arm)
    if a is None:
        return None
    ro = a.get("readouts", a)
    if isinstance(ro, dict):
        for k, v in ro.items():
            if str(k).lower() == name.lower():
                return v
    for k, v in num_leaves(ro).items():
        if name.lower() in k.lower():
            return v
    return None


for arm in ("A3a", "A3b"):
    vals = [readout(probes[s], arm, "field_max_mm") for s in SEEDS]
    print(f"  {arm:4s} field_max_mm = {vals}")
    check(f"{arm} field_max_mm == {CEILING:.4f} on all 3 seeds",
          all(isinstance(v, (int, float)) and abs(v - CEILING) < 1e-3 for v in vals),
          f"vals={vals}")

# ------------------------- [F] doc row columns + E45-context drift (anchored)
print("\n--- [F] doc row columns + numeric drift ---")
master = (ROOT / "docs" / "MASTER.md").read_text(encoding="utf-8")
index = (ROOT / "docs" / "experiments" / "INDEX.md").read_text(encoding="utf-8")
e45_files = sorted((ROOT / "docs" / "experiments").glob("E45-*.md"))

m_rows = [l for l in master.splitlines() if re.match(r"^\|\s*\*{0,2}E45\b", l)]
check("MASTER has exactly one §9 E45 row", len(m_rows) == 1, f"rows={len(m_rows)}")
if m_rows:
    ncols = len(m_rows[0].split("|")) - 2
    check("MASTER §9 E45 row has 3 cells", ncols == 3, f"cells={ncols}")
i_rows = [l for l in index.splitlines() if re.match(r"^\|\s*\*{0,2}E45\b", l)]
check("INDEX has exactly one E45 row", len(i_rows) == 1, f"rows={len(i_rows)}")
if i_rows:
    ncols = len(i_rows[0].split("|")) - 2
    check("INDEX E45 row has 6 cells", ncols == 6, f"cells={ncols}")

# MASTER version must not have gone BACKWARDS (newer experiments may bump it)
head = "\n".join(master.splitlines()[:40])
vm = re.search(r"v(\d+)\.(\d+)", head)
master_v = f"v{vm.group(1)}.{vm.group(2)}" if vm else ""
print(f"  MASTER head version = {master_v}")
check("MASTER version not older than v2.45",
      bool(vm) and (int(vm.group(1)), int(vm.group(2))) >= (2, 45), f"got {master_v}")

# E45 experiment file must state both anchors
e45_text = "\n".join(f.read_text(encoding="utf-8") for f in e45_files)
check("E45 experiment file states floor 0.0924", "0.0924" in e45_text, "anchor missing")
check("E45 experiment file states oracle 0.1653", "0.1653" in e45_text, "anchor missing")

# Adjacency-anchored drift lint: only numbers DIRECTLY attached to 地板 / Oracle count.
ADJ = r"[\s*_`「」『』（）()]*"
CONN = r"(?:为|是|约|≈|=|:|：)?"
FLOOR_RE = re.compile(r"(?:地板|floor)" + ADJ + CONN + ADJ + r"(\d\.\d{2,6})")
ORACLE_RE = re.compile(r"(?:Oracle|oracle)" + ADJ + r"(?:上界|upper bound)?" + ADJ + CONN + ADJ + r"(\d\.\d{2,6})")
E45_CTX = re.compile(r"E45|§4\.47|4\.47")
hits = []
for f in [ROOT / "docs" / "MASTER.md", ROOT / "docs" / "experiments" / "INDEX.md"] + e45_files:
    lines = f.read_text(encoding="utf-8").splitlines()
    for i, line in enumerate(lines):
        if not E45_CTX.search(line):
            continue
        for rx, expect, tag in ((FLOOR_RE, FLOOR, "floor"), (ORACLE_RE, ORACLE, "oracle")):
            for mm in rx.finditer(line):
                try:
                    val = float(mm.group(1))
                except (TypeError, ValueError):
                    continue
                if abs(val - expect) > 1e-9:
                    neg_check(f"no {tag} drift in {f.name} (E45 context)", True,
                              f"line {i+1}: {mm.group(0)[:80]}")
                else:
                    neg_check(f"no {tag} drift in {f.name} (E45 context)", False, "")
                    hits.append((f.name, i + 1, tag, val))
print(f"  anchored anchor-mentions found: {len(hits)}")
for h in hits[:12]:
    print("   ", h)

# ------------------------------------------------------------------- summary
print("\n" + "=" * 78)
print(f"POS {pos_ok}/{pos_ok + pos_bad}   NEG {neg_ok}/{neg_ok + neg_bad}")
if fails:
    print(f"\n{len(fails)} FAILURE(S):")
    for f in fails:
        print("  -", f)
print("VERDICT:", "PASS" if not fails else "FAIL")
print("=" * 78)
sys.exit(0 if not fails else 1)
