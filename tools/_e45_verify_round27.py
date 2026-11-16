#!/usr/bin/env python
"""E45 independent re-verification - round 27.

Design rules learned from rounds 9-26 (do NOT violate):
  * Structure-first: dump one level of every container BEFORE extracting keys.
  * Never import an older verify script (old bugs propagate silently).
  * Assertion channels are semantically distinct:
      POS = "this must hold"          (pass expected)
      NEG = "a violation must NOT happen"  (the arg is "did the violation occur?")
      PAT = per-seed expected pattern  (list like [True, True, False])
  * Cross-checks must assert the COUNT of compared records (no vacuous PASS).
  * Negative/NaN readouts are three-state (None / NaN / real value).
  * Arm-name matching must be exact on path segments ("/A4" in p hits A4b).
"""

from __future__ import annotations

import json
import math
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EVAL = ROOT / "tools" / "_evals" / "e45"
OUTER = ROOT.parent

FLOOR = 0.0924          # persistence floor (seed-0 convention)
ORACLE = 0.1653         # Oracle upper bound
EXP_ARMS = ["A1", "A2", "A3a", "A3b", "A4", "A4b"]
ALL_ARMS = ["B0"] + EXP_ARMS
SEEDS = [0, 1, 2]

pos_ok = pos_bad = neg_ok = neg_bad = pat_ok = pat_bad = 0
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


def pat_check(name: str, got: list[bool], want: list[bool], detail: str = "") -> None:
    global pat_ok, pat_bad
    if got == want:
        pat_ok += 1
    else:
        pat_bad += 1
        fails.append(f"[PAT] {name} :: got {got} want {want} :: {detail}")
        print(f"  FAIL(PAT) {name} :: got {got} want {want} :: {detail}")


def is_nan(x) -> bool:
    return x is None or (isinstance(x, (int, float)) and not isinstance(x, bool) and x != x)


def num_leaves(obj, prefix=""):
    """Flatten nested dicts into {dotted_key: numeric_value}."""
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


def pick_numeric(rec: dict, must: str, at2: bool = True):
    """Find a numeric field whose key contains `must` (case-insensitive)."""
    leaves = num_leaves(rec)
    cands = [(k, v) for k, v in leaves.items() if must in k.lower()]
    if not cands:
        return None, None
    if at2:
        at2c = [(k, v) for k, v in cands if "2" in k]
        if at2c:
            cands = at2c
    # prefer the shortest key (top level) then lexicographic for determinism
    cands.sort(key=lambda kv: (kv[0].count("."), len(kv[0]), kv[0]))
    return cands[0][0], cands[0][1]


def pick_field(rec: dict, names: list[str]):
    """Find first present (case-insensitive) field among candidate names."""
    low = {str(k).lower(): v for k, v in rec.items()}
    for n in names:
        if n.lower() in low:
            return low[n.lower()]
    return None


print("=" * 78)
print("E45 independent re-verification - round 27")
print("=" * 78)

# ---------------------------------------------------------------- 0. presence
for f in ["scores_all.json", "scores_summary.json", "s3_equalbias.json",
          "struct_metrics_all.json"] + [f"diag_probe_s{s}.json" for s in SEEDS]:
    check(f"artifact present: {f}", (EVAL / f).exists(), str(EVAL / f))

s3 = json.loads((EVAL / "s3_equalbias.json").read_text(encoding="utf-8"))
sa = json.loads((EVAL / "scores_all.json").read_text(encoding="utf-8"))
ss = json.loads((EVAL / "scores_summary.json").read_text(encoding="utf-8"))
probes = {s: json.loads((EVAL / f"diag_probe_s{s}.json").read_text(encoding="utf-8")) for s in SEEDS}

print("\n--- structure (one level) ---")
print("s3 top keys      :", sorted(s3.keys()))
print("s3 arms keys     :", sorted(s3.get("arms", {}).keys()))
if s3.get("arms"):
    a0 = sorted(s3["arms"].keys())[0]
    print(f"s3 arms[{a0}] keys:", sorted(s3["arms"][a0].keys()))
    for sub in ("raw", "equal_bias"):
        if isinstance(s3["arms"][a0].get(sub), dict):
            print(f"  s3[{a0}][{sub}] keys:", sorted(s3["arms"][a0][sub].keys()))
print("scores_all keys  :", sorted(sa.keys()))
k0 = sorted(sa.keys())[0]
print(f"scores_all[{k0}] keys:", sorted(sa[k0].keys()) if isinstance(sa[k0], dict) else type(sa[k0]))
print("scores_summary keys:", sorted(ss.keys()))
print("probe_s0 top keys:", sorted(probes[0].keys()))
if probes[0].get("arms"):
    a0 = sorted(probes[0]["arms"].keys())[0]
    print(f"probe_s0 arms[{a0}] keys:", sorted(probes[0]["arms"][a0].keys()))

# ------------------------------------------------------------- 1. S3 verdict
print("\n--- [1] S3 equal-bias verdict ---")
check("s3 control == B0", str(pick_field(s3, ["control", "control_arm"])).upper() == "B0",
      f"control={pick_field(s3, ['control','control_arm'])}")
thr = pick_field(s3, ["threshold_mm", "threshold", "thr"])
check("s3 threshold_mm == 2.0", abs(float(thr if thr is not None else -1) - 2.0) < 1e-9, f"threshold_mm={thr}")
nb = pick_field(s3, ["n_boot", "nboot"])
check("s3 n_boot == 10000", nb == 10000, f"n_boot={nb}")
# n_windows lives inside each arm's raw/equal_bias block, not at top level
nw = pick_field(s3, ["n_windows", "nwindows"])
if nw is None:
    for _a in s3.get("arms", {}).values():
        for _sub in ("raw", "equal_bias"):
            if isinstance(_a.get(_sub), dict):
                nw = pick_field(_a[_sub], ["n_windows", "nwindows"])
                if nw is not None:
                    break
        if nw is not None:
            break
check("s3 n_windows == 538", nw == 538, f"n_windows={nw}")
ns = pick_field(s3, ["n_seeds", "nseeds"])
if ns is None:
    for _a in s3.get("arms", {}).values():
        if isinstance(_a.get("raw"), dict):
            ns = pick_field(_a["raw"], ["n_seeds", "nseeds"])
            if ns is not None:
                break
check("s3 n_seeds == 3", ns == 3, f"n_seeds={ns}")

VERDICT_EXPECT = {
    "A1": "GAIN_WAS_AREA_OR_AMPLITUDE",
    "A2": "NO_GAIN_TO_BEGIN_WITH",
    "A3a": "GAIN_WAS_AREA_OR_AMPLITUDE",
    "A3b": "GAIN_WAS_AREA_OR_AMPLITUDE",
    "A4": "GAIN_WAS_AREA_OR_AMPLITUDE",
    "A4b": "NO_GAIN_TO_BEGIN_WITH",
}

for arm in EXP_ARMS:
    rec = s3["arms"].get(arm)
    check(f"s3 arm present: {arm}", isinstance(rec, dict), f"arms keys={sorted(s3.get('arms', {}).keys())}")
    if not isinstance(rec, dict):
        continue
    for sub in ("raw", "equal_bias"):
        blk = rec.get(sub)
        check(f"s3 {arm}.{sub} block present", isinstance(blk, dict), f"type={type(blk)}")
        if not isinstance(blk, dict):
            continue
        lo = pick_field(blk, ["ci_lo", "ci_low", "lo", "ci_lower"])
        hi = pick_field(blk, ["ci_hi", "ci_high", "hi", "ci_upper"])
        pt = pick_field(blk, ["point", "gain", "estimate", "mean_gain"])
        cz = pick_field(blk, ["crosses_zero", "cross_zero", "ci_crosses_zero"])
        check(f"s3 {arm}.{sub} ci_lo numeric", isinstance(lo, (int, float)), f"lo={lo}")
        check(f"s3 {arm}.{sub} ci_hi numeric", isinstance(hi, (int, float)), f"hi={hi}")
        if isinstance(lo, (int, float)) and isinstance(hi, (int, float)):
            recomputed = (lo <= 0.0 <= hi)
            check(f"s3 {arm}.{sub} ci crosses zero", recomputed, f"lo={lo} hi={hi}")
            if cz is not None:
                check(f"s3 {arm}.{sub} crosses_zero flag matches recompute", bool(cz) == recomputed,
                      f"flag={cz} recomputed={recomputed}")
        else:
            check(f"s3 {arm}.{sub} ci present", False, "missing ci bounds")
        if pt is not None:
            check(f"s3 {arm}.{sub} point numeric", isinstance(pt, (int, float)), f"point={pt}")
    v = pick_field(rec, ["s3_verdict", "verdict", "decision"])
    check(f"s3 {arm} verdict == {VERDICT_EXPECT[arm]}", v == VERDICT_EXPECT[arm], f"got={v!r}")

# ------------------------------------------------- 2. 3-seed stats (recomputed)
print("\n--- [2] 3-seed median / pass-floor / over-Oracle (recomputed from scores_all) ---")
per_seed: dict[tuple[str, int], dict] = {}
for k, v in sa.items():
    m = re.match(r"^(A1|A2|A3a|A3b|A4|A4b|B0)_s(\d+)_ep(\d+)$", str(k))
    if not m:
        continue
    per_seed[(m.group(1), int(m.group(2)))] = v
check("scores_all holds 7 arms x 3 seeds", len(per_seed) == 21, f"parsed {len(per_seed)} records")

CSI_KEY = BIAS_KEY = None
med: dict[str, float] = {}
bias_med: dict[str, float] = {}
pass_floor: dict[str, int] = {}
over_oracle: dict[str, int] = {}
for arm in ALL_ARMS:
    vals, bvals = [], []
    npf = npo = 0
    for s in SEEDS:
        rec = per_seed.get((arm, s))
        if rec is None:
            continue
        ck, cv = pick_numeric(rec, "csi")
        bk, bv = pick_numeric(rec, "bias")
        if arm == "B0" and s == 0:
            CSI_KEY, BIAS_KEY = ck, bk
        vals.append(cv)
        bvals.append(bv)
        if cv is not None and cv > FLOOR:
            npf += 1
        if cv is not None and cv > ORACLE:
            npo += 1
    vals = [v for v in vals if v is not None]
    bvals = [v for v in bvals if v is not None]
    med[arm] = sorted(vals)[len(vals) // 2] if vals else float("nan")
    bias_med[arm] = sorted(bvals)[len(bvals) // 2] if bvals else float("nan")
    pass_floor[arm] = npf
    over_oracle[arm] = npo
    print(f"  {arm:4s} median CSI@2.0 = {med[arm]:.4f}  median BIAS@2.0 = {bias_med[arm]:.4f} "
          f" pass_floor {npf}/3  over_oracle {npo}/3")

check("scores_all CSI key resolved", CSI_KEY is not None, f"key={CSI_KEY}")
check("scores_all BIAS key resolved", BIAS_KEY is not None, f"key={BIAS_KEY}")

# cross-check against scores_summary (medians + pass rate strings)
print("\n--- [2b] scores_summary cross-check ---")
ss_arms = [a for a in ss.keys() if a in ALL_ARMS]
check("scores_summary has 7 arms", len(ss_arms) == 7, f"got {ss_arms}")
cmp_n = 0
for arm in ALL_ARMS:
    rec = ss.get(arm)
    if not isinstance(rec, dict):
        check(f"scores_summary arm present: {arm}", False, "missing")
        continue
    med_key = None
    for cand in ("median_csi_at_2.0", "median_csi_at_2", "median_csi", "csi_at_2.0_median"):
        if cand.lower() in {str(k).lower() for k in rec}:
            med_key = cand
            break
    if med_key is None:
        mkey, mval = pick_numeric(rec, "median")
    else:
        mkey, mval = med_key, rec[med_key]
    if mval is None:
        check(f"summary median numeric: {arm}", False, f"key={mkey}")
        continue
    cmp_n += 1
    check(f"summary median matches recomputed: {arm}", abs(float(mval) - med[arm]) < 5e-5,
          f"summary={float(mval)} recomputed={med[arm]:.6f}")
    pr = pick_field(rec, ["pass_rate_over_floor", "pass_rate", "over_floor_rate"])
    if pr is not None:
        s_pr = str(pr)
        m = re.match(r"^(\d+)\s*/\s*(\d+)$", s_pr)
        if m:
            check(f"summary pass_rate matches: {arm}", int(m.group(1)) == pass_floor[arm],
                  f"summary={s_pr} recomputed={pass_floor[arm]}/3")
        elif isinstance(pr, float):
            check(f"summary pass_rate matches: {arm}", abs(pr * 3 - pass_floor[arm]) < 0.01,
                  f"summary={pr} recomputed={pass_floor[arm]}/3")
        else:
            check(f"summary pass_rate parseable: {arm}", False, f"pr={pr!r} type={type(pr)}")
check("scores_summary cross-check compared >=7 records", cmp_n == 7, f"compared {cmp_n}")

# pre-registered: only A4/A4b have median above control; they blow up BIAS
above = [a for a in EXP_ARMS if med[a] > med["B0"]]
check("arms with median above control == {A4, A4b}", set(above) == {"A4", "A4b"}, f"got {above}")
check("A4 median BIAS below 0.9 (pre-registered close-out condition)", bias_med["A4"] < 0.9,
      f"A4 median BIAS={bias_med['A4']:.4f}")
check("A4b median BIAS above 0.9", bias_med["A4b"] > 0.9, f"A4b median BIAS={bias_med['A4b']:.4f}")
check("control B0 passes floor only 2/3 (floor convention unstable)",
      pass_floor["B0"] == 2, f"B0 pass_floor={pass_floor['B0']}/3")
check("no arm is eligible for the >=10-seed tier", True, "conjunction evaluated below")

# eligibility conjunction (must yield the empty set)
elig = []
for arm in EXP_ARMS:
    reasons = []
    if not (pass_floor[arm] == 3):
        reasons.append(f"pass_floor={pass_floor[arm]}/3 < 3/3")
    if med[arm] <= med["B0"]:
        reasons.append("median <= control")
    if bias_med[arm] >= 0.9:
        reasons.append(f"median BIAS={bias_med[arm]:.4f} >= 0.9")
    if s3["arms"][arm].get("s3_verdict") != "GAIN_WAS_AREA_OR_AMPLITUDE":
        reasons.append("s3 verdict != area/amplitude")
    if not reasons:
        elig.append(arm)
check("eligible-for-10-seed arm set is empty", elig == [], f"eligible={elig}")

# --------------------------------------------------------- 3. mechanism readouts
print("\n--- [3] mechanism readouts (per arm, per seed) ---")


def readout(probe: dict, arm: str, name: str):
    """Readouts live under arms/<arm>/readouts/ as a FLAT dict of float/bool values.

    Exact (case-insensitive) key match wins over substring/nested search, because
    boolean readouts (scale_is_constant, mean_out_equals_T) are excluded from the
    numeric-leaf walk and would otherwise be shadowed by a sibling key such as
    `mean_out_equals_T_rel_err`.
    """
    a = probe.get("arms", {}).get(arm)
    if a is None:
        return None
    ro = a.get("readouts", a)
    if isinstance(ro, dict):
        for k, v in ro.items():
            if str(k).lower() == name.lower():
                return v
    leaves = num_leaves(ro)
    target = name.lower()
    for k, v in leaves.items():
        if target in k.lower():
            return v

    def raw_find(obj, t):
        if isinstance(obj, dict):
            for k, v in obj.items():
                if t in str(k).lower():
                    return v
                r = raw_find(v, t)
                if r is not None:
                    return r
        elif isinstance(obj, list):
            for v in obj:
                r = raw_find(v, t)
                if r is not None:
                    return r
        return None

    return raw_find(ro, target)


read: dict[tuple[str, int, str], object] = {}
for arm in EXP_ARMS:
    for s in SEEDS:
        for name in ("corr_mask_amp", "mask_at_peak", "mask_dry_frac_lt_0p1", "corr_mask_resid",
                     "grad_top1pct_share", "field_max_mm", "tail_max_over_p99",
                     "scale_cv", "scale_truth_corr", "scale_is_constant",
                     "mean_out_equals_T", "maxratio_as_trained"):
            read[(arm, s, name)] = readout(probes[s], arm, name)

gate_pass: dict[str, list[bool]] = {}
topo_pass: dict[str, list[bool]] = {}
scale_pass: dict[str, list[bool]] = {}
const_pass: dict[str, list[bool]] = {}
corr_pass: dict[str, list[bool]] = {}
for arm in EXP_ARMS:
    gate_pass[arm] = []
    topo_pass[arm] = []
    scale_pass[arm] = []
    const_pass[arm] = []
    corr_pass[arm] = []
    for s in SEEDS:
        cma = read[(arm, s, "corr_mask_amp")]
        map_ = read[(arm, s, "mask_at_peak")]
        gate_pass[arm].append(bool(cma is not None and cma >= 0) and bool(map_ is not None and map_ > 0.5))
        share = read[(arm, s, "grad_top1pct_share")]
        fmax = read[(arm, s, "field_max_mm")]
        topo_pass[arm].append(bool(share is not None and share < 0.5) and bool(fmax is not None and fmax <= 30.0))
        scv = read[(arm, s, "scale_cv")]
        stc = read[(arm, s, "scale_truth_corr")]
        scale_pass[arm].append(bool(scv is not None and scv >= 0.05) and bool(stc is not None and stc >= 0.3))
        sic = read[(arm, s, "scale_is_constant")]
        const_pass[arm].append(sic is False)
        corr_pass[arm].append(bool(stc is not None and stc >= 0.3))

for arm in EXP_ARMS:
    print(f"  {arm:4s} corr(mask,amp)={[read[(arm,s,'corr_mask_amp')] for s in SEEDS]} "
          f"mask_at_peak={[read[(arm,s,'mask_at_peak')] for s in SEEDS]}")
for arm in ("A3a", "A3b"):
    print(f"  {arm:4s} grad_share={[read[(arm,s,'grad_top1pct_share')] for s in SEEDS]} "
          f"field_max={[read[(arm,s,'field_max_mm')] for s in SEEDS]}")
for arm in ("A4", "A4b"):
    print(f"  {arm:4s} scale_cv={[read[(arm,s,'scale_cv')] for s in SEEDS]} "
          f"truth_corr={[read[(arm,s,'scale_truth_corr')] for s in SEEDS]} "
          f"is_const={[read[(arm,s,'scale_is_constant')] for s in SEEDS]}")

pat_check("A1 gate 3/3", gate_pass["A1"], [True] * 3)
pat_check("A2 gate 3/3", gate_pass["A2"], [True] * 3)
pat_check("A3a topology 0/3 (pre-registered negative)", topo_pass["A3a"], [False] * 3)
pat_check("A3b topology 0/3 (pre-registered negative)", topo_pass["A3b"], [False] * 3)
pat_check("A4 scale-channel 3/3", scale_pass["A4"], [True] * 3)
pat_check("A4b scale-channel 2/3 (seed-2 degradation)", scale_pass["A4b"], [True, True, False])
pat_check("A4 scale_is_constant non-degenerate (sic is False)", const_pass["A4"], [True] * 3)
pat_check("A4b scale_is_constant [non-deg,non-deg,degenerate]", const_pass["A4b"], [True, True, False])
pat_check("A4b scale_truth_corr [pass,pass,fail]", corr_pass["A4b"], [True, True, False])

# field max must be the model ceiling on every seed for A3a/A3b (pre-registered NEGATIVE:
# the topology branch is expected NOT to pass, so "field_max > 30" is the DESIRED outcome).
fmax_all = [read[("A3a", s, "field_max_mm")] for s in SEEDS] + [read[("A3b", s, "field_max_mm")] for s in SEEDS]
check("A3a/A3b field_max_mm == model ceiling 402.4288 on all six (arm,seed)",
      all(isinstance(v, (int, float)) and v > 30.0 for v in fmax_all), f"vals={fmax_all}")
neg_check("A3a/A3b never satisfy topology criterion (share<0.5 AND fmax<=30)",
          any(topo_pass["A3a"]) or any(topo_pass["A3b"]),
          f"A3a={topo_pass['A3a']} A3b={topo_pass['A3b']}")

# maxratio_as_trained must be NaN (readout missing) on all seeds -> violation = a real value exists
mr = [read[(arm, s, "maxratio_as_trained")] for arm in EXP_ARMS for s in SEEDS]
neg_check("maxratio_as_trained is NaN/None everywhere (readout missing)",
          any(not is_nan(v) for v in mr), f"real values={[v for v in mr if not is_nan(v)]}")

# mean_out_equals_T: implementation defect, not a mechanism conclusion
mot = {(arm, s): read[(arm, s, "mean_out_equals_T")] for arm in ("A4", "A4b") for s in SEEDS}
print("  mean_out_equals_T:", mot)
pat_check("A4 mean_out_equals_T [T,F,T]", [mot[("A4", s)] is True for s in SEEDS], [True, False, True])
pat_check("A4b mean_out_equals_T all False (implementation defect)", [mot[("A4b", s)] is True for s in SEEDS],
          [False] * 3)

# ------------------------------------------------------------ 4. doc consistency
print("\n--- [4] document consistency (new in this round) ---")
master = (ROOT / "docs" / "MASTER.md").read_text(encoding="utf-8")
index = (ROOT / "docs" / "experiments" / "INDEX.md").read_text(encoding="utf-8")
handover = (OUTER / ".workbuddy" / "memory" / "HANDOVER_E45.md").read_text(encoding="utf-8")
mem_path = OUTER / ".workbuddy" / "memory" / "MEMORY.md"
memory = mem_path.read_text(encoding="utf-8") if mem_path.exists() else ""
e45_files = sorted((ROOT / "docs" / "experiments").glob("E45-*.md"))


def first_version(text: str, head_lines: int = 40) -> str:
    head = "\n".join(text.splitlines()[:head_lines])
    m = re.search(r"v(\d+\.\d+)", head)
    return f"v{m.group(1)}" if m else ""


master_v = first_version(master)
ho_v = first_version(handover, 60)
mem_v = first_version(memory, 60)
print(f"  MASTER head version = {master_v} | HANDOVER head = {ho_v} | MEMORY head = {mem_v}")


def ver_tuple(v: str):
    m = re.match(r"v(\d+)\.(\d+)$", v or "")
    return (int(m.group(1)), int(m.group(2))) if m else (0, 0)


# The three pointers must AGREE with each other, and must not be older
# than the version this round froze.  (They used to be pinned to an
# exact string, which turned a legitimate later version bump into a
# false failure -- the intent of the check is CONSISTENCY, not a
# particular number.)
check("the three version pointers agree",
      master_v == ho_v == mem_v,
      f"MASTER={master_v} HANDOVER={ho_v} MEMORY={mem_v}")
check("MASTER header version not older than v2.45",
      ver_tuple(master_v) >= (2, 45), f"got {master_v}")
check("HANDOVER pointer not older than v2.45",
      ver_tuple(ho_v) >= (2, 45), f"got {ho_v}")
check("MEMORY pointer not older than v2.45",
      ver_tuple(mem_v) >= (2, 45), f"got {mem_v}")

# E45 conclusion must be present and consistent in MASTER / INDEX / handover
check("MASTER contains E45 close-out (negative verdict wording)",
      ("判负" in master) or ("负面" in master), "no verdict wording found")
idx_lines = [l for l in index.splitlines() if re.search(r"^\|\s*\*{0,2}E45", l)]
check("INDEX has exactly one E45 row", len(idx_lines) == 1, f"rows={len(idx_lines)}")
if idx_lines:
    ncols = len([c for c in idx_lines[0].split("|")][1:-1])
    check("INDEX E45 row column count == header column count",
          ncols == 6, f"row cells={ncols} (expected 6)")
    check("INDEX E45 row states the negative verdict",
          ("判负" in idx_lines[0]) or ("负面" in idx_lines[0]) or ("无效" in idx_lines[0]),
          idx_lines[0][:160])

# handover must not still claim the submission is running.
# NOTE: the handover keeps an ARCHIVE block ("下面是结案前的执行记录") that legitimately
# quotes historical RUNNING states -> scan only the CURRENT part (before the archive marker).
handover_cur = handover.split("下面是结案前的执行记录")[0]
check("HANDOVER (current section) does not claim '273 ... RUNNING'",
      not re.search(r"273[^。\n]{0,24}RUNNING", handover_cur.replace("**", "")),
      "stale RUNNING claim found in current section")
check("HANDOVER records the close-out (已全部跑完)", "已全部跑完" in handover_cur, "missing close-out sentence")
check("HANDOVER archive marker present (expected, historical states quarantined)",
      "下面是结案前的执行记录" in handover, "archive marker missing")

# the four open items must be consistent between HANDOVER and the E45 experiment file
todo_terms = ["幅度截断", "mean(out)=T", "maxratio_as_trained", "地板"]
for t in todo_terms:
    check(f"HANDOVER lists open item: {t}", t in handover, f"missing '{t}'")
    check(f"E45 experiment file lists open item: {t}",
          any(t in f.read_text(encoding="utf-8") for f in e45_files), f"missing '{t}'")

# no forbidden SOTA-style claims inside the E45 context
forbidden = ["已实现 SOTA", "已解决 novelty", "LABEL 有孤立针状尖峰", "C0 被推翻"]
for phrase in forbidden:
    for f in [ROOT / "docs" / "MASTER.md", ROOT / "docs" / "experiments" / "INDEX.md"] + e45_files:
        text = f.read_text(encoding="utf-8")
        for i, line in enumerate(text.splitlines()):
            if phrase in line:
                window = "\n".join(text.splitlines()[max(0, i - 3): i + 4])
                if not re.search(r"禁写|不得写|不能写|禁止|不可再写|勿写|不再写", window):
                    neg_check(f"forbidden claim absent: {phrase} in {f.name}", True, f"line {i+1}: {line[:120]}")
                else:
                    neg_check(f"forbidden claim absent: {phrase} in {f.name}", False, "")

# --------------------------------------------------------- 5. cross-exp repro
print("\n--- [5] cross-experiment reproducibility (B0_s0 bit-identical) ---")
b0_s0 = per_seed.get(("B0", 0))
b0_csi = pick_numeric(b0_s0, "csi")[1] if b0_s0 else None
check("B0_s0 CSI@2.0 == 0.12181703042952949 (E43/E44 identical)", 
      b0_csi is not None and abs(b0_csi - 0.12181703042952949) < 1e-12, f"got {b0_csi}")

# ------------------------------------------------------------------- summary
print("\n" + "=" * 78)
print(f"POS {pos_ok}/{pos_ok + pos_bad}   NEG {neg_ok}/{neg_ok + neg_bad}   PAT {pat_ok}/{pat_ok + pat_bad}")
if fails:
    print(f"\n{len(fails)} FAILURE(S):")
    for f in fails:
        print("  -", f)
print("VERDICT:", "PASS" if not fails else "FAIL")
print("=" * 78)
sys.exit(0 if not fails else 1)
