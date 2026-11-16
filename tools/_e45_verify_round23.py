#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""E45 round-23 independent re-verification.

Design rules inherited from rounds 9..22 (do not break them):
  * Never import an older verification script (old bugs propagate silently).
  * Re-extract every number straight from the raw artifacts under tools/_evals/e45/.
  * Dump the real key structure before trusting key paths.
  * Three assertion channels: POS (must hold), NEG (pre-registered negative
    results must keep failing), PAT (per-seed expected pattern).
  * "Missing readout" must distinguish None / NaN / valid value.
  * Any cross-check over a collection must also assert the comparison count > 0.
"""
import json
import math
import os
import re
import statistics
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EV = os.path.join(ROOT, "tools", "_evals", "e45")
DOCS = os.path.join(ROOT, "docs")
UPMEM = "R:/1radarNWPprediction/.workbuddy/memory/MEMORY.md"

FLOOR = 0.0924          # persistence floor (seed-0 convention)
ORACLE = 0.1653         # oracle upper bound
ARMS = ["A1", "A2", "A3a", "A3b", "A4", "A4b"]
ALL7 = ["B0"] + ARMS
SEEDS = [0, 1, 2]

POS = []   # (name, ok)
NEG = []   # (name, ok)  -- ok means "the negative result correctly still occurs"
PAT = []   # (name, ok)


def check(name, cond):
    POS.append((name, bool(cond)))
    return bool(cond)


def neg_check(name, violation):
    """violation=True means the pre-registered failure unexpectedly disappeared."""
    NEG.append((name, not violation))


def pat_check(name, got, want):
    PAT.append((name, list(got) == list(want)))


def load(name):
    with open(os.path.join(EV, name), encoding="utf-8") as f:
        return json.load(f)


def is_missing(x):
    return x is None or (isinstance(x, (int, float)) and not isinstance(x, bool) and x != x)


def near(a, b, tol=1e-9):
    if a is None or b is None:
        return False
    return abs(float(a) - float(b)) <= tol * max(1.0, abs(float(b)))


print("=" * 78)
print("E45 round-23 independent re-verification")
print("=" * 78)

# ---------------------------------------------------------------- 0. structure
s3 = load("s3_equalbias.json")
summ = load("scores_summary.json")
sall = load("scores_all.json")
probes = {s: load("diag_probe_s%d.json" % s) for s in SEEDS}

print("\n-- structure (real key paths) --")
print("s3 top keys        :", sorted(s3.keys()))
print("s3 arms            :", sorted(s3["arms"].keys()))
print("summ arms          :", sorted(summ.keys()))
print("summ[B0] keys      :", sorted(summ["B0"].keys()))
print("scores_all keys[:6]:", list(sall.keys())[:6], "... total", len(sall))
print("scores_all rec keys:", sorted(list(sall.values())[0].keys()))
print("probe s0 arms      :", sorted(probes[0]["arms"].keys()))
r0 = probes[0]["arms"]["A1"]["readouts"]
print("A1 readout keys    :", sorted(r0.keys()))

# ---------------------------------------------------------------- 1. S3 verdict
print("\n-- S3 equal-bias control --")
check("s3.threshold==2.0", near(s3["threshold_mm"], 2.0))
check("s3.epoch=='00'", s3["epoch"] == "00")
check("s3.control=='B0'", s3["control"] == "B0")
check("s3.n_boot==10000", s3["n_boot"] == 10000)
check("s3 has exactly 6 arms", sorted(s3["arms"].keys()) == sorted(ARMS))

EXPECT_VERDICT = {
    "A1": "GAIN_WAS_AREA_OR_AMPLITUDE",
    "A2": "NO_GAIN_TO_BEGIN_WITH",
    "A3a": "GAIN_WAS_AREA_OR_AMPLITUDE",
    "A3b": "GAIN_WAS_AREA_OR_AMPLITUDE",
    "A4": "GAIN_WAS_AREA_OR_AMPLITUDE",
    "A4b": "NO_GAIN_TO_BEGIN_WITH",
}
s3_rows = []
for arm in ARMS:
    a = s3["arms"][arm]
    for mode in ("raw", "equal_bias"):
        st = a[mode]
        lo, hi = st["ci_lo"], st["ci_hi"]
        recomputed = (lo <= 0.0 <= hi)
        check("s3.%s.%s crosses_zero" % (arm, mode), recomputed)
        check("s3.%s.%s flag agrees" % (arm, mode), bool(st["crosses_zero"]) == recomputed)
        check("s3.%s.%s n_seeds==3" % (arm, mode), st["n_seeds"] == 3)
        check("s3.%s.%s n_windows==538" % (arm, mode), st["n_windows"] == 538)
        # pre-registered negative: gain CI must straddle zero
        neg_check("s3.%s.%s CI must straddle zero" % (arm, mode), not recomputed)
        s3_rows.append((arm, mode, st["point"], lo, hi, recomputed))
    check("s3.%s verdict" % arm, a["s3_verdict"] == EXPECT_VERDICT[arm])
    check("s3.%s seeds 0/1/2" % arm, sorted(a["seeds"].keys()) == ["0", "1", "2"])

print("\n-- S3 table (point / CI / crosses zero) --")
for arm, mode, p, lo, hi, cz in s3_rows:
    print("  %-4s %-10s point=%+.6f  CI=[%+.6f, %+.6f]  crosses_zero=%s" % (arm, mode, p, lo, hi, cz))

# ---------------------------------------------------------------- 2. 3-seed stats
print("\n-- 3-seed statistics (recomputed from scores_all.json) --")
# scores_all keys look like '<arm>_s<seed>_ep00'
per_seed = {}
for k, v in sall.items():
    m = re.match(r"^([A-Za-z0-9]+)_s(\d+)_ep(\d+)$", k)
    if not m:
        continue
    arm, sd, ep = m.group(1), int(m.group(2)), m.group(3)
    per_seed[(arm, sd)] = v
check("scores_all parsed 21 (arm,seed) records", len(per_seed) == 21)
check("scores_all covers 7 arms x 3 seeds",
      sorted(set(a for a, _ in per_seed)) == sorted(ALL7) and
      sorted(set(s for _, s in per_seed)) == SEEDS)


def csi_of(arm, sd):
    rec = per_seed.get((arm, sd))
    if rec is None:
        return None
    for key in ("csi_at_2.0", "CSI@2.0", "csi@2.0"):
        if isinstance(rec, dict) and key in rec:
            return rec[key]
    if isinstance(rec, dict):
        for k, v in rec.items():
            if "csi" in str(k).lower() and "2.0" in str(k):
                return v
    return rec


stats = {}
for arm in ALL7:
    vals = [csi_of(arm, s) for s in SEEDS]
    if any(v is None for v in vals):
        check("scores_all csi present for %s" % arm, False)
        continue
    med = statistics.median(vals)
    n_floor = sum(1 for v in vals if v > FLOOR)
    n_oracle = sum(1 for v in vals if v > ORACLE)
    spread = max(vals) / min(vals) if min(vals) > 0 else float("inf")
    stats[arm] = dict(vals=vals, med=med, n_floor=n_floor, n_oracle=n_oracle, spread=spread)
    # cross-check against scores_summary.json
    srec = summ.get(arm)
    check("summ has %s" % arm, srec is not None)
    if srec:
        check("summ[%s].median == recomputed" % arm, near(srec["median_csi_at_2.0"], med))
        pr = srec.get("pass_rate_over_floor")
        check("summ[%s].pass_rate == '%d/3'" % (arm, n_floor), str(pr) == "%d/3" % n_floor)
        check("summ[%s].seeds == [0,1,2]" % arm, sorted(srec.get("seeds", [])) == SEEDS)
        for sd in SEEDS:
            check("summ[%s].csi_at_2.0.s%d == scores_all" % (arm, sd),
                  near(srec["csi_at_2.0"]["s%d" % sd], vals[sd]))

print("  arm   median CSI@2.0   range                    >floor  >oracle  spread")
for arm in ALL7:
    if arm not in stats:
        continue
    st = stats[arm]
    print("  %-4s  %.6f          %.6f-%.6f   %d/3     %d/3     %.2fx" % (
        arm, st["med"], min(st["vals"]), max(st["vals"]),
        st["n_floor"], st["n_oracle"], st["spread"]))

# B0 = control: its own floor pass-rate must be < 3/3 -> floor convention is seed dependent
check("B0 itself does not clear floor 3/3 (floor convention unstable)", stats["B0"]["n_floor"] < 3)
above = [a for a in ARMS if a in stats and stats[a]["med"] > stats["B0"]["med"]]
check("arms above control median == {A4,A4b}", set(above) == {"A4", "A4b"})
b0_bias = summ["B0"]["median_bias_at_2.0"]
for a in ("A4", "A4b"):
    bias_rec = summ[a]["bias_at_2.0"]
    bmed = statistics.median([bias_rec["s%d" % s] for s in SEEDS])
    check("%s median BIAS inflated vs B0" % a, bmed > b0_bias)
# pre-registered branch-closing condition (E45 doc: "A4/A4b 的 BIAS@2.0 < 0.9" -> 该支就此结掉)
a4_bias = statistics.median([summ["A4"]["bias_at_2.0"]["s%d" % s] for s in SEEDS])
a4b_bias = statistics.median([summ["A4b"]["bias_at_2.0"]["s%d" % s] for s in SEEDS])
neg_check("A4 median BIAS < 0.9 (amount branch closes)", a4_bias >= 0.9)
check("A4b median BIAS >= 0.9 (passes that one condition)", a4b_bias >= 0.9)
print("  arms above B0 median:", above,
      "| median BIAS: B0=%.4f A4=%.4f A4b=%.4f" % (b0_bias, a4_bias, a4b_bias))

# ---------------------------------------------------------------- 3. mechanism
print("\n-- mechanism readouts (per arm per seed, keys under arms/<arm>/readouts/) --")


def rd(arm, sd, key):
    try:
        return probes[sd]["arms"][arm]["readouts"][key]
    except (KeyError, TypeError):
        return None


def k(arm, sd, key):
    return str(key).lower()


# -- gate arms A1/A2 : corr(mask,amp) >= 0 and mask_at_peak > 0.5
for arm in ("A1", "A2"):
    corr = [rd(arm, s, "corr_mask_amp") for s in SEEDS]
    peak = [rd(arm, s, "mask_at_peak") for s in SEEDS]
    check("%s corr(mask,amp) present 3/3" % arm, not any(is_missing(v) for v in corr))
    check("%s mask_at_peak present 3/3" % arm, not any(is_missing(v) for v in peak))
    pat_check("%s corr(mask,amp) >= 0 all seeds" % arm,
              [c is not None and c >= 0 for c in corr], [True] * 3)
    pat_check("%s mask_at_peak > 0.5 all seeds" % arm,
              [p is not None and p > 0.5 for p in peak], [True] * 3)
    print("  %-4s corr(mask,amp)=%s  mask_at_peak=%s" % (
        arm, [None if is_missing(v) else round(v, 4) for v in corr],
        [None if is_missing(v) else round(v, 4) for v in peak]))

# -- topology arms A3a/A3b : grad_top1pct_share < 0.5 AND field_max_mm <= 30
for arm in ("A3a", "A3b"):
    share = [rd(arm, s, "grad_top1pct_share") for s in SEEDS]
    fmax = [rd(arm, s, "field_max_mm") for s in SEEDS]
    check("%s grad_top1pct_share present 3/3" % arm, not any(is_missing(v) for v in share))
    check("%s field_max_mm present 3/3" % arm, not any(is_missing(v) for v in fmax))
    pat_check("%s grad_top1pct_share<0.5 pattern" % arm,
              [v is not None and v < 0.5 for v in share],
              [False, False, True] if arm == "A3a" else [False, False, False])
    pat_check("%s field_max_mm<=30 pattern" % arm,
              [v is not None and v <= 30.0 for v in fmax], [False, False, False])
    # pre-registered negative: topology criterion (grad<0.5 AND field_max<=30)
    # must still fail for every seed -> violation = some seed fully passes it
    violation = any(
        (sh is not None and sh < 0.5) and (fm is not None and fm <= 30.0)
        for sh, fm in zip(share, fmax))
    neg_check("%s topology criterion must still fail" % arm, violation)
    print("  %-4s grad_top1pct_share=%s  field_max_mm=%s" % (
        arm, [None if is_missing(v) else round(v, 4) for v in share],
        [None if is_missing(v) else round(v, 4) for v in fmax]))

# -- amount arms A4/A4b : scale_cv >= 0.05, scale_truth_corr >= 0.3, scale_is_constant False
for arm in ("A4", "A4b"):
    cv = [rd(arm, s, "scale_cv") for s in SEEDS]
    tc = [rd(arm, s, "scale_truth_corr") for s in SEEDS]
    ic = [rd(arm, s, "scale_is_constant") for s in SEEDS]
    check("%s scale_cv present 3/3" % arm, not any(is_missing(v) for v in cv))
    check("%s scale_truth_corr present 3/3" % arm, not any(is_missing(v) for v in tc))
    pat_check("%s scale_cv>=0.05 all seeds" % arm,
              [v is not None and v >= 0.05 for v in cv], [True] * 3)
    pat_check("%s scale_truth_corr>=0.3 pattern" % arm,
              [v is not None and v >= 0.3 for v in tc],
              [True] * 3) if arm == "A4" else pat_check(
              "%s scale_truth_corr>=0.3 pattern" % arm,
              [v is not None and v >= 0.3 for v in tc], [True, True, False])
    pat_check("%s scale_is_constant pattern" % arm, list(ic), [False, False, False] if arm == "A4" else [False, False, True])
    print("  %-4s scale_cv=%s  scale_truth_corr=%s  scale_is_constant=%s" % (
        arm, [None if is_missing(v) else round(v, 4) for v in cv],
        [None if is_missing(v) else round(v, 4) for v in tc], list(ic)))

# -- implementation defect: mean(out)=T
for arm in ("A4", "A4b"):
    mot = [rd(arm, s, "mean_out_equals_T") for s in SEEDS]
    check("%s mean_out_equals_T present 3/3" % arm, not any(v is None for v in mot))
    pat_check("%s mean_out_equals_T pattern" % arm, list(mot),
              [True, False, True] if arm == "A4" else [False, False, False])
    print("  %-4s mean_out_equals_T=%s" % (arm, list(mot)))

# -- missing readout: maxratio_as_trained must be NaN/None in all 3 seeds
for arm in ("A1", "A2"):
    mr = [rd(arm, s, "maxratio_as_trained") for s in SEEDS]
    check("%s maxratio_as_trained missing in 3/3" % arm, all(is_missing(v) for v in mr))
    print("  %-4s maxratio_as_trained=%s (expected all-missing)" % (arm, mr))

# ---------------------------------------------------------------- 4. write-back state
print("\n-- Gate-3 write-back state --")
master_p = os.path.join(DOCS, "MASTER.md")
index_p = os.path.join(DOCS, "experiments", "INDEX.md")
master = open(master_p, encoding="utf-8").read()
index = open(index_p, encoding="utf-8").read()
check("MASTER exists", os.path.exists(master_p))
head_rows = [l for l in master.splitlines()[:20] if "最后更新" in l]
check("MASTER head has 最后更新 row", len(head_rows) == 1)
if head_rows:
    check("MASTER head version row says v2.45", "v2.45" in head_rows[0])
check("MASTER has 4.47 card", "4.47" in master)
m_rows = [l for l in master.splitlines() if re.match(r"^\|\s*\*{0,2}E45", l)]
check("MASTER section-9 E45 row present (1)", len(m_rows) == 1)
if m_rows:
    cells = [c for c in m_rows[0].split("|")[1:-1]]
    check("MASTER section-9 E45 row has 3 cells", len(cells) == 3)
chg = [l for l in master.splitlines() if re.match(r"^\|\s*\d{4}-\d{2}-\d{2}\s*\|\s*\*\*v2\.4[45]\*\*", l)]
check("MASTER changelog has v2.44 and v2.45 rows", len(chg) == 2)
if len(chg) == 2:
    for l in chg:
        check("changelog row has 4 cells", len([c for c in l.split("|")[1:-1]]) == 4)
i_rows = [l for l in index.splitlines() if re.match(r"^\|\s*\*{0,2}E45", l)]
check("INDEX E45 row present (1)", len(i_rows) == 1)
if i_rows:
    cells = [c for c in i_rows[0].split("|")[1:-1]]
    check("INDEX E45 row has 6 cells", len(cells) == 6)
check("upstream MEMORY.md exists", os.path.exists(UPMEM))
if os.path.exists(UPMEM):
    up = open(UPMEM, encoding="utf-8").read()
    check("upstream MEMORY.md points at v2.45", "v2.45" in up)

# forbidden phrasing lint (assertion vs prohibition-reference aware)
FORBIDDEN = ["已实现 SOTA", "已解决 novelty", "LABEL 有孤立针状尖峰", "C0 被推翻"]
PROHIBITION = re.compile(r"禁写|不得写|不能写|禁止|不可再写|不可写|不再写|原文一条不删")
lint_hits = []
for path in (master_p, index_p):
    lines = open(path, encoding="utf-8").read().splitlines()
    for i, l in enumerate(lines):
        for f in FORBIDDEN:
            if f in l:
                ctx = "\n".join(lines[max(0, i - 3):i + 2])
                if not PROHIBITION.search(ctx):
                    lint_hits.append((os.path.basename(path), i + 1, f))
check("forbidden-phrase lint clean", not lint_hits)
if lint_hits:
    print("   lint hits:", lint_hits)

# ---------------------------------------------------------------- 5. report
def report(title, items):
    bad = [n for n, ok in items if not ok]
    print("  %-18s %3d/%3d passed" % (title, len(items) - len(bad), len(items)))
    for n in bad:
        print("      FAILED: %s" % n)
    return not bad


print("\n" + "=" * 78)
ok_pos = report("POS", POS)
ok_neg = report("NEG", NEG)
ok_pat = report("PAT", PAT)
allok = ok_pos and ok_neg and ok_pat
print("=" * 78)
print("VERDICT: %s  (POS %d/%d, NEG %d/%d, PAT %d/%d)" % (
    "PASS" if allok else "FAIL",
    sum(1 for _, o in POS if o), len(POS),
    sum(1 for _, o in NEG if o), len(NEG),
    sum(1 for _, o in PAT if o), len(PAT)))
sys.exit(0 if allok else 1)
