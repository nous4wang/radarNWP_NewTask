#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""E45 independent re-verification - round 29 (NEW checks only).

New in this round (not covered by rounds 12-28):
  [1] 待办四件事 (four outstanding items) registration completeness across the
      five authoritative documents (MASTER / E45 gate file / INDEX / MEMORY.md /
      HANDOVER_E45.md).
  [2] 结论措辞 (verdict wording) coverage across the same five documents:
      六臂判负 / 不投十 seed 算力 / 重开须重新提机制族 / 拓扑支结掉 /
      门指向结构上已解决 / 量通道不是瓶颈.

Design rules carried over from earlier rounds:
  * does NOT import any earlier verification script (old bugs would propagate);
  * reads raw artefacts directly (runs/e45 json under tools/_evals/e45);
  * every presence matrix prints the matched excerpt so a miss can be judged
    by the reader, not by a silent boolean;
  * assertion channels are separated: POS (must hold), NEG (must not hold),
    MAT (coverage matrix rows that must be non-empty).
"""

import json
import os
import re
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EVALS = os.path.join(REPO, "tools", "_evals", "e45")
OUTER = r"R:\1radarNWPprediction\.workbuddy\memory"

DOCS = {
    "MASTER": os.path.join(REPO, "docs", "MASTER.md"),
    "E45GATE": os.path.join(REPO, "docs", "experiments",
                            "E45-gate-direction-amount-separation-baseline.md"),
    "INDEX": os.path.join(REPO, "docs", "experiments", "INDEX.md"),
    "MEMORY": os.path.join(OUTER, "MEMORY.md"),
    "HANDOVER": os.path.join(OUTER, "HANDOVER_E45.md"),
}

POS = []
NEG = []
MAT = []


def check(name, cond, detail=""):
    POS.append((name, bool(cond), detail))
    return bool(cond)


def neg_check(name, violation, detail=""):
    """violation = 'the bad thing happened' (True => FAIL)."""
    NEG.append((name, bool(violation), detail))
    return not violation


def mat(name, value, detail=""):
    MAT.append((name, value, detail))


def load_text(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def find_excerpt(text, pattern, window=70):
    m = re.search(pattern, text)
    if not m:
        return None
    s = max(0, m.start() - 20)
    e = min(len(text), m.end() + window)
    return text[s:e].replace("\n", " / ")


# ---------------------------------------------------------------- documents
print("=" * 78)
print("E45 independent re-verification - round 29 (registration completeness)")
print("=" * 78)

texts = {}
for k, p in DOCS.items():
    if not os.path.exists(p):
        print("MISSING DOC: %s -> %s" % (k, p))
        sys.exit(2)
    texts[k] = load_text(p)
    print("read %-9s %7d chars  %s" % (k, len(texts[k]), p))

# ---------------------------------------------------- [1] 待办四件事登记
# four outstanding items carried by the closed E45 case (see HANDOVER §4)
ITEMS = [
    ("T1-clamp-not-on-output-field",
     r"幅度截断",
     "幅度截断未作用到输出场（场最大值 402.43 mm = 模型上限）"),
    ("T2-mean-out-eq-T-defect",
     r"mean\(out\)|mean_out_equals_T",
     "量通道 mean(out)=T 实现缺陷（修后重测，不当机制结论）"),
    ("T3-maxratio-as-trained-missing",
     r"maxratio_as_trained",
     "门那组 maxratio_as_trained 三 seed 全空值（读数缺失）"),
    ("T4-floor-baseline-unstable",
     r"地板口径|臂不稳",
     "判据工具要区分「臂不稳」与「地板口径不稳」"),
]

# documents that MUST carry the three implementation-defect items
# (T4 is a tooling item -> only required in the two "open work" documents)
REQUIRED = {
    "T1-clamp-not-on-output-field": ["MASTER", "E45GATE", "MEMORY", "HANDOVER"],
    "T2-mean-out-eq-T-defect": ["MASTER", "E45GATE", "MEMORY", "HANDOVER"],
    "T3-maxratio-as-trained-missing": ["MASTER", "E45GATE", "HANDOVER"],
    "T4-floor-baseline-unstable": ["E45GATE", "HANDOVER"],
}

# An item counts as registered if ANY of its alternative wordings is present.
# (round 29 lesson: MASTER's item ③ names the readout only in prose —
#  "门那组的第三个读数探针没测出来" — and does not spell out the key
#  `maxratio_as_trained`; that is still a registered outstanding item.)
ALTERNATES = {
    "T3-maxratio-as-trained-missing": [
        r"maxratio_as_trained",
        r"门那组的第三个读数",
        r"第三个读数.{0,20}空值",
    ],
}

print("\n--- [1] 待办四件事 登记矩阵 (rows=item, cols=doc, value=hit count) ---")
hdr = "  %-34s" % "item" + "".join("%-11s" % k for k in DOCS)
print(hdr)
for key, pat, desc in ITEMS:
    row = "  %-34s" % key
    for k in DOCS:
        hits = re.findall(pat, texts[k])
        row += "%-11s" % ("x%d" % len(hits) if hits else "-")
        if k in REQUIRED[key]:
            pats = ALTERNATES.get(key, [pat])
            hit = any(re.search(p, texts[k]) for p in pats)
            check("%s @ %s" % (key, k), hit, "patterns=%s" % pats)
    print(row + "  <- " + desc)

print("\n  excerpts (one per item, MASTER / E45GATE / MEMORY / HANDOVER):")
for key, pat, desc in ITEMS:
    for k in ("MASTER", "E45GATE", "MEMORY", "HANDOVER"):
        ex = find_excerpt(texts[k], pat)
        if ex:
            print("    [%s] %-22s %s" % (key[:22], k, ex[:150]))
    print("")

# ---------------------------------------------------- [2] 结论措辞覆盖
VERDICTS = [
    ("V1-six-arms-negative",
     r"六臂.{0,12}判负|六臂全部判负|判负",
     "六臂判负"),
    ("V2-no-ten-seed-compute",
     r"不投.{0,8}十 ?seed|不投十seed|无一臂.{0,10}十 ?seed",
     "不投十 seed 算力"),
    ("V3-reopen-needs-new-family",
     r"重新提机制族",
     "重开须重新提机制族"),
    ("V4-topology-branch-closed",
     r"拓扑支.{0,10}结掉|拓扑支就此结掉",
     "拓扑支按预注册结掉"),
    ("V5-gate-solved-but-not-bottleneck",
     r"结构上已解决",
     "门指向结构上已解决（但不是瓶颈）"),
    ("V6-amount-not-bottleneck",
     r"量通道.{0,12}不是瓶颈|量与结构分离",
     "量通道不是瓶颈（E45GATE 措辞为「量与结构分离…该支结掉」）"),
]

print("--- [2] 结论措辞 覆盖矩阵 ---")
print(hdr)
for key, pat, desc in VERDICTS:
    row = "  %-34s" % key
    for k in DOCS:
        hits = re.findall(pat, texts[k])
        row += "%-11s" % ("x%d" % len(hits) if hits else "-")
    print(row + "  <- " + desc)
    for k in ("MASTER", "E45GATE", "MEMORY", "HANDOVER"):
        ex = find_excerpt(texts[k], pat)
        if ex:
            print("      %-9s %s" % (k, ex[:130]))

# the closed-case verdict must be stated in every authoritative document
for key, pat, desc in VERDICTS:
    check("%s @ MASTER" % key, re.search(pat, texts["MASTER"]) is not None)
    check("%s @ E45GATE" % key, re.search(pat, texts["E45GATE"]) is not None)
    check("%s @ MEMORY" % key, re.search(pat, texts["MEMORY"]) is not None)

# ---------------------------------------------------- [3] raw artefacts sanity
print("\n--- [3] 原始产物复核（不 import 旧脚本，直接重取）---")
s3 = json.load(open(os.path.join(EVALS, "s3_equalbias.json"), encoding="utf-8"))
arms = s3["arms"]
print("  s3 arms:", sorted(arms.keys()))
for a in sorted(arms.keys()):
    blk = arms[a]
    raw, eb = blk["raw"], blk["equal_bias"]
    raw_cz = raw["ci_lo"] <= 0.0 <= raw["ci_hi"]
    eb_cz = eb["ci_lo"] <= 0.0 <= eb["ci_hi"]
    check("S3 raw crosses zero %s" % a, raw_cz == bool(raw["crosses_zero"]))
    check("S3 equal-bias crosses zero %s" % a, eb_cz == bool(eb["crosses_zero"]))
    neg_check("S3 raw gain %s still crosses zero" % a, not raw_cz,
              "point=%.6f ci=[%.6f, %.6f]" % (raw["point"], raw["ci_lo"], raw["ci_hi"]))
    neg_check("S3 equal-bias gain %s still crosses zero" % a, not eb_cz,
              "point=%.6f ci=[%.6f, %.6f]" % (eb["point"], eb["ci_lo"], eb["ci_hi"]))
    print("    %-4s raw=%+.6f [%.6f, %.6f] cz=%s | equal_bias=%+.6f [%.6f, %.6f] cz=%s | %s"
          % (a, raw["point"], raw["ci_lo"], raw["ci_hi"], raw["crosses_zero"],
             eb["point"], eb["ci_lo"], eb["ci_hi"], eb["crosses_zero"], blk["s3_verdict"]))

# three-seed medians recomputed from scores_all.json
sa = json.load(open(os.path.join(EVALS, "scores_all.json"), encoding="utf-8"))
FLOOR, ORACLE = 0.0924, 0.1653
per_arm = {}
for k, v in sa.items():
    m = re.match(r"^(.+)_s(\d+)_ep00$", k)
    if not m:
        continue
    per_arm.setdefault(m.group(1), {})[int(m.group(2))] = v
print("\n  3-seed median / pass-floor / over-Oracle (floor %.4f, oracle %.4f):" % (FLOOR, ORACLE))
for a in sorted(per_arm.keys()):
    cs = [per_arm[a][s]["csi_at_2.0"] for s in sorted(per_arm[a])]
    bs = [per_arm[a][s]["bias_at_2.0"] for s in sorted(per_arm[a])]
    med_c = sorted(cs)[len(cs) // 2]
    med_b = sorted(bs)[len(bs) // 2]
    pf = sum(1 for c in cs if c > FLOOR)
    oo = sum(1 for c in cs if c > ORACLE)
    print("    %-4s median CSI=%.4f  median BIAS=%.4f  pass_floor %d/%d  over_oracle %d/%d"
          % (a, med_c, med_b, pf, len(cs), oo, len(cs)))
    mat("3seed stats %s" % a, (med_c, med_b, pf, oo))
check("scores_all covers 7 arms x 3 seeds", len(per_arm) == 7 and
      all(len(v) == 3 for v in per_arm.values()),
      "arms=%d" % len(per_arm))

# mechanism readouts straight from the three probe files
print("\n  机制读数（逐臂逐 seed，键在 arms/<臂>/readouts/）:")
for s in (0, 1, 2):
    p = json.load(open(os.path.join(EVALS, "diag_probe_s%d.json" % s), encoding="utf-8"))
    for arm in ("A1", "A2"):
        ro = p["arms"][arm]["readouts"]
        cma = ro.get("corr_mask_amp")
        map_ = ro.get("mask_at_peak")
        print("    s%d %-4s corr(mask,amp)=%+.4f  mask_at_peak=%.4f" % (s, arm, cma, map_))
        check("gate corr>=0 %s s%d" % (arm, s), cma >= 0.0)
        check("gate mask_at_peak>0.5 %s s%d" % (arm, s), map_ > 0.5)
    for arm in ("A3a", "A3b"):
        ro = p["arms"][arm]["readouts"]
        share = ro.get("grad_top1pct_share")
        fmax = ro.get("field_max_mm")
        print("    s%d %-4s grad_top1pct_share=%.4f  field_max_mm=%.2f" % (s, arm, share, fmax))
        # pre-registered NEGATIVE result: topo branch must NOT pass the gate
        violation = (share < 0.5 and fmax <= 30.0)
        neg_check("topo branch still fails %s s%d" % (arm, s), violation,
                  "share=%.4f fmax=%.2f" % (share, fmax))
    for arm in ("A4", "A4b"):
        ro = p["arms"][arm]["readouts"]
        cv = ro.get("scale_cv")
        tc = ro.get("scale_truth_corr")
        ic = ro.get("scale_is_constant")
        print("    s%d %-4s scale_cv=%.4f  scale_truth_corr=%.4f  is_constant=%s"
              % (s, arm, cv, tc, ic))
        check("scale_cv>=0.05 %s s%d" % (arm, s), cv >= 0.05)
        if arm == "A4":
            check("scale_truth_corr>=0.3 A4 s%d" % s, tc >= 0.3)
            check("scale_is_constant A4 s%d is False" % s, ic is False)
        else:
            # known pattern: A4b degenerates on the third seed only
            #   scale_truth_corr = 0.2789 < 0.3  ->  scale_is_constant = True
            if s == 2:
                check("A4b s2 known degeneration (truth_corr<0.3)", tc < 0.3,
                      "value=%.4f" % tc)
                check("A4b s2 is_constant is True", ic is True)
            else:
                check("scale_truth_corr>=0.3 A4b s%d" % s, tc >= 0.3)
                check("scale_is_constant A4b s%d is False" % s, ic is False)
    # maxratio_as_trained: the gate readout that was never produced (A1)
    mr = p["arms"]["A1"]["readouts"].get("maxratio_as_trained")
    missing = (mr is None) or (isinstance(mr, float) and mr != mr)
    neg_check("maxratio_as_trained still missing (s%d)" % s, not missing,
              "value=%r" % (mr,))

# ---------------------------------------------------- summary
print("\n" + "=" * 78)
pos_fail = [n for n, ok, _ in POS if not ok]
neg_fail = [n for n, bad, _ in NEG if bad]
print("POS %d/%d   NEG %d/%d   MAT %d rows"
      % (len(POS) - len(pos_fail), len(POS),
         len(NEG) - len(neg_fail), len(NEG), len(MAT)))
for n in pos_fail:
    print("  POS FAIL:", n)
for n in neg_fail:
    print("  NEG FAIL:", n)
verdict = "PASS" if (not pos_fail and not neg_fail) else "FAIL"
print("VERDICT: %s" % verdict)
print("=" * 78)
sys.exit(0 if verdict == "PASS" else 1)
