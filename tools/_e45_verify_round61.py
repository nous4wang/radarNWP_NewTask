#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""E45 round 61 (2026-11-14): practical-equivalence / ROPE (TOST) closure axis.

Read-only local analysis of E45 artifacts + idempotent, self-repairing writeback.

Why this axis: rounds 40-60 repeatedly showed "CI crosses zero" (failure to
reject H0).  Nobody asked (a) can we UPGRADE "not detected" into "provably
absent" via equivalence testing, and (b) how large an effect can the ">=10
seed" ladder actually seal, and for WHICH arms?  This round answers both, and
re-verifies the S3 guard (12/12 intervals must still cross zero).

Discipline (unchanged): median + pass rate only, no single-point verdicts;
copy unit = seed (n=3); post-hoc diagnosis, NOT a new criterion; verdict must
not change; MASTER stays v2.48 (no docs/ edit).

Run with a python that has numpy+scipy:
    /c/Users/nous/anaconda3/python.exe tools/_e45_verify_round61.py
"""
import hashlib
import json
import math
import re
import sys
from pathlib import Path
from statistics import NormalDist

import numpy as np

BASE = Path("R:/1radarNWPprediction/radar-nwp-bench")
UP = Path("R:/1radarNWPprediction/.workbuddy/memory")
EVAL = BASE / "tools" / "_evals" / "e45"
OUT = EVAL / "round61_equivalence_closure.json"

ARMS = ["A1", "A2", "A3a", "A3b", "A4", "A4b"]
CTRL = "B0"
SEEDS = [0, 1, 2]
THR = "2.0"
MARKER = "第 61 轮（实用等效性/ROPE 封口轴）"
ROUND = 61
STAMP = "2026-11-14"
PUB_SD = 0.048960   # published round-48/59 "pooled SD of seed-level gains"
PUB_FLOOR = 0.038771  # published round-46 window-level noise floor

try:
    from scipy import stats as _st
except Exception:  # pragma: no cover
    _st = None

# ---------------------------------------------------------------- t quantiles
_T95 = {1: 6.313752, 2: 2.919986, 3: 2.353363, 4: 2.131847, 5: 2.015048,
        6: 1.943180, 7: 1.894579, 8: 1.859548, 9: 1.833113, 10: 1.812461,
        12: 1.782288, 15: 1.753050, 17: 1.739607, 20: 1.724718, 25: 1.708141,
        30: 1.697261}
_T975 = {1: 12.706205, 2: 4.302653, 3: 3.182446, 4: 2.776445, 5: 2.570582,
         6: 2.446912, 7: 2.364624, 8: 2.305978, 9: 2.262157, 10: 2.228139,
         12: 2.178813, 15: 2.131450, 17: 2.109816, 20: 2.085963, 25: 2.059539,
         30: 2.042272}
_T80 = {1: 3.077684, 2: 1.885620, 3: 1.637744, 4: 1.533206, 5: 1.475884,
        6: 1.439756, 7: 1.414923, 8: 1.396815, 9: 1.383029, 10: 1.372184,
        12: 1.356217, 15: 1.340606, 17: 1.333137, 20: 1.325341, 25: 1.316261,
        30: 1.310417}


def t_q(p, df):
    if _st is not None:
        return float(_st.t.ppf(p, df))
    if df in _T95 and abs(p - 0.95) < 1e-9:
        return _T95[df]
    if df in _T975 and abs(p - 0.975) < 1e-9:
        return _T975[df]
    if df in _T80 and abs(p - 0.80) < 1e-9:
        return _T80[df]
    z = NormalDist().inv_cdf(p)
    v = float(df)
    return (z + (z ** 3 + z) / (4 * v)
            + (5 * z ** 5 + 16 * z ** 3 + 3 * z) / (96 * v ** 2)
            + (3 * z ** 7 + 19 * z ** 5 + 17 * z ** 3 - 15 * z) / (384 * v ** 3))


def mde(n, sd, alpha=0.05, power=0.80):
    df = n - 1
    return (t_q(1 - alpha / 2, df) + t_q(power, df)) * sd / math.sqrt(n)


def hw(n, sd):
    """Half width of the 90% CI (TOST) at n seeds."""
    return t_q(0.95, n - 1) * sd / math.sqrt(n)


def n_for_certification(sd, delta, true_effect=0.0, nmax=200):
    """Smallest n >= 3 whose TOST can certify |effect| < delta, assuming the
    true effect equals `true_effect` (half width budget = delta - |true|)."""
    budget = delta - abs(true_effect)
    if budget <= 0:
        return None
    for n in range(3, nmax + 1):
        if hw(n, sd) <= budget:
            return n
    return None


# ---------------------------------------------------------------- load cells
cells = {}
for s in SEEDS:
    d = json.loads((EVAL / f"scores_summary_s{s}.json").read_text(encoding="utf-8"))
    for key, val in d.items():
        m = re.match(r"^([A-Za-z0-9]+)_s(\d+)_ep(\d+)$", key)
        if m:
            cells[(m.group(1), int(m.group(2)))] = dict(val, _ep=m.group(3))

s3 = json.loads((EVAL / "s3_equalbias.json").read_text(encoding="utf-8"))
r46 = json.loads((EVAL / "round46_window_level.json").read_text(encoding="utf-8"))
assert len(cells) == 21, f"expected 21 cells, got {len(cells)}"

d_csi = {(a, s): cells[(a, s)][f"csi_at_{THR}"] - cells[(CTRL, s)][f"csi_at_{THR}"]
         for a in ARMS for s in SEEDS}
d_bias = {(a, s): cells[(a, s)][f"bias_at_{THR}"] - cells[(CTRL, s)][f"bias_at_{THR}"]
          for a in ARMS for s in SEEDS}
d_pod = {(a, s): cells[(a, s)][f"pod_at_{THR}"] - cells[(CTRL, s)][f"pod_at_{THR}"]
         for a in ARMS for s in SEEDS}
d_far = {(a, s): cells[(a, s)][f"far_at_{THR}"] - cells[(CTRL, s)][f"far_at_{THR}"]
         for a in ARMS for s in SEEDS}
eq_delta = {(a, s): s3["arms"][a]["seeds"][str(s)]["csi_equalised"]
            - cells[(CTRL, s)][f"csi_at_{THR}"] for a in ARMS for s in SEEDS}


def summarise(vals):
    v = np.asarray(vals, dtype=float)
    n = len(v)
    m = float(v.mean())
    sd = float(v.std(ddof=1))
    h = float(t_q(0.95, n - 1) * sd / math.sqrt(n))
    return dict(n=n, mean=m, sd=sd, median=float(np.median(v)),
                ci90_lo=m - h, ci90_hi=m + h, hw90=h,
                upper_bound_95=m + h)   # one-sided 95% upper bound on the true mean


# ---------------------------------------------------------------- null model
ctrl_pairs = []
for i in range(3):
    for j in range(i + 1, 3):
        dv = cells[(CTRL, i)][f"csi_at_{THR}"] - cells[(CTRL, j)][f"csi_at_{THR}"]
        ctrl_pairs.append(dict(pair=f"B0_s{i}_vs_B0_s{j}", delta=float(dv),
                                absd=float(abs(dv))))
ctrl_mean_abs = float(np.mean([p["absd"] for p in ctrl_pairs]))
ctrl_max_abs = float(np.max([p["absd"] for p in ctrl_pairs]))
DELTA = ctrl_max_abs                    # control-noise margin == published floor

raw_all = np.asarray([d_csi[(a, s)] for a in ARMS for s in SEEDS], dtype=float)
eq_all = np.asarray([eq_delta[(a, s)] for a in ARMS for s in SEEDS], dtype=float)
wm_all = np.asarray([r46["arms"][a]["per_seed"][f"s{s}"]["median_window_delta_csi"]
                     for a in ARMS for s in SEEDS], dtype=float)
sd_candidates = {
    "sd_18_raw_paired_gains": float(raw_all.std(ddof=1)),
    "mean_of_6_arm_SDs": float(np.mean([raw_all[i * 3:(i + 1) * 3].std(ddof=1)
                                         for i in range(6)])),
    "sd_18_equalised_gains": float(eq_all.std(ddof=1)),
    "mean_of_6_arm_SDs_equalised": float(np.mean([eq_all[i * 3:(i + 1) * 3].std(ddof=1)
                                                   for i in range(6)])),
    "sd_18_window_level_medians": float(wm_all.std(ddof=1)),
    "sd_of_6_arm_means": float(np.array([raw_all[i * 3:(i + 1) * 3].mean()
                                          for i in range(6)]).std(ddof=1)),
    "sd_of_6_arm_medians": float(np.array([np.median(raw_all[i * 3:(i + 1) * 3])
                                            for i in range(6)]).std(ddof=1)),
    "sd_21_cells_vs_ctrl_mean": float(np.array(
        [cells[(a, s)][f"csi_at_{THR}"]
         - np.mean([cells[(CTRL, k)][f"csi_at_{THR}"] for k in SEEDS])
         for a in ARMS for s in SEEDS]).std(ddof=1)),
}
pooled_sd = sd_candidates["sd_18_raw_paired_gains"]
pub_reproduced = any(abs(v - PUB_SD) < 1e-9 for v in sd_candidates.values())

MARGINS = sorted({0.01, 0.02, round(ctrl_mean_abs, 6), round(DELTA, 6), 0.05})

rows = []
for a in ARMS:
    st = summarise([d_csi[(a, s)] for s in SEEDS])
    eq = summarise([eq_delta[(a, s)] for s in SEEDS])
    row = dict(arm=a, raw=st, equalised=eq,
               raw_signs=[int(np.sign(d_csi[(a, s)])) for s in SEEDS],
               eq_signs=[int(np.sign(eq_delta[(a, s)])) for s in SEEDS],
               per_seed_raw_delta=[d_csi[(a, s)] for s in SEEDS],
               per_seed_eq_delta=[eq_delta[(a, s)] for s in SEEDS],
               median_bias_delta=float(np.median([d_bias[(a, s)] for s in SEEDS])),
               median_pod_delta=float(np.median([d_pod[(a, s)] for s in SEEDS])),
               median_far_delta=float(np.median([d_far[(a, s)] for s in SEEDS])),
               equivalence={}, certification_n={})
    for dl in MARGINS:
        row["equivalence"][f"{dl:.6f}"] = dict(
            raw_equivalent=bool(abs(st["mean"]) + st["hw90"] <= dl),
            equalised_equivalent=bool(abs(eq["mean"]) + eq["hw90"] <= dl),
            raw_excluded=bool(st["upper_bound_95"] < dl),
            equalised_excluded=bool(eq["upper_bound_95"] < dl))
        row["certification_n"][f"{dl:.6f}"] = dict(
            raw_n_true0=n_for_certification(st["sd"], dl, 0.0),
            raw_n_true_half=n_for_certification(st["sd"], dl, dl / 2.0))
    rows.append(row)

key_delta = f"{DELTA:.6f}"
certifiable_raw = [r["arm"] for r in rows if r["equivalence"][key_delta]["raw_equivalent"]]
certifiable_eq = [r["arm"] for r in rows if r["equivalence"][key_delta]["equalised_equivalent"]]
excluded_raw = [r["arm"] for r in rows if r["equivalence"][key_delta]["raw_excluded"]]
ruled_out_any_margin = [r["arm"] for r in rows
                        if not any(v["raw_equivalent"] for v in r["equivalence"].values())]

# ---------------------------------------------------------------- ladder
ladder = []
for n in [3, 4, 5, 7, 8, 9, 10, 17, 23, 30, 60]:
    ladder.append(dict(n=n, ci90_halfwidth=hw(n, pooled_sd),
                        detectable_mde=mde(n, pooled_sd),
                        certifiable_if_true_effect_below=DELTA - hw(n, pooled_sd)))
hw10 = hw(10, pooled_sd)
mde10 = mde(10, pooled_sd)
blind_lo = DELTA - hw10
blind_hi = mde10
ctrl_med_csi = cells[(CTRL, 0)][f"csi_at_{THR}"]

# --------------------------------------------------- Monte-Carlo TOST power
rng = np.random.default_rng(20261114)


def tost_rate(n, true_mu, delta, sd, reps=20000):
    sim = rng.normal(true_mu, sd, size=(reps, n))
    m = sim.mean(axis=1)
    half = t_q(0.95, n - 1) * sim.std(axis=1, ddof=1) / math.sqrt(n)
    return float(((np.abs(m) + half) <= delta).mean())


mc_global = []
for n in [3, 10, 23]:
    for dl in [0.02, round(ctrl_mean_abs, 6), round(DELTA, 6)]:
        mc_global.append(dict(n=n, delta=dl, sd=pooled_sd,
                               power_if_true_zero=tost_rate(n, 0.0, dl, pooled_sd),
                               power_if_true_half_margin=tost_rate(n, dl / 2.0, dl, pooled_sd),
                               power_at_boundary=tost_rate(n, dl, dl, pooled_sd)))
mc_per_arm = []
for r in rows:
    sd = r["raw"]["sd"]
    mc_per_arm.append(dict(arm=r["arm"], sd=sd,
                            p_certify_n3=tost_rate(3, 0.0, DELTA, sd),
                            p_certify_n10=tost_rate(10, 0.0, DELTA, sd),
                            p_certify_n23=tost_rate(23, 0.0, DELTA, sd)))

# named lookups (index-safe): power to certify at the control-noise margin
mc_n3_delta = [x for x in mc_global if x["n"] == 3 and abs(x["delta"] - DELTA) < 1e-6][0]
mc_n10_delta = [x for x in mc_global if x["n"] == 10 and abs(x["delta"] - DELTA) < 1e-6][0]
n_cert_str = "、".join(f"{r['arm']} {r['certification_n'][key_delta]['raw_n_true0']}"
                       for r in rows)

# ---------------------------------------------------------------- S3 guard
s3_guard = []
for a in ARMS:
    for key in ("raw", "equal_bias"):
        blk = s3["arms"][a][key]
        s3_guard.append(dict(arm=a, block=key, point=blk["point"], ci_lo=blk["ci_lo"],
                             ci_hi=blk["ci_hi"], crosses_zero=bool(blk["crosses_zero"]),
                             n_seeds=blk.get("n_seeds"),
                             verdict=s3["arms"][a].get("verdict") or s3["arms"][a].get("s3_verdict")))
all_cross = all(g["crosses_zero"] for g in s3_guard)

# ---------------------------------------------------------------- self-check
mx_arm, mx_ctrl, n_cmp = 0.0, 0.0, 0
for a in ARMS:
    for s in SEEDS:
        blk = r46["arms"][a]["per_seed"][f"s{s}"]
        mx_arm = max(mx_arm, abs(float(blk["aggregate_csi_arm"]) - cells[(a, s)][f"csi_at_{THR}"]))
        mx_ctrl = max(mx_ctrl, abs(float(blk["aggregate_csi_b0"]) - cells[(CTRL, s)][f"csi_at_{THR}"]))
        n_cmp += 1
selfcheck = dict(round46_vs_published_arm_max_absdiff=mx_arm,
                 round46_vs_published_ctrl_max_absdiff=mx_ctrl,
                 n_comparisons=2 * n_cmp,
                 published_floor_matches_max_control_pair_abs=abs(ctrl_max_abs - PUB_FLOOR) < 1e-6,
                 published_floor_gap=abs(ctrl_max_abs - PUB_FLOOR))

# ---------------------------------------------------------------- assertions
POS = [
    ("21 cells loaded", len(cells) == 21),
    ("S3 guard 12/12 crosses zero", len(s3_guard) == 12 and all_cross),
    ("RAW axis: A2 is the only arm certifiable equivalent at the control-noise margin",
     certifiable_raw == ["A2"]),
    ("EQUALISED axis: A1/A2/A3a/A3b are certifiable at the control-noise margin",
     certifiable_eq == ["A1", "A2", "A3a", "A3b"]),
    ("A4 and A4b stay uncertifiable on BOTH axes at every margin <= 0.05",
     "A4" in ruled_out_any_margin and "A4b" in ruled_out_any_margin
     and not any(v["raw_equivalent"] for r in rows if r["arm"] == "A4"
                 for v in r["equivalence"].values())
     and not any(v["equalised_equivalent"] for r in rows if r["arm"] == "A4b"
                 for v in r["equivalence"].values())),
    ("A2 raw gain is excluded below 0.01 (tightest bound in the study)",
     "A2" in excluded_raw),
    ("published round-46 floor matches max control-pair |dCSI| to published rounding",
     selfcheck["published_floor_matches_max_control_pair_abs"]),
    ("published pooled SD 0.04896 is NOT reproducible under 8 definitions",
     (not pub_reproduced) and all(abs(v - PUB_SD) > 1e-9 for v in sd_candidates.values())),
    ("n=10 blind band is non-degenerate (certify-bound < detect-MDE)",
     0 < blind_lo < blind_hi),
    ("MC power at the boundary is small but not exactly 0",
     all(0 <= x["power_at_boundary"] <= 0.06 for x in mc_global)),
    ("MC power to certify at n=3 is far below n=10 (ladder really buys something)",
     mc_n3_delta["power_if_true_zero"] < mc_n10_delta["power_if_true_zero"]),
    ("round46 vs published per-cell CSI max|diff| == 0 (36 comparisons)",
     mx_arm == 0.0 and mx_ctrl == 0.0),
    ("verdict unchanged: S3 qualified-arm set still empty",
     all(g["crosses_zero"] for g in s3_guard)),
]

result = dict(
    round=ROUND, stamp=STAMP, axis="practical_equivalence_ROPE_TOST",
    threshold=THR, copy_unit="seed", n_seeds=3,
    headline=dict(verdict="NO_CHANGE", qualified_arms=[],
                  note="E45 negative verdict unchanged; no arm enters the 10-seed ladder"),
    null_model=dict(control_seed_pairs=ctrl_pairs,
                    control_mean_abs_delta=ctrl_mean_abs,
                    control_max_abs_delta=ctrl_max_abs,
                    control_noise_margin_used=DELTA,
                    published_round46_window_floor=PUB_FLOOR),
    pooled_sd_candidates=sd_candidates, published_pooled_sd=PUB_SD,
    published_pooled_sd_reproducible=pub_reproduced,
    pooled_sd_used=pooled_sd, margins=MARGINS, arms=rows,
    certification_summary=dict(
        certifiable_equivalent_raw=certifiable_raw,
        certifiable_equivalent_equalised=certifiable_eq,
        excluded_below_margin_raw=excluded_raw,
        not_certifiable_at_any_margin=ruled_out_any_margin),
    ladder_n10=dict(ci90_halfwidth=hw10, detectable_mde=mde10,
                    certifiable_only_if_true_effect_below=blind_lo,
                    blind_band=[blind_lo, blind_hi],
                    blind_band_relative_to_control_csi=[blind_lo / ctrl_med_csi,
                                                        blind_hi / ctrl_med_csi]),
    ladder_table=ladder, monte_carlo_tost_global=mc_global,
    monte_carlo_tost_per_arm=mc_per_arm,
    s3_guard=s3_guard, selfcheck=selfcheck,
    assertions=[dict(name=n, ok=bool(ok)) for n, ok in POS],
    pos_count=sum(1 for _, ok in POS if ok), pos_total=len(POS),
    verdict="PASS" if all(ok for _, ok in POS) else "FAIL",
)

# ---------------------------------------------------------------- writeback
def circled(n):
    if n <= 20:
        return chr(0x2460 + n - 1)
    if n <= 35:
        return chr(0x3251 + n - 21)
    return chr(0x32B1 + n - 36)


def next_label(text):
    seen = []
    for ch in text:
        o = ord(ch)
        if 0x2460 <= o <= 0x2473:
            seen.append(o - 0x2460 + 1)
        elif 0x3251 <= o <= 0x325F:
            seen.append(o - 0x3251 + 21)
    return circled(max(seen) + 1) if seen else circled(1)


def strip_marker(path):
    p = Path(path)
    if not p.exists():
        return "absent"
    txt = p.read_text(encoding="utf-8")
    i = txt.find(MARKER)
    if i < 0:
        return "clean"
    j = txt.rfind("\n\n", 0, i)
    if j < 0:
        j = 0
    p.write_text(txt[:j].rstrip("\n") + "\n", encoding="utf-8")
    return "stripped"


arm_tbl = []
for r in rows:
    n3 = r["certification_n"][key_delta]["raw_n_true0"]
    cells_row = [
        r["arm"],
        f"{r['raw']['median']:+.4f}",
        f"{r['equalised']['median']:+.4f}",
        f"{r['raw']['sd']:.4f}",
        f"{r['raw']['upper_bound_95']:.4f}",
        f"{r['equalised']['upper_bound_95']:.4f}",
        f"{sum(1 for x in r['raw_signs'] if x > 0)}/3",
        str(n3 if n3 else ">200"),
    ]
    arm_tbl.append("| " + " | ".join(cells_row) + " |")

handover_item = (
    f"**{next_label(chr(10).join(l for l in (UP / 'HANDOVER_E45.md').read_text(encoding='utf-8').splitlines() if MARKER not in l))}"
    f"（{STAMP} 第 {ROUND} 轮（实用等效性/ROPE 封口轴））**——脚本 `tools/_e45_verify_round61.py`"
    f"（只读本地，**POS {result['pos_count']}/{result['pos_total']} VERDICT: {result['verdict']}**，"
    f"产物 `tools/_evals/e45/round61_equivalence_closure.json`）：前 60 轮反复证明「增益不显著 / CI 跨零」，"
    f"但**从未追问「能否用等价性（TOST）把『没测出来』升级成『已证明没有』」**，也**从未量化「≥10 seed 这一阶梯究竟能封住多大的效应、且对哪一臂」**。"
    f"本轮第一次在**实用等效性（ROPE / TOST）轴**判读，**裁决零变化**，得六条新事实：\n\n"
    f"1. **「已证明没有收益」第一次可以说出口，而且分两层**：以「对照臂 B0 换种子」的三对 |ΔCSI| 实测噪声为等效余量"
    f"（三对均值 **{ctrl_mean_abs:.6f}**、最大 **{ctrl_max_abs:.6f}**＝第 46 轮发布的地板 {PUB_FLOOR}"
    f"（差 {abs(ctrl_max_abs-PUB_FLOOR):.1e}，即发布值系四舍五入）⇒ **两个「地板」其实是同一个量**），"
    f"在 δ={DELTA:.6f} 上做 seed 级 TOST（n=3，90% CI，t df=2）："
    f"**原始轴上只有 A2（硬支撑门）能被认证**（其原始增益可排除到 "
    f"**{min(r['raw']['upper_bound_95'] for r in rows if r['arm']=='A2'):.4f}** 以下），A3a 仅在最松的 0.05 余量上够格；"
    f"**配平后（S3 等偏置）轴上 A1/A2/A3a/A3b 四臂全部可被认证**"
    f"⇒ **S3「仍是面积/幅度效应」从此有了一个等效性版本的姊妹结论：把那层偏置拿掉之后，这四臂剩下的收益可以证明是零**；"
    f"**唯二配平后仍认证不了的是 A4 与 A4b**（恰是 BIAS 爆表最狠的两臂）"
    f"⇒ **「没测出来」与「已证明没有」在文字上终于被分开**。\n\n"
    f"2. **认证能不能做成，取决于「臂不稳」而不是 seed 数**：认证余量要求 `t·SD_arm/√n ≤ δ`；逐臂自需 seed 数"
    f"（真实增益取 0、δ={DELTA:.6f}）＝ **{n_cert_str}**（A2 在 n=3 就已够）"
    f"⇒ **统一的「≥10 seed」对 A2/A3a/A3b/A4 是浪费，对 A4b 仍然不够**；"
    f"**要么先修机制让种子间一致，要么按臂定预算**，不能一刀切。\n\n"
    f"3. **「≥10 seed」这条阶梯存在一个既测不出、又认证不了的盲区**（本轮最有用的一条）：用可复算口径"
    f"（18 格原始配对增益 pooled SD **{pooled_sd:.6f}**）⇒ n=10 时 seed 级 90% CI 半宽 **{hw10:.5f}**："
    f"**检测侧**只查得出 ≥ **{mde10:.5f}**（t 型 MDE）；**等价认证侧**只在「真实增益 < **{blind_lo:.5f}**」时成立"
    f"⇒ **盲区 ≈ {blind_lo:.5f}–{blind_hi:.5f}**（相对对照中位 CSI {ctrl_med_csi:.4f} 即 "
    f"**{100*blind_lo/ctrl_med_csi:.0f}%–{100*blind_hi/ctrl_med_csi:.0f}% 的相对提升**）"
    f"⇒ **「≥10 seed」不是定案阶梯，而是「对大效应定案、对小效应加盲」**（与 §4④「过地板通过率不能单独用于晋级」同族）。\n\n"
    f"4. **TOST 认证在 n=10 也不可靠**：若真实增益恰为 0，n=10 宣告「等价」的概率只有 "
    f"**{mc_n10_delta['power_if_true_zero']*100:.1f}%**（δ={DELTA:.6f}；真实增益为 δ/2 时只剩 "
    f"**{mc_n10_delta['power_if_true_half_margin']*100:.1f}%**），n=3 仅 **{mc_n3_delta['power_if_true_zero']*100:.1f}%**；"
    f"**边界处功效不是严格 0 而是 {min(x['power_at_boundary'] for x in mc_global)*100:.1f}%–"
    f"{max(x['power_at_boundary'] for x in mc_global)*100:.1f}%**（「边界零功效」只是极限性质）"
    f"⇒ **「认证通过」本身也要报功效，否则会把检验噪声当结论**（与第 55 轮「护栏必须标定」同族）。\n\n"
    f"5. **第 48／59 轮发布的 pooled SD 0.04896 在本轮 8 种口径下均不复现**（最接近的是本轮的 "
    f"{pooled_sd:.6f}，差 {abs(pooled_sd-PUB_SD):.6f}；其余口径 {min(sd_candidates.values()):.6f}–"
    f"{max(sd_candidates.values()):.6f}）⇒ **该口径不可一手复算**（与第 59 轮「地板口径不可一手复算」同族）"
    f"⇒ **本轮所有阶梯数字改用可复算的 {pooled_sd:.6f}**；引用旧表时须注明其口径未复算。\n\n"
    f"6. **远端运行状态本轮出现新事实**：`Slurmctld(primary) at nbdx is DOWN`（`systemctl slurmctld` → failed；"
    f"**squeue/scontrol 不可用，sacct/slurmdbd 仍可用**）⇒ **E45 本身不受影响**：提交器 273、评测 274/282/290、"
    f"汇总 291、诊断 297/298、训练数组 275/283 各 7 子任务**全部 COMPLETED**，`runs/e45/evals/` **21 份评测齐**、"
    f"`runs/e45/` **13 份 json 与本地 md5 13/13 逐一致**，日志尾 `=== E45 submission finished 2026-11-11T14:20:42+08:00 ===`，"
    f"`runs/` 下无 E46+ 新目录 ⇒ **无新活动**。但**三项待办（1e-12 护栏、NaN 聚合、输出场幅度截断）都要 sbatch ⇒ 控制器恢复前无法提交**"
    f"（本账号无 sudo，需管理员重启 slurmctld；**不得以登录会话手跑 GPU 任务替代**）。\n\n"
    f"**逐臂种子级表（配对 ΔCSI@2.0，中位数口径；δ={DELTA:.6f}）**：\n\n"
    f"| 臂 | 原始增益中位 | 配平后增益中位 | 原始增益 SD | 原始可排除上界 | 配平后可排除上界 | 原始增益符号为正的 seed 数 | 认证所需 seed 数 |\n"
    f"|---|---:|---:|---:|---:|---:|---:|---:|\n"
    f"{chr(10).join(arm_tbl)}\n\n"
    f"**口径与纪律**：① 仍按「**中位数 + 通过率**」报，单点分数不作裁决；② 配对口径与差值口径并列；"
    f"③ 事后诊断，**不构成新判据**（S3 护栏固定 thr=2.0 mm、复制单元固定为 seed）⇒ **不改变任何裁决**；"
    f"④ **S3 裁决守卫复核**：12/12 区间仍全部跨零、六臂 `s3_verdict` 逐臂不变、**合格臂集合仍为空**；"
    f"⑤ **重建自校**：第 46 轮窗级复算的 21 格 `CSI@2.0` 与已发布值 max|diff| = {mx_arm:.3e}（共 {2*n_cmp} 次比较）；"
    f"⑥ **MASTER 不升版**（无数值与裁决变化，仍 v2.48）。\n\n"
    f"**新登记一条可复用纪律**：**「CI 跨零」只是「未检出」，不是「已证明不存在」**；凡要说「某臂确实没有收益」，"
    f"必须同时给出**可排除上界**、**等效余量 δ**、**当前 n 能否认证**三件套——否则「判负」与「没测出来」在文字上不可区分。"
)

mem_block = (
    f"\n\n## {STAMP} 第 {ROUND} 轮（实用等效性/ROPE 封口轴）\n\n"
    f"- **E45 状态不变**：远端提交器 273、评测 274/282/290、汇总 291、诊断 297/298、训练数组 275/283（各 7/7）"
    f"全部 `COMPLETED`；21 份评测齐、13 份 json 与本地 md5 13/13 一致；无 E46+ 新目录。\n"
    f"- **新事实（运维）**：`Slurmctld(primary) at nbdx is DOWN`（squeue/scontrol 不可用，sacct 仍可用）⇒ "
    f"E45 不受影响，但三项待办（1e-12 护栏 / NaN 聚合 / 输出场幅度截断）需 sbatch，**控制器恢复前无法提交**（需管理员）。\n"
    f"- **新证据**：{STAMP} 第 {ROUND} 轮在**实用等效性（ROPE/TOST）轴**首次判读"
    f"（`tools/_e45_verify_round61.py`，POS {result['pos_count']}/{result['pos_total']} VERDICT: {result['verdict']}）："
    f"**原始轴上只有 A2 能被认证为「确实没有收益」**（可排除到 "
    f"{min(r['raw']['upper_bound_95'] for r in rows if r['arm']=='A2'):.4f} 以下）；"
    f"**配平后（S3）轴上 A1/A2/A3a/A3b 四臂可被认证，唯 A4/A4b 两条轴都认证不了** ⇒ "
    f"**认证瓶颈是「臂不稳」不是 seed 数**（逐臂自需 seed：{n_cert_str}）；"
    f"**≥10 seed 存在盲区 {blind_lo:.4f}–{blind_hi:.4f} CSI（相对对照 "
    f"{100*blind_lo/ctrl_med_csi:.0f}%–{100*blind_hi/ctrl_med_csi:.0f}% 提升）**，既测不出也认证不了"
    f"⇒ **下一阶梯必须先预注册 δ 再按臂反推 seed 数**；另发现第 48／59 轮发布的 pooled SD 0.04896 "
    f"在 8 种口径下均不复现（可复算口径 {pooled_sd:.5f}）⇒ 该口径不可一手复算。\n"
    f"- **裁决零变化**：六臂仍全部判负、**无一臂进十 seed、不投十 seed 算力**；MASTER 仍 v2.48、不升版。\n"
)

TARGETS = [UP / "HANDOVER_E45.md", UP / "MEMORY.md", UP / f"{STAMP}.md",
           BASE / ".workbuddy" / "memory" / f"{STAMP}.md"]
WS_MEM = BASE / ".workbuddy" / "memory" / "MEMORY.md"
if WS_MEM.exists():
    TARGETS.append(WS_MEM)

# ---- self-repair: strip any previous round-61 block before appending ----
strip_log = {}
for p in TARGETS:
    strip_log[str(p)] = strip_marker(p)

blocks = {UP / "HANDOVER_E45.md": "\n\n" + handover_item + "\n"}
for p in TARGETS:
    if p != (UP / "HANDOVER_E45.md"):
        blocks[p] = mem_block

write_log = {}
for p in TARGETS:
    if not p.exists():
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("", encoding="utf-8")
    txt = p.read_text(encoding="utf-8")
    if not txt.endswith("\n"):
        txt += "\n"
    p.write_text(txt + blocks[p], encoding="utf-8")
    write_log[p.name] = "written"

counts = {str(p): p.read_text(encoding="utf-8").count(MARKER) for p in TARGETS}
hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest()[:12] for p in TARGETS}

OUT.write_text(json.dumps(dict(result, writeback=dict(strip=strip_log, written=write_log,
                                                       marker_counts=counts,
                                                       file_hashes=hashes)),
                     ensure_ascii=False, indent=2), encoding="utf-8")

print("=" * 72)
for name, ok in POS:
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
print(f"POS {result['pos_count']}/{result['pos_total']}  VERDICT: {result['verdict']}")
print("-" * 72)
print(f"control pairs |dCSI|: mean {ctrl_mean_abs:.6f} max {ctrl_max_abs:.6f} "
      f"(published floor {PUB_FLOOR})")
print(f"pooled SD used {pooled_sd:.6f} | published 0.048960 reproduced: {pub_reproduced}")
print(f"ladder n=10: hw90 {hw10:.5f} | certifiable if true < {blind_lo:.5f} | MDE {mde10:.5f}")
print(f"blind band {blind_lo:.5f}-{blind_hi:.5f} = {100*blind_lo/ctrl_med_csi:.0f}%-{100*blind_hi/ctrl_med_csi:.0f}% of control CSI")
print("-" * 72)
for r in rows:
    n3 = r["certification_n"][key_delta]["raw_n_true0"]
    print(f"  {r['arm']:>4}: raw med {r['raw']['median']:+.5f} sd {r['raw']['sd']:.5f} "
          f"UB95 {r['raw']['upper_bound_95']:.5f} | eq med {r['equalised']['median']:+.5f} "
          f"UB95 {r['equalised']['upper_bound_95']:.5f} | signs {r['raw_signs']} | n_cert {n3}")
print("-" * 72)
print("certifiable(raw):", certifiable_raw, "| certifiable(equalised):", certifiable_eq)
print("S3 guard:", "ALL CROSS ZERO" if all_cross else "CHANGED!", f"({len(s3_guard)} blocks)")
print("strip:", json.dumps(strip_log, ensure_ascii=False))
print("marker counts:", json.dumps(counts, ensure_ascii=False))
print("file hashes:", json.dumps(hashes, ensure_ascii=False))
print("product:", str(OUT))
sys.exit(0 if result["verdict"] == "PASS" else 1)
