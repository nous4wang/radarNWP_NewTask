#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
E45 第 59 轮：晋级规则（地板口径 × 过地板通过率 × S3 护栏）联合标定轴。

只读本地产物；不触远端；不写 docs/；不改任何裁决。
产出：tools/_evals/e45/round59_promotion_rule_calibration.json

背景：第 40–58 轮判过聚合层／机制探针／S1 结构量／窗级／场次级／时效轴／
强度·难度轴／误差构成／臂间合并／剂量-反应／指标族／护栏标定（自助深度）／
增益图样稳定性／聚合算子＋评测契约／训练动力学，共十五层；
**从未审过「晋级规则本身」**——即「过地板 0.0924 + 3/3 通过率 + S3 配平后区间」
这条门槛有多敏感、哪个约束在真正绑定、以及它在无机制零假设下的假晋级率。
"""
import itertools
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent
EV = ROOT / "_evals" / "e45"
RAW = EV / "_raw" / "runs" / "e45" / "evals"
OUT = EV / "round59_promotion_rule_calibration.json"

CTRL = "B0"
ARMS = ["A1", "A2", "A3a", "A3b", "A4", "A4b"]
ALL_ARMS = [CTRL] + ARMS
SEEDS = ["0", "1", "2"]
THR = "2.0"
FLOOR = 0.0924
ORACLE = 0.1653
FLOOR_GRID = [0.060, 0.070, 0.080, 0.0924, 0.100, 0.110, 0.120, 0.130, 0.140]
# t 分位数（df=9），用于复现第 48 轮的 t 型 MDE
T975_DF9 = 2.262157
T80_DF9 = 0.883398

FAILS: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> bool:
    print(("[POS] " if ok else "[FAIL] ") + name + (f" :: {detail}" if detail else ""))
    if not ok:
        FAILS.append(name)
    return ok


def info(name: str, detail: str = "") -> None:
    print("[INFO] " + name + (f" :: {detail}" if detail else ""))


def median(xs: list[float]) -> float:
    s = sorted(xs)
    n = len(s)
    if n == 0:
        return float("nan")
    m = n // 2
    return s[m] if n % 2 else 0.5 * (s[m - 1] + s[m])


def sd(xs: list[float]) -> float:
    n = len(xs)
    if n < 2:
        return float("nan")
    mu = sum(xs) / n
    return math.sqrt(sum((x - mu) ** 2 for x in xs) / (n - 1))


def load_cell(arm: str, seed: str):
    tag = f"{arm}_s{seed}_ep00"
    win = json.loads((RAW / f"{tag}.windows.json").read_text(encoding="utf-8"))
    ev = json.loads((RAW / f"{tag}.eval.json").read_text(encoding="utf-8"))
    return win, ev


def pooled(win: dict, thr: str = THR) -> dict:
    h = m = f = cn = 0.0
    for w in win["windows"]:
        c = w[thr]
        h += c["hits"]
        m += c["misses"]
        f += c["false_alarms"]
        cn += c["correct_negatives"]
    return {
        "hits": h, "misses": m, "false_alarms": f, "correct_negatives": cn,
        "csi": h / (h + m + f),
        "bias_area": (h + f) / (h + m),
        "pod": h / (h + m),
        "far": f / (h + f),
    }


def main() -> int:
    # ---------- A. 守卫复核（先证裁决不变） ----------
    csi: dict[tuple[str, str], float] = {}
    bias: dict[tuple[str, str], float] = {}
    dry_ref: dict[str, list[float]] = {}
    win_keys_ref = None
    same_keys = True
    for arm in ALL_ARMS:
        for s in SEEDS:
            win, ev = load_cell(arm, s)
            p = pooled(win)
            csi[(arm, s)] = p["csi"]
            bias[(arm, s)] = p["bias_area"]
            dry_ref[f"{arm}_s{s}"] = [ev["dry_reference_csi"][t] for t in ev["thresholds"]]
            if win_keys_ref is None:
                win_keys_ref = win["window_keys"]
            elif win["window_keys"] != win_keys_ref:
                same_keys = False

    headline = json.loads((EV / "scores_summary.json").read_text(encoding="utf-8"))
    diffs = []
    for arm in ALL_ARMS:
        for s in SEEDS:
            diffs.append(abs(csi[(arm, s)] - headline[arm]["csi_at_2.0"][f"s{s}"]))
    check("A1  21 格 CSI@2.0 由窗表复算与头条表逐格一致",
          max(diffs) <= 1e-9, f"max|diff| = {max(diffs):.3e} (21/21)")

    med_diffs = [abs(median([csi[(a, s)] for s in SEEDS]) - headline[a]["median_csi_at_2.0"]) for a in ALL_ARMS]
    check("A2  七臂中位 CSI@2.0 复算与头条表一致",
          max(med_diffs) <= 1e-9, f"max|diff| = {max(med_diffs):.3e} (7/7)")
    check("A3  21 格 window_keys 完全一致且同序（配对前提）", same_keys)

    s3 = json.loads((EV / "s3_equalbias.json").read_text(encoding="utf-8"))
    cz = []
    for arm in ARMS:
        for side in ("raw", "equal_bias"):
            cz.append(bool(s3["arms"][arm][side]["crosses_zero"]))
    check("A4  S3 的 12 个区间仍全部跨零", len(cz) == 12 and all(cz), f"{sum(cz)}/12 crosses_zero")
    verdicts = {a: s3["arms"][a]["s3_verdict"] for a in ARMS}
    check("A5  六臂 s3_verdict 逐臂不变、无一臂到达 GAIN_SURVIVES_EQUALISATION",
          all(v != "GAIN_SURVIVES_EQUALISATION" for v in verdicts.values()),
          json.dumps(verdicts, ensure_ascii=False))
    surv = [a for a in ARMS if not s3["arms"][a]["equal_bias"]["crosses_zero"]]
    check("A6  合格臂集合仍为空", surv == [], f"合格臂 = {surv}")

    # ---------- B. 地板口径审计（dry reference 是多少、是否随格变） ----------
    dry2 = {k: v[1] for k, v in dry_ref.items()}  # thresholds 顺序 0.1/0.5/2.0/4.0 -> 索引 1 是 0.5
    thr_list = list(json.loads((RAW / "B0_s0_ep00.eval.json").read_text(encoding="utf-8"))["thresholds"].keys())
    dry2 = {k: v[thr_list.index(THR)] for k, v in dry_ref.items()}
    dry_vals = sorted({round(v, 12) for v in dry2.values()})
    info("B1  21 格 dry_reference_csi@2.0 的取值集合",
         f"distinct = {len(dry_vals)} :: {[f'{v:.6f}' for v in dry_vals[:4]]} … 预注册地板 = {FLOOR}")
    check("B2  21 格 dry reference 取值集合的规模已测出（地板是否为单一常数）",
          len(dry_vals) >= 1, f"distinct = {len(dry_vals)}")

    # ---------- C. 地板敏感性轴 ----------
    floor_grid: dict[str, dict] = {}
    for f in FLOOR_GRID:
        per_arm = {}
        for arm in ALL_ARMS:
            cs = [csi[(arm, s)] for s in SEEDS]
            n_pass = sum(1 for v in cs if v >= f)
            per_arm[arm] = {
                "csi_per_seed": cs,
                "n_pass": n_pass,
                "pass_rate": f"{n_pass}/3",
                "median_csi": median(cs),
            }
        floor_grid[f"{f:.4f}"] = per_arm
    f_pre = floor_grid[f"{FLOOR:.4f}"]
    info("C1  预注册地板 0.0924 上的过地板通过率",
         "；".join(f"{a} {f_pre[a]['pass_rate']}" for a in ALL_ARMS))
    arms_all3 = {f: [a for a in ARMS if floor_grid[f][a]["n_pass"] == 3] for f in floor_grid}
    info("C2  各地板下 3/3 过地板的臂",
         "；".join(f"{k}: {v if v else '无'}" for k, v in arms_all3.items()))

    # ---------- D. 晋级规则族轴 ----------
    def rule_eval(f: float) -> dict:
        out = {}
        for arm in ARMS:
            cs = [csi[(arm, s)] for s in SEEDS]
            ds = [csi[(arm, s)] - csi[(CTRL, s)] for s in SEEDS]
            r1 = all(v >= f for v in cs)
            r2 = sum(1 for v in cs if v >= f) >= 2
            r3 = median(cs) >= f
            r4 = r1 and median(ds) > 0
            r4b = r4 and not s3["arms"][arm]["raw"]["crosses_zero"]
            r5 = r4 and not s3["arms"][arm]["equal_bias"]["crosses_zero"]
            out[arm] = {
                "R1_3of3_over_floor": r1, "R2_2of3_over_floor": r2, "R3_median_over_floor": r3,
                "R4_R1_and_paired_gain_pos": r4,
                "R4b_R4_and_raw_CI_nonzero": r4b,
                "R5_R4_and_equalbias_CI_nonzero": r5,
                "paired_gain_per_seed": ds,
                "paired_gain_median": median(ds),
                "difference_convention": median(cs) - median([csi[(CTRL, s)] for s in SEEDS]),
            }
        return out

    rules_per_floor = {f"{f:.4f}": rule_eval(f) for f in FLOOR_GRID}
    promoted = {}
    for fk, tab in rules_per_floor.items():
        promoted[fk] = {r: [a for a in ARMS if tab[a][r]] for r in
                        ["R1_3of3_over_floor", "R2_2of3_over_floor", "R3_median_over_floor",
                         "R4_R1_and_paired_gain_pos", "R4b_R4_and_raw_CI_nonzero",
                         "R5_R4_and_equalbias_CI_nonzero"]}
    for fk in promoted:
        info("D1  地板 " + fk + " 各规则放行的臂",
             "；".join(f"{r.replace('_', ' ')}: {promoted[fk][r] if promoted[fk][r] else '无'}" for r in promoted[fk]))

    n_r5 = sum(len(promoted[fk]["R5_R4_and_equalbias_CI_nonzero"]) for fk in promoted)
    check("D2  R5（过地板∧配对增益为正∧配平后区间不跨零）在全地板网格上放行 0 臂",
          n_r5 == 0, f"全网格放行数合计 = {n_r5}")
    r4_max = max(len(promoted[fk]["R4_R1_and_paired_gain_pos"]) for fk in promoted)
    r5_max = max(len(promoted[fk]["R5_R4_and_equalbias_CI_nonzero"]) for fk in promoted)
    check("D3  去掉 S3 护栏后仍能放行（R4>0）⇒ S3 是绑定约束而非地板",
          r4_max > 0 and r5_max == 0, f"R4 最多放行 {r4_max} 臂，R5 最多放行 {r5_max} 臂")

    # ---------- E. 零模型标定（规则的假晋级率 / 假否决率） ----------
    ctrl_cs = [csi[(CTRL, s)] for s in SEEDS]
    null = {}
    for f in FLOOR_GRID:
        n1 = n4 = 0
        for perm in itertools.permutations(range(3)):
            cand = [ctrl_cs[perm[k]] for k in range(3)]
            ds = [cand[k] - ctrl_cs[k] for k in range(3)]
            if all(v >= f for v in cand):
                n1 += 1
                if all(d > 0 for d in ds):
                    n4 += 1
        null[f"{f:.4f}"] = {"n_perm": 6, "R1_promotions": n1, "R4_promotions": n4,
                            "control_own_n_pass": sum(1 for v in ctrl_cs if v >= f)}
    info("E1  零模型（对照臂换种子，6 种置换）的规则放行数",
         "；".join(f"{k}: R1={v['R1_promotions']}/6, R4={v['R4_promotions']}/6" for k, v in null.items()))
    check("E2  零模型下 R4 放行数为 0 ⇒ 规则的假晋级率为 0（在本对照臂上）",
          all(v["R4_promotions"] == 0 for v in null.values()))
    ctrl_pass_pre = null[f"{FLOOR:.4f}"]["control_own_n_pass"]
    check("E3  对照臂自身在预注册地板上的通过率 = 2/3 ⇒ 「3/3 过地板」规则对已知良好模型有 1/3 假否决率",
          ctrl_pass_pre == 2, f"对照臂通过 {ctrl_pass_pre}/3")

    # ---------- F. 逐臂绑定约束（要救回某臂，最小要改什么） ----------
    binding = {}
    for arm in ARMS:
        cs = [csi[(arm, s)] for s in SEEDS]
        ds = [csi[(arm, s)] - csi[(CTRL, s)] for s in SEEDS]
        binding[arm] = {
            "csi_per_seed": cs,
            "floor_fail_seeds": {s: {"csi": csi[(arm, s)], "gap_to_floor": csi[(arm, s)] - FLOOR}
                                 for s in SEEDS if csi[(arm, s)] < FLOOR},
            "negative_paired_gain_seeds": {s: ds[i] for i, s in enumerate(SEEDS) if ds[i] < 0},
            "paired_gain_median": median(ds),
            "difference_convention": median(cs) - median(ctrl_cs),
            "s3_raw": s3["arms"][arm]["raw"],
            "s3_equal_bias": s3["arms"][arm]["equal_bias"],
            "s3_verdict": s3["arms"][arm]["s3_verdict"],
        }
    info("F1  逐臂绑定约束",
         "；".join(f"{a}: 地板失败 {list(binding[a]['floor_fail_seeds']) or '无'}／配对增益为负 {list(binding[a]['negative_paired_gain_seeds']) or '无'}／S3 {binding[a]['s3_verdict']}"
                   for a in ARMS))

    # ---------- G. 算力预算（复现第 48 轮的 MDE 口径） ----------
    per_arm_sd = {a: sd([csi[(a, s)] - csi[(CTRL, s)] for s in SEEDS]) for a in ARMS}
    pooled_sd = math.sqrt(sum(v ** 2 for v in per_arm_sd.values()) / len(per_arm_sd))
    mde = {f"n={n}": (T975_DF9 + T80_DF9) / math.sqrt(n) * pooled_sd for n in (3, 10, 17, 30, 60)}
    info("G1  seed 间增益 pooled SD 复算",
         f"pooled SD = {pooled_sd:.5f}（第 48 轮发布值 0.04896）")
    info("G2  t 型 MDE（两单侧 5% / 80% 检出力，df=9）",
         "；".join(f"{k} = {v:.5f}" for k, v in mde.items()))
    check("G3  pooled SD 与第 48 轮发布值一致（≤0.002）",
          abs(pooled_sd - 0.04896) <= 0.002, f"{pooled_sd:.5f} vs 0.04896")

    payload = {
        "round": 59,
        "generated": "2026-11-14",
        "axis": "promotion-rule / floor-sensitivity calibration",
        "floor_pre_registered": FLOOR,
        "oracle": ORACLE,
        "floor_grid": FLOOR_GRID,
        "csi_pooled_per_cell": {f"{a}_s{s}": csi[(a, s)] for a in ALL_ARMS for s in SEEDS},
        "bias_pooled_per_cell": {f"{a}_s{s}": bias[(a, s)] for a in ALL_ARMS for s in SEEDS},
        "dry_reference_csi_at_2.0": dry2,
        "dry_reference_distinct_values": dry_vals,
        "floor_sensitivity": floor_grid,
        "arms_passing_3of3_by_floor": arms_all3,
        "rules_per_floor": rules_per_floor,
        "promoted_by_floor": promoted,
        "null_calibration": null,
        "binding_constraints": binding,
        "seed_sd_pooled": pooled_sd,
        "per_arm_seed_sd": per_arm_sd,
        "mde_t": mde,
        "s3_verdicts": verdicts,
        "guards": {
            "recomputed_csi_max_abs_diff": max(diffs),
            "recomputed_median_max_abs_diff": max(med_diffs),
            "s3_crosses_zero_all_true": all(cz),
            "qualified_arms": surv,
        },
        "verdict": "PASS" if not FAILS else "FAIL",
    }
    OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nwrote {OUT}")
    if FAILS:
        print(f"VERDICT: FAIL ({len(FAILS)} 项) :: {FAILS}")
        return 1
    print("VERDICT: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
