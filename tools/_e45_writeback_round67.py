#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""E45 第 67 轮写回：HANDOVER_E45.md（㉚）＋ 上层 MEMORY.md 状态段 ＋ 上层/项目级 2026-11-14 日志 ＋ 上层 2026-11-11.md ＋ 本自动化记忆文件。

纪律（沿用第 61/63/64/65/66 轮教训）：
  * 写回在校验之后；断言不许假设数据会迎合结论。
  * 幂等：整块标记自剥后追加，连跑两次 sha256 必须逐位相同。
  * 表格行必须用**单元格列表**构造，分隔符由代码插入（手拼竖线会让列数错）。
  * 不动 docs/（本轮无数值与裁决变化 => MASTER 不升版，仍 v2.48）。

本轮特殊：epoch 轴从 n=1（只有 seed 0）升到 **n=2（seed 0+1 整列，42 格 + 6 探针）**，
所以写回必须把「n=2 的读数」与「n=1 的旧结论」分开写：
  * 不把 n=2 的读数写成三 seed 结论；
  * 不把「第 66 轮 4/7 臂」写成已定论（n=2 下被推翻）；
  * 也不把「作业仍在跑」写成「失败」。
"""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
UP = Path("R:/1radarNWPprediction/.workbuddy/memory")
PROJ = ROOT / ".workbuddy" / "memory"
AUTO = PROJ / "automations" / "e45-slurm-status" / "memory.md"
AUTO = PROJ / "automations" / "e3cc6636-3e60-4a58-8f7e-67d4c896b0b6" / "memory.md"
J = json.loads((ROOT / "tools" / "_evals" / "e45" / "round67_epoch_axis_n2.json").read_text(encoding="utf-8"))

ORDER = ["B0", "A1", "A2", "A3a", "A3b", "A4", "A4b"]
NAMES = {"B0": "B0 对照", "A1": "A1 幅度单调门", "A2": "A2 硬支撑门",
         "A3a": "A3a 拓扑截断", "A3b": "A3b 拓扑平滑+面积归一",
         "A4": "A4 量通道新头", "A4b": "A4 量通道自带头"}
EPOCHS = ["ep00", "ep01", "ep02"]
FLOOR = 0.0924

ARM_E = J["arm_epoch"]
PEAK = J["peak_epoch"]
CRIT = J["criteria"]
FLR = J["zero_model_floor_per_epoch"]
EPK = J["ep00_peak_count"]


def row(cells):
    """表格行必须由单元格列表构造，分隔符由代码插入。"""
    assert isinstance(cells, list) and len(cells) >= 2, "row() needs a cell list"
    return "| " + " | ".join(str(c) for c in cells) + " |"


def head(cells):
    return row(cells) + "\n" + row(["---"] * len(cells)) + "\n"


def read(p):
    raw = p.read_bytes()
    nl = "\r\n" if b"\r\n" in raw else "\n"
    return raw.decode("utf-8"), nl


def write(p, text, nl):
    p.write_bytes(text.replace("\n", nl).encode("utf-8"))


def strip(text, begin, end):
    """自剥旧块；找不到就原样返回（幂等追加的基础）。"""
    out, i, n_removed = [], 0, 0
    while True:
        a = text.find(begin, i)
        if a < 0:
            out.append(text[i:])
            break
        b = text.find(end, a)
        if b < 0:
            out.append(text[i:])
            break
        out.append(text[i:a])
        i = b + len(end)
        n_removed += 1
    return "".join(out), n_removed


# ---------------------------------------------------------------- assertions
assert J["n_checks_pass"] == J["n_checks_total"], "analysis checks not all positive"
assert J["n_cells"] == 42, "expected 42 two-seed cells"
assert J["control_md5"]["n_identical"] == J["control_md5"]["n_compared"] == 16
assert J["probe_control_max_abs_diff"] == 0.0
assert J["s3_guard"]["n_intervals"] == 12 and J["s3_guard"]["n_cross_zero"] == 12
assert J["published_md5"]["scores_summary"].startswith("57bcd511a537ab85")
assert J["published_md5"]["s3_equalbias"].startswith("3aba999d5ef6dcc4a")
assert J["published_headline"]["median"] and J["published_headline"]["pass_rate"]
assert EPK["n1"] == 4 and EPK["n2"] == 7, "epoch-peak counts changed unexpectedly"
assert J["verdict"]["eligible_arm_set"] == [] and J["verdict"]["e45_verdict_changed"] is False

# ------------------------------------------------------------------- tables
t1 = head(["臂", "epoch", "中位 CSI@2.0 (n=2)", "配对中位 ΔCSI", "差值口径 ΔCSI",
           "正号率", "种子跨度", "中位 BIAS@2.0", "过地板"])
for arm in ORDER:
    for ep in EPOCHS:
        r = ARM_E[f"{arm}_{ep}"]
        vals = r["per_seed_csi"]
        n_over = sum(1 for v in vals if v >= FLOOR)
        t1 += row([NAMES[arm], ep, f"{r['median_csi']:.6f}",
                   f"{r['paired_median_delta']:+.6f}", f"{r['diff_convention_delta']:+.6f}",
                   f"{r['pos_sign_rate']:.2f}", f"{r['seed_spread']:.6f}",
                   f"{r['median_bias']:.4f}", f"{n_over}/{len(vals)}"]) + "\n"

t2 = head(["臂", "n=1(seed0) 最高 epoch", "n=2 最高 epoch", "argmax 是否改变",
           "n=2 中位 CSI: ep00 / ep01 / ep02"])
for arm in ORDER:
    p = PEAK[arm]
    m = p["n2_median"]
    t2 += row([NAMES[arm], p["best_n1"], p["best_n2"],
               "是" if p["argmax_changed_n1_to_n2"] else "否",
               f"{m['00']:.6f} / {m['01']:.6f} / {m['02']:.6f}"]) + "\n"

CRIT_ORDER = [("A1", "corr_mask_amp"), ("A1", "mask_at_peak"), ("A1", "maxratio_as_trained"),
              ("A2", "corr_mask_amp"), ("A2", "mask_at_peak"), ("A2", "maxratio_as_trained"),
              ("A3a", "grad_top1pct_share"), ("A3a", "field_max_mm"),
              ("A3b", "grad_top1pct_share"), ("A3b", "field_max_mm"),
              ("A4", "scale_cv"), ("A4", "scale_truth_corr"),
              ("A4b", "scale_cv"), ("A4b", "scale_truth_corr")]
t3 = head(["臂", "判据", "过", "不过", "读数缺失", "通过率(可判格)",
           "ep00 s0/s1", "ep01 s0/s1", "ep02 s0/s1"])
for arm, cname in CRIT_ORDER:
    r = CRIT[f"{arm}:{cname}"]
    cells = []
    for ep in EPOCHS:
        cells.append(f"{r['values'][f'{arm}_s0_{ep}']} / {r['values'][f'{arm}_s1_{ep}']}")
    rate = r["pass_rate_over_available"]
    t3 += row([NAMES[arm], r["rule"], r["n_pass"], r["n_fail"], r["n_missing"],
               ("n/a" if rate is None else f"{rate:.3f}")] + cells) + "\n"

# -------------------------------------------------------------- HANDOVER ㉚
HANDOVER_HEAD = "**㉚（2026-11-14 22:2x 第 67 轮新增证据"
ho = []
ho.append(f"**㉚（2026-11-14 22:2x 第 67 轮新增证据（epoch 轴：n=2 受控判读——第 66 轮「ep00 ＝ 峰值端只对 4/7 臂成立」是 n=1 的种子噪声，本轮在 n=2 下被推翻））**——脚本 `tools/_e45_verify_round67.py`（只读本地，产物 `tools/_evals/e45/round67_epoch_axis_n2.json`，**VERDICT: PASS {J['n_checks_pass']}/{J['n_checks_total']}**）；远端作业 **315/316/317** 仍在跑（第 65 轮提交；本轮末各 **{16}/21** 格、合计 **48/63**，正在补 seed 2 列）。**E45 本身无新活动**（提交器 273、训练数组 275/283、评测 274/282/290、汇总 291、只读诊断 297/298 全部 `COMPLETED`；`runs/e45/evals/` 21 份评测齐；`runs/e45/` 13 份 json 未变，`scores_summary.json` md5 `57bcd511a537ab85`、`s3_equalbias.json` md5 `3aba999d5ef6dcc4a`、`scores_all.json` md5 `80655963622b09cd5231588e26468a28`）⇒ **裁决零变化：六臂仍全部判负、合格臂集合仍为空、不投十 seed 算力、MASTER 不升版（仍 v2.48）**。\n")
ho.append("1. **本轮把 epoch 轴从 n=1 升到 n=2**：回取 **42 格评测**（7 臂 × 2 seed × 3 epoch）＋ **6 份机制探针** ＋ 已发布 E45 产物（一个 tar、一次下载）。第 66 轮只有 seed 0（机制臂 n=1），本轮 seed 1 整列已齐 ⇒ **机制臂 n=2**。**n=2 仍然只给方向与幅度，不作臂级结论**（n=3 才是任何检验构造上可能的第一档，见第 64 轮分辨率下界 1/8）。\n")
ho.append(f"2. **同协议基准全部通过（零漂移）**：ep00 重跑与 E45 已发布文件 **md5 逐位相同 {J['control_md5']['n_identical']}/{J['control_md5']['n_compared']} 格**；机制探针 ep00 与已发布 `diag_probe_s0.json` 的可比读数 **max|diff| = {J['probe_control_max_abs_diff']:.3e}**；已发布头条表七臂中位 CSI 与过地板通过率 **复算 max|diff| = 0.000e+00**（过地板通过率字符串逐臂一致）⇒ **ep01/ep02 与 ep00 的任何差异只可能来自 epoch 本身**。\n")
ho.append(f"3. **最重要的一条，它是对第 66 轮的订正**：「ep00 ＝ 峰值端」在 **n=1（只有 seed 0）下只对 {EPK['n1']}/7 臂成立**（第 66 轮结论，本轮逐位复现）；但在 **n=2（seed 0+1 取中位）下 {EPK['n2']}/7 臂都以 ep00 为最高 epoch**——**B0、A1、A4b 三臂的 argmax 从 ep01/ep02 翻回 ep00**（{EPK['argmax_changed']} 臂 argmax 改变）。⇒ **第 66 轮那句「ep00 ＝ 峰值端是只对少数臂成立的便利约定、连对照臂都违反它」是单种子噪声造成的，n=2 下不成立**；**同时对照臂自身的技能随 epoch 单调塌陷**（B0 中位 CSI ep00 {ARM_E['B0_ep00']['median_csi']:.6f} → ep01 {ARM_E['B0_ep01']['median_csi']:.6f} → ep02 {ARM_E['B0_ep02']['median_csi']:.6f}）⇒ **「ep00 是峰值端、last.pt 是崩塌端」这条规则在 n=2 下重新得到支持**。**裁决不变**：E45 用 ep00 评的这张表**没有被推翻**。\n")
ho.append(f"4. **「换个 epoch 能更好」依然不成立**：零模型地板（＝同一 epoch 下把对照臂换个 seed）**ep00 {FLR['00']:.6f}／ep01 {FLR['01']:.6f}／ep02 {FLR['02']:.6f}**（与第 66 轮发布值逐位相同 ⇒ 复现）；**七臂的 ep01/ep02 中位 CSI 全部低于自己的 ep00**（差值口径 7/7 同向为负），且跨 epoch 的 |ΔCSI| 全部落在地板之内 ⇒ **ep00 是免费的正确选择，但不是「唯一可选」——任何换 epoch 的收益都小于种子噪声**。\n")
ho.append("5. **机制读数在 epoch 轴 × 种子轴上的三条新事实**（每臂每判据 6 格 = 2 seed × 3 epoch）：\n")
ho.append(f"   - **门族（A1/A2）方向稳、但 A2 的读数缺失随 epoch 变**：A1 `corr_mask_amp` **6/6 过**（{CRIT['A1:corr_mask_amp']['values']['A1_s0_ep00']}–{CRIT['A1:corr_mask_amp']['values']['A1_s1_ep02']}）、`mask_at_peak` **6/6 过**（{CRIT['A1:mask_at_peak']['values']['A1_s1_ep00']}–{CRIT['A1:mask_at_peak']['values']['A1_s1_ep02']}）；A2 `mask_at_peak` **6 格全是 1.0**（硬支撑门在每个 epoch 都完整保住峰值）但 `corr_mask_amp` **只有 3/3 可判格全过、另 3 格缺失**（ep01 两格全 NaN、ep02 s1 NaN）⇒ **这是读数缺失（探针的非 nan 感知聚合缺陷，第 41 轮同族），不是判据翻转**。`maxratio_as_trained > 0.9` 仍是 **6/6 缺失** ⇒ 该预注册判据**从探针自身仍不可判**（第 41 轮的旁路诊断 A1 0.72–0.76 不过／A2 1.0 过仍然有效）。\n")
ho.append(f"   - **拓扑族：第 66 轮的「A3a 在 ep02 勉强过」是单种子现象**：A3a `grad_top1pct_share` < 0.5 **只有 1/6 格过**（唯一一格 A3a_s0_ep02 = {CRIT['A3a:grad_top1pct_share']['values']['A3a_s0_ep02']}），**同臂 seed 1 的三个 epoch 是 {CRIT['A3a:grad_top1pct_share']['values']['A3a_s1_ep00']}／{CRIT['A3a:grad_top1pct_share']['values']['A3a_s1_ep01']}／{CRIT['A3a:grad_top1pct_share']['values']['A3a_s1_ep02']}（离判据极远）** ⇒ **n=2 下 A3a 的梯度集中判据通过率只有 1/6，不是「ep02 通过」**；A3b `grad_top1pct_share` **0/6**（{CRIT['A3b:grad_top1pct_share']['values']['A3b_s0_ep00']}–{CRIT['A3b:grad_top1pct_share']['values']['A3b_s1_ep02']}）；**两臂 `field_max_mm` ≤ 30 都是 0/6（六格恒为 402.428802 mm ＝ `expm1(6)` 的 clamp 上限）** ⇒ **拓扑支结案结论不变，且理由现在按「epoch × 种子」两轴都写清**：不是「两臂都不过」，而是「**幅度截断根本没作用在输出场上**，这条判据在任何 epoch、任何种子下都不可能过」。\n")
ho.append(f"   - **量族：量通道在任何 epoch、任何种子下都没退化成常数，但相关性分支又触发了一格**：A4 `scale_cv` **6/6 过**（{CRIT['A4:scale_cv']['values']['A4_s1_ep00']}–{CRIT['A4:scale_cv']['values']['A4_s0_ep01']}）、`scale_truth_corr` **6/6 过**（{CRIT['A4:scale_truth_corr']['values']['A4_s1_ep02']}–{CRIT['A4:scale_truth_corr']['values']['A4_s0_ep00']}）；A4b `scale_cv` **6/6 过**（{CRIT['A4b:scale_cv']['values']['A4b_s0_ep00']}–{CRIT['A4b:scale_cv']['values']['A4b_s1_ep01']}），`scale_truth_corr` **5/6**——**唯一不过的一格是 A4b_s1_ep02 = {CRIT['A4b:scale_truth_corr']['values']['A4b_s1_ep02']}** ⇒ **相关性分支在第 43 轮那格（种子 2 @ ep00）之外的另一个 (seed, epoch) 组合上又触发了一次** ⇒ 再次确认 **退化的形式是「量级塌缩＋与真值脱钩」而不是「退化成常数」**（cv 分支仍未触发），且 **该现象随 checkpoint 移动** ⇒ 与 `1e-12` 护栏缺陷的假设同向（**仍不构成因果主张**）。\n")
ho.append(f"   - **`mean_out_equals_T` 缺陷在 epoch 轴上同样存在**：A4 只有 seed 0 为 True（3/3 epoch），seed 1 三个 epoch 全为 False；A4b **6/6 全 False** ⇒ **修 `1e-12` 护栏后重测必须逐 (seed, epoch) 报告，不能只报一个 epoch**。\n")
ho.append("6. **口径与纪律（本轮新增三条）**：① **n=1 上发布过的规则性结论，发布前必须先看 n=2**——本轮直接把第 66 轮的「4/7 臂」翻成「7/7 臂」；② **判据贴边时（同族另一 seed 离判据很远）不能写成「该 epoch 通过」**，只能写成通过率（A3a 是 1/6 而不是「ep02 过」）；③ **零模型地板要连 n 一起给**（本轮地板值与第 66 轮逐位相同，是因为地板本身只用一对种子）。\n")
ho.append("7. **下一步**：作业 315/316/317 预计本轮结束后不久补齐 seed 2 列（63/63），`tools/_e45_epochaxis_fetch.sh` 会自动打包下载；**下一轮可在 n=3（完整 21 格 × 3 epoch）上判读**——那才是构造上可能出现任何证据的第一档（n=3 的精确 p 下界 1/8，第 64 轮）；**仍不改变「六臂判负、不投十 seed 算力」的结论**，除非 S3 配平轴在 n=3 上给出 3/3 同号正增益。\n")
ho.append("\n**epoch 轴逐臂 × 逐 epoch 读数（n=2，seed 0+1 中位；地板 0.0924；零模型地板＝同 epoch 下 B0 换 seed）**\n\n")
ho.append(t1 + "\n")
ho.append("**每个臂的峰值 epoch：n=1（seed 0）vs n=2（seed 0+1 中位）**\n\n")
ho.append(t2 + "\n")
ho.append("**预注册机制判据在 epoch 轴 × 种子轴上的通过情况（每判据 6 格 ＝ 2 seed × 3 epoch）**\n\n")
ho.append(t3 + "\n")
ho.append("> **边界声明**：本轮是**事后诊断**，**不构成新判据、不改变任何裁决**；六臂仍全部判负、合格臂集合仍为空、**不投十 seed 算力**；`docs/MASTER.md` **不升版（仍 v2.48）**，`docs/` 未改。\n")
HANDOVER_BLOCK = "\n".join(ho)

# ------------------------------------------------------- shared summary text
sum_line = ("**第 67 轮（2026-11-14 22:2x，epoch 轴 n=2 受控判读）**：远端 E45 无新活动（273/274/275/282/283/290/291/297/298 全 `COMPLETED`；21 份评测齐；13 份 json md5 未变）；"
            "**唯一新活动＝epoch 轴作业 315/316/317 仍在跑**（各 16/21 格、合计 48/63）。本轮回取 **42 格（7 臂 × 2 seed × 3 epoch）+ 6 份探针**，"
            "脚本 `tools/_e45_verify_round67.py`（只读本地，**VERDICT: PASS 9/9**，产物 `tools/_evals/e45/round67_epoch_axis_n2.json`）。"
            f"**同协议基准零漂移**（ep00 重跑 md5 {J['control_md5']['n_identical']}/{J['control_md5']['n_compared']} 格逐位相同；探针 max|diff| = {J['probe_control_max_abs_diff']:.3e}）。"
            f"**关键订正**：第 66 轮「ep00＝峰值端只对 4/7 臂成立」是 **n=1 的种子噪声**——n=2 下 **{EPK['n2']}/7 臂都以 ep00 为最高 epoch**（B0/A1/A4b 三臂 argmax 翻回 ep00），"
            f"且对照臂技能随 epoch 单调塌陷（{ARM_E['B0_ep00']['median_csi']:.6f}→{ARM_E['B0_ep01']['median_csi']:.6f}→{ARM_E['B0_ep02']['median_csi']:.6f}）⇒ 「ep00 是峰值端」这条规则重新得到支持、**E45 用 ep00 评的表没有被推翻**。"
            f"**换 epoch 仍无收益**（零模型地板 ep00 {FLR['00']:.6f}／ep01 {FLR['01']:.6f}／ep02 {FLR['02']:.6f}，与第 66 轮逐位相同；七臂 ep01/ep02 中位全部更低）。"
            "**机制三条新事实**：① A3a 梯度集中判据在 n=2 下只有 **1/6** 格过（唯一一格 A3a_s0_ep02 ＝ 0.499906，seed 1 三 epoch 为 0.5666／0.8915／0.8813）⇒ 第 66 轮「A3a 在 ep02 勉强过」是单种子现象；"
            "A3a/A3b `field_max_mm` ≤ 30 **0/6**（恒为 402.428802 mm ＝ clamp 上限）⇒ 拓扑支结论不变、理由按「epoch × 种子」两轴写清；"
            "② A2 `mask_at_peak` 六格全 1.0 但 `corr_mask_amp` **3 格缺失**（读数缺失 ≠ 判据翻转）；`maxratio_as_trained` 仍 **6/6 缺失**；"
            "③ A4b `scale_truth_corr` **5/6**，唯一不过者 **A4b_s1_ep02 ＝ 0.2582**（相关性分支在另一个 (seed, epoch) 组合上再次触发 ⇒ 退化的形式是「塌缩＋脱钩」不是「退化成常数」，且随 checkpoint 移动 ⇒ 与 `1e-12` 护栏缺陷同向、**不构成因果主张**）；"
            "`mean_out_equals_T` 缺陷在 epoch 轴上同样存在（A4 仅 seed 0 为 True、A4b 6/6 全 False）。"
            "**裁决零变化**：S3 **12/12 区间仍全部跨零**（4 × `GAIN_WAS_AREA_OR_AMPLITUDE` ＋ 2 × `NO_GAIN_TO_BEGIN_WITH`）、合格臂集合仍为空、"
            "**六臂判负、不投十 seed 算力、MASTER 仍 v2.48 不升版**、`docs/` 未改。**纪律三条**：n=1 上要发布的规则性结论发布前先看 n=2；判据贴边时只能报通过率不能写「通过」；零模型地板要连 n 一起给。\n")

MEM_BLOCK = ("<!-- E45-R67-BEGIN -->\n- " + sum_line.strip() +
             "\n<!-- E45-R67-END -->\n")

LOG_BLOCK = ("<!-- E45-R67-BEGIN -->\n### 2026-11-14 22:2x｜第 67 轮（epoch 轴：n=2 受控判读 —— 第 66 轮的「4/7 臂」是单种子噪声）\n\n"
             + sum_line.strip() + "\n"
             "**逐格读数与表格见** `HANDOVER_E45.md` §4㉚；**产物** `tools/_evals/e45/round67_epoch_axis_n2.json`；"
             "**脚本** `tools/_e45_verify_round67.py`（只读本地，VERDICT: PASS 9/9）；**回取物** `tools/_evals/e45/epoch_axis/`（42 格评测 + 6 份探针 + 已发布产物，tar 由远端一次性打包）。\n"
             "<!-- E45-R67-END -->\n")

AUTO_BLOCK = ("<!-- E45-R67-BEGIN -->\n### 2026-11-14 22:2x｜第 67 轮（epoch 轴 n=2 受控判读）：状态复核 + 首个 n=2 判读，裁决不变\n\n"
              "- **判定**：远端 **E45 本身全部完成且与第 50–66 轮完全一致**（273/274/275/282/283/290/291/297/298 全 `COMPLETED`；`runs/e45/evals/` 21 份齐；"
              "`runs/e45/` 13 份 json 未变：`scores_summary.json` md5 `57bcd511a537ab85`、`s3_equalbias.json` md5 `3aba999d5ef6dcc4a`、`scores_all.json` md5 `80655963622b09cd5231588e26468a28`）"
              "⇒ **E45 无新活动**。**唯一新活动＝第 65 轮提交的 epoch 轴诊断作业 315/316/317，仍在跑**（本轮末各 **16/21 格**、合计 **48/63**，正在补 seed 2 列；提交 2026-11-14 19:51:54、时限 6 h）。"
              "**裁决不变**（六臂判负、无一臂进十 seed、不投十 seed 算力）；**未改 `docs/`**（MASTER 仍 v2.48）。\n"
              "- **本轮取回了 epoch 轴的两 seed 整列并做了首个 n=2 判读**：回取 **42 格评测**（7 臂 × 2 seed × 3 epoch）＋ **6 份机制探针** ＋ 已发布 E45 产物（远端一次打包 tar、一次 `--download`，`--out` 显式指定）。"
              "产物 `tools/_evals/e45/round67_epoch_axis_n2.json`（脚本 `tools/_e45_verify_round67.py`，**POS 9/9 VERDICT: PASS**）。\n"
              "- **同协议基准零漂移**：ep00 重跑与已发布文件 md5 **16/16 逐位相同**；ep00 探针与已发布 `diag_probe_s0.json` 可比读数 **max|diff| = 0.000e+00**；已发布头条表七臂中位 CSI 与过地板通过率复算 **max|diff| = 0.000e+00**（通过率字符串逐臂一致）。\n"
              f"- **关键订正（本轮最重要）**：第 66 轮「「ep00 ＝ 峰值端」只对 4/7 臂成立、连对照臂都违反」是 **n=1（只有 seed 0）的种子噪声**——n=2 下 **{EPK['n2']}/7 臂都以 ep00 为最高 epoch**（B0/A1/A4b 三臂 argmax 翻回 ep00），"
              f"且对照臂技能随 epoch 单调塌陷（B0 中位 {ARM_E['B0_ep00']['median_csi']:.6f}→{ARM_E['B0_ep01']['median_csi']:.6f}→{ARM_E['B0_ep02']['median_csi']:.6f}）⇒ **该规则在 n=2 下重新得到支持，E45 用 ep00 评的表没有被推翻**；"
              f"零模型地板（ep00 {FLR['00']:.6f}／ep01 {FLR['01']:.6f}／ep02 {FLR['02']:.6f}）与第 66 轮发布值逐位相同 ⇒ **换 epoch 仍无收益**（七臂 ep01/ep02 中位全部更低，跨 epoch |ΔCSI| 全在地板内）。\n"
              "- **机制三条新事实（每判据 6 格 = 2 seed × 3 epoch）**：① **A3a 梯度集中判据只有 1/6 格过**（唯一一格 A3a_s0_ep02 ＝ 0.499906；seed 1 为 0.5666／0.8915／0.8813）⇒ 第 66 轮「A3a 在 ep02 勉强过」是单种子现象；"
              "**A3a/A3b `field_max_mm` ≤ 30 为 0/6**（恒为 402.428802 mm ＝ `expm1(6)` clamp 上限）⇒ 拓扑支结论不变、理由须按「epoch × 种子」两轴写清；"
              "② A2 `mask_at_peak` 六格全 1.0 但 `corr_mask_amp` **3 格缺失**（读数缺失 ≠ 判据翻转）；`maxratio_as_trained > 0.9` 仍 **6/6 缺失**；"
              "③ A4b `scale_truth_corr` **5/6**（唯一不过者 A4b_s1_ep02 ＝ 0.2582）⇒ 相关性分支在第 43 轮那格之外的另一个 (seed, epoch) 组合上再次触发 ⇒ 退化形式是「塌缩＋脱钩」不是「退化成常数」，且随 checkpoint 移动 ⇒ 与 `1e-12` 护栏缺陷同向、**不构成因果主张**；"
              "`mean_out_equals_T` 缺陷在 epoch 轴上同样存在（A4 仅 seed 0 为 True、A4b 6/6 全 False）。\n"
              "- **守卫**：S3 **12/12 区间仍全部跨零**（4 × `GAIN_WAS_AREA_OR_AMPLITUDE` ＋ 2 × `NO_GAIN_TO_BEGIN_WITH`）、六臂 `s3_verdict` 逐臂不变、**合格臂集合仍为空**；已发布头条表复算 **max|diff| = 0.000e+00** ⇒ **MASTER 仍 v2.48、不升版**。\n"
              "- **写回**：`HANDOVER_E45.md` ㉚ ＋ 上层 `MEMORY.md` 状态段 ＋ 上层/项目级 2026-11-14 日志 ＋ 上层 2026-11-11.md ＋ 本文件（标记块自剥后幂等追加；表格行由单元格列表构造、分隔符由代码插入）。\n"
              "- **本轮踩坑（可复用规则）**：① **远端 `tar` 不支持 `--wildcards`** ⇒ 改用 shell 展开后再打包（`tar -czf out.tar.gz evals/*_s0_ep*.json …`），并且**多个 `-C` 可以串在一个 tar 里**（用于把已发布产物与 md5 清单一起打进同一个包、只做一次 `--download`）；"
              "② **一次调用里塞超长 heredoc 会被截断**（本轮首版把整段脚本塞进 bash heredoc，直接 EOF 报错）⇒ 长脚本改用文件写入再执行；"
              "③ **已发布 `scores_summary.json` 的结构是 `{arm: {csi_at_2.0: {s0:…, s1:…, s2:…}, median_csi_at_2.0, pass_rate_over_floor}}`**（不是 `{cell: {csi_at_2.0}}`）⇒ 取数要按「臂 → 逐 seed」两层走；"
              "④ **S3 产物的区间布尔字段名不是 `crosses_zero`**（本轮改为「键名含 cross 的布尔量」通用收集）⇒ 断言要按事实枚举而不是硬编码键名。\n"
              "- **口径与纪律**：① 仍按「**中位数 + 通过率**」报，单点分数不作裁决；② 配对口径与差值口径并列；③ 事后诊断、**不构成新判据**、**不改变任何裁决**；"
              "④ **新登记三条纪律**：n=1 上要发布的规则性结论发布前先看 n=2；判据贴边时只能报通过率不能写「通过」；零模型地板要连 n 一起给。\n"
              "<!-- E45-R67-END -->\n")

# ------------------------------------------------------------------- writing
targets = [
    (UP / "HANDOVER_E45.md", None, HANDOVER_BLOCK),
    (UP / "MEMORY.md", ("<!-- E45-R67-BEGIN -->", "<!-- E45-R67-END -->"), MEM_BLOCK),
    (UP / "2026-11-14.md", ("<!-- E45-R67-BEGIN -->", "<!-- E45-R67-END -->"), LOG_BLOCK),
    (PROJ / "2026-11-14.md", ("<!-- E45-R67-BEGIN -->", "<!-- E45-R67-END -->"), LOG_BLOCK),
    (UP / "2026-11-11.md", ("<!-- E45-R67-BEGIN -->", "<!-- E45-R67-END -->"), LOG_BLOCK),
    (AUTO, ("<!-- E45-R67-BEGIN -->", "<!-- E45-R67-END -->"), AUTO_BLOCK),
]

written = []
for path, markers, block in targets:
    if not path.exists():
        print(f"SKIP (missing): {path}")
        continue
    text, nl = read(path)
    if markers is None:
        # HANDOVER: 整块文本作幂等标记，追加到文件末尾。
        # 若上一版已写入（本轮首版表格分隔行缺换行），先按块首标记裁掉再重写。
        cut = text.find(HANDOVER_HEAD)
        if cut >= 0:
            text = text[:cut].rstrip("\n") + "\n"
        if block.strip() in text:
            print(f"IDEMPOTENT (already present): {path.name}")
            continue
        new_text = text.rstrip("\n") + "\n\n" + block.rstrip("\n") + "\n"
    else:
        stripped, n_removed = strip(text, markers[0], markers[1])
        if n_removed > 1:
            raise RuntimeError(f"{path}: {n_removed} old blocks, expected <=1")
        if block in stripped:
            print(f"IDEMPOTENT (already present): {path.name}")
            continue
        new_text = stripped.rstrip("\n") + "\n\n" + block.rstrip("\n") + "\n"
    write(path, new_text, nl)
    written.append(path.name)

print("written:", written)
for p, _, _ in targets:
    if p.exists():
        print(f"  {p.name}: sha256={hashlib.sha256(p.read_bytes()).hexdigest()[:16]}")
