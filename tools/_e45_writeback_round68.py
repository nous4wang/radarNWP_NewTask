#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""E45 第 68 轮写回：HANDOVER_E45.md（㉛）＋ 上层 MEMORY.md 状态段 ＋ 上层/项目级 2026-11-14 日志
＋ 上层 2026-11-11.md ＋ 本自动化记忆文件。

纪律（沿用第 61/63/64/65/66/67 轮教训）：
  * 写回在校验之后；断言不许假设数据会迎合结论。
  * 幂等：整块标记自剥后追加，连跑两次 sha256 必须逐位相同。
  * 表格行必须用**单元格列表**构造，分隔符由代码插入（手拼竖线会让列数错）。
  * 不动 docs/（本轮无数值与裁决变化 => MASTER 不升版，仍 v2.48）。

本轮特殊：epoch 轴从 n=2 升到 **n=3（三 seed 整列，63 格评测 + 9 份探针）**，
所以写回必须把「n=3 的读数」与「n=1／n=2 的旧结论」分开写：
  * 不把 n=3 的读数写成「机制成立」（S3 仍 12/12 跨零、六臂仍判负）；
  * 不把「第 66 轮 4/7 臂」当成定论（第 67 轮已在 n=2 推翻，n=3 复现 7/7）；
  * 不把「作业已提交」写成「失败」——三个 epoch 作业本轮已全部 COMPLETED。
"""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
UP = Path("R:/1radarNWPprediction/.workbuddy/memory")
PROJ = ROOT / ".workbuddy" / "memory"
AUTO = PROJ / "automations" / "e3cc6636-3e60-4a58-8f7e-67d4c896b0b6" / "memory.md"
J = json.loads((ROOT / "tools" / "_evals" / "e45" / "round68_epoch_axis_n3.json").read_text(encoding="utf-8"))

BEGIN = "<!-- E45-R68-BEGIN -->"
END = "<!-- E45-R68-END -->"

ORDER = ["B0", "A1", "A2", "A3a", "A3b", "A4", "A4b"]
NAMES = {"B0": "B0 对照", "A1": "A1 幅度单调门", "A2": "A2 硬支撑门",
         "A3a": "A3a 拓扑截断", "A3b": "A3b 拓扑平滑+面积归一",
         "A4": "A4 量通道新头", "A4b": "A4 量通道自带头"}
EPOCHS = ["00", "01", "02"]
FLOOR = 0.0924

ARM_E = J["arm_epoch"]
PEAK = J["peak_epoch"]
CRIT = J["criteria"]
FLR = J["zero_model_floor_per_epoch"]
FP = J["floor_pass_counts_per_epoch"]
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
    text = text.replace("\r\n", "\n").replace("\r", "\n")
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


def append_block(path, body):
    """幂等：先自剥本轮旧块，再整块追加到文件末尾。"""
    text, nl = read(path)
    text, n_removed = strip(text, BEGIN, END)
    assert n_removed <= 1, f"{path.name}: found {n_removed} old R68 blocks"
    text = text.rstrip("\r\n") + "\n\n" + BEGIN + "\n" + body.rstrip("\n") + "\n" + END + "\n"
    write(path, text, nl)
    after, _ = read(path)
    cnt = after.count(BEGIN)
    assert cnt == 1, f"{path.name}: marker count {cnt} != 1"
    return path, n_removed


# ---------------------------------------------------------------- assertions
assert J["n_checks_pass"] == J["n_checks_total"] == 17, "analysis checks not all positive"
assert J["n_cells"] == 63, "expected 63 three-seed cells"
assert J["control_md5"]["n_identical"] == J["control_md5"]["n_compared"] == 21
assert J["control_csi_max_abs_diff"] == 0.0
assert J["probe_control_max_abs_diff"] == 0.0
assert J["s3_guard"]["n_intervals"] == 12 and J["s3_guard"]["n_cross_zero"] == 12
assert J["published_md5"]["scores_summary"].startswith("57bcd511a537ab85")
assert J["published_md5"]["s3_equalbias"].startswith("3aba999d5ef6dcc4a")
assert J["published_per_seed_consistency_max_abs_diff"] < 1e-12
assert EPK["n2"] == 7 and EPK["n3"] == 7, "epoch-peak counts changed unexpectedly"
assert EPK["argmax_changed_n2_to_n3"] == 0, "an arm changed its peak epoch between n=2 and n=3"
assert J["verdict"]["eligible_arm_set"] == [] and J["verdict"]["e45_verdict_changed"] is False
assert all(not r["paired_delta_exceeds_floor"]
           for k, r in ARM_E.items() if k != "B0_ep00"), "an arm/epoch beat the zero-model floor"
assert max(FP["00"].values()) == 3 and FP["00"]["A3a"] == 3 and FP["00"]["A3b"] == 3
assert max(FP["01"].values()) <= 2 and max(FP["02"].values()) <= 2

# ------------------------------------------------------------------- tables
t1 = head(["臂", "epoch", "中位 CSI@2.0 (n=3)", "配对中位 ΔCSI", "差值口径 ΔCSI",
            "正号率", "过地板", "超零模型地板"])
for arm in ORDER:
    for ep in EPOCHS:
        r = ARM_E[f"{arm}_ep{ep}"]
        t1 += row([NAMES[arm], ep, f"{r['median_csi']:.6f}",
                   f"{r['paired_median_delta']:+.6f}", f"{r['diff_convention_delta']:+.6f}",
                   f"{r['pos_sign_rate']:.2f}", f"{r['n_over_floor']}/3",
                   "是" if r["paired_delta_exceeds_floor"] else "否"]) + "\n"

t2 = head(["臂", "n=1(seed0) 最高 epoch", "n=2 最高 epoch", "n=3 最高 epoch",
            "n=3 中位 CSI: ep00 / ep01 / ep02", "n=3 过地板"])
for arm in ORDER:
    p = PEAK[arm]
    m = p["n3_median"]
    t2 += row([NAMES[arm], p["best_n1"], p["best_n2"], p["best_n3"],
               f"{m['00']:.6f} / {m['01']:.6f} / {m['02']:.6f}",
               f"{FP['00'][arm]}/3"]) + "\n"

CRIT_ORDER = [("A1", "corr_mask_amp"), ("A1", "mask_at_peak"), ("A1", "maxratio_as_trained"),
              ("A2", "corr_mask_amp"), ("A2", "mask_at_peak"), ("A2", "maxratio_as_trained"),
              ("A3a", "grad_top1pct_share"), ("A3a", "field_max_mm"),
              ("A3b", "grad_top1pct_share"), ("A3b", "field_max_mm"),
              ("A4", "scale_cv"), ("A4", "scale_truth_corr"),
              ("A4b", "scale_cv"), ("A4b", "scale_truth_corr")]
t3 = head(["臂", "判据", "过", "不过", "读数缺失", "通过率(可判格)", "不过/缺失的格"])
for arm, cname in CRIT_ORDER:
    r = CRIT[f"{arm}:{cname}"]
    bad = [k for k, v in r["per_cell"].items() if v["verdict"] != "pass"]
    rate = "n/a" if r["pass_rate_over_available"] is None else f"{r['pass_rate_over_available']:.3f}"
    t3 += row([NAMES[arm], f"{cname} {r['rule'].split(cname)[1].strip()}",
               r["n_pass"], r["n_fail"], r["n_missing"], rate,
               "、".join(bad) if bad else "—"]) + "\n"

t4 = head(["epoch", "零模型地板中位(3 对)", "零模型地板最大", "逐对绝对 ΔCSI（B0 换 seed）"])
for ep in EPOCHS:
    f = FLR[ep]
    t4 += row([ep, f"{f['median']:.6f}", f"{f['max']:.6f}",
               "、".join(f"{d:.6f}" for d in f["pairs"])]) + "\n"

A3A_PASS = [k for k, v in CRIT["A3a:grad_top1pct_share"]["per_cell"].items()
             if v["verdict"] == "pass"]
A4B_BAD = [k for k, v in CRIT["A4b:scale_truth_corr"]["per_cell"].items()
           if v["verdict"] != "pass"]

# --------------------------------------------------------------- block texts
handover_body = f"""**㉛（2026-11-14 23:2x 第 68 轮新增证据（epoch 轴：n=3 受控判读——三 seed 整列齐了，epoch 轴判读封口））**——脚本 `tools/_e45_verify_round68.py`（只读本地，产物 `tools/_evals/e45/round68_epoch_axis_n3.json`，**VERDICT: PASS 17/17**）；远端 epoch 轴作业 **315/316/317 已全部 `COMPLETED`**（2026-11-14 23:01:46／23:02:34／23:03:15，各约 3h10m）。**E45 本身无新活动**（提交器 273、训练数组 275/283、评测 274/282/290、汇总 291、只读诊断 297/298 全部 `COMPLETED`；`runs/e45/evals/` 21 份评测齐；`runs/e45/` 13 份 json 未变——`scores_summary.json` md5 `57bcd511a537ab85`、`s3_equalbias.json` md5 `3aba999d5ef6dcc4a`）⇒ **裁决零变化：六臂仍全部判负、合格臂集合仍为空、不投十 seed 算力、MASTER 不升版（仍 v2.48）**。

1. **本轮把 epoch 轴从 n=2 升到 n=3（三 seed 整列）**：回取 **63 格评测**（7 臂 × 3 seed × 3 epoch）＋ **9 份机制探针**（6 臂 × 3 seed × 3 epoch ＝ 54 格读数），远端一次打包、一次下载（tar md5 `734857236b13142f02d6174ee6d8fc8f`，本地逐位相同）。

2. **同协议基准全部通过（零漂移，并从 16 格扩到 21 格）**：ep00 重跑与 E45 已发布同名文件 **md5 逐位相同 21/21 格**；ep00 重跑的 21 格 `CSI@2.0` 与已发布逐 seed 值 **max|diff| = 0.000e+00**；**三个 seed** 的机制探针 ep00 与已发布 `diag_probe_s{{0,1,2}}.json` 可比读数 **max|diff| = 0.000e+00**；已发布头条表七臂中位 CSI 与过地板通过率复算 **max|diff| = 0.000e+00**（通过率字符串逐臂一致）⇒ **ep01/ep02 与 ep00 的任何差异只可能来自 epoch 本身**。

3. **最重要的一条：「ep00 ＝ 峰值端」在 n=3 下仍然成立**：七臂 **7/7** 以 ep00 为最高 epoch（第 66 轮 n=1 的 4/7 是种子噪声，第 67 轮 n=2 的 7/7 本轮复现，**没有任何一臂的 argmax 在 n=2→n=3 之间改变**）；七臂 n=3 中位 CSI **全部 ep00 最高、ep01 次之、ep02 最低**；**对照臂 B0 自身技能随 epoch 单调塌陷**（中位 0.121817 → 0.121067 → 0.055067）⇒ **E45 用 ep00 评的这张表没有被推翻**。

4. **「换个 epoch 能更好」在 n=3 下依然不成立，且证据比 n=2 更强**：零模型地板（＝同一 epoch 下把对照臂换个 seed，本轮 **3 对**）中位 **ep00 0.030118／ep01 0.060224／ep02 0.058907**；**21 个（臂 × epoch）的配对中位 ΔCSI 中，没有一个的绝对值超过该 epoch 的零模型地板**（逐格判定全部「否」）⇒ **任何换 epoch 带来的「增益」都小于种子噪声**。

5. **过地板（0.0924）通过率按 epoch 收缩**：ep00 **A3a 3/3、A3b 3/3**（其余五臂含对照臂 B0 自身各 2/3），ep01 最多 2/3、ep02 最多 2/3 ⇒ **「3 seed 全过地板」只在 ep00 出现，且只对 A3a/A3b**——两臂早已被 S3 判负（12/12 区间跨零）⇒ **不投十 seed 算力、不得择优**。

6. 机制判据在 epoch × seed 两轴上的通过情况（每判据 **9 格 ＝ 3 seed × 3 epoch**）：
   - **门族**：A1 `corr_mask_amp` **9/9 过**、`mask_at_peak` **9/9 过**；A2 `mask_at_peak` **9 格全 1.0**、`corr_mask_amp` **5 格过 / 4 格缺失（NaN）/ 0 格不过** ⇒ 仍是**读数缺失**（探针非 nan 感知聚合缺陷）**不是判据翻转**；`maxratio_as_trained > 0.9` 仍是 **9/9 缺失**（A1、A2 各 9 格）⇒ 该预注册判据**从探针自身仍不可判**（第 41 轮旁路诊断 A1 0.72–0.76 不过／A2 1.0 过仍然有效）。
   - **拓扑族**：**A3a `grad_top1pct_share < 0.5` 只有 2/9 格过**（{'、'.join(A3A_PASS)}）⇒ 第 66/67 轮的「勉强过」始终是**单格现象**、不构成臂级结论；**A3a/A3b `field_max_mm ≤ 30` 0/9**（恒 402.428802 mm ＝ clamp 上限）⇒ **拓扑支结案结论不变**，理由按「epoch × 种子」两轴写清。
   - **量族**：A4/A4b `scale_cv ≥ 0.05` **9/9 过**；A4 `scale_truth_corr ≥ 0.3` **9/9 过**；A4b **7/9 过**（不过者 {'、'.join(A4B_BAD)}，与第 43/67 轮登记的两格同格）⇒ **量通道三 epoch 三 seed 都没退化成常数**，退化形式仍是「塌缩＋脱钩」且随 checkpoint 移动 ⇒ 与 `1e-12` 护栏缺陷同向、**不构成因果主张**。

7. **守卫**：S3 **12/12 区间仍全部跨零**（4 × `GAIN_WAS_AREA_OR_AMPLITUDE` ＋ 2 × `NO_GAIN_TO_BEGIN_WITH`）、六臂 `s3_verdict` 逐臂不变、**合格臂集合仍为空**；`docs/` 未改、**MASTER 不升版（仍 v2.48）**。

8. **口径与纪律**：① 仍按「**中位数 + 通过率**」报，单点分数不作裁决；② 配对口径与差值口径并列；③ 事后诊断、**不构成新判据**、不改变任何裁决；④ **新登记两条纪律**：**同协议重跑基准必须随复制单元同步扩格**（本轮把零漂移基准从 16/16 扩到 21/21 格、探针从 1 seed 扩到 3 seed）；**换 epoch 的收益必须同时报该 epoch 的零模型地板及其配对数**（n=3 的地板有 3 对，比 n=2 的 1 对更稳）。

{t1}
{t2}
{t3}
{t4}
> **边界声明**：本轮是**事后诊断**，**不构成新判据、不改变任何裁决**；六臂仍全部判负、合格臂集合仍为空、**不投十 seed 算力**；`docs/MASTER.md` **不升版（仍 v2.48）**，`docs/` 未改。"""

mem_body = f"""### 2026-11-14｜第 68 轮（epoch 轴：n=3 受控判读 —— 三 seed 整列齐了，epoch 轴判读封口）

- **判定**：远端 **E45 本身全部完成且与第 50–67 轮完全一致**（273/274/275/282/283/290/291/297/298 全 `COMPLETED`；`runs/e45/evals/` 21 份齐；`runs/e45/` 13 份 json 未变，`scores_summary.json` md5 `57bcd511a537ab85`、`s3_equalbias.json` md5 `3aba999d5ef6dcc4a`）⇒ **E45 无新活动**。**唯一新活动＝第 65 轮提交的 epoch 轴诊断作业 315/316/317 本轮全部 `COMPLETED`**（23:01:46／23:02:34／23:03:15，各约 3h10m）⇒ **epoch 轴数据首次到齐（63 格评测 + 9 份探针）**。**裁决不变**（六臂判负、无一臂进十 seed、不投十 seed 算力）；**未改 `docs/`**（MASTER 仍 v2.48）。
- **回取与基准**：远端一次打包、一次下载（tar md5 `734857236b13142f02d6174ee6d8fc8f`，本地逐位相同）；脚本 `tools/_e45_verify_round68.py`（只读本地，**POS 17/17 VERDICT: PASS**，产物 `tools/_evals/e45/round68_epoch_axis_n3.json`）。**同协议基准零漂移**：ep00 重跑与已发布文件 md5 **21/21 逐位相同**；ep00 的 21 格 `CSI@2.0` 与已发布逐 seed 值 **max|diff| = 0.000e+00**；**三个 seed** 的探针 ep00 与已发布 `diag_probe_s{{0,1,2}}.json` 可比读数 **max|diff| = 0.000e+00**；已发布头条表七臂中位 CSI 与过地板通过率复算 **max|diff| = 0.000e+00**。
- **关键结论三条**：① **「ep00 ＝ 峰值端」在 n=3 下成立**（七臂 **7/7** 以 ep00 为最高 epoch，n=2→n=3 无一臂 argmax 改变；第 66 轮 n=1 的 4/7 确系种子噪声）；对照臂 B0 技能随 epoch 单调塌陷（0.121817→0.121067→0.055067）⇒ **E45 用 ep00 评的表未被推翻**。② **换 epoch 无收益且证据更强**：零模型地板（同 epoch 换 seed，3 对）中位 **ep00 0.030118／ep01 0.060224／ep02 0.058907**，**21 个（臂 × epoch）配对中位 ΔCSI 无一超过该 epoch 地板**。③ **「3 seed 全过地板」只在 ep00 出现且只对 A3a/A3b**（两臂早已被 S3 判负）⇒ **不投十 seed 算力、不得择优**。
- **机制三条新事实**：① A3a `grad_top1pct_share < 0.5` **只有 2/9 格过**（A3a_s2_ep00 = 0.478171、A3a_s0_ep02 = 0.499906）⇒ 始终是单格现象；**A3a/A3b `field_max_mm ≤ 30` 0/9**（恒 402.428802 mm ＝ clamp 上限）⇒ **拓扑支结论不变**。② A2 `corr_mask_amp` **5 格过 / 4 格缺失 / 0 格不过**、`mask_at_peak` 9/9 全 1.0；`maxratio_as_trained` **9/9 缺失** ⇒ 仍是**读数缺失不是判据翻转**。③ A4/A4b `scale_cv` 9/9 过、A4 `scale_truth_corr` 9/9 过、A4b **7/9 过**（不过者 A4b_s2_ep00、A4b_s1_ep02，与第 43/67 轮同格）⇒ **量通道三 epoch 三 seed 都没退化成常数**。
- **守卫**：S3 **12/12 区间仍全部跨零**、六臂 `s3_verdict` 逐臂不变、**合格臂集合仍为空**；`docs/` 未改、**MASTER 不升版（仍 v2.48）**。
- **纪律**：① 仍按「**中位数 + 通过率**」报，单点分数不作裁决；② 配对口径与差值口径并列；③ 事后诊断、**不构成新判据**、**不改变任何裁决**；④ **新登记两条纪律**：**同协议重跑基准必须随复制单元同步扩格**；**换 epoch 的收益必须同时报该 epoch 的零模型地板及其配对数**。
"""
short_body = f"""- **第 68 轮（2026-11-14 23:2x，epoch 轴 n=3 受控判读）**：远端 E45 无新活动（273/274/275/282/283/290/291/297/298 全 `COMPLETED`；21 份评测齐；13 份 json md5 未变）；**epoch 轴作业 315/316/317 本轮全部 `COMPLETED`**（23:01–23:03）⇒ **63 格评测 + 9 份探针首次到齐**。回取 tar md5 `734857236b13142f02d6174ee6d8fc8f`，脚本 `tools/_e45_verify_round68.py`（只读本地，**VERDICT: PASS 17/17**，产物 `tools/_evals/e45/round68_epoch_axis_n3.json`）。**同协议基准零漂移**（ep00 重跑 md5 **21/21** 逐位相同；21 格 `CSI@2.0` 与已发布 **max|diff| = 0.000e+00**；三 seed 探针 ep00 **max|diff| = 0.000e+00**）。**「ep00 ＝ 峰值端」在 n=3 下成立**（七臂 **7/7**，n=2→n=3 无 argmax 改变；第 66 轮 n=1 的 4/7 确系种子噪声）⇒ **E45 用 ep00 评的表未被推翻**；**换 epoch 无收益**（零模型地板 3 对，中位 ep00 0.030118／ep01 0.060224／ep02 0.058907；21 个（臂 × epoch）配对中位 ΔCSI **无一超过地板**）；**「3 seed 全过地板」只在 ep00 且只对 A3a/A3b**（早已被 S3 判负）⇒ **不投十 seed 算力、不得择优**。机制：A3a `grad_top1pct_share` **2/9 过**（A3a_s2_ep00 0.478171、A3a_s0_ep02 0.499906）、A3a/A3b `field_max_mm` **0/9**（402.428802 mm ＝ clamp 上限）⇒ **拓扑支结论不变**；A2 `corr_mask_amp` 5 过／4 缺失／0 不过、`mask_at_peak` 9/9、`maxratio` 9/9 缺失 ⇒ 读数缺失 ≠ 判据翻转；A4b `scale_truth_corr` 7/9（不过者 A4b_s2_ep00、A4b_s1_ep02，与第 43/67 轮同格）⇒ **量通道三 epoch 三 seed 都没退化成常数**。**裁决零变化**：S3 **12/12 跨零**、合格臂集合仍为空、**六臂判负、不投十 seed 算力、MASTER 仍 v2.48 不升版**、`docs/` 未改。**纪律两条**：同协议重跑基准必须随复制单元同步扩格；换 epoch 的收益必须同时报该 epoch 的零模型地板及其配对数。"""

# 上层 2026-11-11.md 用独立标记，避免与该文件其它轮次混标
BEGIN_111 = "<!-- E45-R68-111-BEGIN -->"
END_111 = "<!-- E45-R68-111-END -->"
text111 = f"""{BEGIN_111}
- **第 68 轮（2026-11-14 23:2x，epoch 轴 n=3 受控判读）**：**epoch 轴三 seed 整列齐了**（作业 315/316/317 全部 `COMPLETED`）⇒ 63 格评测 + 9 份探针首次到齐；脚本 `tools/_e45_verify_round68.py`（**VERDICT: PASS 17/17**，产物 `tools/_evals/e45/round68_epoch_axis_n3.json`）。**「ep00 ＝ 峰值端」在 n=3 下成立**（七臂 7/7；n=1 的 4/7 确系种子噪声）⇒ **E45 用 ep00 评的表未被推翻**；**换 epoch 无收益**（零模型地板 3 对，21 个（臂 × epoch）配对中位 ΔCSI 无一超过地板）；A3a 梯度集中判据 **2/9 过**、`field_max_mm` **0/9** ⇒ **拓扑支结论不变**。**裁决零变化**（S3 12/12 跨零、六臂判负、不投十 seed 算力、MASTER 仍 v2.48 不升版）。
{END_111}
"""


def append_block_custom(path, begin, end, body):
    text, nl = read(path)
    text, n_removed = strip(text, begin, end)
    assert n_removed <= 1, f"{path.name}: found {n_removed} old blocks"
    text = text.rstrip("\r\n") + "\n\n" + body.rstrip("\n") + "\n"
    write(path, text, nl)
    after, _ = read(path)
    assert after.count(begin) == 1, f"{path.name}: marker count != 1"
    return n_removed


targets = [
    (UP / "HANDOVER_E45.md", handover_body, BEGIN, END),
    (UP / "MEMORY.md", mem_body, BEGIN, END),
    (UP / "2026-11-14.md", short_body, BEGIN, END),
    (PROJ / "2026-11-14.md", short_body, BEGIN, END),
    (AUTO, mem_body, BEGIN, END),
]
written = []
for path, body, b, e in targets:
    n = append_block(path, body) if (b, e) == (BEGIN, END) else None
    written.append((path.name, n))

n111 = append_block_custom(UP / "2026-11-11.md", BEGIN_111, END_111, text111)
written.append(("2026-11-11.md", n111))

print("written:")
for name, n in written:
    print(f"  {name}: stripped {n} old block(s)")
print()
for path in [UP / "HANDOVER_E45.md", UP / "MEMORY.md", UP / "2026-11-14.md",
             UP / "2026-11-11.md", PROJ / "2026-11-14.md", AUTO]:
    raw = path.read_bytes()
    print(f"{hashlib.sha256(raw).hexdigest()}  {path}")
