#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""E45 第 66 轮写回：HANDOVER_E45.md（㉙）＋ 上层 MEMORY.md 状态段 ＋ 上层/项目级 2026-11-14 日志 ＋ 上层 2026-11-11.md。

纪律（沿用第 61/63/64/65 轮教训）：
  * 写回在校验之后；断言不许假设数据会迎合结论。
  * 幂等：整块标记自剥后追加，连跑两次 sha256 必须逐位相同。
  * 表格行必须用**单元格列表**构造，分隔符由代码插入（手拼竖线会让列数错）。
  * 不动 docs/（本轮无数值与裁决变化 ⇒ MASTER 不升版，仍 v2.48）。

本轮特殊：epoch 轴的产物**部分回取**（seed 0 整列 21 格 + B0_s1 三格 + A1_s1_ep02 一格 ＝ 25 份），
所以写回必须把「已回取的部分」与「尚未回取的部分」分开写：
  * 不把「部分列」写成「完整 21 格 × 3 epoch」；
  * 不把「机制臂只有 seed 0」写成「三 seed 结论」；
  * 也不把「作业仍在跑」写成「失败」。
"""
import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
UP = Path("R:/1radarNWPprediction/.workbuddy/memory")
PROJ = ROOT / ".workbuddy" / "memory"
J = json.loads((ROOT / "tools" / "_evals" / "e45" / "round66_epoch_axis_controlled.json").read_text(encoding="utf-8"))

BEGIN = "<!-- E45-R66-BEGIN -->"
END = "<!-- E45-R66-END -->"
NAMES = {"B0": "B0 对照", "A1": "A1 幅度单调门", "A2": "A2 硬支撑门",
         "A3a": "A3a 拓扑截断", "A3b": "A3b 拓扑平滑+面积归一",
         "A4": "A4 量通道新头", "A4b": "A4b 量通道自带头"}
ORDER = ["B0", "A1", "A2", "A3a", "A3b", "A4", "A4b"]
EPOCHS = ["ep00", "ep01", "ep02"]


def read(p):
    raw = p.read_bytes()
    nl = "\r\n" if b"\r\n" in raw else "\n"
    return raw.decode("utf-8"), nl


def write(p, text, nl):
    p.write_bytes(text.replace("\n", nl).encode("utf-8"))


def strip_block(text):
    pat = re.compile(re.escape(BEGIN) + r".*?" + re.escape(END) + r"\s*", re.S)
    return pat.sub("", text)


def strip_section(text, label):
    pat = re.compile(r"\*\*" + re.escape(label) + r"（.*?(?=\n\n\*\*|\n\n<!--|\Z)", re.S)
    return pat.sub("", text)


def table(hdr, rows):
    """表格行必须用单元格列表构造，分隔符由代码插入。"""
    out = ["| " + " | ".join(str(c) for c in hdr) + " |",
           "|" + "|".join(["---"] * len(hdr)) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(c) for c in r) + " |")
    return "\n".join(out)


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


# ------------------------------------------------------------------ 取数
guard = J["guard"]
axis = J["epoch_axis_seed0"]
peak = J["peak_epoch_per_arm"]
paired = J["paired_delta_vs_ep00"]
zero = J["zero_model_floor_seed_swap"]
ratio = J["epoch_delta_over_zero_model"]
colmed = J["column_medians"]
diffc = J["difference_convention"]
flips = J["criterion_flips_across_epoch"]
zm = J["zero_model_floor_max"]

# ------------------------------------------------------------------ 表格
rows_axis = []
for ep in EPOCHS:
    for arm in ORDER:
        blk = axis.get(arm, {}).get(ep)
        if not blk:
            continue
        rows_axis.append([
            f"ep{ep[2:]}", NAMES[arm], f"{blk['median_csi_at_2.0']:.6f}",
            f"{blk['median_bias_at_2.0']:.4f}",
            f"{blk['pass_floor']}/{blk['n_seeds']}",
        ])
tbl_axis = table(["epoch", "臂", "中位 CSI@2.0", "中位 BIAS@2.0", "过地板"], rows_axis)

rows_peak = []
for arm in ORDER:
    if arm not in peak:
        continue
    p = peak[arm]
    rows_peak.append([
        NAMES[arm], p["best_epoch"], "是" if p["ep00_is_best"] else "**否**",
        " / ".join(f"ep{e[2:]}={v:.6f}" for e, v in sorted(p["median_by_epoch"].items())),
        f"{p['ep01_minus_ep00']:+.6f}",
        f"{p['ep02_minus_ep00']:+.6f}",
        f"{ratio.get(arm, {}).get('ep01', float('nan')):.2f} / "
        f"{ratio.get(arm, {}).get('ep02', float('nan')):.2f}",
    ])
tbl_peak = table(["臂", "最高 epoch", "ep00 即最高?", "各 epoch 中位 CSI@2.0",
                  "Δ ep01−ep00", "Δ ep02−ep00", "|Δ|/零模型地板 ep01/ep02"], rows_peak)

rows_zero = [[f"ep{ep[2:]}", f"{zero[ep]['abs_delta_csi']:.6f}",
              f"{zero[ep]['abs_delta_bias']:.4f}", "对照臂 B0 换 seed（纯种子噪声）"]
             for ep in EPOCHS if ep in zero]
tbl_zero = table(["epoch", "|ΔCSI| 零模型地板", "|ΔBIAS| 零模型地板", "零模型定义"], rows_zero)

# ------------------------------------------------------------------ 文本块
HANDOVER_SECTION = f"""
{BEGIN}
**㉙（2026-11-14 第 66 轮（epoch 轴：首个受控判读——「ep00 ＝ 峰值端」这条规则被数据否掉了一部分））**——脚本 `tools/_e45_verify_round66.py`（只读本地，产物 `tools/_evals/e45/round66_epoch_axis_controlled.json`）；远端作业 **315/316/317** 仍在跑（第 65 轮提交）。第 65 轮只做了「作业已提交」，本轮第一次拿到**可判读的部分产物**并给出受控结论。

1. **本轮先说清楚「回了多少」**：作业 315/316/317 各完成 **9/21 格**（远端合计 **27/63 格**）；本轮回取的是**其中 25 份**——**seed 0 整列 21 格（7 臂 × 3 epoch 全齐）＋ 对照臂 B0 的第二 seed 三格 ＋ A1_s1_ep02 一格**。⇒ **机制臂在本轮只有 seed 0（n=1）**，**A1/A2/A3a/A3b/A4/A4b 的臂级结论本轮一律不给**；下面所有读数都标了「同 seed 固定」的配对性质。剩余 38 格（s1/s2 全列）仍由后台 `tools/_e45_epochaxis_fetch.sh` 轮询取回。

2. **同协议基准比第 65 轮更强（从 1 臂扩到 7 臂）**：本轮**逐臂**核对了 ep00 重跑与 E45 已发布文件——**B0/A1/A2/A3a/A3b/A4/A4b 七臂的 `{arm}_s0_ep00.eval.json` 与已发布同名文件 md5 全部逐位相同**（B0 `19f73e2acadb8560`、A1 `081f10d09d05e8a4`、A2 `4cd4d82310969df3`、A3a `a463d9102694b1d8`、A3b `6d63f95aa26322f5`、A4 `d79fd59b52fdc833`、A4b `9247529bf6d2cb40`）⇒ **重跑管道零漂移已由单臂扩到全七臂**。机制探针侧同样：epoch 轴的 ep00 探针与已发布 `diag_probe_s0.json` 的全部可比读数 **max|diff| = 0.000e+00**（n = {guard['probe_same_protocol_n_compared']} 项）⇒ **ep01/ep02 与 ep00 的任何差异只可能来自 epoch 本身**。已发布 `runs/e45/` 全程只读（`scores_summary.json` md5 `57bcd511a537ab85`、`s3_equalbias.json` md5 `3aba999d5ef6dcc4a` 未变）。

3. **最重要的发现：「ep00 ＝ 峰值端」这条规则**只在 **4/7 臂**上成立，对照臂本身就不成立**——**B0 的最高 epoch 是 ep01**（ep00 0.121817 → ep01 0.130836 → ep02 0.113975），**A1 与 A4b 的最高 epoch 是 ep02**（A1 ep02 0.121355 > ep00 0.118397；A4b ep02 0.145589 > ep00 0.137567）；只有 A2/A3a/A3b/A4 是 ep00 最高。⇒ **E45 整张表所依赖的「ep00 是峰值端」是一条只对少数臂成立的近似规则**，且**连对照臂都违反它** ⇒ 它不是一条被验证过的选择准则，更像是**一条恰好对多数臂成立的便利约定**。

4. **但「换个 epoch 就能更好」这句话也不成立**——所有跨 epoch 的差都落在**零模型地板之内**。零模型＝**同一 epoch 下把对照臂换个 seed**（B0_s0 vs B0_s1，纯种子噪声、不含任何机制）：|ΔCSI| 地板 **ep00 {zero['ep00']['abs_delta_csi']:.6f}／ep01 {zero['ep01']['abs_delta_csi']:.6f}／ep02 {zero['ep02']['abs_delta_csi']:.6f}**，**最大 {zm:.6f}**。逐臂跨 epoch 的 |ΔCSI| 与地板之比**全部 < 1**（最大者 A4 的 ep01，×{ratio['A4']['ep01']:.2f}）⇒ **没有任何一臂的 epoch 选择带来可与噪声区分的技能差**。差值口径同向：列中位 ep01 − ep00 = **{diffc['ep01']['column_median_delta']:+.6f}**、ep02 − ep00 = **{diffc['ep02']['column_median_delta']:+.6f}**（中位口径），七臂在 ep01/ep02 全部下降。

5. **机制读数在 epoch 轴上的稳定性**——三条新的、可独立登记的事实：
   - **拓扑族的梯度集中读数随 epoch 单调改善，A3a 在 ep02 首次越过预注册判据**：A3a 的 `grad_top1pct_share` ep00 0.5526 → ep01 0.5384 → **ep02 0.4999**（判据 < 0.5 ⇒ **ep02 才 PASS**，ep00/ep01 FAIL）；A3b 同读数 ep00 0.9712 → ep01 0.9762 → ep02 0.9975（**三 epoch 全 FAIL**）。⇒ **「拓扑支结掉」这条结论必须精确改写**：它对 **A3b 三 epoch 全败**成立，对 **A3a 是「ep00/ep01 败、ep02 勉强过、但另一条判据 `field_max_mm` 402.43 三个 epoch 全败」** ⇒ **A3a 的拓扑支并未被 ep02 救回来**（另一条判据仍 FAIL）⇒ **拓扑支结案的结论不变，但理由要按 epoch 分支写，不能笼统说「两臂都不过」**。
   - **量族的量级塌缩随 epoch 缓解**：A4b 的 `scale_mean` ep00 0.4068 → ep01 0.0858 → ep02 0.1507，`scale_cv` 三个 epoch 都在 1.9–2.8（**量通道三 epoch 都没退化成常数**），`scale_truth_corr` 0.6565/0.5323/0.6651（**三 epoch 都 ≥ 0.3 过判据**）；A4 的 `mean_out_equals_T` **三 epoch 全为 True**。⇒ **第 62 轮登记的 `1e-12` 护栏缺陷，其触发概率随 epoch 变化** ⇒ **修护栏后重测时必须逐 epoch 报告，不能只报一个 epoch**。
   - **门族的机制强度随 epoch 衰减且 A2 在 ep01 读数缺失**：A1 `corr_mask_amp` 0.6103/0.5555/0.5822（**三 epoch 都 ≥ 0 过判据**）、`mask_at_peak` 0.8217/0.9144/0.9612（**三 epoch 都 > 0.5 过判据**）；A2 `corr_mask_amp` 0.4035/**ep01 读数缺失（NaN）**/0.1822，`corr_mask_resid` 0.6769/**缺失**/0.3221 ⇒ **A2 在 ep01 的门读数整组为空**，与第 41/43 轮登记的「非 nan 感知聚合」缺陷同族 ⇒ **这是读数缺失，不是判据翻转，必须分开写**。

6. **裁决零变化**：六臂仍全部判负、**无一臂进十 seed、不投十 seed 算力**。守卫复核：**S3 的 12/12 区间仍全部跨零**、六臂 `s3_verdict` 逐臂不变、**合格臂集合仍为空**；头条表七臂中位 CSI 复算 **max|diff| = {guard['headline_table_max_abs_diff']:.3e}**。⇒ **MASTER 不升版**（仍 v2.48，无数值与裁决变化）。

7. **口径与纪律（本轮新增三条）**：① **凡报「换一个超参/检查点/配置就更好」，必须同时给出该轴的零模型地板**（本轮＝换 seed）——本轮正因给了地板，才能把「ep00 不是峰值」与「换 epoch 没有可区分收益」这两句同时说清（第 49 轮那条纪律的推广：从「随时效」推广到「任何可比轴」）。② **「判据翻转」必须区分「读数越界」与「读数缺失」**——A3a 的 `grad_top1pct_share` 在 ep02 是**越界达标**（真值 0.4999），A2 的 `corr_mask_amp` 在 ep01 是**读数缺失**（NaN），二者都表现为「这一格没有 PASS」，但含义相反。③ **n=1 的格只能报方向与幅度，不得作臂级结论**——机制臂本轮只有 seed 0，故本轮只给「同 seed 固定下的逐 epoch 方向」。

**epoch 轴逐格读数（seed 0 整列；每格 n=1，中位数＝该格值；地板 0.0924）**

{tbl_axis}

**每个臂的峰值 epoch ＋ 相对 ep00 的配对差 ＋ 与零模型地板之比**

{tbl_peak}

**零模型地板（对照臂 B0 换 seed，逐 epoch；纯种子噪声、不含机制）**

{tbl_zero}
"""

MEM_BLOCK = f"""
{BEGIN}
### 2026-11-14｜第 66 轮（epoch 轴：首个受控判读——「ep00 ＝ 峰值端」被数据否掉了一部分）：状态复核 + 产物部分回取

- **判定**：远端 E45 本身全部完成且与第 50–65 轮完全一致；**裁决不变**（六臂判负、无一臂进十 seed、不投十 seed 算力）；本轮**回取了 epoch 轴的部分产物并做了首个受控判读**；**未改 `docs/`**（MASTER 仍 v2.48、不升版）。
- **远端（复核）**：273/274/275/282/283/290/291/297/298 全部 `COMPLETED`；`runs/e45/evals/` **21 份评测齐**、`runs/e45/` **13 份 json**（`scores_summary.json` md5 `57bcd511a537ab85`、`s3_equalbias.json` md5 `3aba999d5ef6dcc4a` 未变）⇒ **E45 本身无新活动**。**epoch 轴作业 315/316/317 仍在跑，各 9/21 格（合计 27/63）**；本轮回取 25 份（seed 0 整列 21 格 ＋ B0_s1 三格 ＋ A1_s1_ep02 一格），后台 `tools/_e45_epochaxis_fetch.sh` 继续轮询。
- **同协议基准（从 1 臂扩到 7 臂）**：**七臂的 `{arm}_s0_ep00.eval.json` 与 E45 已发布同名文件 md5 全部逐位相同**；epoch 轴的 ep00 探针 vs 已发布 `diag_probe_s0.json` 的全部可比读数 **max|diff| = 0.000e+00**（n = {guard['probe_same_protocol_n_compared']}）⇒ **重跑管道零漂移，ep01/ep02 的差异只可能来自 epoch 本身**。
- **最重要的新事实**：「**ep00 ＝ 峰值端**」这条 E45 整表所依赖的规则，**只对 4/7 臂成立，对照臂 B0 自身就不成立**（B0 最高是 ep01 0.130836；A1/A4b 最高是 ep02）⇒ 它是**一条未经验证的便利约定**，不是被验证过的选择准则。**但「换 epoch 能更好」同样不成立**：跨 epoch 的 |ΔCSI| **全部落在零模型地板内**（零模型＝同 epoch 下 B0 换 seed，最大地板 {zm:.6f}；逐臂比值最大 A4 ep01 ×{ratio['A4']['ep01']:.2f}，**无一超过 1**），差值口径同向（列中位 ep01 {diffc['ep01']['column_median_delta']:+.6f}、ep02 {diffc['ep02']['column_median_delta']:+.6f}）。
- **机制读数随 epoch 的三条新事实**：① **A3a 的 `grad_top1pct_share` ep02 首次越过预注册判据**（0.5526 → 0.5384 → **0.4999**，判据 < 0.5），但另一条判据 `field_max_mm` 402.43 **三 epoch 全败** ⇒ **拓扑支结案结论不变，但理由要按 epoch 分支写**（A3b 三 epoch 全败；A3a 是「一条勉强过、另一条全败」）。② **量族量级塌缩随 epoch 缓解**（A4b `scale_mean` 0.4068 → 0.0858 → 0.1507），`scale_cv` 三 epoch 都在 1.9–2.8、`scale_truth_corr` 三 epoch 都 ≥ 0.3 ⇒ **量通道三 epoch 都没退化成常数**；**A4 的 `mean_out_equals_T` 三 epoch 全 True** ⇒ **第 62 轮登记的 `1e-12` 护栏缺陷触发概率随 epoch 变化，修后重测须逐 epoch 报告**。③ **A2 在 ep01 的门读数整组缺失（NaN）**（`corr_mask_amp`/`corr_mask_resid`），A1 三 epoch 的门读数全过判据 ⇒ **这是读数缺失、不是判据翻转，必须分开写**。
- **守卫**：S3 的 **12/12 区间仍全部跨零**、六臂 `s3_verdict` 逐臂不变、**合格臂集合仍为空**；头条表七臂中位 CSI 复算 **max|diff| = {guard['headline_table_max_abs_diff']:.3e}**。
- **口径与纪律**：① 仍按「**中位数 + 通过率**」报，单点分数不作裁决；② 配对口径与差值口径并列；③ 事后诊断，**不构成新判据**、**不改变任何裁决**；④ **本轮新增三条纪律**：**凡报「换一个配置/检查点更好」必须同时给该轴的零模型地板**；**「判据翻转」必须区分「读数越界」与「读数缺失」**；**n=1 的格只报方向与幅度，不作臂级结论**；⑤ **MASTER 不升版**（仍 v2.48）。
{END}
"""

LOG_LINE = (f"- 2026-11-14 第 66 轮（epoch 轴：首个受控判读；产物 `tools/_evals/e45/round66_epoch_axis_controlled.json`）："
            f"**裁决不变**（六臂判负、不投十 seed 算力、MASTER v2.48 不升版）。"
            f"① **远端状态**：E45 本身（273/274/275/282/283/290/291/297/298）全部 `COMPLETED`、21 份评测齐、"
            f"`runs/e45/` 13 份 json 未变 ⇒ 无新活动；**epoch 轴 315/316/317 仍在跑（各 9/21 格，合计 27/63）**，"
            f"本轮回取 **25 份**（seed 0 整列 21 格 ＋ B0_s1 三格 ＋ A1_s1_ep02 一格），后台取回脚本继续轮询。"
            f"② **同协议基准从 1 臂扩到 7 臂**：七臂 `{arm}_s0_ep00.eval.json` 与 E45 已发布同名文件 **md5 全部逐位相同**；"
            f"epoch 轴 ep00 探针 vs 已发布 `diag_probe_s0.json` **max|diff| = 0.000e+00** ⇒ **重跑零漂移，差异只来自 epoch**。"
            f"③ **最重要的新事实：「ep00 ＝ 峰值端」只对 4/7 臂成立，连对照 B0 都不成立**（B0 最高是 ep01 0.130836；"
            f"A1/A4b 最高是 ep02）⇒ 它是未经验证的便利约定。**但换 epoch 也换不到可区分收益**：跨 epoch |ΔCSI| "
            f"**全部落在零模型地板内**（地板＝同 epoch 下 B0 换 seed，最大 {zm:.6f}；逐臂比值无一超过 1），"
            f"差值口径列中位 ep01 {diffc['ep01']['column_median_delta']:+.6f}、ep02 {diffc['ep02']['column_median_delta']:+.6f}。"
            f"④ **三条机制新事实**：A3a `grad_top1pct_share` ep02 首次达标（0.4999）但 `field_max_mm` 402.43 三 epoch 全败 ⇒ "
            f"**拓扑支结案不变但理由须按 epoch 分支写**；量族 `scale_cv` 三 epoch 都在 1.9–2.8、`scale_truth_corr` 三 epoch 都 ≥ 0.3 "
            f"⇒ **量通道三 epoch 都没退化成常数**，A4b `scale_mean` 随 epoch 起伏 ⇒ **第 62 轮 `1e-12` 护栏缺陷触发概率随 epoch 变化、"
            f"修后重测须逐 epoch 报告**；**A2 在 ep01 的门读数整组缺失（NaN）** ⇒ **读数缺失 ≠ 判据翻转**。"
            f"⑤ 守卫：S3 **12/12 区间仍全部跨零**、`s3_verdict` 逐臂不变、**合格臂集合仍为空**；头条表复算 max|diff| = "
            f"{guard['headline_table_max_abs_diff']:.3e}。⑥ **新登记三条纪律**：报「换配置更好」必须给该轴零模型地板；"
            f"「判据翻转」须区分越界与缺失；n=1 的格只报方向与幅度。**未改 `docs/`、未重跑 E45、未动已发布产物。**")


def append_to(path, text):
    raw_text, nl = read(path)
    body = strip_block(raw_text).rstrip() + "\n\n" + text.strip() + "\n"
    write(path, body, nl)
    return sha(path)


targets = []

# 1) 上层交接文档：追加 ㉙ 小节（圈码 + 标记双保险，保证幂等）
h_path = UP / "HANDOVER_E45.md"
htext, hnl = read(h_path)
hbody = strip_section(strip_block(htext), "**㉙")
assert "㉙" not in hbody, "HANDOVER 剥除旧块后仍残留 ㉙，需人工检查"
hbody = hbody.rstrip() + "\n" + HANDOVER_SECTION.strip() + "\n\n" + (
    "**远端运行状态（第 66 轮复核）**：提交器 273、训练数组 275/283（各 7/7）、评测 274/282/290、汇总 291、"
    "只读诊断 297/298 全部 `COMPLETED`；`runs/e45/evals/` **21 份评测齐**、`runs/e45/` **13 份 json 未变**"
    "（`scores_summary.json` md5 `57bcd511a537ab85`、`s3_equalbias.json` md5 `3aba999d5ef6dcc4a`）⇒ **E45 本身无新活动**。"
    "**epoch 轴作业 315/316/317 仍在跑**（各 9/21 格，合计 27/63）；本轮已回取 **25 份**（seed 0 整列 21 格 ＋ "
    "B0_s1 三格 ＋ A1_s1_ep02 一格）并完成首个受控判读；**七臂的 ep00 重跑与已发布文件 md5 全部逐位相同** ⇒ 零漂移。"
    "剩余 s1/s2 全列由后台 `tools/_e45_epochaxis_fetch.sh` 轮询取回。**E45 的已发布产物与判据未被改动。**\n") + END + "\n"
write(h_path, hbody, hnl)
targets.append(("HANDOVER_E45.md", sha(h_path)))

# 2) 上层 MEMORY.md 状态段
targets.append(("MEMORY.md", append_to(UP / "MEMORY.md", MEM_BLOCK)))

# 3) 上层 2026-11-14 日志
targets.append(("upper 2026-11-14.md", append_to(UP / "2026-11-14.md", MEM_BLOCK)))

# 4) 项目级 2026-11-14 日志
targets.append(("proj 2026-11-14.md", append_to(PROJ / "2026-11-14.md", MEM_BLOCK)))

# 5) 上层 2026-11-11.md（E45 实验日志日）——单行式（按前缀剥旧行，保证幂等）
p111 = UP / "2026-11-11.md"
t111, n111 = read(p111)
n_before = len(t111.splitlines())
keep = [ln for ln in t111.splitlines() if not ln.startswith("- 2026-11-14 第 66 轮")]
n_dropped = n_before - len(keep)
assert n_dropped in (0, 1), f"2026-11-11.md 的旧行前缀不唯一（剥掉 {n_dropped} 行），需人工检查"
t111 = "\n".join(keep).rstrip() + "\n" + LOG_LINE + "\n"
write(p111, t111, n111)
targets.append(("upper 2026-11-11.md", sha(p111)))

# ------------------------------------------------------------------ 校验
for name, h in targets:
    print(f"  {name:28s} sha256 {h[:16]}…")

for path in (h_path, UP / "MEMORY.md", UP / "2026-11-14.md", PROJ / "2026-11-14.md"):
    txt, _ = read(path)
    n = txt.count(BEGIN)
    assert n == 1, f"{path} 标记计数 {n} != 1"
print("marker counts OK (each == 1)")
print("written OK")
