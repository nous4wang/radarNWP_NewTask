#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""E45 第 64 轮写回：HANDOVER_E45.md（㉗）＋ 上层 MEMORY.md 状态段 ＋ 上层/项目级 2026-11-14 日志 ＋ 上层 2026-11-11.md。

纪律（沿用第 61/63 轮教训）：
  * 写回在校验之后；断言不许假设数据会迎合结论。
  * 幂等：整块标记自剥后追加，连跑两次 sha256 必须逐位相同。
  * 表格行必须用**单元格列表**构造，分隔符由代码插入（手拼竖线会让列数错）。
  * 不动 docs/（本轮无数值与裁决变化 ⇒ MASTER 不升版，仍 v2.48）。
"""
import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
UP = Path("R:/1radarNWPprediction/.workbuddy/memory")
PROJ = ROOT / ".workbuddy" / "memory"
J = json.loads((ROOT / "tools" / "_evals" / "e45" / "round64_exact_seed_inference.json").read_text(encoding="utf-8"))

BEGIN = "<!-- E45-R64-BEGIN -->"
END = "<!-- E45-R64-END -->"
ARMS = ["A1", "A2", "A3a", "A3b", "A4", "A4b"]
NAMES = {"A1": "A1 幅度单调门", "A2": "A2 硬支撑门", "A3a": "A3a 拓扑截断",
         "A3b": "A3b 拓扑平滑+面积归一", "A4": "A4 量通道新头", "A4b": "A4b 量通道自带头"}


def read(p):
    """保留原始换行风格读回。"""
    raw = p.read_bytes()
    nl = "\r\n" if b"\r\n" in raw else "\n"
    return raw.decode("utf-8"), nl


def write(p, text, nl):
    p.write_bytes(text.replace("\n", nl).encode("utf-8"))


def strip_block(text):
    """剥掉本轮旧块（幂等）。"""
    pat = re.compile(re.escape(BEGIN) + r".*?" + re.escape(END) + r"\s*", re.S)
    return pat.sub("", text)


def strip_section(text, label):
    """剥掉 HANDOVER 里已有的同名圈码小节（幂等）。
    圈码小节一律追加在文末，故从标签处一直剥到文末。"""
    i = text.find(label)
    if i < 0:
        return text
    return text[:i].rstrip() + "\n"


def table(header, rows):
    """表格行由单元格列表构造，分隔符由代码插入。"""
    sep = "|" + "|".join(["---"] * len(header)) + "|"
    out = ["| " + " | ".join(str(c) for c in header) + " |", sep]
    for r in rows:
        assert len(r) == len(header), f"列数不符: {len(r)} vs {len(header)} :: {r}"
        out.append("| " + " | ".join(str(c) for c in r) + " |")
    return "\n".join(out)


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


# ------------------------------------------------------------------ 取数
ex = J["exact_inference"]
head = J["headline"]


def fmt_ex(axis, arm, key, nd=6):
    return f"{ex[axis][arm][key]:+.{nd}f}"


rows_raw = []
rows_eq = []
for a in ARMS:
    e = ex["raw"][a]
    per = " / ".join(f"{e['per_seed_paired_gain'][f's{s}']:+.4f}" for s in (0, 1, 2))
    rows_raw.append([NAMES[a], per, fmt_ex("raw", a, "paired_median"), fmt_ex("raw", a, "diff_median_arm_level"),
                     f"{e['signs_positive']}/3", f"{e['exact_sign_test_p_one_sided']:.4f}",
                     f"{e['exact_label_permutation_p_two_sided']:.4f}",
                     f"{e['exact_resample_frac_positive_median']:.4f}"])
    e = ex["equal_bias"][a]
    per = " / ".join(f"{e['per_seed_paired_gain'][f's{s}']:+.4f}" for s in (0, 1, 2))
    rows_eq.append([NAMES[a], per, fmt_ex("equal_bias", a, "paired_median"), fmt_ex("equal_bias", a, "diff_median_arm_level"),
                    f"{e['signs_positive']}/3", f"{e['exact_sign_test_p_one_sided']:.4f}",
                    f"{e['exact_label_permutation_p_two_sided']:.4f}",
                    f"{e['exact_resample_frac_positive_median']:.4f}"])

hdr = ["臂", "逐 seed 配对增益 (s0/s1/s2)", "配对中位", "差值口径", "正号数",
       "符号检验 p", "置换检验 p", "重抽正比例"]
tbl_raw = table(hdr, rows_raw)
tbl_eq = table(hdr, rows_eq)

hl_rows = []
for a in ["B0"] + ARMS:
    h = head[a]
    hl_rows.append([a, f"{h['median_csi_at_2.0']:.6f}", f"{h['pass_floor_count']}/3"])
tbl_head = table(["臂", "中位 CSI@2.0", "过地板通过率"], hl_rows)

n_check = J["n_pos"]
n_neg = J["n_neg"]
verdict = J["verdict"]

# ------------------------------------------------------------------ 文本块
HANDOVER_SECTION = f"""
{BEGIN}
**㉗（2026-11-14 第 64 轮（精确种子级推断轴 ＋ Slurm 控制器崩溃时间线））**——脚本 `tools/_e45_verify_round64.py`（只读本地，**POS {n_check}/{n_check + n_neg} VERDICT: {verdict}**，产物 `tools/_evals/e45/round64_exact_seed_inference.json`）：前 63 轮判过二十三层（聚合层／机制探针／S1 结构量／窗级／场次级／时效轴／强度·难度轴／误差构成／臂间合并／剂量-反应／指标族／护栏标定／增益图样／聚合算子与评测契约／训练动力学／晋级规则标定／决策价值／ROPE 认证／待办可执行性／权重空间），**全部用「抽样式」的区间与重抽**（bootstrap 10000 次、置换 20000 次）⇒ 本轮第一次把同一批逐 seed 数据放到**完全枚举**的精确推断下，并顺带做了一条**数据完整性审计**。**裁决零变化**（六臂判负、无一臂进十 seed、不投十 seed 算力；**MASTER 不升版，仍 v2.48**）。

1. **最要紧的远端新事实：Slurm 控制器不是「一直挂着不动」，而是今天崩过一次，且不会自愈**。`systemctl show slurmctld`：`ExecMainStartTimestamp=2026-11-14 07:00:11`、`ExecMainExitTimestamp=2026-11-14 14:23:28`、退出码 **1**、`NRestarts=0`、**`Restart=no`** ⇒ **它已死 4 小时 12 分，不会自己回来**（第 61/62/63 轮只记了「DOWN」，没记死因与时间线）。⇒ **三项待办（`1e-12` 护栏、NaN 聚合、输出场幅度截断）仍然一个都提交不了**，且**这次连"等它自己恢复"都不成立**（必须管理员手动重启 `slurmctld`）。

2. **崩溃时间点与一个「别的项目线的作业」相差 1 秒**：作业 **312**（`/home/wf/radarNWPQML` 的 qlfn2 / f36_nwp 线，`Submit=Start=2026-11-14T14:23:27`）在控制器死前 **1 秒**被派发 ⇒ 成为**孤儿作业**（`312.batch` 仍在节点上跑：`qlfn2_epoch_08.json` 18:38 才落盘 ⇒ 活儿本身还在推进），但控制器已死 ⇒ **无法受理任何新 sbatch、无法 requeue、无法记账**。⇒ **登记一条运维纪律：控制器崩溃时，正在跑的作业会变成「无主孤儿」——它的计算不中断，但它的排队、续跑与结算全部失效**（本轮实测）。

3. **观测口径坑（可直接复用）**：`sacct -u wf -S 2026-11-14T13:30:00` 与 `sacct -u wf`（无过滤）**都列不出作业 312**，只有 `sacct -j 312` 能列出 ⇒ **「过滤查不到」≠「没有作业在跑」**；查在跑的作业必须用 `sacct -j <id>` 或直接 `ps`（本轮实测）。

4. **数据完整性审计（此前从未做过）**：`runs/e45/scores_summary_s{{0,1,2}}.json` 实为**累积快照**、不是「每 seed 一份」——**s0 = 7 键（仅 seed0）、s1 = 14 键（seed0+1）、s2 = 21 键（seed0+1+2）**，且 **s2 与 `scores_all.json` 逐位相同**（远端 md5 `80655963622b09cd5231588e26468a28`）⇒ **逐 seed 取值必须按 key 里的 `_s<seed>_` 过滤**，若按文件名直接当「该 seed 的表」会**静默多算一倍的格**。本轮验证：累积链 **s0 ⊆ s1 ⊆ s2 逐值相等**（无缺失、无值漂移），且**按 key 过滤可逐位复现已发布的 21 格 `CSI@2.0` 与七臂中位数、过地板通过率** ⇒ 已发布数字无误，但**这条命名陷阱是真实存在的**。

5. **新轴：精确种子级推断（穷举，不抽样）**——三种检验全部在 n=3 上**完全枚举**：① **精确符号检验**（一侧，`Binom(3, 1/2)`）；② **标签置换检验**（「臂 vs 对照」逐 seed 翻转标签，**2^3 = 8** 种全枚举，统计量＝逐 seed 配对差中位数）；③ **种子重抽完全枚举**（**3^3 = 27** 种带重复重抽，统计「配对增益中位 > 0」的比例）。

   - **决定性的一条：n = 3 时「显著」在构造上就不可达**——任何精确检验的 p 值下界是 **1/8 = 0.125**（标签置换）或 **1/8 = 0.125**（符号检验）⇒ **连「观测到最强的那一臂」都不可能被精确检验判成显著**；S3 的「区间跨零」是当前唯一可用的口径，而它同样欠功效 ⇒ **E45 的判负不是「测了没测到」，而是「在 3 个复制单元下，任何检验都给不出证据」**（与第 48/52 轮的 MDE 结论、第 61 轮的 TOST 结论同族，但本条是**分辨率下界**，不依赖任何分布假设）。
   - **六臂 × 2 轴 = 12 组的精确 p 值全部落在 {{0.5000, 0.8750, 1.0000}}**，无一 < 0.125。
   - **逐 seed 正号数**：原始轴 A1/A2 **1/3**、A3a/A3b/A4/A4b **2/3**；配平轴 A3b/A4b **1/3**、其余 **2/3** ⇒ **没有一臂 3/3 同号为正**（与第 40 轮「六臂没有任何一臂在三个 seed 上都是正增益」独立吻合）。
   - **重抽完全枚举的正比例**：2/3 正号的臂 **0.7407（20/27）**、1/3 正号的臂 **0.2593（7/27）** ⇒ 与符号计数互为印证，且**与配平口径下 A3b/A4b 落到 0.2593 同向**（这两臂正是第 40/51 轮登记的符号翻转臂）。

6. **与前 52/54 轮的交叉一致性（本轮的外部校验）**：本轮重建的配对中位与差值口径增益**逐项与已发布值相同**——原始轴 A1 −0.003420／A2 −0.009814／A3a +0.015529／A3b +0.024357／A4 +0.013426／A4b +0.015750；配平轴 A1 +0.006443／A2 +0.001795／A3a +0.000249／A3b −0.005454／A4 +0.004395／A4b −0.009842 ⇒ **本轮脚本与既往 63 轮完全自洽**（S3 的 12 个点估复算 max|diff| = {J['s3_points_reproduced_max_abs_diff']:.3e}；逐 seed CSI 复算 max|diff| = {J['per_seed_csi_reproduced_max_abs_diff']:.3e}）。

7. **配平轴的「差值口径」此前从未报过**（本轮首次）：A1 −0.012769｜A2 −0.000945｜A3a −0.000152｜A3b −0.002205｜**A4 +0.004395**｜A4b −0.026410 ⇒ **差值口径下只有 A4 为正，其余五臂为负** ⇒ 与「配平后仍无存活增益」一致；**A4b 最负**（与第 45 轮场级塌缩、第 49 轮长时效过预报最重、第 60 轮 `r*` 最大同向）。

**原始增益轴（配对口径 ＝ 逐 seed 差值取中位数；差值口径 ＝ 两条中位数相减；thr = 2.0 mm）**

{tbl_raw}

**配平后（等偏置 S3）增益轴**

{tbl_eq}

**头条表（中位数 + 过地板通过率；地板 0.0924）**

{tbl_head}

**口径与纪律**：① 仍按「**中位数 + 通过率**」报，单点分数不作裁决；② 配对口径与差值口径并列；③ 事后诊断，**不构成新判据**（S3 护栏固定 thr=2.0 mm、复制单元固定为 seed）⇒ **不改变任何裁决**；④ **S3 裁决守卫复核**：12/12 区间仍全部跨零、六臂 `s3_verdict` 逐臂不变、**合格臂集合仍为空**；⑤ **MASTER 不升版**（无数值与裁决变化，仍 v2.48）；⑥ 本轮**未 sbatch、未下载、未改远端状态、未动 `docs/`**；远端 13 份 json 与本地 **md5 13/13 全同**，四个文档闸门全 `VERDICT: PASS`。

**新登记四条可复用纪律**：① **「区间跨零」在 n=3 下要连同「分辨率下界」一起说**——精确 p 的最小可达值是 **1/8 = 0.125**，所以「没测到」与「测不出」在 3 个复制单元下是**同一件事**（判语不能写成「无显著差异」）；② **累积快照必须按 key 过滤取数**，不能按文件名取「该 seed 的表」；③ **查在跑的作业不要只靠 `sacct -u -S`**（过滤会漏掉在跑作业），要用 `sacct -j` 或 `ps`；④ **控制器崩溃时在跑的作业会变孤儿**（计算不中断，但排队/续跑/结算失效）⇒ 崩溃时刻附近的作业必须单独核查。
"""

MEM_BLOCK = f"""
{BEGIN}
### 2026-11-14｜第 64 轮（精确种子级推断轴 ＋ Slurm 控制器崩溃时间线）：状态复核 + 只读判读，裁决不变

- **判定**：远端 E45 全部完成且与第 50–63 轮完全一致；**裁决不变**（六臂判负、无一臂进十 seed、不投十 seed 算力）；本轮远端动作**仅只读探针**（状态查询 + 控制器日志时间线 + 作业 312 核查），**未 sbatch、未下载、未改远端状态、未动 `docs/`**（MASTER 仍 v2.48）。
- **远端**：273/274/275/282/283/290/291/297/298 全部 `COMPLETED`；`runs/e45/evals/` **21 份评测齐**、`runs/e45/` **13 份 json 与本地 md5 13/13 逐一致**；日志尾 `=== E45 submission finished 2026-11-11T14:20:42+08:00 ===`；`runs/` 下无 E46+ 目录 ⇒ **无新活动**；四个文档闸门（`_check_tables`／`_check_v245_row`／`_check_v248_row`／`_e45_docs_numbers`）全 `VERDICT: PASS`。
- **新事实（运维，最要紧）**：`slurmctld` **今天崩过一次且不会自愈**——`ExecMainStartTimestamp=07:00:11`、`ExecMainExitTimestamp=14:23:28`、退出码 **1**、`NRestarts=0`、**`Restart=no`** ⇒ 已死 4h12m，**须管理员手动重启**（本账号 `sudo -n` 需密码、无 sudo）。崩溃前 **1 秒**派发了**别的项目线**的作业 **312**（`/home/wf/radarNWPQML` 的 qlfn2/f36_nwp）⇒ 成为**孤儿作业**（batch 仍在跑，`qlfn2_epoch_08.json` 18:38 落盘），但**排队/续跑/结算全部失效**、新 sbatch 无法受理 ⇒ 第 62 轮登记的三项待办**仍一个都提交不了**。
- **观测口径坑**：`sacct -u wf -S <日期>` 与 `sacct -u wf` **都列不出在跑的作业 312**，只有 `sacct -j 312` 能列出 ⇒ **「过滤查不到」≠「没有作业在跑」**。
- **数据完整性审计（首次）**：`runs/e45/scores_summary_s{{0,1,2}}.json` 是**累积快照**（s0=7 键、s1=14 键、s2=21 键），**s2 与 `scores_all.json` 逐位相同** ⇒ 逐 seed 取值必须**按 key 里的 `_s<seed>_` 过滤**，按文件名取会静默多算一倍格。本轮验证累积链 **s0⊆s1⊆s2 逐值相等**，且按 key 过滤**逐位复现已发布的 21 格 `CSI@2.0` 与七臂中位数/过地板通过率** ⇒ 已发布数字无误。
- **新轴（精确种子级推断，完全枚举）**：脚本 `tools/_e45_verify_round64.py`（只读本地，**POS {n_check}/{n_check + n_neg} VERDICT: {verdict}**，产物 `tools/_evals/e45/round64_exact_seed_inference.json`）。① **决定性的一条：n=3 时「显著」在构造上不可达**——精确 p 的下界是 **1/8 = 0.125** ⇒ **判负不是「测了没测到」，而是「3 个复制单元下任何检验都给不出证据」**（与第 48/52 轮 MDE、第 61 轮 TOST 同族，但本条是**分辨率下界**、不依赖分布假设）。② 六臂 × 2 轴 = 12 组精确 p 全部落在 {{0.5000, 0.8750, 1.0000}}。③ 逐 seed 正号数：原始轴 A1/A2 **1/3**、其余 **2/3**；配平轴 A3b/A4b **1/3**、其余 **2/3** ⇒ **无一臂 3/3 同号为正**（与第 40 轮独立吻合）。④ 重抽完全枚举正比例：2/3 正号臂 **0.7407（20/27）**、1/3 正号臂 **0.2593（7/27）**。⑤ **配平轴差值口径首次报**：A1 −0.012769｜A2 −0.000945｜A3a −0.000152｜A3b −0.002205｜**A4 +0.004395**｜A4b −0.026410 ⇒ 只有 A4 为正、**A4b 最负**。
- **守卫与自校**：S3 的 **12/12 区间仍全部跨零**、`crosses_zero` 12/12 为 true、六臂 `s3_verdict` 逐臂不变、**合格臂集合仍为空**；S3 的 12 个点估复算 max|diff| = {J['s3_points_reproduced_max_abs_diff']:.3e}；逐 seed `CSI@2.0` 复算 max|diff| = {J['per_seed_csi_reproduced_max_abs_diff']:.3e}；配对中位与差值口径增益**逐项与第 52/54 轮已发布值相同**。
- **口径与纪律**：① 仍按「**中位数 + 通过率**」报，单点分数不作裁决；② 配对口径与差值口径并列；③ 事后诊断，**不构成新判据**、**不改变任何裁决**；④ **新登记四条纪律**：报「区间跨零」须连同 n=3 的分辨率下界 **1/8** 一起说（不能写成「无显著差异」）；累积快照按 key 过滤取数；查在跑作业用 `sacct -j`/`ps` 而非只靠 `sacct -u -S`；控制器崩溃时在跑作业变孤儿须单独核查；⑤ **MASTER 不升版**（仍 v2.48）。
{END}
"""

LOG_LINE = (f"- 2026-11-14 第 64 轮（精确种子级推断轴 ＋ Slurm 控制器崩溃时间线，只读本地，"
            f"POS {n_check}/{n_check + n_neg} VERDICT: {verdict}，产物 "
            f"`tools/_evals/e45/round64_exact_seed_inference.json`）：**裁决不变**（六臂判负、不投十 seed 算力、"
            f"MASTER v2.48 不升版）。① **远端新事实（运维）**：`slurmctld` 今天崩过一次且**不会自愈**"
            f"（`ExecMainStartTimestamp=07:00:11`→`ExecMainExitTimestamp=14:23:28`、退出码 1、`NRestarts=0`、"
            f"`Restart=no`）⇒ **须管理员手动重启**；崩溃前 **1 秒**派发了别的项目线的**作业 312**"
            f"（`/home/wf/radarNWPQML` qlfn2/f36_nwp）⇒ **孤儿作业**（batch 仍在跑但排队/续跑/结算失效）；"
            f"**观测坑**：`sacct -u -S` 列不出在跑作业，只有 `sacct -j` 能列。② **数据完整性审计（首次）**："
            f"`scores_summary_s{{0,1,2}}.json` 是**累积快照**（7/14/21 键，s2 与 `scores_all.json` 逐位相同）"
            f"⇒ 逐 seed 取值必须**按 key 过滤**；累积链 s0⊆s1⊆s2 逐值相等，按 key 过滤**逐位复现已发布数字**。"
            f"③ **新轴（精确种子级推断，完全枚举）**：**n=3 时「显著」在构造上不可达**（精确 p 下界 **1/8 = 0.125**）"
            f"⇒ 判负＝「3 个复制单元下任何检验都给不出证据」；六臂 × 2 轴 12 组精确 p 全落在 {{0.5000, 0.8750, 1.0000}}；"
            f"逐 seed 正号数原始轴 A1/A2 **1/3**、其余 **2/3**，配平轴 A3b/A4b **1/3**、其余 **2/3** ⇒ **无一臂 3/3 同号为正**；"
            f"重抽完全枚举正比例 0.7407（20/27）vs 0.2593（7/27）；**配平轴差值口径首次报**（A4 +0.004395 唯一为正、A4b −0.026410 最负）。"
            f"④ 守卫：S3 的 12/12 区间仍全部跨零、`s3_verdict` 逐臂不变、**合格臂集合仍为空**；"
            f"12 个点估复算 max|diff| = {J['s3_points_reproduced_max_abs_diff']:.3e}。"
            f"**未 sbatch、未下载、未改远端状态、未动 `docs/`。**")


def append_to(path, text, marker_mode=True, section_label=None):
    raw_text, nl = read(path)
    if marker_mode:
        body = strip_block(raw_text).rstrip() + "\n\n" + text.strip() + "\n"
    else:
        body = strip_section(raw_text, section_label).rstrip() + "\n" + text.strip() + "\n"
    write(path, body, nl)
    return sha(path)


targets = []

# 1) 上层交接文档：追加 ㉗ 小节（圈码 + 标记双保险，保证幂等）
h_path = UP / "HANDOVER_E45.md"
htext, hnl = read(h_path)
hbody = strip_section(strip_block(htext), "**㉗")
assert "㉗" not in hbody, "HANDOVER 剥除旧块后仍残留 ㉗，需人工检查"
hbody = hbody.rstrip() + "\n" + HANDOVER_SECTION.strip() + "\n\n" + (
    "**远端运行状态（第 64 轮复核）**：提交器 273、训练数组 275/283（各 7/7）、评测 274/282/290、汇总 291、"
    "只读诊断 297/298 全部 `COMPLETED`；`runs/e45/evals/` **21 份评测齐**、`runs/e45/` **13 份 json 与本地 md5 13/13 逐一致**；"
    "日志尾 `=== E45 submission finished 2026-11-11T14:20:42+08:00 ===`；`runs/` 下无 E46+ 目录 ⇒ **无新活动**。"
    "**`Slurmctld(primary) at nbdx` 已于今日 14:23:28 崩溃（退出码 1、`Restart=no`、`NRestarts=0` ⇒ 不会自愈）**，"
    "崩溃前 1 秒派发的作业 312（别的项目线）成为**孤儿作业** ⇒ 第 62 轮登记的三项待办仍**无法提交**"
    "（需管理员手动重启控制器；本账号无 sudo，**不得以登录会话手跑 GPU 任务替代**）。"
    "本轮远端动作**仅只读探针**，**无训练、无 GPU 占用、无状态修改**。\n") + END + "\n"
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
keep = [ln for ln in t111.splitlines() if not ln.startswith("- 2026-11-14 第 64 轮")]
assert len(keep) == len(t111.splitlines()) - 1, "2026-11-11.md 的旧行前缀不唯一，需人工检查"
t111 = "\n".join(keep).rstrip() + "\n" + LOG_LINE + "\n"
write(p111, t111, n111)
targets.append(("upper 2026-11-11.md", sha(p111)))

# ------------------------------------------------------------------ 校验
for name, h in targets:
    print(f"  {name:28s} sha256 {h[:16]}…")

for path, label in [(h_path, "㉗"), (UP / "MEMORY.md", None),
                    (UP / "2026-11-14.md", None), (PROJ / "2026-11-14.md", None)]:
    txt, _ = read(path)
    n = txt.count(BEGIN)
    assert n == 1, f"{path} 标记计数 {n} != 1"
print("marker counts OK (each == 1)")
print("written OK")
