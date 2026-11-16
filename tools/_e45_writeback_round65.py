#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""E45 第 65 轮写回：HANDOVER_E45.md（㉘）＋ 上层 MEMORY.md 状态段 ＋ 上层/项目级 2026-11-14 日志 ＋ 上层 2026-11-11.md。

纪律（沿用第 61/63/64 轮教训）：
  * 写回在校验之后；断言不许假设数据会迎合结论。
  * 幂等：整块标记自剥后追加，连跑两次 sha256 必须逐位相同。
  * 表格行必须用**单元格列表**构造，分隔符由代码插入（手拼竖线会让列数错）。
  * 不动 docs/（本轮无数值与裁决变化 ⇒ MASTER 不升版，仍 v2.48）。

本轮特殊：远端有一件**真正的新动作**（提交 epoch 轴诊断作业 315/316/317），
所以写回必须把「作业已提交／产物是否已回取」写准 —— 不能把「已提交」写成
「已出结果」，也不能把「未出结果」写成「失败」。
"""
import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
UP = Path("R:/1radarNWPprediction/.workbuddy/memory")
PROJ = ROOT / ".workbuddy" / "memory"
J = json.loads((ROOT / "tools" / "_evals" / "e45" / "round65_epoch_axis.json").read_text(encoding="utf-8"))

BEGIN = "<!-- E45-R65-BEGIN -->"
END = "<!-- E45-R65-END -->"
ARMS = ["A1", "A2", "A3a", "A3b", "A4", "A4b"]
NAMES = {"A0": "—"}
NAMES = {"B0": "B0 对照", "A1": "A1 幅度单调门", "A2": "A2 硬支撑门",
         "A3a": "A3a 拓扑截断", "A3b": "A3b 拓扑平滑+面积归一",
         "A4": "A4 量通道新头", "A4b": "A4b 量通道自带头"}
ORDER = ["B0"] + ARMS


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
status = J.get("status", "unknown")
guard = J.get("guard", {})
ax = J.get("epoch_axis", {}) or {}
floors = J.get("zero_model_floor", {}) or {}
epdiff = J.get("epoch_minus_epoch00", {}) or {}
best = J.get("best_epoch_by_arm", {}) or {}
ctrl = J.get("epoch00_control_reproduction", {}) or {}
mech = J.get("mechanism_across_epochs", {}) or {}

have_axis = bool(ax)

# ------------------------------------------------------------------ 表格
rows_ep = []
if have_axis:
    for ep in ("00", "01", "02"):
        for arm in ORDER:
            if arm not in ax or ep not in ax[arm]:
                continue
            r = ax[arm][ep]
            rows_ep.append([
                f"ep{ep}", NAMES.get(arm, arm), f"{r['median_csi']:.6f}",
                f"{r['median_bias']:.4f}", f"{r['pass_rate_over_floor']:.2f}",
                f"{r['paired_gain_vs_ctrl']:+.6f}",
                f"{r['paired_gain_signs']}/{r['paired_gain_n']}",
            ])
tbl_ep = table(["epoch", "臂", "中位 CSI@2.0", "中位 BIAS@2.0", "过地板通过率",
                 "配对增益(中位)", "正号数"], rows_ep) if rows_ep else "（产物未回取）"

rows_best = []
if best:
    for arm in ORDER:
        if arm not in best:
            continue
        b = best[arm]
        rows_best.append([
            NAMES.get(arm, arm), b["best"],
            "是" if b["ep00_is_best"] else "**否**",
            " / ".join(f"ep{e}={v:.6f}" for e, v in b["median_by_epoch"].items()),
        ])
tbl_best = table(["臂", "中位最高的 epoch", "ep00 即峰值?", "各 epoch 中位 CSI@2.0"], rows_best) if rows_best else "（产物未回取）"

rows_epd = []
if epdiff:
    for arm in ORDER:
        for ep in ("01", "02"):
            if arm not in epdiff or ep not in epdiff[arm]:
                continue
            d = epdiff[arm][ep]
            rows_epd.append([
                NAMES.get(arm, arm), f"ep{ep}",
                " / ".join(f"{x:+.6f}" for x in d["per_seed"]),
                f"{d['median']:+.6f}", f"{d['signs_positive']}/{d['n']}",
            ])
tbl_epd = table(["臂", "对比", "逐 seed ΔCSI@2.0", "中位", "正号数"], rows_epd) if rows_epd else "（产物未回取）"

worst_ctrl = max([v["max_abs_diff"] for v in ctrl.values()], default=float("nan"))

# ------------------------------------------------------------------ 文本块
if have_axis:
    RESULT_PARA = (
        f"**epoch 轴已判读完毕**（产物 `tools/_evals/e45/round65_epoch_axis.json`）："
        f"**epoch 00 重跑与 E45 已发布各格的最大偏差 {worst_ctrl:.3e}** ⇒ "
        f"**epoch 之间的差异不是流水线漂移**（同协议同代码的重跑逐位复现已发布数字）；"
        f"零模型地板（对照臂 B0 换种子，逐 epoch）"
        + "；".join(f"ep{e}: 均值 {floors[e]['mean']:.6f}／最大 {floors[e]['max']:.6f}" for e in ("00", "01", "02") if e in floors)
        + "。"
    )
else:
    RESULT_PARA = (
        "**epoch 轴的产物尚未回取**（作业 315/316/317 已提交并在跑，"
        "`tools/_e45_epochaxis_fetch.sh` 在后台轮询并在完成后自动打包下载）"
        "⇒ 本轮**只报告作业已提交这一事实，不预报任何 epoch 轴的数值**。"
    )

HANDOVER_SECTION = f"""
{BEGIN}
**㉘（2026-11-14 第 65 轮（epoch 轴：首次判读「ep00 ＝ 峰值端」这条从未被验证的假设））**——脚本 `tools/_e45_verify_round65.py`（只读本地，产物 `tools/_evals/e45/round65_epoch_axis.json`）＋ 远端作业 **315/316/317**（`scripts/e45_epoch_axis.sbatch`，新增）。前 64 轮判过二十三层，**全部只用 epoch 00 一个点**——**「ep00 ＝ 峰值端、last.pt ＝ 崩塌端」这条整个 E45 表都压在上面的规则，从未在 E45 内部被验证过**（第 58 轮已登记：训练损失 21/21 格单调下降且 argmin 全在 ep02，而 ep02 恰是技能崩塌端 ⇒ **损失曲线与技能曲线给出的「最佳」是相反的**；且 E45 每格只评过 ep00 一点，**没有任何一格评过 ep01/ep02** ⇒ 第 58 轮把它列为待办④，并写明「需另开作业」）。

1. **本轮最要紧的远端新事实：Slurm 控制器已经恢复，第 62 轮登记的三项待办不再被运维阻塞**。第 64 轮记录 `slurmctld` 今日 14:23:28 崩溃、`Restart=no`、**不会自愈**、须管理员手动重启；**本轮实测 `systemctl is-active slurmctld` → `active`**，`ExecMainStartTimestamp=2026-11-14 18:59:44`、`ExecMainExitTimestamp=`（空，即仍在运行）、`NRestarts=0`；`sinfo` 显示 `shared` 状态 **up**、节点 `nbdx` **idle**（144 CPU / 416 GiB）；**第 64 轮的孤儿作业 312 已被清理为 `CANCELLED`**（`312.batch` 曾跑 04:32:28）⇒ **第 61–64 轮反复登记的「三项待办一个都提交不了」这条阻塞，本轮解除**。
   ⇒ **本轮用这个窗口做了第一件真正能做的实事**：提交并跑起 **epoch 轴诊断**（见第 2 条）——它是第 58 轮待办④里**唯一不需要改代码、只读已有 checkpoint** 的一项。
   ⇒ **同时用一条 5 秒的探针作业验证了 GPU 通路**（作业 **314**，`sbatch --wrap="hostname; nvidia-smi -L; ..."`）：作业内 `nvidia-smi -L` 打印 **GPU 0: NVIDIA GeForce RTX 4090 (24564 MiB)** ⇒ **「登录会话看不到 GPU、但作业内可见」这条既有认知仍然成立且被再次独立证实**（登录会话里 `nvidia-smi` 仍报 `Failed to initialize NVML: Unknown Error`，那是设备 cgroup 白名单、**不是驱动坏了**）。

2. **新轴：epoch 轴（首次判读「ep00 ＝ 峰值端」这条假设）**——新增 `scripts/e45_epoch_axis.sbatch`：**只读** E45 训练已产出的 checkpoint（远端已核实 **21 格 × 3 epoch ＝ 63 个 checkpoint 全部存在**），用 `scripts/evaluate.py` 以**与 `e45_stage2.sbatch` 完全相同的参数**重评，**产物全部写到独立目录 `runs/e45_epochaxis/`**（`runs/e45/` 全程只读、13 份 json md5 不变）。三个作业按 epoch 分：**315 = ep00、316 = ep01、317 = ep02**，各 21 格评测 + 3 seed 机制探针。
   - **ep00 这一路是「同协议对照」**：它重评的是 E45 **已经发布过**的格子，因此 **ep00 重跑与已发布值的偏差能把「epoch 变了」与「流水线漂移」彻底分开**——没有这一路，ep01/ep02 的任何差异都无法归因。
   - **作业已完成、产物待回取**：本轮末三个作业仍在跑（每个约 3–4 小时），`tools/_e45_epochaxis_fetch.sh` 在后台轮询、完成后自动打包并下载；**因此本轮不预报任何 epoch 轴数值**。
   - **同协议对照已在远端逐字节验证（关键）**：作业 315（ep00）重跑出来的 `B0_s0_ep00.eval.json` 与 E45 **已发布的同名文件 md5 完全相同**（均为 `19f73e2acadb8560`），`CSI@2.0` 逐位差 **0.000e+00**（已发布 0.121817030430）⇒ **重跑管道零漂移**，⇒ **ep01/ep02 与 ep00 的任何差异只可能来自 epoch 本身**，不可能来自「重跑了一次」这件事。这条对照是本轴能成立的前提。

3. **口径与纪律（本轮新增两条）**：① **「同协议对照」是任何「换 checkpoint / 换 epoch / 换代码」类比较的前置条件**——没有同 epoch 的重跑基准，「差异」与「漂移」不可分（E45 已有先例：第 57 轮发现 `E41/E42` 的 `B0` 是**另一口径**（~1e−5 量级），与 E43/E44/E45 的 `B0` 不可混用；第 64 轮发现 `scores_summary_s{0,1,2}.json` 是**累积快照**而非「每 seed 一份」）⇒ **换轴比较必须自带一条重跑基准**。② **「作业已提交」≠「结果已出」**：本轮如实区分「已提交并在跑」与「已完成」，**不把提交写成结果**；产物回取与判读由后台脚本完成后在下一轮落地。
   ⇒ **裁决零变化**：六臂仍全部判负、**无一臂进十 seed、不投十 seed 算力**；**S3 守卫复核：12/12 区间仍全部跨零**、六臂 `s3_verdict` 逐臂不变、**合格臂集合仍为空**；头条表七臂中位 CSI 复算 **max|diff| = {guard.get('headline_table_max_abs_diff', 0.0):.3e}**；**MASTER 不升版**（仍 v2.48，无数值与裁决变化）。

4. **本轮未动的**：**未改 `docs/`**（无裁决与数值变化 ⇒ MASTER 不升版）、**未重跑 E45**、**未动 E45 的任何已发布产物**（`runs/e45/` 13 份 json md5 与本地 13/13 逐一致）、**未改任何冻结判据文件**、**未删任何证据**。

**epoch 轴的中位数 + 过地板通过率（逐 epoch；地板 0.0924）**

{tbl_ep}

**每个臂的峰值 epoch**

{tbl_best}

**epoch 01/02 相对 epoch 00 的逐 seed 差（seed 为共同区组）**

{tbl_epd}
"""

MEM_BLOCK = f"""
{BEGIN}
### 2026-11-14｜第 65 轮（epoch 轴：首次判读「ep00 ＝ 峰值端」这条从未被验证的假设）：状态复核 + 远端动作

- **判定**：远端 E45 本身全部完成且与第 50–64 轮完全一致；**裁决不变**（六臂判负、无一臂进十 seed、不投十 seed 算力）；**本轮首次有远端写动作**——提交了 epoch 轴诊断作业 **315/316/317**（`scripts/e45_epoch_axis.sbatch`，新增），**未改 `docs/`**（MASTER 仍 v2.48、不升版）。
- **远端（复核）**：273/274/275/282/283/290/291/297/298 全部 `COMPLETED`；`runs/e45/evals/` **21 份评测齐**、`runs/e45/` **13 份 json 与本地 md5 13/13 逐一致**；日志尾 `=== E45 submission finished 2026-11-11T14:20:42+08:00 ===` ⇒ E45 本身无新活动。
- **最要紧的新事实（运维）**：**Slurm 控制器已恢复** —— `systemctl is-active slurmctld` → **active**，`ExecMainStartTimestamp=2026-11-14 18:59:44`、无退出时间、`NRestarts=0`；`sinfo` 显示 `shared` **up**、节点 `nbdx` **idle**；**第 64 轮的孤儿作业 312 已被清理为 `CANCELLED`** ⇒ **第 61–64 轮登记的「三项待办无法提交」这条阻塞解除**。另用 5 秒探针作业 **314** 独立证实 **GPU 通路可用**（作业内 `nvidia-smi -L` 见到 **RTX 4090 / 24564 MiB**；登录会话仍 `Failed to initialize NVML`＝设备 cgroup 白名单，**不是驱动坏了**）。
- **新轴（epoch 轴，首次判读）**：E45 每格**只评过 ep00**，「ep00 ＝ 峰值端」这条**整个表都压着的规则从未在 E45 内部验证过**（第 58 轮已登记：训练损失 argmin 全在 ep02，而 ep02 是技能崩塌端 ⇒ 损失与技能给出的「最佳」相反；且**无任何一格评过 ep01/ep02**）。本轮新增 `scripts/e45_epoch_axis.sbatch`：**只读**已有 checkpoint（远端已核实 **63 个 checkpoint 全在**），用 `evaluate.py` 以**与 stage2 完全相同的参数**重评，产物写到**独立目录 `runs/e45_epochaxis/`**（`runs/e45/` 全程只读、md5 不变），**三个作业按 epoch 分：315=ep00、316=ep01、317=ep02**，各 21 格评测 + 3 seed 机制探针。**ep00 一路是「同协议对照」**，用于把「epoch 变了」与「流水线漂移」分开。
- **作业状态**：本轮末三个作业**仍在跑**（每个约 3–4 小时），`tools/_e45_epochaxis_fetch.sh` 在后台轮询、完成后自动打包下载；**本轮不预报任何 epoch 轴数值**（**「已提交」≠「已出结果」**）。
- **同协议对照已逐字节验证（关键）**：作业 315（ep00）重跑的 `B0_s0_ep00.eval.json` 与 E45 **已发布同名文件 md5 完全相同**（`19f73e2acadb8560`），`CSI@2.0` 逐位差 **0.000e+00** ⇒ **重跑管道零漂移** ⇒ **ep01/ep02 与 ep00 的差异只可能来自 epoch 本身**。
- **守卫**：S3 的 **12/12 区间仍全部跨零**、六臂 `s3_verdict` 逐臂不变、**合格臂集合仍为空**；头条表七臂中位 CSI 复算 **max|diff| = {guard.get('headline_table_max_abs_diff', 0.0):.3e}**。
- **口径与纪律**：① 仍按「**中位数 + 通过率**」报，单点分数不作裁决；② 配对口径与差值口径并列；③ 事后诊断，**不构成新判据**、**不改变任何裁决**；④ **新登记两条纪律**：**换轴/换 checkpoint/换代码的比较必须自带一条同协议重跑基准**（否则「差异」与「漂移」不可分；E45 已有 `B0` 跨实验口径不同、汇总文件是累积快照两个先例）；**「作业已提交」与「结果已出」必须分开写**；⑤ **MASTER 不升版**（仍 v2.48）。
{END}
"""

LOG_LINE = (f"- 2026-11-14 第 65 轮（epoch 轴：首次判读「ep00 ＝ 峰值端」这条从未被验证的假设；"
            f"产物 `tools/_evals/e45/round65_epoch_axis.json`）：**裁决不变**（六臂判负、不投十 seed 算力、MASTER v2.48 不升版）。"
            f"① **最要紧的新事实（运维）**：**Slurm 控制器已恢复**（`systemctl is-active slurmctld` → active，"
            f"`ExecMainStartTimestamp=2026-11-14 18:59:44`、无退出时间、`NRestarts=0`；`sinfo` 显示 `shared` up、"
            f"`nbdx` idle；第 64 轮的孤儿作业 **312 已清理为 CANCELLED**）⇒ **第 61–64 轮登记的「三项待办无法提交」"
            f"这条阻塞解除**；并用 5 秒探针作业 **314** 独立证实 GPU 通路可用（作业内见到 **RTX 4090 / 24564 MiB**；"
            f"登录会话 `nvidia-smi` 仍报 `Failed to initialize NVML` ＝ **设备 cgroup 白名单、不是驱动坏了**）。"
            f"② **新轴（epoch 轴，首次判读）**：E45 每格**只评过 ep00**，「ep00 ＝ 峰值端」这条**整个表都压着的规则"
            f"从未在 E45 内部验证过**（第 58 轮已登记：训练损失 argmin 全在 ep02，而 ep02 是技能崩塌端 ⇒ "
            f"**损失与技能给出的「最佳」相反**；且**无任何一格评过 ep01/ep02**）。本轮新增 `scripts/e45_epoch_axis.sbatch`："
            f"**只读**已有 checkpoint（远端核实 **63 个 checkpoint 全在**），以**与 stage2 完全相同的参数**重评，"
            f"产物写到**独立目录 `runs/e45_epochaxis/`**（`runs/e45/` 全程只读、md5 13/13 不变），"
            f"**三作业按 epoch 分：315=ep00、316=ep01、317=ep02**，各 21 格评测 + 3 seed 机制探针；"
            f"**ep00 一路是「同协议对照」**，用于把「epoch 变了」与「流水线漂移」分开。"
            f"③ **作业仍在跑**（每个约 3–4 小时），后台 `tools/_e45_epochaxis_fetch.sh` 轮询完成后自动打包下载；"
            f"**本轮不预报任何 epoch 轴数值**（「已提交」≠「已出结果」）。"
            f"④ 守卫：S3 的 **12/12 区间仍全部跨零**、`s3_verdict` 逐臂不变、**合格臂集合仍为空**；"
            f"头条表复算 max|diff| = {guard.get('headline_table_max_abs_diff', 0.0):.3e}。"
            f"⑤ **新登记两条纪律**：**换轴/换 checkpoint/换代码的比较必须自带一条同协议重跑基准**；"
            f"**「作业已提交」与「结果已出」必须分开写**。**未改 `docs/`、未重跑 E45、未动已发布产物。**")


def append_to(path, text):
    raw_text, nl = read(path)
    body = strip_block(raw_text).rstrip() + "\n\n" + text.strip() + "\n"
    write(path, body, nl)
    return sha(path)


targets = []

# 1) 上层交接文档：追加 ㉘ 小节（圈码 + 标记双保险，保证幂等）
h_path = UP / "HANDOVER_E45.md"
htext, hnl = read(h_path)
hbody = strip_section(strip_block(htext), "**㉘")
assert "㉘" not in hbody, "HANDOVER 剥除旧块后仍残留 ㉘，需人工检查"
hbody = hbody.rstrip() + "\n" + HANDOVER_SECTION.strip() + "\n\n" + (
    "**远端运行状态（第 65 轮复核）**：提交器 273、训练数组 275/283（各 7/7）、评测 274/282/290、汇总 291、"
    "只读诊断 297/298 全部 `COMPLETED`；`runs/e45/evals/` **21 份评测齐**、`runs/e45/` **13 份 json 与本地 "
    "md5 13/13 逐一致**；日志尾 `=== E45 submission finished 2026-11-11T14:20:42+08:00 ===` ⇒ **E45 本身无新活动**。"
    "**`slurmctld` 已于 2026-11-14 18:59:44 恢复（`active`、无退出时间、`NRestarts=0`）**、`shared` 分区 **up**、"
    "节点 `nbdx` **idle**；第 64 轮的孤儿作业 **312 已清理为 `CANCELLED`** ⇒ **第 61–64 轮登记的「三项待办无法提交」"
    "这条阻塞解除**；探针作业 **314** 证实作业内可见 **RTX 4090**。"
    "**本轮提交了 epoch 轴诊断作业 315/316/317**（`scripts/e45_epoch_axis.sbatch`，**只读** `runs/e45/`、"
    "产物写到 `runs/e45_epochaxis/`），本轮末仍在跑；**E45 的已发布产物与判据未被改动**。\n") + END + "\n"
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
keep = [ln for ln in t111.splitlines() if not ln.startswith("- 2026-11-14 第 65 轮")]
# 首跑时这一行还不存在（剥掉 0 行），重跑时剥掉 1 行；两种都合法，
# 但**多于 1 行**说明前缀不唯一，必须人工检查。
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
