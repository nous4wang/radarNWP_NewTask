#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""E45 第 90 轮写回：带标记的幂等追加（标记已存在即跳过，绝不静默重复追加）。

写回目标：
  1. R:/1radarNWPprediction/.workbuddy/memory/2026-11-11.md   （上层逐日日志）
  2. R:/1radarNWPprediction/.workbuddy/memory/HANDOVER_E45.md（交接文档）
  3. R:/1radarNWPprediction/.workbuddy/memory/MEMORY.md      （状态段追加 bullet）
  4. radar-nwp-bench/.workbuddy/memory/2026-11-16.md         （项目级逐日日志，新建）
  5. radar-nwp-bench/.workbuddy/memory/automations/<id>/memory.md（本自动化 memory）

不改 docs/（无新数字、无裁决变化 => MASTER 不升版，仍 v2.50）。
"""
import re
from pathlib import Path

REPO = Path(r"R:/1radarNWPprediction/radar-nwp-bench")
UPPER = Path(r"R:/1radarNWPprediction/.workbuddy/memory")
PROJ = REPO / ".workbuddy" / "memory"
AUTO = PROJ / "automations" / "e3cc6636-3e60-4a58-8f7e-67d4c896b0b6" / "memory.md"

BEGIN = "<!-- E45-R90-BEGIN -->"
END = "<!-- E45-R90-END -->"
MARK = "E45-R90-BEGIN"

BODY = """### 2026-11-16｜第 90 轮（E45 远端运行状态复核 ＋ **S3 护栏数值级重算** ＋ 主跑评测文件路径独立复算）

- **判定**：远端 E45 全部完成、无新活动 ⇒ **裁决不变＝六臂判负、无一臂进十 seed、不投十 seed 算力**。**远端动作＝0**（只读查询：`squeue`/`sacct`/`md5sum`/`find`/`ls`；未 sbatch／未同步／未下载）。
- **远端终态**：权威链 273／275·283（训练数组 7/7+7/7）／274·282·290（stage2）／291（summary）／297·298／314／315·316·317 全 `COMPLETED`；**`squeue` 中 E45 作业数 = 0**（队列里 8 个 `qlfn2-f38` 属同一 `wf` 账号下别的项目线、不干预）；`runs/e45/evals/` **21 份 `.eval.json`**、`runs/e45/` **13 份** run 级 json；`runs/e45` 共 **244** 份文件、**结案后（> 2026-11-11 14:20:42）被修改 = 0 份**；`sacct -S 2026-11-12` 主链 **0 个** e45-train／e45-stage2／e45-submit ⇒ **主链从未重启**；无 E46+ 正式训练目录。**S3 由 summary 291 当年自动跑完 ⇒ 本轮无需手动 sbatch**。
- **产物一致性**：13 份 run 级 json **远端↔本地 md5 13/13 逐位全同**（本轮实测远端 md5 与本地重算对照，非复用登记值）。
- **本轮新增审计面 ①（S3 数值级重算）**：既往轮次读 `crosses_zero` 布尔标志；本轮改由 **CI 端点 `ci_lo ≤ 0 ≤ ci_hi` 的符号包含关系数值级重算**，并逐格核对存储标志与数值是否自洽 ⇒ **12/12 自洽、12/12 区间数值上全部包含 0、合格臂集合为空**；护栏参数逐格核对（thr 2.0／control B0／n_boot 10000／rng 20261109／epoch 00／n_windows 538／n_seeds 3）全部未改写。
- **本轮新增审计面 ②（头条表换数据路径）**：头条表不再取 `scores_all.json`，改从 **21 份主跑 `*.eval.json` 的 `thresholds["2.0"].csi` 逐格解析**复算，并对该路径做**自校验**（21 格与 `scores_all.json` 逐格 max|dev| = 0.000e+00）⇒ 七臂中位数复算 **max|dev| = 0.000e+00**、过地板通过率逐臂一致（B0 2/3、A1 2/3、A2 2/3、A3a 3/3、A3b 3/3、A4 2/3、A4b 2/3）；逐 seed 判「超 Oracle 上界 0.1653」的仍是 **A3b／A4**。
- **本轮新增审计面 ③（失效模式分分支清点）**：`s3_verdict` 是**预注册的两分支文本**，本轮首次按分支清点并核对与 raw 点估符号自洽 ⇒ **4 臂 `GAIN_WAS_AREA_OR_AMPLITUDE`**（A1 raw +0.004190、A3a +0.007576、A3b +0.013653、A4 +0.018037）＋ **2 臂 `NO_GAIN_TO_BEGIN_WITH`**（A2 raw −0.006099、A4b raw −0.017100）；**两分支都是判负**，但**失效模式不同**（前者＝有增益但被等偏置判为面积／幅度效应；后者＝本来就没有增益）⇒ 报告须分分支写，**不得合并成一个「全判负」**。
- **机制判据重判（从 diag_probe 原始 JSON）**：A1／A2 **3/3 过**（`corr(mask,amp)` 与 `mask_at_peak`，门指向已结构上解决）；A3a／A3b **0/3 过**（`field_max_mm` 六格全 402.4288 mm ＝模型 clamp 上限 ⇒ 幅度截断仍未作用在输出场上；梯度集中未解决 ⇒ 拓扑支按预注册结掉）；A4 **3/3 过**、A4b **2/3 过**（种子 2 触发的是**相关性分支**，其 `scale_cv` **3.0188** 为六格最高 ⇒ **E45 全程没有任何一格真的退化成常数**）；`mean(out)=T` 不达格仍只记**实现缺陷**、不当机制结论（A4s1／A4bs0／A4bs1／A4bs2）。
- **文档闸门**：`_check_tables` **PASS**（MASTER／INDEX／E45 实验文件 mismatched=0）；`_check_v245_row` **PASS**；`_e45_docs_numbers` **PASS**；`_check_v248_row` **59/61**（余 2 条属 **SNAPSHOT** 类、按设计过期：版本指针已由 v2.48 推进到 v2.50＝E46 属正常、v2.48 变更日志行后面不再是空行；**CONTENT 类失败 = 0**）⇒ **未改 `docs/`**（无新数字、无裁决变化 ⇒ MASTER 仍 **v2.50**，不升版）。
- **本轮自身失误与修正（记录以防重犯）**：新写的审计脚本首跑报 3 条 FAIL，**全部是脚本自身的错误预期**（把「本地有第 4 份 seed-0 列副本」误判为缺失、用**中位数**而非**逐 seed** 判超 Oracle、误以为 `s3_verdict` 只有一种文本）⇒ 逐条核对原始数据后改正判据，**远端与产物本身无任何问题**。**纪律：新增审计项必须先核对原始布局再写断言，不得照抄既有脚本的预期值。**
- **下一步（不变）**：交接 §4.5 三件实现缺陷修复（① 幅度截断接到输出场上；② 量通道改相对护栏破绝对 `1e-12`；③ `maxratio` 改 `np.nanmedian` ＋ 报可用帧数）＋ 重开须**重新提机制族**，不是派生。
- **产物**：`tools/_e45_round90.py`、`tools/_evals/e45/round90_status_audit.json`。
"""


def append_block(path: Path, header: str = ""):
    """带标记幂等追加：标记已存在则跳过。"""
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("", encoding="utf-8")
    txt = path.read_text(encoding="utf-8")
    if MARK in txt:
        print(f"  SKIP (标记已存在) -> {path}")
        return False
    block = (header + BEGIN + "\n" + BODY + END + "\n") if header else (BEGIN + "\n" + BODY + END + "\n")
    if txt and not txt.endswith("\n"):
        txt += "\n"
    path.write_text(txt + block, encoding="utf-8")
    print(f"  APPENDED -> {path}")
    return True


def append_bullet(path: Path, bullet: str, after_marker: str):
    """在 MEMORY.md 状态段追加 bullet（幂等：bullet 文本已存在则跳过）。"""
    txt = path.read_text(encoding="utf-8")
    if bullet.strip()[:60] in txt:
        print(f"  SKIP (bullet 已存在) -> {path}")
        return False
    lines = txt.split("\n")
    idx = None
    for i, ln in enumerate(lines):
        if after_marker in ln:
            idx = i
            break
    if idx is None:
        raise SystemExit(f"未找到锚点 {after_marker!r} 于 {path} —— 报错，不静默跳过")
    lines.insert(idx + 1, bullet)
    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"  APPENDED bullet -> {path}")
    return True


print("=== 写回（幂等） ===")
append_block(UPPER / "2026-11-11.md")
append_block(UPPER / "HANDOVER_E45.md")
append_block(PROJ / "2026-11-16.md")
append_block(AUTO)

BULLET = ("- **E45 状态复核（第 90 轮，2026-11-16）**：远端权威链全 `COMPLETED`（273／275·283／274·282·290／291／297·298／314–317）、"
          "**`squeue` 中 E45 作业数 = 0**、21 份 `.eval.json`＋13 份 run 级 json 齐、**结案后（> 11-11 14:20:42）被修改文件 = 0 份** ＋ "
          "`sacct -S 2026-11-12` 主链 0 个作业 ⇒ **主链从未重启**；**S3 由 summary 291 当年自动跑完 ⇒ 无需手动 sbatch**；产物 md5 **13/13 逐位全同**（本轮实测）。"
          "**本轮三条新审计面**：① **S3 护栏数值级重算**——不看 `crosses_zero` 布尔、改为由 `ci_lo ≤ 0 ≤ ci_hi` 数值级重算 ⇒ **12/12 自洽且全部跨零**、合格臂集合为空；"
          "② **头条表换数据路径**——改从 21 份主跑 `*.eval.json` 的 `thresholds[\"2.0\"].csi` 复算并自校验（与 `scores_all.json` 21 格 max|dev| = 0）⇒ 七臂中位数 max|dev| = 0、"
          "过地板通过率逐臂一致（B0/A1/A2/A4/A4b 2/3，A3a/A3b 3/3）；③ **失效模式分分支清点**——`s3_verdict` 为**预注册两分支**：**4 臂 `GAIN_WAS_AREA_OR_AMPLITUDE`**"
          "（A1 +0.004190／A3a +0.007576／A3b +0.013653／A4 +0.018037）＋ **2 臂 `NO_GAIN_TO_BEGIN_WITH`**（A2 −0.006099／A4b −0.017100），"
          "**两分支均判负但失效模式不同 ⇒ 报告须分分支写、不得合并**。机制判据重判：A1／A2 3/3、A3a／A3b 0/3（场最大六格全 402.4288 mm）、A4 3/3、A4b 2/3"
          "（种子 2 触发**相关性分支**，其 `scale_cv` **3.0188** 为六格最高 ⇒ 未退化成常数）。文档闸门 tables／v245／docs-numbers 全 PASS、v248 **59/61**（CONTENT 类失败＝0）。"
          "**裁决不变**（六臂判负、不投十 seed 算力）；**未改 `docs/`**（MASTER 仍 v2.50）。")
append_bullet(UPPER / "MEMORY.md", BULLET, "- **E45 状态复核（第 89 轮")

print("\n=== 复核标记唯一性 ===")
for p in [UPPER / "2026-11-11.md", UPPER / "HANDOVER_E45.md", PROJ / "2026-11-16.md", AUTO]:
    n = p.read_text(encoding="utf-8").count(MARK)
    print(f"  {p.name}: {MARK} x{n}")
mb = (UPPER / "MEMORY.md").read_text(encoding="utf-8")
print(f"  MEMORY.md: 第 90 轮 bullet x{mb.count('E45 状态复核（第 90 轮')}")
print("  MEMORY.md 承重引用 scale_cv 3.0188 仍在:", "3.0188" in mb)
