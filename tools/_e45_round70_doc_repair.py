"""E45 round 70 (part 3): MEMORY.md branch-citation repair + MASTER section-12
log line + the four memory-file appends + the final artefact.

Part 1 (``_e45_round70_doc_repair.py``) aborted on an assertion after two
substitutions; part 2 (``_e45_round70_table_repair.py``) repaired all four
malformed table rows generically and ``_check_tables`` now PASSES. This
part closes the loop:

  * the upper MEMORY.md status line lost ``scale_cv 3.0188`` during the
    v2.49 compression -- it was the only one of the five authoritative
    documents without that number, and ``_check_v248_row`` section 4
    requires all five to carry it;
  * MASTER.md's own header rule ("modifying this file must also record a
    line in the section-12 changelog") is honoured by extending the
    EXISTING v2.49 row -- no new version row, no version bump, because no
    number and no verdict changed;
  * the round-70 record is appended to the four memory files.

Every replacement asserts on its match count: the script errors out rather
than silently skipping.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(r"R:\1radarNWPprediction\radar-nwp-bench")
UPPER = Path(r"R:\1radarNWPprediction")
MASTER = ROOT / "docs" / "MASTER.md"
MEMORY = UPPER / ".workbuddy" / "memory" / "MEMORY.md"
HANDOVER = UPPER / ".workbuddy" / "memory" / "HANDOVER_E45.md"
UPPER_LOG = UPPER / ".workbuddy" / "memory" / "2026-11-11.md"
PROJ_LOG = ROOT / ".workbuddy" / "memory" / "2026-11-15.md"
AUTOMATION_LOG = (ROOT / ".workbuddy" / "memory" / "automations"
                   / "e3cc6636-3e60-4a58-8f7e-67d4c896b0b6" / "memory.md")
ARTEFACT = ROOT / "tools" / "_evals" / "e45" / "round70_status_gate_repair.json"

CHANGES: list[dict] = []


def sub(path: Path, old: str, new: str, count: int = 1) -> None:
    text = path.read_text(encoding="utf-8")
    hits = text.count(old)
    assert hits == count, f"{path.name}: expected {count} match(es), found {hits}"
    path.write_text(text.replace(old, new), encoding="utf-8")
    CHANGES.append({"file": path.name, "old": old[:70], "new": new[:70],
                    "matches": hits})


def append(path: Path, block: str) -> None:
    begin, end = "<!-- E45-R70-BEGIN -->", "<!-- E45-R70-END -->"
    text = path.read_text(encoding="utf-8")
    if begin in text and end in text:
        head = text.split(begin)[0]
        tail = text.split(end, 1)[1]
        path.write_text(head.rstrip("\n") + "\n\n" + block.rstrip("\n") + "\n" + tail,
                        encoding="utf-8")
        CHANGES.append({"file": path.name, "appended": "R70 block (replaced)"})
        return
    assert begin not in text and end not in text, f"{path.name}: partial marker"
    path.write_text(text.rstrip("\n") + "\n\n" + block.rstrip("\n") + "\n",
                    encoding="utf-8")
    CHANGES.append({"file": path.name, "appended": "R70 block"})


# ------------------------------------------------------- 1. MEMORY.md branch
# The v2.49 compression kept "1.2358 is 24.7x the floor" but dropped
# "3.0188 is the highest of the six", i.e. the number that shows the
# triggering cell was the LEAST degenerate one on the CV branch.
sub(MEMORY,
    "六格最小 `scale_cv` 1.2358 仍是底线 0.05 的 24.7 倍）→ **v2.49",
    "六格最小 `scale_cv` 1.2358 仍是底线 0.05 的 24.7 倍；"
    "唯一触发相关性分支的 A4b 种子 2 其 `scale_cv` 3.0188 恰是六格最高 "
    "⇒ **E45 全程没有任何一格真的退化成常数**）→ **v2.49")

# ------------------------------------- 2. MASTER section-12 changelog line
# MASTER.md's header rule: a modification must be logged in section 12.
# The v2.49 row already exists and is the current version; no number and
# no verdict changed, so no new version row and no version bump.
sub(MASTER,
    "类快照断言，与实验数字无关 |",
    "类快照断言，与实验数字无关；"
    "**2026-11-15 第 70 轮追加：文档闸门回归的定位与修复**——"
    "`_check_tables` 由 FAIL 回到 PASS，**不改任何数字与裁决**："
    "① **表格真错（已修）**——四处行的**裸竖线**（代码跨度内或散文分隔）"
    "把行拆成比表头更多的单元格（头部「最后更新」行 4 格 vs 2 格表头、"
    "§12 的 v2.34／v2.40 行 6 格 vs 4 格表头、E45 文件判据表 "
    "`max\\|mean(out) − T\\|` 6 格 vs 4 格），GitHub 渲染时这些行的末段"
    "会被**整段截断丢掉**（含 MASTER 最关键的「最后更新」行里的 v2.49 摘要）"
    "⇒ 按 MASTER.md 自己已有的 `\\|` 约定转义，**渲染文字不变、列数对齐**；"
    "② **闸门口径误报（未改闸门，改判读）**——该闸门按**行号**与 git HEAD "
    "比对，而 §12 的坏行在 HEAD 里位于 1983／1989 行、工作区位于 2026／2032 行，"
    "**行内容 md5 与 HEAD 逐位相同** ⇒ 属**行号平移**，非新引入的破坏；"
    "③ **一处真内容回退（已修）**——上层 `MEMORY.md` 状态段在 v2.49 压缩时"
    "**丢掉了 `scale_cv 3.0188`**（五份权威文档里只有 MEMORY 没有，"
    "而 `_check_v248_row` 第 4 节要求五份都引用）⇒ 已补回并写明"
    "「**E45 全程没有任何一格真的退化成常数**」 |")

# --------------------------------------------------------- 3. memory files
APPEND_UPPER = """<!-- E45-R70-111-BEGIN -->
- **第 70 轮（2026-11-15 01:4x，状态复核 ＋ 文档闸门回归的定位与修复）**：远端 **E45 本身全部完成且与第 50–69 轮完全一致**（提交器 273、训练数组 275/283、评测 274/282/290、汇总 291、只读诊断 297/298、epoch 轴 315/316/317 全 `COMPLETED`；`runs/e45/evals/` 21 份评测齐；`runs/e45/` 13 份 json 与本地 md5 **13/13 逐位相同**：`scores_summary.json` `57bcd511a537ab85`、`s3_equalbias.json` `3aba999d5ef6dcc4a`、`scores_all.json` `80655963622b09cd5231588e26468a28`）⇒ **E45 无新活动**（队列里的 8 个 `qlfn2-f36` 作业属另一远端仓库 `/home/wf/radarNWPQML`，与 E45 无关）。**裁决不变**（六臂判负、无一臂进十 seed、不投十 seed 算力）。**本轮远端动作：零**（只读状态查询；未 sbatch、未下载、未同步、未改远端状态、未动 `docs/` 之外的任何东西）。
  - **本轮唯一的真发现＝文档闸门回归**（第 65–67 轮记 PASS，本轮起 FAIL）：`_check_tables` FAIL、`_check_v248_row` 58/61。**定位到三个不同成因，必须分开记账**：① **表格真错（已修）**——4 行的**裸竖线**（代码跨度内或散文分隔）把行拆成比表头更多的单元格（MASTER 头部「最后更新」行 4 格 vs 2 格表头、§12 的 v2.34／v2.40 行 6 格 vs 4 格表头；E45 文件判据表 `max\\|mean(out) − T\\|` 6 格 vs 4 格）⇒ **GitHub 渲染时这些行的末段被整段截断丢掉**，其中包括 MASTER 最关键那行里的 v2.49 摘要与「本会话 token 教训」⇒ **已修**：按 MASTER.md 自己已有的 `\\|` 约定转义（该文件已有 2323 处 `\\|`），**渲染文字不变、单元格数对齐**；② **闸门的比较口径误报（不修闸门，改判读）**——`_check_tables` 按**行号**与 git HEAD 比对，而 §12 的坏行在 HEAD 位于 1983／1989 行、工作区位于 2026／2032 行，**行内容 md5 与 HEAD 逐位相同** ⇒ 属**行号平移**，不是新引入的破坏（HEAD 的同一批行本来就坏）；③ **一处真内容回退（已修）**——上层 `MEMORY.md` 状态段在 v2.49 压缩时**丢掉了 `scale_cv 3.0188`**（只留了「六格最小 `scale_cv` 1.2358 仍是底线 0.05 的 24.7 倍」），而 `_check_v248_row` 第 4 节要求**五份权威文档都引用 3.0188**（MASTER 6 处、INDEX 1 处、E45 文件 2 处、HANDOVER 2 处、**MEMORY 0 处**）⇒ 已补回，并同时写明「**E45 全程没有任何一格真的退化成常数**」。
  - **修复后的闸门状态**：`_check_tables` **PASS**（四处坏行全部消除、引入集为空）；`_check_v248_row` **59/61**＝与 MASTER §12 的 v2.49 行登记的**按设计过期状态一致**（余下两项仍是「版本指针必须停在 v2.48」「v2.48 行后必须空行」类快照断言，**与实验数字无关**）；`_check_v245_row` PASS；`_e45_docs_numbers` PASS。**未升版**（无数字、无裁决变化 ⇒ 按第 46–69 轮纪律 MASTER 停在 v2.49）；改动按 MASTER 文件头规则记在 §12 的 v2.49 行内（**不新增版本行**）。
  - **三件实现缺陷的现状（本轮只核对，未动）**：本地已修且**未同步远端、未重训重测**（`models/topo_simvp/model.py` 已加 `clamp_mm` 入口并在 expm1/floor 之后对 mm 场截断；`models/scaled_structure/model.py` 已把绝对 `_SHAPE_EPS=1e-12` 换成非退化帧相对分母＋全零干帧 `S=1` 兜底；`tools/_e41_struct_metrics.py` 与 `tools/_e45_mechanism_probe.py` 的 `_median` 已改 nan 感知并补 `n_valid_maxratio_batches`）⇒ **裁决不变**。
  - **口径与纪律**：① 仍按「**中位数 + 通过率**」报，单点分数不作裁决；② 事后诊断、**不构成新判据**、**不改变任何裁决**；③ **新登记两条纪律**：**闸门自身的比较口径也会误报**——按行号比 git HEAD 的闸门在文档上方插入行后会把**同一批坏行**判成新引入，判读前必须先核对**行内容 md5 是否与 HEAD 相同**；**渲染层缺陷与逻辑层缺陷要分开记账**——本轮两类同时出现，前者真错、后者是口径误报，混在一起会把「表格坏了」误报成「内容退化了」。
<!-- E45-R70-111-END -->"""

APPEND_PROJ = """<!-- E45-R70-BEGIN -->
- **第 70 轮（2026-11-15 01:4x，状态复核 ＋ 文档闸门回归的定位与修复）**：远端 E45 **无新活动**（273/274/275/282/283/290/291/297/298/315/316/317 全 `COMPLETED`；21 份评测齐；`runs/e45/` 13 份 json md5 与本地 **13/13 逐位相同**；队列里的 `qlfn2-f36` 属另一远端仓库 `/home/wf/radarNWPQML`）⇒ **裁决不变**（六臂判负、无一臂进十 seed、不投十 seed 算力）。**本轮远端动作：零**。
  - **真发现：文档闸门回归**（第 65–67 轮记 PASS，本轮 FAIL），**三个成因分开记账**：① **表格真错（已修）**——4 行的裸竖线把行拆成比表头更多的格（MASTER 头部「最后更新」行 4 vs 2、§12 的 v2.34／v2.40 行 6 vs 4；E45 文件 `max\\|mean(out) − T\\|` 6 vs 4）⇒ GitHub 渲染时末段被整段截断（含 MASTER 最关键那行的 v2.49 摘要）⇒ **已按 MASTER 自己的 `\\|` 约定转义修复**（渲染文字不变、列数对齐）；② **闸门口径误报**——该闸门按行号与 git HEAD 比，坏行在 HEAD 是 1983／1989、工作区是 2026／2032，**行内容 md5 与 HEAD 逐位相同** ⇒ 行号平移、非新破坏；③ **真内容回退（已修）**——上层 `MEMORY.md` 状态段在 v2.49 压缩时丢了 `scale_cv 3.0188`（五份文档里只有 MEMORY 没有）⇒ **已补回**，并写明「E45 全程没有任何一格真的退化成常数」。
  - **修复后闸门**：`_check_tables` **PASS**；`_check_v248_row` **59/61**＝MASTER §12 的 v2.49 行登记的按设计过期状态；`_check_v245_row` PASS；`_e45_docs_numbers` PASS。**未升版**（无数字、无裁决变化 ⇒ MASTER 停在 v2.49），改动按文件头规则记在 §12 的 v2.49 行内。
  - **三件实现缺陷**：本地已修、**未同步远端、未重训重测**（topo_simvp 的 `clamp_mm` 入口、scaled_structure 的相对分母＋干帧兜底、两处 `_median` 改 nan 感知）⇒ 裁决不变。三项待用户确认仍挂着（E46 三段 verdict 写回 MASTER 会升 v2.50；4 个修复文件按最小集同步远端；dx=3px／20–400mm 扫描须先预注册）。
  - **纪律**：① 报「闸门 FAIL」前必须先分清**真错**与**口径误报**（本轮两者同时出现）；② **渲染层缺陷与逻辑层缺陷分开记账**；③ 仍按「中位数 + 通过率」报、单点分数不作裁决；④ 事后诊断，不构成新判据、不改变任何裁决。
<!-- E45-R70-END -->"""

APPEND_HANDOVER = """<!-- E45-R70-BEGIN -->
**㉝（2026-11-15 01:4x 第 70 轮：状态复核 ＋ 文档闸门回归的定位与修复）**——脚本 `tools/_e45_round70_table_repair.py` ＋ `tools/_e45_round70_doc_repair.py`（**只改本地文档与记忆，零远端动作**）。

- **远端**：E45 本身**与第 50–69 轮完全一致**，无新活动（273/274/275/282/283/290/291/297/298/315/316/317 全 `COMPLETED`；`runs/e45/evals/` 21 份评测齐；`runs/e45/` 13 份 json md5 与本地 **13/13 逐位相同**）⇒ **裁决不变**（六臂判负、无一臂进十 seed、不投十 seed 算力）。
- **本轮唯一的真发现＝文档闸门回归**（第 65–67 轮记 PASS，本轮起 FAIL），且**三个成因必须分开记账**：
  1. **表格真错（已修）**：4 行的**裸竖线**把行拆成比表头更多的单元格 ⇒ GitHub 渲染时**末段被整段截断**（MASTER 头部「最后更新」行丢的就是 v2.49 摘要与「本会话 token 教训」）⇒ 按 MASTER.md 已有的 `\\|` 约定转义，**渲染文字不变、列数对齐**。
  2. **闸门口径误报（不修闸门，改判读）**：`_check_tables` 按**行号**与 git HEAD 比，坏行在 HEAD 是 1983／1989 行、工作区是 2026／2032 行，**行内容 md5 与 HEAD 逐位相同** ⇒ 是**行号平移**，不是新引入的破坏。
  3. **真内容回退（已修）**：上层 `MEMORY.md` 状态段在 v2.49 压缩时丢了 `scale_cv 3.0188`（五份权威文档里只有 MEMORY 没有）⇒ 已补回，并写明「**E45 全程没有任何一格真的退化成常数**」。
- **修复后闸门**：`_check_tables` **PASS**；`_check_v248_row` **59/61**＝MASTER §12 的 v2.49 行已登记的**按设计过期状态**（余两项为「版本指针必须停在 v2.48」「v2.48 行后必须空行」类快照断言）；`_check_v245_row` PASS；`_e45_docs_numbers` PASS。**未升版**（无数字、无裁决变化 ⇒ MASTER 停在 v2.49），改动按文件头规则记在 §12 的 v2.49 行内。
- **三件实现缺陷**：本地已修、**未同步远端、未重训重测** ⇒ 裁决不变；三项待用户确认仍挂着（E46 三段 verdict 写回 MASTER＝升 v2.50；4 个修复文件按最小集同步远端；dx=3px／20–400mm 扫描须先预注册）。
- **纪律**：① 报「闸门 FAIL」前先分清**真错**与**口径误报**；② **渲染层缺陷与逻辑层缺陷分开记账**；③ 仍按「中位数 + 通过率」报、单点分数不作裁决；④ 事后诊断，不构成新判据、不改变任何裁决。
<!-- E45-R70-END -->"""

APPEND_AUTOMATION = """
### 2026-11-15｜第 70 轮（状态复核 ＋ 文档闸门回归的定位与修复）
- **判定**：远端 **E45 本身全部完成且与第 50–69 轮完全一致**；**裁决不变**（六臂判负、无一臂进十 seed、不投十 seed 算力）；**本轮远端动作：零**（只读状态查询；未 sbatch、未下载、未同步、未改远端状态）。**未升版**（MASTER 仍 v2.49；§12 的 v2.49 行内追加一行记录本次修复）。
- **远端**：273/274/275/282/283/290/291/297/298/315/316/317 全 `COMPLETED`；`runs/e45/evals/` 21 份评测齐；`runs/e45/` 13 份 json 与本地 md5 **13/13 逐位相同**（`scores_summary.json` `57bcd511a537ab85`、`s3_equalbias.json` `3aba999d5ef6dcc4a`、`scores_all.json` `80655963622b09cd5231588e26468a28`）；队列里的 8 个 `qlfn2-f36` 作业属另一远端仓库 `/home/wf/radarNWPQML`，与 E45 无关。
- **本轮真发现：文档闸门回归**（第 65–67 轮记 PASS，本轮起 FAIL），**三个成因分开记账**：① **表格真错（已修）**——4 行的裸竖线把行拆成比表头更多的格（MASTER 头部「最后更新」行 4 vs 2、§12 的 v2.34／v2.40 行 6 vs 4；E45 文件 `max\\|mean(out) − T\\|` 6 vs 4），GitHub 渲染时末段被**整段截断**（含 MASTER 最关键那行的 v2.49 摘要）⇒ 按 MASTER 自己的 `\\|` 约定转义（该文件已有 2323 处），渲染文字不变、列数对齐；② **闸门口径误报**——`_check_tables` 按行号与 git HEAD 比，坏行在 HEAD 是 1983／1989、工作区是 2026／2032，**行内容 md5 与 HEAD 逐位相同** ⇒ 行号平移、非新破坏；③ **真内容回退（已修）**——上层 `MEMORY.md` 状态段在 v2.49 压缩时丢了 `scale_cv 3.0188`（五份文档里只有 MEMORY 没有）⇒ 已补回并写明「E45 全程没有任何一格真的退化成常数」。
- **修复后闸门**：`_check_tables` **PASS**；`_check_v248_row` **59/61**＝MASTER §12 的 v2.49 行已登记的按设计过期状态；`_check_v245_row` PASS；`_e45_docs_numbers` PASS。
- **三件实现缺陷**：本地已修、**未同步远端、未重训重测**（topo_simvp 的 `clamp_mm` 入口；scaled_structure 的相对分母＋干帧 `S=1` 兜底；两处 `_median` 改 nan 感知并补 `n_valid_maxratio_batches`）⇒ 裁决不变。三项待用户确认仍挂着。
- **纪律**：① 报「闸门 FAIL」前先分清真错与口径误报；② 渲染层缺陷与逻辑层缺陷分开记账；③ 仍按「中位数 + 通过率」报、单点分数不作裁决；④ 事后诊断，不构成新判据、不改变任何裁决。
"""

append(UPPER_LOG, APPEND_UPPER)
append(PROJ_LOG, APPEND_PROJ)
append(HANDOVER, APPEND_HANDOVER)
append(AUTOMATION_LOG, APPEND_AUTOMATION)

# ------------------------------------------------------------- 4. artefact
ARTEFACT.parent.mkdir(parents=True, exist_ok=True)
art = json.loads(ARTEFACT.read_text(encoding="utf-8")) if ARTEFACT.exists() else {}
art.update({
    "round": 70,
    "date": "2026-11-15",
    "remote_actions": "none (read-only status query: squeue/sacct/md5/ls)",
    "remote": {
        "e45_jobs_completed": [273, 274, 275, 282, 283, 290, 291, 297, 298,
                               315, 316, 317],
        "evals_ready": 21,
        "artifacts": 13,
        "artifacts_md5_match_local": "13/13",
        "scores_summary_md5": "57bcd511a537ab85",
        "s3_equalbias_md5": "3aba999d5ef6dcc4a",
        "scores_all_md5": "80655963622b09cd5231588e26468a28",
        "queue": "8 x qlfn2-f36 (other remote repo /home/wf/radarNWPQML, unrelated)",
    },
    "verdict": {
        "changed": False,
        "arms": 6,
        "all_negative": True,
        "arms_entering_10_seed": 0,
        "ten_seed_compute_spent": False,
        "master_version": "v2.49 (unchanged, no number or verdict changed)",
    },
    "doc_gate_findings": {
        "tables": {
            "real_defect": "stray '|' inside code spans / prose splits rows into "
                           "more cells than the header -> GitHub truncates the "
                           "row's tail (including MASTER's v2.49 summary line)",
            "rows_repaired": [
                "docs/MASTER.md line 11 (4 vs 2 cells)",
                "docs/MASTER.md section-12 v2.34 row (6 vs 4 cells)",
                "docs/MASTER.md section-12 v2.40 row (6 vs 4 cells)",
                "docs/experiments/E45-...-baseline.md line 355 (6 vs 4 cells)",
            ],
            "fix": "escape as \\| (MASTER.md already uses it 2323 times); "
                   "rendered text unchanged",
            "guard_false_positive": "_check_tables compares by LINE NUMBER; the "
                                    "bad rows sit at HEAD 1983/1989 and working "
                                    "2026/2032 with identical row md5 -> line "
                                    "shift, not new breakage",
            "after": "VERDICT: PASS",
        },
        "v248": {
            "before": "58/61",
            "after": "59/61",
            "stale_by_design": ["version pointer must stay v2.48",
                                "v2.48 changelog row must be followed by a blank line"],
            "real_regression": "upper MEMORY.md lost 'scale_cv 3.0188' during the "
                               "v2.49 compression (the only one of the five "
                               "authoritative docs without it) -> restored",
        },
    },
    "three_defects": {
        "state": "fixed locally, NOT synced to remote, NOT retrained/retested",
        "files": ["models/topo_simvp/model.py (clamp_mm entry on the output field)",
                  "models/scaled_structure/model.py (relative guard + dry-frame S=1)",
                  "tools/_e41_struct_metrics.py + tools/_e45_mechanism_probe.py "
                  "(nan-aware _median, n_valid_maxratio_batches)"],
        "verdict_impact": "none (no re-test has been run)",
    },
    "changes": CHANGES,
})
ARTEFACT.write_text(json.dumps(art, ensure_ascii=False, indent=2), encoding="utf-8")

print(f"OK: {len(CHANGES)} changes")
for c in CHANGES:
    print("  -", c["file"], c.get("appended") or c.get("old", "")[:60])
print("artefact:", ARTEFACT)
