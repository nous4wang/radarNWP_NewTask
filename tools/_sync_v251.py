"""Write back the round-92 finding (v2.51).

What v2.51 records
------------------
Round 92 re-derived the E45 verdict from the raw artefacts (POS 89/89)
and confirmed it is unchanged: six arms negative, no arm into the
10-seed line, no ten-seed compute.  While doing so it found something
the previous ninety-one rounds never looked at, because they never
looked at it:  **the local fix for defect (1) is not reachable.**

MASTER v2.50 records that ``models/topo_simvp/model.py`` now takes a
``clamp_mm`` and clamps the millimetre field after the expm1/floor
transform.  That is the MODEL-level half.  The path that would carry a
value from the E45 arm table into that constructor does not exist:

* ``configs/models/topo_simvp.yaml`` does not carry ``clamp_mm``, so
  ``model_kwargs()`` never passes it;
* ``--topo-clamp-mm`` is consumed by the LOSS (``TopoCriterion``) and
  never reaches the model constructor;
* the arm table's A3a row still says ``--topo-clamp-mm 20`` and the
  table passes no ``--model-kwargs`` at all;
* the evaluation side restores only ``head_bins`` and ``level_mode``
  from the checkpoint's ``config.json`` -- neither ``clamp_mm`` nor
  ``model_kwargs`` comes back;
* ``clamp_mm`` is a plain float attribute, so it does NOT change
  ``state_dict``: a checkpoint trained with the clamp loads cleanly into
  a model without it and is then scored on the UNCLAMPED field.

Consequently, re-running A3a today would reproduce the E45 result
verbatim (field maximum still 402.43 mm).  Syncing the four fixed files
is necessary but NOT sufficient: the training side and the evaluation
side must both carry the value, or the round is wasted.

The same round also established that the queued seed-robustness job
(Slurm 343) is ready to run -- script, runner, all 21 window tables,
all 21 dumped prediction/truth pairs and the guard-rail module are in
place, the guard parameters match the stored ones, and the local and
remote copies of the guard rail differ only in line endings (the
normalised md5 is identical, so the queued job imports the very
implementation that produced the published numbers).  Its only
blocker is the account quota shared with another project line.

Scope of this write-back
------------------------
* MASTER.md -- header version line, the section 4.47 card row, the
  section 4.48 status bullet, the section 9 row, a section 12
  changelog row (the five places Gate 3 names, plus the one place
  that carries the three fixes);
* INDEX.md -- the E45 row;
* MEMORY.md -- the version pointer, the status section, the to-do item;
* HANDOVER_E45.md -- the version pointer and the to-do item;
* both daily logs -- the round note.

The historical section 12 rows (v2.44-v2.50) are deliberately NOT
rewritten: they are the audit trail.  Every replacement is anchored on
a literal that must occur exactly the expected number of times; a
missing anchor is an error, never a silent skip.  The new section 12
row is built from its CELLS; the separators are inserted by the code
and the pipe count is compared against the row it follows.  Running
the script twice is a no-op.
"""
from pathlib import Path
import re
import sys

MASTER = Path("docs/MASTER.md")
INDEX = Path("docs/experiments/INDEX.md")
E45DOC = Path("docs/experiments/E45-gate-direction-amount-separation-baseline.md")
MEMORY = Path("R:/1radarNWPprediction/.workbuddy/memory/MEMORY.md")
HANDOVER = Path("R:/1radarNWPprediction/.workbuddy/memory/HANDOVER_E45.md")
DAILY = Path("R:/1radarNWPprediction/.workbuddy/memory/2026-11-11.md")
DAILY_PROJECT = Path(".workbuddy/memory/2026-11-16.md")

TAG = "v2.51"
OLD_TAG = "v2.50"
#: A distinctive marker for the idempotency guard.  The bare version string
#: is NOT usable: earlier rounds left the text "（＝升 v2.51）" in the
#: handover and the daily log (a pending user decision that said the next
#: write-back would bump to v2.51), and a guard on the bare string would
#: skip both files WITHOUT writing anything -- a silent false positive.
MARKER = "v2.51 第 92 轮"
#: Per-file markers, because the bare version string is ambiguous: earlier
#: rounds left "（＝升 v2.51）" in several files, and a guard on it would
#: skip them WITHOUT writing anything -- a silent false positive.
MARK_MASTER = "| 2026-11-16 | **v2.51** |"
MARK_INDEX = "v2.51 第 92 轮"
MARK_MEMORY = "v2.51 第 92 轮"
MARK_HANDOVER = "v2.51 追加（可达性缺口，重开前必须先补）"
MARK_DAILY = "## v2.51 第 92 轮（本次运行"
MARK_E45DOC = "v2.51 追加（可达性缺口，重开前必须先补）"


def row(*cells):
    """One markdown table row from its cells (separators inserted by code)."""
    return "| " + " | ".join(c.strip() for c in cells) + " |"


def unescaped_pipes(line):
    return len(re.findall(r"(?<!\\)\|", line))


def apply(text, old, new, count=1, label=""):
    """Replace `old` with `new`, exactly `count` times (idempotent)."""
    n_old = text.count(old)
    n_new = text.count(new)
    if n_old == count:
        if n_new:
            raise SystemExit(f"BOTH ANCHORS PRESENT ({label}): old={n_old} new={n_new}")
        return text.replace(old, new)
    if n_new >= 1 and n_old == 0:
        return text  # already applied by an earlier run
    raise SystemExit(
        f"ANCHOR MISS ({label}): expected {count} occurrence(s), found {n_old}"
        f" (new text present {n_new}x)\n  {old[:90]!r}"
    )


def replace_in_line(text, line_prefix, old, new, label=""):
    """Replace `old` by `new` inside the single line starting with `line_prefix`."""
    lines = text.split("\n")
    hits = [i for i, l in enumerate(lines) if l.startswith(line_prefix)]
    if len(hits) != 1:
        raise SystemExit(
            f"LINE ANCHOR MISS ({label}): prefix {line_prefix[:60]!r} "
            f"matched {len(hits)} line(s), expected 1")
    i = hits[0]
    if new in lines[i] and old not in lines[i]:
        return text          # already applied
    if lines[i].count(old) != 1:
        raise SystemExit(
            f"ANCHOR MISS ({label}): {old[:80]!r} occurs "
            f"{lines[i].count(old)}x in the target line, expected 1")
    lines[i] = lines[i].replace(old, new, 1)
    return "\n".join(lines)


def append_to_line(text, line_prefix, addition, label=""):
    """Append `addition` to the single line starting with `line_prefix`."""
    lines = text.split("\n")
    hits = [i for i, l in enumerate(lines) if l.startswith(line_prefix)]
    if len(hits) != 1:
        raise SystemExit(
            f"LINE ANCHOR MISS ({label}): prefix {line_prefix[:60]!r} "
            f"matched {len(hits)} line(s), expected 1")
    i = hits[0]
    if addition in lines[i]:
        return text          # already applied
    lines[i] = lines[i] + addition
    return "\n".join(lines)


# --------------------------------------------------------------------------
# the new statement, written once
# --------------------------------------------------------------------------
REACHABILITY = (
    "**v2.51 新发现（可达性缺口）**：缺陷① 的修复只落了**模型层**——`models/topo_simvp/model.py` "
    "确有 `clamp_mm` 入口并在 `expm1/floor_at_zero` 之后对毫米场 `clamp(max=clamp_mm)`；"
    "**但从 E45 臂表到该构造参数没有任何通路**：① `configs/models/topo_simvp.yaml` **不带** "
    "`clamp_mm` ⇒ `model_kwargs()` 永不传它；② `--topo-clamp-mm` 只被**损失侧**"
    "（`TopoCriterion`）消费，**不进模型构造**；③ 臂表 A3a 仍是 `--topo-clamp-mm 20`、"
    "全表**不传** `--model-kwargs`；④ 评测端从 checkpoint 的 `config.json` 只复原 "
    "`head_bins`／`level_mode`，**不复原** `clamp_mm`／`model_kwargs`；⑤ `clamp_mm` 是普通 float "
    "属性、**不改 state_dict** ⇒ 即便训练侧打开，评测漏传也会**顺利加载 checkpoint 并按未截断的场打分**"
    "（正是本项目吃过的亏：运行时开关不改变 state_dict ⇒ 评测端漏传 ⇒ 静默用回被关掉的机制）。"
    "⇒ **照现状重跑 A3a 会逐位复现 E45 的结果**（场最大仍 402.43 mm）；"
    "**同步那 4 个文件是必要但不充分**：训练侧与评测侧**必须同时**携带该值"
    "（或等价地写进 yaml 并让评测端复原），否则白跑一轮。"
)

# --------------------------------------------------------------------------
# MASTER
# --------------------------------------------------------------------------
HEADER_OLD = (
    "；**v2.50（2026-11-15，最新）**：用户裁决把 E46 立项前预探针与三件实现缺陷修复写回 §4.48"
    "（三段 verdict 全 FAIL ⇒ 不投训练算力），并预登记新排序／y_hist 探针判据；"
    "**E45 裁决不变＝六臂判负、不投十 seed 算力** |"
)
HEADER_NEW = (
    "；**v2.50（2026-11-15）**：用户裁决把 E46 立项前预探针与三件实现缺陷修复写回 §4.48"
    "（三段 verdict 全 FAIL ⇒ 不投训练算力），并预登记新排序／y_hist 探针判据；"
    "**v2.51（2026-11-16，最新）**：E45 第 92 轮从原始产物**独立重导裁决**（POS 89/89）"
    "⇒ **裁决不变＝六臂判负、不投十 seed 算力**；**新发现缺陷① 修复的可达性缺口**——"
    "模型层有 `clamp_mm` 入口、但臂表→构造参数无通路、评测端也不复原它 ⇒ "
    "**照现状重跑 A3a 会逐位复现 E45 结果**；同时**预检了排队中的作业 343**"
    "（脚本／runner／21 份窗表／21 对场／护栏实现齐备、参数与已存一致、护栏本地与远端**仅差行尾**）"
    "⇒ 其唯一阻塞是**同账号配额**（另项目线的 8 个 `qlfn2-f38`，不干预），配额释放即可跑、**无需重投** |"
)

CARD_PREFIX = "| §4.47 | 修三处机制缺陷 + 把比较基准摆正"
CARD_OLD = (
    "**裁决不变**（A4b 种子 2 的标志仍为真 ⇒ 预注册舍弃照样触发）。 |"
    " `docs/experiments/E45-gate-direction-amount-separation-baseline.md`；"
    "v2.49 压缩总览见 §4.47；v2.50 见 §4.48 |"
)
CARD_NEW = (
    "**裁决不变**（A4b 种子 2 的标志仍为真 ⇒ 预注册舍弃照样触发）。｜ **v2.51**："
    "第 92 轮从原始产物**独立重导裁决**（POS 89/89）⇒ **裁决不变**；"
    "并**首次核到缺陷① 修复的可达性缺口**（模型层有 `clamp_mm` 入口、但臂表→构造参数无通路、"
    "评测端也不复原它 ⇒ 照现状重跑 A3a 会逐位复现 E45 的结果）⇒ "
    "**同步 4 个文件是必要但不充分**。 |"
    " `docs/experiments/E45-gate-direction-amount-separation-baseline.md`；"
    "v2.49 压缩总览见 §4.47；v2.50 见 §4.48；v2.51 见 §4.48 状态行 |"
)

S448_STATUS_PREFIX = "- **状态**：4 个修复文件只在本地"
S448_STATUS_OLD = "最小同步集仍是这 4 个文件。"
S448_STATUS_NEW = (
    "最小同步集仍是这 4 个文件。" + REACHABILITY +
    " **v2.51 另记**：排队中的作业 343（`scripts/e45_s3_seedrobust.sbatch`）已预检——"
    "脚本与 runner 已同步、21 份窗表与 21 对导出场在位、护栏参数与已存一致、"
    "且 `tools/_e41_s3_equalbias.py` 本地与远端**仅差行尾（归一化 md5 相同）**"
    "⇒ 它导入的正是产出已发布数字的那份实现 ⇒ **唯一阻塞是同账号配额**，配额释放即可跑、**无需重投**。"
)

S9_PREFIX = "| **E45 / §4.47 修三处机制缺陷 + 摆正基准** |"
S9_OLD = (
    "｜ 交接：`HANDOVER_E45.md`；v2.49 压缩总览见 §4.47；v2.50 见 §4.48 |"
)
S9_NEW = (
    "｜ 交接：`HANDOVER_E45.md`；v2.49 压缩总览见 §4.47；v2.50 见 §4.48；"
    "v2.51 见 §4.48 状态行（缺陷① 修复的**可达性缺口**） |"
)

CHANGELOG_ROW = row(
    "2026-11-16",
    f"**{TAG}**",
    "**E45 第 92 轮：裁决从原始产物独立重导（不变）＋ 缺陷① 修复的可达性缺口（首次核到）**："
    "远端主链终态不变（提交器 273／训练数组 275·283／评测 274·282·290／汇总 291 全 `COMPLETED`、"
    "21 份 `.eval.json` ＋ 13 份 run 级 json 齐、**结案后（> 11-11 14:20:42）被修改文件 = 0 份**）"
    "⇒ **无新活动**。**远端动作 = 0**（只读查询；未下载、未 sbatch、未改远端状态）。"
    "**产物一致性**：13 份 run 级 json 远端↔本地 md5 **13/13 逐位全同**（本轮实测）。"
    "**裁决独立重导**（`tools/_e45_round92.py`，只读本地，**POS 89/89 VERDICT: PASS**）："
    "① 头条表由 **21 份主跑 `*.eval.json`** 的 `thresholds[\"2.0\"].csi` 复算 ⇒ 七臂中位数与"
    "过地板通过率**逐臂一致**（B0/A1/A2/A4/A4b 2-3，A3a/A3b 3-3）；"
    "② **3/3 过地板的只有 A3a／A3b**：A3a 中位数 0.110015、A3b 中位数 0.116055，"
    "**两者均低于同批对照 B0 的 0.121817**、且都低于 Oracle 上界 0.1653 ⇒ **不晋级**；③ S3 护栏参数未改写、**12/12 区间跨零**、"
    "**合格臂集合为空**、失效模式**分两支**（`GAIN_WAS_AREA_OR_AMPLITUDE` 4 臂／"
    "`NO_GAIN_TO_BEGIN_WITH` 2 臂，**不合并**）；④ 机制判据逐格重判＝**与已登记结论完全一致**"
    "（A1／A2 各 3/3 过；A3a 梯度集中 1/3、A3b 0/3 且两臂 `field_max_mm` **0/3 过**、六格全 402.4288 mm；"
    "A4 3/3 过；A4b 2/3 过、**种子 2 触发相关性分支**且其 `scale_cv` 3.0188 为六格最高 "
    "⇒ **量通道全程未退化成常数**）。"
    "**本轮新发现（这是 91 轮来第一次核「修复能不能生效」）**：缺陷① 的修复**只落了模型层**——"
    "`models/topo_simvp/model.py` 确有 `clamp_mm` 入口并在 `expm1/floor_at_zero` 之后对毫米场 "
    "`clamp(max=clamp_mm)`；**但从 E45 臂表到该构造参数没有任何通路**："
    "`configs/models/topo_simvp.yaml` 不带 `clamp_mm`（⇒ `model_kwargs()` 永不传它）；"
    "`--topo-clamp-mm` 只被**损失侧** `TopoCriterion` 消费、**不进模型构造**；"
    "臂表 A3a 仍是 `--topo-clamp-mm 20` 且全表**不传** `--model-kwargs`；"
    "评测端从 checkpoint 的 `config.json` 只复原 `head_bins`／`level_mode`、"
    "**不复原** `clamp_mm`／`model_kwargs`；而 `clamp_mm` 是普通 float 属性、**不改 state_dict** "
    "⇒ 即便训练侧打开，评测漏传也会**顺利加载 checkpoint 并按未截断的场打分**"
    "（本项目吃过的亏：运行时开关不改变 state_dict ⇒ 评测端漏传 ⇒ 静默用回被关掉的机制）。"
    "⇒ **照现状重跑 A3a 会逐位复现 E45 的结果**（场最大仍 402.43 mm）"
    "⇒ **同步那 4 个文件是必要但不充分**：训练侧与评测侧**必须同时**携带该值，"
    "否则重开 E45 的第一候选修复点会**白跑一轮**。"
    "**作业 343 预检**（排队中，`QOSMaxJobsPerUserLimit`：队列里 8 个 `qlfn2-f38` 属同一 `wf` 账号下的"
    "**别的项目线**，不干预）：脚本 `scripts/e45_s3_seedrobust.sbatch` 与 runner "
    "`tools/_e45_s3_seedrobust.py` **已同步**、21 份窗表与 21 对导出场**在位**、"
    "护栏参数（thr 2.0／n_boot 10000／rng 20261109／epoch 00／control B0）与已存**一致**、"
    "且 `tools/_e41_s3_equalbias.py` 本地与远端**仅差行尾（归一化 md5 相同）** "
    "⇒ 它导入的正是产出已发布数字的那份实现，其「注册种子复现已存端点」自检**有效** "
    "⇒ **唯一阻塞是配额**，释放即可跑、**无需重投**。"
    "**裁决零变化**（六臂仍全部判负、无一臂进十 seed、**不投十 seed 算力**）；"
    "**下一步（变更一处）**：§4.5 待办② 补上**训练侧与评测侧同时携带 `clamp_mm`** 的接线，"
    "其余三件缺陷修复与「重开须重新提机制族」不变。",
    "`tools/_e45_round92.py` · `tools/_evals/e45/round92_status_audit.json` · "
    "`docs/experiments/E45-gate-direction-amount-separation-baseline.md` · "
    "`scripts/e45_s3_seedrobust.sbatch` · `tools/_e45_s3_seedrobust.py`",
)

# --------------------------------------------------------------------------
# INDEX
# --------------------------------------------------------------------------
INDEX_PREFIX = "| E45 | §4.47（v2.49 压缩总览） |"
INDEX_OLD = (
    "｜ **v2.47 数字对账**：文档里的数字已与原始产物逐格对账（远端从 21 份评测报告独立重算，"
    "再逐格核对三张表）⇒ 查出并订正 §8.2 seed-0 表 A4 `CSI@2.0` 的末位笔误"
    "（0.175501 → 0.175500），**裁决不变** |"
)
INDEX_NEW = (
    INDEX_OLD[:-1] +
    " ｜ **v2.51 第 92 轮**：裁决从 21 份主跑 `*.eval.json` **独立重导**（POS 89/89）"
    "⇒ **裁决不变＝六臂判负、不投十 seed 算力**；**新发现缺陷① 修复的可达性缺口**——"
    "模型层有 `clamp_mm` 入口但**臂表→构造参数无通路、评测端也不复原** ⇒ "
    "**照现状重跑 A3a 会逐位复现 E45 的结果**（同步 4 个文件必要但不充分）；"
    "**作业 343 已预检就绪**，唯一阻塞是同账号配额 |"
)

# --------------------------------------------------------------------------
# E45 experiment file (version pointer + the defect-(1) to-do)
# --------------------------------------------------------------------------
E45DOC_PTR_OLD = (
    "> **2026-11-15（v2.50）**：**E45 内容与判据自 v2.49 起未变**（S0–S6 一字未改；"
    "裁决不变：六臂判负、不投十 seed 算力）；MASTER 已至 v2.50（新增 §4.48 属 E46，与本文件无关）。"
)
E45DOC_PTR_NEW = (
    "> **2026-11-16（v2.51）**：**E45 内容与判据自 v2.49 起未变**（S0–S6 一字未改；"
    "裁决不变：六臂判负、不投十 seed 算力）；第 92 轮从原始产物独立重导裁决（POS 89/89）"
    "**裁决不变**，并**首次核到缺陷① 修复的可达性缺口**（模型层有 `clamp_mm` 入口、"
    "但臂表→构造参数无通路、评测端也不复原它 ⇒ 照现状重跑 A3a 会逐位复现 E45 的结果）"
    "⇒ **同步 4 个文件是必要但不充分**（详见 §8.4 三件待办第 1 条）。\n"
    "> **2026-11-15（v2.50）**：**E45 内容与判据自 v2.49 起未变**（S0–S6 一字未改；"
    "裁决不变：六臂判负、不投十 seed 算力）；MASTER 已至 v2.50（新增 §4.48 属 E46，与本文件无关）。"
)

E45DOC_TODO_PREFIX = "  ⇒ 下一轮若重开，这是第一个候选修复点"
E45DOC_TODO_OLD = (
    "  ⇒ 下一轮若重开，这是第一个候选修复点（把截断作用在输出场上，而不是只作用在拓扑损失里）。"
)
E45DOC_TODO_NEW = (
    "  ⇒ 下一轮若重开，这是第一个候选修复点（把截断作用在输出场上，而不是只作用在拓扑损失里）。\n"
    "  **v2.51 追加（可达性缺口，重开前必须先补）**：模型层的 `clamp_mm` 入口已落地"
    "（`models/topo_simvp/model.py` 在 `expm1/floor_at_zero` 之后对毫米场 `clamp(max=clamp_mm)`），"
    "**但从 E45 臂表到该构造参数没有任何通路**：`configs/models/topo_simvp.yaml` 不带 `clamp_mm`"
    "（⇒ `model_kwargs()` 永不传它）；`--topo-clamp-mm` 只被**损失侧** `TopoCriterion` 消费、"
    "**不进模型构造**；臂表 A3a 仍是 `--topo-clamp-mm 20` 且全表**不传** `--model-kwargs`；"
    "评测端从 checkpoint 的 `config.json` 只复原 `head_bins`／`level_mode`、**不复原** "
    "`clamp_mm`／`model_kwargs`；而 `clamp_mm` 是普通 float 属性、**不改 state_dict** "
    "⇒ 即便训练侧打开，评测漏传也会**顺利加载 checkpoint 并按未截断的场打分** "
    "（本项目吃过的亏：运行时开关不改变 state_dict ⇒ 评测端漏传 ⇒ 静默用回被关掉的机制）。"
    "⇒ **照现状重跑 A3a 会逐位复现 E45 的结果**（场最大仍 402.43 mm）"
    "⇒ **训练侧与评测侧必须同时携带该值**（或等价地写进 yaml 并让评测端复原），否则白跑一轮。\n"
)

# --------------------------------------------------------------------------
# MEMORY.md
# --------------------------------------------------------------------------
MEM_PTR_OLD = "> 权威 `docs/MASTER.md` **v2.50**（§4.47＝全部进展压缩总览，§4.48＝E46 预探针、三件缺陷修复与新探针判据先写后跑）；"
MEM_PTR_NEW = (
    "> 权威 `docs/MASTER.md` **v2.51**（§4.47＝全部进展压缩总览，§4.48＝E46 预探针、三件缺陷修复与新探针判据先写后跑、"
    "缺陷① 修复的**可达性缺口**）；"
)

MEM_STATUS_PREFIX = "- **E45 结案＝负面**："
MEM_STATUS_OLD = (
    "合格臂集合为空。裁决历 v2.44→v2.50 未变。"
)
MEM_STATUS_NEW = (
    "合格臂集合为空。裁决历 v2.44→v2.51 未变。"
    "**v2.51 第 92 轮（2026-11-16）**：从 21 份主跑 `*.eval.json` **独立重导裁决**"
    "（`tools/_e45_round92.py`，POS 89/89）⇒ 头条表逐臂一致、**3/3 过地板的只有 A3a／A3b "
    "但两臂中位数均低于同批对照 B0**（0.110015／0.116055 < 0.121817）⇒ 不晋级、"
    "S3 **12/12 跨零**、合格臂集合为空、机制判据逐格重判与已登记结论**完全一致**。"
    "**新发现（91 轮来第一次核「修复能不能生效」）**：缺陷① 的修复**只落了模型层**——"
    "`clamp_mm` 有入口、但 `configs/models/topo_simvp.yaml` 不带它、`--topo-clamp-mm` 只进"
    "**损失侧**、臂表不传 `--model-kwargs`、**评测端也不复原**它，而 `clamp_mm` **不改 state_dict** "
    "⇒ **照现状重跑 A3a 会逐位复现 E45 的结果**（场最大仍 402.43 mm）"
    "⇒ **同步 4 个文件必要但不充分**，训练侧与评测侧必须同时携带该值。"
    "**作业 343 已预检就绪**（脚本／runner／21 份窗表／21 对场／护栏参数齐备且与已存一致、"
    "护栏本地与远端**仅差行尾**⇒ 导入的正是产出已发布数字的实现），**唯一阻塞是同账号配额**、"
    "释放即可跑、**无需重投**。"
)

MEM_TODO_PREFIX = "1. **三件实现缺陷**：① `models/topo_simvp/model.py` 输出场 `clamp_mm`；"
MEM_TODO_OLD = "护栏与幅度截断都在 forward 内 ⇒ 若据此改判据，A4/A4b/A3a/A3b 四臂必须重训。"
MEM_TODO_NEW = (
    "护栏与幅度截断都在 forward 内 ⇒ 若据此改判据，A4/A4b/A3a/A3b 四臂必须重训。"
    "**v2.51 追加（可达性缺口）**：① 的修复**只落了模型层**——`clamp_mm` 有入口、"
    "但 yaml 不带它、`--topo-clamp-mm` 只进**损失侧**、臂表不传 `--model-kwargs`、"
    "**评测端也不复原**它，而 `clamp_mm` **不改 state_dict** "
    "⇒ **照现状重跑 A3a 会逐位复现 E45 的结果** ⇒ **除同步 4 个文件外，还须补训练侧与评测侧"
    "同时携带 `clamp_mm` 的接线**（否则评测会按未截断的场打分），否则白跑一轮。"
)

# --------------------------------------------------------------------------
# HANDOVER_E45.md
# --------------------------------------------------------------------------
HO_PTR_OLD = "Gate 3 写回已完成（MASTER **v2.48**）。"
HO_PTR_NEW = "Gate 3 写回已完成（MASTER **v2.51**）。"

HO_TODO2_OLD = (
    "**② 幅度截断作用到输出场上**（重开时的第一个候选修复点；场最大值三 seed 全是 402.43 mm＝模型上限）。"
)
HO_TODO2_NEW = (
    "**② 幅度截断作用到输出场上**（重开时的第一个候选修复点；场最大值三 seed 全是 402.43 mm＝模型上限）。"
    "**v2.51 追加（可达性缺口，重开前必须先补）**：模型层的 `clamp_mm` 入口已落地，"
    "**但臂表→构造参数无通路**（yaml 不带 `clamp_mm`、`--topo-clamp-mm` 只进**损失侧**、"
    "臂表不传 `--model-kwargs`），**且评测端不复原**它、`clamp_mm` **不改 state_dict** "
    "⇒ **照现状重跑 A3a 会逐位复现 E45 的结果** ⇒ **训练侧与评测侧必须同时携带该值**。"
)


# --------------------------------------------------------------------------
# patchers
# --------------------------------------------------------------------------
def patch_master(text):
    if MARK_MASTER in text:
        print("MASTER already carries the v2.51 changelog row -> skip")
        return text
    text = apply(text, HEADER_OLD, HEADER_NEW, 1, "master.header.version")
    text = replace_in_line(text, CARD_PREFIX, CARD_OLD, CARD_NEW, "master.card")
    text = replace_in_line(text, S448_STATUS_PREFIX, S448_STATUS_OLD, S448_STATUS_NEW,
                           "master.s448.status")
    text = replace_in_line(text, S9_PREFIX, S9_OLD, S9_NEW, "master.s9.row")
    # insert the v2.51 changelog row right after the v2.50 row
    lines = text.split("\n")
    idx = [i for i, l in enumerate(lines) if l.startswith("| 2026-11-15 | **v2.50** |")]
    if len(idx) != 1:
        raise SystemExit(f"CHANGELOG ANCHOR MISS: found {len(idx)} v2.50 changelog rows")
    i = idx[0]
    prev = unescaped_pipes(lines[i])
    new = unescaped_pipes(CHANGELOG_ROW)
    if prev != new:
        raise SystemExit(f"PIPE COUNT MISMATCH: v2.50 row {prev} cells vs new row {new} cells")
    lines.insert(i + 1, CHANGELOG_ROW)
    return "\n".join(lines)


def patch_index(text):
    if MARK_INDEX in text:
        print("INDEX already carries", MARK_INDEX, "-> skip")
        return text
    return replace_in_line(text, INDEX_PREFIX, INDEX_OLD, INDEX_NEW, "index.e45")


def patch_e45doc(text):
    if MARK_E45DOC in text:
        print("E45DOC already carries", MARK_E45DOC, "-> skip")
        return text
    text = apply(text, E45DOC_PTR_OLD, E45DOC_PTR_NEW, 1, "e45doc.pointer")
    text = replace_in_line(text, E45DOC_TODO_PREFIX, E45DOC_TODO_OLD,
                           E45DOC_TODO_NEW, "e45doc.todo")
    return text


def patch_memory(text):
    if MARK_MEMORY in text:
        print("MEMORY already carries", MARK_MEMORY, "-> skip")
        return text
    text = apply(text, MEM_PTR_OLD, MEM_PTR_NEW, 1, "memory.header.version")
    text = replace_in_line(text, MEM_STATUS_PREFIX, MEM_STATUS_OLD, MEM_STATUS_NEW,
                           "memory.status")
    text = replace_in_line(text, MEM_TODO_PREFIX, MEM_TODO_OLD, MEM_TODO_NEW,
                           "memory.todo")
    return text


def patch_handover(text):
    if MARK_HANDOVER in text:
        print("HANDOVER already carries", MARK_HANDOVER, "-> skip")
        return text
    text = apply(text, HO_PTR_OLD, HO_PTR_NEW, 1, "handover.pointer")
    text = apply(text, HO_TODO2_OLD, HO_TODO2_NEW, 1, "handover.todo2")
    return text


def append_daily(text):
    if MARK_DAILY in text:
        print("daily log already carries", MARK_DAILY, "-> skip")
        return text
    note = (
        f"\n\n## {MARKER}（本次运行，2026-11-16）· E45 状态复核 + 裁决独立重导 + "
        "**缺陷① 修复的可达性缺口**\n\n"
        "**远端状态无变化、远端动作＝0**：提交器 **273** `COMPLETED`、训练数组 275／283 与评测 "
        "274／282／290、汇总 **291** 全 `COMPLETED`；21 份 `.eval.json` ＋ 13 份 run 级 json 齐；"
        "**结案后（> 2026-11-11 14:20:42）被修改文件 = 0 份**；无 E46+ 训练目录 "
        "⇒ **无新活动**（只读查询，未下载、未 sbatch、未改远端状态）。"
        "13 份 run 级 json 远端↔本地 md5 **13/13 逐位全同**（本轮实测）。\n\n"
        "**裁决从原始产物独立重导**（`tools/_e45_round92.py`，只读本地，**POS 89/89 VERDICT: PASS**）："
        "头条表由 **21 份主跑 `*.eval.json`** 复算 ⇒ 七臂中位数与过地板通过率逐臂一致；"
        "**3/3 过地板的只有 A3a／A3b，但两臂中位数都低于同批对照 B0**"
        "（0.110015／0.116055 < 0.121817）且低于 Oracle 上界 ⇒ **不晋级**；"
        "S3 **12/12 区间跨零**、合格臂集合为空、失效模式**分两支报**（不合并）；"
        "机制判据逐格重判与已登记结论**完全一致**（A3a／A3b `field_max_mm` 六格全 402.4288 mm；"
        "A4b 种子 2 触发**相关性分支**、`scale_cv` 3.0188 为六格最高 ⇒ 量通道全程未退化成常数）。"
        "⇒ **裁决不变＝六臂判负、不投十 seed 算力**。\n\n"
        "**本轮新发现（91 轮来第一次核「修复能不能生效」）**：缺陷① 的修复**只落了模型层**。"
        "`models/topo_simvp/model.py` 确有 `clamp_mm` 入口并在 `expm1/floor_at_zero` 之后对毫米场 "
        "`clamp(max=clamp_mm)`；**但从 E45 臂表到该构造参数没有任何通路**——"
        "`configs/models/topo_simvp.yaml` **不带** `clamp_mm`（⇒ `model_kwargs()` 永不传它）；"
        "`--topo-clamp-mm` 只被**损失侧** `TopoCriterion` 消费、**不进模型构造**；"
        "臂表 A3a 仍是 `--topo-clamp-mm 20` 且全表**不传** `--model-kwargs`；"
        "评测端从 checkpoint 的 `config.json` 只复原 `head_bins`／`level_mode`、"
        "**不复原** `clamp_mm`／`model_kwargs`；而 `clamp_mm` 是普通 float 属性、**不改 state_dict** "
        "⇒ 即便训练侧打开，评测漏传也会**顺利加载 checkpoint 并按未截断的场打分**"
        "（本项目吃过的亏：运行时开关不改变 state_dict ⇒ 评测端漏传 ⇒ 静默用回被关掉的机制）。"
        "⇒ **照现状重跑 A3a 会逐位复现 E45 的结果**（场最大仍 402.43 mm）"
        "⇒ **同步那 4 个文件是必要但不充分**：训练侧与评测侧**必须同时**携带该值，"
        "否则重开 E45 的第一候选修复点会**白跑一轮**。\n\n"
        "**作业 343 预检就绪**（仍 `PENDING`，`QOSMaxJobsPerUserLimit`：队列里 8 个 `qlfn2-f38` "
        "属同一 `wf` 账号下的**别的项目线**，不干预）：脚本与 runner **已同步**、"
        "21 份窗表与 21 对导出场**在位**、护栏参数（thr 2.0／n_boot 10000／rng 20261109／epoch 00／"
        "control B0）与已存**一致**、且 `tools/_e41_s3_equalbias.py` 本地与远端**仅差行尾"
        "（归一化 md5 相同）** ⇒ 它导入的正是产出已发布数字的那份实现，"
        "其「注册种子复现已存端点」自检**有效** ⇒ **唯一阻塞是配额**，释放即可跑、**无需重投**。\n\n"
        "**下一步（变更一处）**：§4.5 待办② 补上**训练侧与评测侧同时携带 `clamp_mm`** 的接线；"
        "其余三件缺陷修复与「重开须重新提机制族」不变。"
        "**产物**：`tools/_e45_round92.py`、`tools/_evals/e45/round92_status_audit.json`；"
        "写回 `tools/_sync_v251.py`（MASTER 升 **v2.51**，Gate 3 五处 + INDEX + MEMORY + HANDOVER + 两份日志）。\n"
    )
    return text + note


def main():
    plan = [
        (MASTER, patch_master),
        (INDEX, patch_index),
        (E45DOC, patch_e45doc),
        (MEMORY, patch_memory),
        (HANDOVER, patch_handover),
        (DAILY, append_daily),
        (DAILY_PROJECT, append_daily),
    ]
    changed = []
    for path, fn in plan:
        if not path.exists():
            raise SystemExit(f"FILE MISSING: {path}")
        text = path.read_text(encoding="utf-8")
        new = fn(text)
        if new != text:
            path.write_text(new, encoding="utf-8")
            changed.append(str(path))
            print(f"changed: {path}")
        else:
            print(f"unchanged: {path}")
    print()
    print(f"{len(changed)} file(s) changed")
    if not changed:
        print("(idempotent: nothing to do)")


if __name__ == "__main__":
    sys.exit(main())
