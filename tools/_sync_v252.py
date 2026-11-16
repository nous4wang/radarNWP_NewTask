"""Write back the round-93 finding (v2.52).

What v2.52 records
------------------
Round 92 established, by reading the source, that the local fix for
defect (1) -- putting the amplitude cap on the OUTPUT field -- cannot
reach the model constructor from the E45 arm table, and that the
evaluation side would not restore it either.  That was a statement
about the code as it stands.

Round 93 checks the same thing against the RUN RECORDS: the
``config.json`` each of the 21 E45 training runs wrote next to its
checkpoint.  Those files are the training run's own account of how it
built its model, and they are what an evaluation would read.  The
answer is unambiguous: **all 21 runs recorded ``model_kwargs: None``**,
and A3a's three recorded ``topo_clamp_mm: 20.0`` -- the LOSS got the
cap and the MODEL did not.  So the amplitude cap was off in the very
runs that produced the published numbers.  That upgrades the finding
from an inference about the code to a fact about the experiment.

Round 93 also lands the half of the wiring that can be landed without
touching the frozen E45 arm table:

* the EVALUATION side now restores the constructor kwargs the training
  run recorded (``scripts/evaluate.py``, new helper
  ``_trained_model_kwargs``), with a new test file
  ``tests/test_eval_model_kwargs_restore.py`` (12 tests, all passing);
* the TRAINING side is deliberately NOT changed: the E45 arm table is
  the pre-registered definition of what E45 ran, and editing it would
  change what E45 means.  The route the next round's arm table will
  use -- ``--model-kwargs {"clamp_mm":20.0}`` -- is verified here to
  survive the arm table's own ``read -r -a`` splitting as ONE token
  and to reach the constructor through ``model_kwargs()``.

A remote-wide scan confirms the evaluation-side restore changes nothing
for any existing run: of the 43 ``config.json`` files on the remote,
**0** record any constructor kwargs.

Scope of this write-back
------------------------
* MASTER.md -- header version line, the section 4.47 card row, the
  section 4.48 status bullet, the section 9 row, a section 12
  changelog row (the five places Gate 3 names);
* INDEX.md -- the E45 row;
* E45 experiment file -- the version pointer and the defect-(1) to-do;
* MEMORY.md -- the version pointer, the status section, the to-do item;
* HANDOVER_E45.md -- the version pointer, the to-do item, the round note;
* both daily logs -- the round note.

The historical section 12 rows (v2.44-v2.51) are deliberately NOT
rewritten: they are the audit trail.  Every replacement is anchored on
a literal that must occur exactly the expected number of times; a
missing anchor is an error, never a silent skip.  The new section 12
row is built from its CELLS; the separators are inserted by the code
and the pipe count is compared against the row it follows.  Running
the script twice is a no-op.

Every marker is PER-FILE and distinctive.  The bare version string is
not usable as a guard: earlier rounds left the text "（＝升 v2.51）" in
the handover and the daily logs, and a guard on the bare string would
skip those files WITHOUT writing anything -- a silent false positive.
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

TAG = "v2.52"
MARKER = "v2.52 第 93 轮"
MARK_MASTER = "| 2026-11-16 | **v2.52** |"
MARK_INDEX = "v2.52 第 93 轮"
MARK_MEMORY = "v2.52 第 93 轮"
#: The marker must match the text EXACTLY, spaces included: the paragraph
#: it guards is written with spaces around the full-width plus sign.
MARK_HANDOVER = "v2.52 追加（运行记录证实 ＋ 评测侧接线已落地）"
MARK_DAILY = "## v2.52 第 93 轮（本次运行"
MARK_E45DOC = "v2.52 追加（运行记录证实 ＋ 评测侧接线已落地）"


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
        raise SystemExit(f"LINE ANCHOR MISS ({label}): prefix {line_prefix[:60]!r} "
                         f"matched {len(hits)} line(s), expected 1")
    i = hits[0]
    if new in lines[i] and old not in lines[i]:
        return text          # already applied
    if lines[i].count(old) != 1:
        raise SystemExit(f"ANCHOR MISS ({label}): {old[:80]!r} occurs "
                         f"{lines[i].count(old)}x in the target line, expected 1")
    lines[i] = lines[i].replace(old, new, 1)
    return "\n".join(lines)


# --------------------------------------------------------------------------
# the new statement, written once
# --------------------------------------------------------------------------
RUNRECORD = (
    "**v2.52 运行记录证实（比 v2.51 的源码推断更强）**：v2.51 说缺陷① 的修复"
    "「从臂表到构造参数没有通路」——那是**读源码**得出的。本轮改查 **21 份训练运行"
    "自己写的 `config.json`**（就放在 checkpoint 旁边，正是评测会读的那份记录）："
    "**21 份全部记 `model_kwargs: None`**；A3a 的三份记 `topo_clamp_mm: 20.0` ——"
    "**损失侧拿到了幅度上限、模型侧没有** ⇒ **产出已发布数字的那批运行里，幅度上限根本没开**。"
    "这条从「代码现在通不通」升级成「那次实验里到底开没开」的事实。"
    "另：远端全库扫了 43 份 `config.json`，**0 份**记过任何构造参数 ⇒ "
    "**评测侧复原对任何既有运行都没有影响**（零回归风险）。"
)

WIRING = (
    "**v2.52 接线（评测侧已落地、训练侧按冻结纪律不动）**：可落地的那一半已落地——"
    "`scripts/evaluate.py` 现在会**复原训练运行记录下来的构造参数**"
    "（新helper `_trained_model_kwargs`，与既有的 `head_bins`／`level_mode` 复原同一处），"
    "配套新测试 `tests/test_eval_model_kwargs_restore.py`（12 项全过），"
    "其中一项直接钉住「**clamp_mm 不是权重、不进 `state_dict`**」⇒ "
    "这正是**必须复原**、且**光靠 checkpoint 传不过来**的原因。"
    "**训练侧不动**：E45 臂表是**预注册的「E45 跑的是什么」**，改它＝改 E45 的定义 ⇒ "
    "下一轮的臂表才带 `--model-kwargs {\"clamp_mm\":20.0}`；本轮已实测该写法**经臂表自己的 "
    "`read -r -a` 分词仍是**一个** token**、且经 `model_kwargs()` 真的进到构造参数。"
)

CHANGELOG_ROW = row(
    "2026-11-16",
    f"**{TAG}**",
    "**E45 第 93 轮：用 21 份运行记录证实缺陷① 从未生效（比源码推断更强）＋ 评测侧接线已落地**"
    "：远端主链终态不变（提交器 273／训练数组 275·283／评测 274·282·290／汇总 291 全 `COMPLETED`、"
    "21 份 `.eval.json` ＋ 13 份 run 级 json 齐、**结案后（> 11-11 14:20:42）被修改文件 = 0 份**）"
    "⇒ **无新活动**。**远端动作 = 0**（只读查询；未下载、未 sbatch、未同步、未改远端状态）。"
    "**产物一致性**：13 份 run 级 json 远端↔本地 md5 **13/13 逐位全同**（本轮实测）。"
    "**裁决独立重导**（`tools/_e45_round93.py`，只读本地，**POS 73/73 VERDICT: PASS**）："
    "头条表由 21 份主跑 `*.eval.json` 复算 ⇒ 七臂中位数与过地板通过率**逐臂一致**；"
    "**3/3 过地板的只有 A3a／A3b，但两臂中位数均低于同批对照 B0** ⇒ **不晋级**；"
    "S3 护栏参数未改写、**12/12 区间跨零**、合格臂集合为空、失效模式**分两支**（不合并）。"
    "**本轮新证据（91、92 轮以来第一次查「那次实验里到底开没开」）**：查 **21 份训练运行自己写的 "
    "`config.json`** ⇒ **21 份全部记 `model_kwargs: None`**；A3a 的三份记 `topo_clamp_mm: 20.0` "
    "⇒ **损失侧拿到幅度上限、模型侧没有** ⇒ **产出已发布数字的那批运行里幅度上限根本没开**。"
    "这把 v2.51 的**源码推断**升级成**实验事实**。远端全库 43 份 `config.json` 中 "
    "**0 份**记过构造参数 ⇒ 评测侧复原**对既有运行零影响**。"
    "**接线（可落地的一半已落地）**：`scripts/evaluate.py` 现在**复原训练运行记录下来的构造参数**"
    "（新helper `_trained_model_kwargs`，与 `head_bins`／`level_mode` 同一处复原），"
    "新测试 `tests/test_eval_model_kwargs_restore.py`（**12 项全过**），其中一项钉住 "
    "**clamp_mm 不是权重、不进 `state_dict`**（这正是必须复原、且光靠 checkpoint 传不过来的原因）；"
    "**训练侧不动**（E45 臂表是预注册的「E45 跑的是什么」，改它＝改 E45 的定义）"
    "⇒ 下一轮臂表带 `--model-kwargs {\"clamp_mm\":20.0}`，本轮已实测该写法**经臂表自己的 "
    "`read -r -a` 分词仍是一个 token**、且经 `model_kwargs()` 真的进到构造参数。"
    "**作业 343 仍排队**（`QOSMaxJobsPerUserLimit`：队列里 8 个 `qlfn2-f38` 属同一 `wf` 账号下的"
    "**别的项目线**，不干预；脚本／runner／21 份窗表／21 对场／护栏实现已预检齐备）"
    "⇒ **唯一阻塞仍是同账号配额**，释放即可跑、**无需重投**。"
    "**裁决零变化**（六臂仍全部判负、无一臂进十 seed、**不投十 seed 算力**）；"
    "**下一步（变更一处）**：§4.5 待办② 只需**训练侧**再接一次（评测侧已落地）"
    "＋ **下一轮须重新提机制族**，其余两件缺陷修复不变。",
    "`tools/_e45_round93.py` · `tools/_evals/e45/round93_status_audit.json` · "
    "`scripts/evaluate.py` · `tests/test_eval_model_kwargs_restore.py` · "
    "`docs/experiments/E45-gate-direction-amount-separation-baseline.md`",
)

# --------------------------------------------------------------------------
# MASTER
# --------------------------------------------------------------------------
HEADER_OLD = (
    "；**v2.51（2026-11-16，最新）**：E45 第 92 轮从原始产物**独立重导裁决**（POS 89/89）"
    "⇒ **裁决不变＝六臂判负、不投十 seed 算力**；**新发现缺陷① 修复的可达性缺口**——"
    "模型层有 `clamp_mm` 入口、但臂表→构造参数无通路、评测端也不复原它 ⇒ "
    "**照现状重跑 A3a 会逐位复现 E45 结果**；同时**预检了排队中的作业 343**"
    "（脚本／runner／21 份窗表／21 对场／护栏实现齐备、参数与已存一致、护栏本地与远端**仅差行尾**）"
    "⇒ 其唯一阻塞是**同账号配额**（另项目线的 8 个 `qlfn2-f38`，不干预），配额释放即可跑、**无需重投** |"
)
HEADER_NEW = (
    "；**v2.51（2026-11-16）**：E45 第 92 轮从原始产物**独立重导裁决**（POS 89/89）"
    "⇒ **裁决不变＝六臂判负、不投十 seed 算力**；**新发现缺陷① 修复的可达性缺口**——"
    "模型层有 `clamp_mm` 入口、但臂表→构造参数无通路、评测端也不复原它 ⇒ "
    "**照现状重跑 A3a 会逐位复现 E45 结果**；同时**预检了排队中的作业 343**"
    "（脚本／runner／21 份窗表／21 对场／护栏实现齐备、参数与已存一致、护栏本地与远端**仅差行尾**）"
    "⇒ 其唯一阻塞是**同账号配额**（另项目线的 8 个 `qlfn2-f38`，不干预），配额释放即可跑、**无需重投**；"
    "**v2.52（2026-11-16，最新）**：E45 第 93 轮改查 **21 份训练运行自己写的 `config.json`** ⇒ "
    "**21 份全部记 `model_kwargs: None`**、A3a 记 `topo_clamp_mm: 20.0` "
    "⇒ **产出已发布数字的那批运行里，幅度上限根本没开**（把 v2.51 的源码推断升级成实验事实）；"
    "**评测侧接线已落地**（`scripts/evaluate.py` 复原训练侧记录的构造参数 ＋ 新测试 12 项全过）"
    "⇒ 训练侧只剩「下一轮臂表接一次」，**裁决不变＝六臂判负、不投十 seed 算力** |"
)

CARD_PREFIX = "| §4.47 | 修三处机制缺陷 + 把比较基准摆正"
CARD_OLD = (
    "⇒ **同步 4 个文件是必要但不充分**。 |"
    " `docs/experiments/E45-gate-direction-amount-separation-baseline.md`；"
    "v2.49 压缩总览见 §4.47；v2.50 见 §4.48；v2.51 见 §4.48 状态行 |"
)
CARD_NEW = (
    "⇒ **同步 4 个文件是必要但不充分**。 ｜ **v2.52**：改查 **21 份训练运行自己写的 "
    "`config.json`** ⇒ **21 份全部记 `model_kwargs: None`**、A3a 记 `topo_clamp_mm: 20.0` "
    "⇒ **那批运行里幅度上限根本没开**（源码推断 → 实验事实）；"
    "**评测侧接线已落地**（`scripts/evaluate.py` 复原记录下来的构造参数 ＋ 新测试 12 项全过）"
    "⇒ 训练侧只剩「下一轮臂表接一次」。 |"
    " `docs/experiments/E45-gate-direction-amount-separation-baseline.md`；"
    "v2.49 压缩总览见 §4.47；v2.50 见 §4.48；v2.51 见 §4.48 状态行；v2.52 见 §4.48 状态行 |"
)

S448_STATUS_PREFIX = "- **状态**：4 个修复文件只在本地"
S448_STATUS_OLD = "最小同步集仍是这 4 个文件。"
S448_STATUS_NEW = (
    "最小同步集仍是这 4 个文件。" + RUNRECORD + WIRING +
    " **v2.52 另记**：排队中的作业 343（`scripts/e45_s3_seedrobust.sbatch`）状态不变"
    "（仍 `PENDING`／`QOSMaxJobsPerUserLimit`，队列里 8 个 `qlfn2-f38` 属别的项目线、不干预）"
    "⇒ **唯一阻塞仍是同账号配额**，释放即可跑、**无需重投**。"
)

S9_PREFIX = "| **E45 / §4.47 修三处机制缺陷 + 摆正基准** |"
S9_OLD = (
    "｜ 交接：`HANDOVER_E45.md`；v2.49 压缩总览见 §4.47；v2.50 见 §4.48；"
    "v2.51 见 §4.48 状态行（缺陷① 修复的**可达性缺口**） |"
)
S9_NEW = (
    "｜ 交接：`HANDOVER_E45.md`；v2.49 压缩总览见 §4.47；v2.50 见 §4.48；"
    "v2.51 见 §4.48 状态行（缺陷① 修复的**可达性缺口**）；"
    "v2.52 见 §4.48 状态行（**21 份运行记录证实幅度上限从未生效** ＋ **评测侧接线已落地**） |"
)

# --------------------------------------------------------------------------
# INDEX
# --------------------------------------------------------------------------
INDEX_PREFIX = "| E45 | §4.47（v2.49 压缩总览） |"
INDEX_OLD = (
    "**作业 343 已预检就绪**，唯一阻塞是同账号配额 |"
)
INDEX_NEW = (
    "**作业 343 已预检就绪**，唯一阻塞是同账号配额 ｜ **v2.52 第 93 轮**："
    "改查 **21 份训练运行自己写的 `config.json`** ⇒ **21 份全部记 `model_kwargs: None`**、"
    "A3a 记 `topo_clamp_mm: 20.0` ⇒ **产出已发布数字的那批运行里幅度上限根本没开**"
    "（v2.51 的源码推断 → 实验事实）；**评测侧接线已落地**"
    "（`scripts/evaluate.py` 复原记录下来的构造参数 ＋ 新测试 12 项全过）"
    "⇒ 训练侧只剩「下一轮臂表接一次」；**裁决不变＝六臂判负、不投十 seed 算力** |"
)

# --------------------------------------------------------------------------
# E45 experiment file
# --------------------------------------------------------------------------
E45DOC_PTR_OLD = (
    "> **2026-11-16（v2.51）**：**E45 内容与判据自 v2.49 起未变**（S0–S6 一字未改；"
    "裁决不变：六臂判负、不投十 seed 算力）；第 92 轮从原始产物独立重导裁决（POS 89/89）"
    "**裁决不变**，并**首次核到缺陷① 修复的可达性缺口**（模型层有 `clamp_mm` 入口、"
    "但臂表→构造参数无通路、评测端也不复原它 ⇒ 照现状重跑 A3a 会逐位复现 E45 的结果）"
    "⇒ **同步 4 个文件是必要但不充分**（详见 §8.4 三件待办第 1 条）。\n"
)
E45DOC_PTR_NEW = (
    "> **2026-11-16（v2.52）**：**E45 内容与判据自 v2.49 起未变**（S0–S6 一字未改；"
    "裁决不变：六臂判负、不投十 seed 算力）；第 93 轮改查 **21 份训练运行自己写的 "
    "`config.json`** ⇒ **21 份全部记 `model_kwargs: None`**、A3a 记 `topo_clamp_mm: 20.0` "
    "⇒ **产出已发布数字的那批运行里，幅度上限根本没开**（把 v2.51 的源码推断升级成实验事实）；"
    "**评测侧接线已落地**（`scripts/evaluate.py` 复原训练侧记录的构造参数 ＋ 新测试 12 项全过）"
    "⇒ 训练侧只剩「下一轮臂表接一次」（详见 §8.4 三件待办第 1 条）。\n"
    "> **2026-11-16（v2.51）**：**E45 内容与判据自 v2.49 起未变**（S0–S6 一字未改；"
    "裁决不变：六臂判负、不投十 seed 算力）；第 92 轮从原始产物独立重导裁决（POS 89/89）"
    "**裁决不变**，并**首次核到缺陷① 修复的可达性缺口**（模型层有 `clamp_mm` 入口、"
    "但臂表→构造参数无通路、评测端也不复原它 ⇒ 照现状重跑 A3a 会逐位复现 E45 的结果）"
    "⇒ **同步 4 个文件是必要但不充分**（详见 §8.4 三件待办第 1 条）。\n"
)

E45DOC_TODO_PREFIX = "  **v2.51 追加（可达性缺口，重开前必须先补）**："
E45DOC_TODO_OLD = (
    "⇒ **训练侧与评测侧必须同时携带该值**（或等价地写进 yaml 并让评测端复原），否则白跑一轮。"
)
E45DOC_TODO_NEW = (
    "⇒ **训练侧与评测侧必须同时携带该值**（或等价地写进 yaml 并让评测端复原），否则白跑一轮。\n"
    "  **v2.52 追加（运行记录证实 ＋ 评测侧接线已落地）**：改查 **21 份训练运行自己写的 "
    "`config.json`** ⇒ **21 份全部记 `model_kwargs: None`**、A3a 记 `topo_clamp_mm: 20.0` "
    "⇒ **产出已发布数字的那批运行里，幅度上限根本没开**（本条从「源码现在通不通」升级成"
    "「那次实验里到底开没开」）。**评测侧接线已落地**：`scripts/evaluate.py` 现在**复原训练侧"
    "记录下来的构造参数**（新helper `_trained_model_kwargs`，与 `head_bins`／`level_mode` 同一处），"
    "新测试 `tests/test_eval_model_kwargs_restore.py`（12 项全过），其中一项钉住 "
    "**clamp_mm 不是权重、不进 `state_dict`** ⇒ 这正是必须复原、且光靠 checkpoint 传不过来的原因；"
    "远端 43 份 `config.json` 中 **0 份**记过构造参数 ⇒ **对既有运行零影响**。"
    "**训练侧不动**（E45 臂表是预注册的「E45 跑的是什么」，改它＝改 E45 的定义）"
    "⇒ **下一轮臂表**接 `--model-kwargs {\"clamp_mm\":20.0}` 即可（已实测该写法经臂表自己的 "
    "`read -r -a` 分词**仍是一个 token**、且经 `model_kwargs()` 真的进到构造参数）。"
)

# --------------------------------------------------------------------------
# MEMORY.md
# --------------------------------------------------------------------------
MEM_PTR_OLD = (
    "> 权威 `docs/MASTER.md` **v2.51**（§4.47＝全部进展压缩总览，§4.48＝E46 预探针、三件缺陷修复与新探针判据先写后跑、"
    "缺陷① 修复的**可达性缺口**）；"
)
MEM_PTR_NEW = (
    "> 权威 `docs/MASTER.md` **v2.52**（§4.47＝全部进展压缩总览，§4.48＝E46 预探针、三件缺陷修复与新探针判据先写后跑、"
    "缺陷① 修复的**可达性缺口**、**21 份运行记录证实幅度上限从未生效**、**评测侧接线已落地**）；"
)

MEM_STATUS_PREFIX = "- **E45 结案＝负面**："
MEM_STATUS_OLD = (
    "**作业 343 已预检就绪**（脚本／runner／21 份窗表／21 对场／护栏参数齐备且与已存一致、"
    "护栏本地与远端**仅差行尾**⇒ 导入的正是产出已发布数字的实现），**唯一阻塞是同账号配额**、"
    "释放即可跑、**无需重投**。"
)
MEM_STATUS_NEW = (
    "**作业 343 已预检就绪**（脚本／runner／21 份窗表／21 对场／护栏参数齐备且与已存一致、"
    "护栏本地与远端**仅差行尾**⇒ 导入的正是产出已发布数字的实现），**唯一阻塞是同账号配额**、"
    "释放即可跑、**无需重投**。"
    "**v2.52 第 93 轮（2026-11-16）**：改查 **21 份训练运行自己写的 `config.json`** ⇒ "
    "**21 份全部记 `model_kwargs: None`**、A3a 记 `topo_clamp_mm: 20.0` "
    "⇒ **产出已发布数字的那批运行里，幅度上限根本没开**（v2.51 的源码推断 → **实验事实**）；"
    "远端全库 43 份 `config.json` 中 **0 份**记过构造参数 ⇒ 评测侧复原**对既有运行零影响**。"
    "**评测侧接线已落地**：`scripts/evaluate.py` 复原训练侧记录的构造参数（`tools/_e45_round93.py`，"
    "POS 73/73；新测试 `tests/test_eval_model_kwargs_restore.py` 12 项全过，"
    "其中一项钉住 **clamp_mm 不是权重、不进 `state_dict`**）。"
    "**训练侧不动**（E45 臂表是预注册的「E45 跑的是什么」）⇒ 下一轮臂表接 "
    "`--model-kwargs {\"clamp_mm\":20.0}` 即可（已实测经 `read -r -a` 仍是一个 token）。"
    "**裁决不变＝六臂判负、不投十 seed 算力**。"
)

MEM_TODO_PREFIX = "1. **三件实现缺陷**：① `models/topo_simvp/model.py` 输出场 `clamp_mm`；"
MEM_TODO_OLD = (
    "**v2.51 追加（可达性缺口）**：① 的修复**只落了模型层**——`clamp_mm` 有入口、"
    "但 yaml 不带它、`--topo-clamp-mm` 只进**损失侧**、臂表不传 `--model-kwargs`、"
    "**评测端也不复原**它，而 `clamp_mm` **不改 state_dict** "
    "⇒ **照现状重跑 A3a 会逐位复现 E45 的结果** ⇒ **除同步 4 个文件外，还须补训练侧与评测侧"
    "同时携带 `clamp_mm` 的接线**（否则评测会按未截断的场打分），否则白跑一轮。"
)
MEM_TODO_NEW = (
    "**v2.51 追加（可达性缺口）**：① 的修复**只落了模型层**——`clamp_mm` 有入口、"
    "但 yaml 不带它、`--topo-clamp-mm` 只进**损失侧**、臂表不传 `--model-kwargs`、"
    "**评测端也不复原**它，而 `clamp_mm` **不改 state_dict** "
    "⇒ **照现状重跑 A3a 会逐位复现 E45 的结果** ⇒ **除同步 4 个文件外，还须补训练侧与评测侧"
    "同时携带 `clamp_mm` 的接线**（否则评测会按未截断的场打分），否则白跑一轮。"
    "**v2.52 追加（运行记录证实 ＋ 评测侧已接线）**：① 的**评测侧已落地**"
    "（`scripts/evaluate.py` 复原训练侧记录的构造参数 ＋ 新测试 12 项全过）；"
    "**训练侧只剩「下一轮臂表接一次」**（`--model-kwargs {\"clamp_mm\":20.0}`，已实测经臂表的 "
    "`read -r -a` 仍是一个 token）⇒ **待办① 从「两侧都要补」缩到「只差训练侧一次」**；"
    "**训练侧按冻结纪律不动**（E45 臂表是预注册的「E45 跑的是什么」，改它＝改 E45 的定义）。"
)

# --------------------------------------------------------------------------
# HANDOVER_E45.md
# --------------------------------------------------------------------------
HO_PTR_OLD = "Gate 3 写回已完成（MASTER **v2.51**）。"
HO_PTR_NEW = "Gate 3 写回已完成（MASTER **v2.52**）。"

HO_TODO2_OLD = (
    "⇒ **照现状重跑 A3a 会逐位复现 E45 的结果** ⇒ **训练侧与评测侧必须同时携带该值**。"
)
HO_TODO2_NEW = (
    "⇒ **照现状重跑 A3a 会逐位复现 E45 的结果** ⇒ **训练侧与评测侧必须同时携带该值**。"
    "**v2.52 追加（运行记录证实 ＋ 评测侧接线已落地）**：改查 **21 份训练运行自己写的 "
    "`config.json`** ⇒ **21 份全部记 `model_kwargs: None`**、A3a 记 `topo_clamp_mm: 20.0` "
    "⇒ **产出已发布数字的那批运行里，幅度上限根本没开**（源码推断 → 实验事实）；"
    "**评测侧接线已落地**（`scripts/evaluate.py` 复原训练侧记录的构造参数 ＋ 新测试 12 项全过）"
    "⇒ **待办② 从「两侧都要补」缩到「只差训练侧一次」**（下一轮臂表接 "
    "`--model-kwargs {\"clamp_mm\":20.0}`，已实测经臂表的 `read -r -a` 仍是一个 token）。"
)


# --------------------------------------------------------------------------
# patchers
# --------------------------------------------------------------------------
def patch_master(text):
    if MARK_MASTER in text:
        print("MASTER already carries the v2.52 changelog row -> skip")
        return text
    text = apply(text, HEADER_OLD, HEADER_NEW, 1, "master.header.version")
    text = replace_in_line(text, CARD_PREFIX, CARD_OLD, CARD_NEW, "master.card")
    text = replace_in_line(text, S448_STATUS_PREFIX, S448_STATUS_OLD, S448_STATUS_NEW,
                           "master.s448.status")
    text = replace_in_line(text, S9_PREFIX, S9_OLD, S9_NEW, "master.s9.row")
    # insert the v2.52 changelog row right after the v2.51 row
    lines = text.split("\n")
    idx = [i for i, l in enumerate(lines) if l.startswith("| 2026-11-16 | **v2.51** |")]
    if len(idx) != 1:
        raise SystemExit(f"CHANGELOG ANCHOR MISS: found {len(idx)} v2.51 changelog rows")
    i = idx[0]
    prev = unescaped_pipes(lines[i])
    new = unescaped_pipes(CHANGELOG_ROW)
    if prev != new:
        raise SystemExit(f"PIPE COUNT MISMATCH: v2.51 row {prev} cells vs new row {new} cells")
    lines.insert(i + 1, CHANGELOG_ROW)
    return "\n".join(lines)


def patch_index(text):
    if MARK_INDEX in text:
        print("INDEX already carries", MARK_INDEX, "-> skip")
        return text
    return replace_in_line(text, INDEX_PREFIX, INDEX_OLD, INDEX_NEW, "index.e45")


def patch_e45doc(text):
    if MARK_E45DOC in text:
        print("E45DOC already carries the marker -> skip")
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
        print("HANDOVER already carries the marker -> skip")
        return text
    text = apply(text, HO_PTR_OLD, HO_PTR_NEW, 1, "handover.pointer")
    text = apply(text, HO_TODO2_OLD, HO_TODO2_NEW, 1, "handover.todo2")
    return text


ROUND_NOTE = """
<!-- E45-R93-BEGIN -->
### 2026-11-16｜第 93 轮（E45 远端状态复核 ＋ **用 21 份运行记录证实缺陷① 从未生效** ＋ **评测侧接线已落地**）

- **判定**：远端 E45 主链终态不变（提交器 273／训练数组 275·283／评测 274·282·290／汇总 291／297·298／314–317 全 `COMPLETED`、21 份 `.eval.json` ＋ 13 份 run 级 json 齐、**结案后（> 2026-11-11 14:20:42）被修改文件 = 0 份**）⇒ **无新活动**。**远端动作 = 0**（只读查询；未下载、未 sbatch、未同步、未改远端状态）。
- **产物一致性**：13 份 run 级 json **远端↔本地 md5 13/13 逐位全同**（本轮实测）。
- **裁决独立重导**（`tools/_e45_round93.py`，只读本地，**POS 73/73 VERDICT: PASS**）：头条表由 21 份主跑 `*.eval.json` 复算 ⇒ 七臂中位数与过地板通过率逐臂一致；**3/3 过地板的只有 A3a／A3b，但两臂中位数均低于同批对照 B0** ⇒ **不晋级**；S3 **12/12 跨零**、合格臂集合为空、失效模式**分两支报**。⇒ **裁决不变＝六臂判负、不投十 seed 算力**。
- **本轮新证据（91、92 轮以来第一次查「那次实验里到底开没开」）**：查 **21 份训练运行自己写的 `config.json`**（就放在 checkpoint 旁边、正是评测会读的那份记录）⇒ **21 份全部记 `model_kwargs: None`**；A3a 的三份记 `topo_clamp_mm: 20.0` ⇒ **损失侧拿到了幅度上限、模型侧没有** ⇒ **产出已发布数字的那批运行里，幅度上限根本没开**。这把 v2.51 的**源码推断**升级成**实验事实**。另：远端全库扫了 43 份 `config.json`，**0 份**记过任何构造参数 ⇒ **评测侧复原对任何既有运行都零影响**。
- **接线（可落地的一半已落地）**：`scripts/evaluate.py` 现在**复原训练运行记录下来的构造参数**（新helper `_trained_model_kwargs`，与既有 `head_bins`／`level_mode` 复原同一处），配套新测试 `tests/test_eval_model_kwargs_restore.py`（**12 项全过**），其中一项直接钉住「**clamp_mm 不是权重、不进 `state_dict`**」⇒ 这正是**必须复原**、且**光靠 checkpoint 传不过来**的原因。**训练侧不动**：E45 臂表是**预注册的「E45 跑的是什么」**，改它＝改 E45 的定义 ⇒ 下一轮臂表才接 `--model-kwargs {"clamp_mm":20.0}`；本轮已实测该写法**经臂表自己的 `read -r -a` 分词仍是一个 token**、且经 `model_kwargs()` 真的进到构造参数。
- **作业 343 仍排队**（`PENDING`／`QOSMaxJobsPerUserLimit`：队列里 8 个 `qlfn2-f38` 属同一 `wf` 账号下的**别的项目线**，不干预；脚本／runner／21 份窗表／21 对场／护栏实现已预检齐备）⇒ **唯一阻塞仍是同账号配额**，释放即可跑、**无需重投**。
- **下一步（变更一处）**：§4.5 待办② 只需**训练侧**再接一次（评测侧已落地）＋ **下一轮须重新提机制族**；其余两件缺陷修复不变。
- **产物**：`tools/_e45_round93.py`、`tools/_evals/e45/round93_status_audit.json`、`tests/test_eval_model_kwargs_restore.py`、`scripts/evaluate.py`。
<!-- E45-R93-END -->
"""


def append_daily(text):
    if MARK_DAILY in text:
        print("daily log already carries", MARK_DAILY, "-> skip")
        return text
    note = (
        f"\n\n## {MARKER}（本次运行，2026-11-16）· E45 状态复核 ＋ "
        "**21 份运行记录证实缺陷① 从未生效** ＋ **评测侧接线已落地**\n\n" + ROUND_NOTE
    )
    return text + note


def append_handover_round(text):
    """Append the round note to the handover, under its own marker."""
    if "<!-- E45-R93-BEGIN -->" in text:
        print("HANDOVER already carries the R93 round note -> skip")
        return text
    return text + ROUND_NOTE


def main():
    plan = [
        (MASTER, patch_master),
        (INDEX, patch_index),
        (E45DOC, patch_e45doc),
        (MEMORY, patch_memory),
        (HANDOVER, patch_handover),
        (HANDOVER, append_handover_round),
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
