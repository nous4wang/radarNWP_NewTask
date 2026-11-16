"""Write back the round-95 findings (v2.54).

What v2.54 records
------------------
Round 95 re-checked the remote Slurm state and re-derived the verdict
from the raw artifacts (POS 84/84, VERDICT: PASS).  The remote is
unchanged: the E45 main chain is still COMPLETED, the queue is empty,
job 343 finished at 06:02:25 and NOTHING under runs/e45 changed after
it, and all 14 artifacts are byte-identical to the local copies
(md5 14/14).  The verdict is unchanged: six arms negative, none to ten
seeds, no ten-seed compute.

Two things are NEW in this round, and both are corrections to text that
is already published:

1.  **The minimal sync set is FIVE files, not four.**  Rounds 71-75
    audited the local-vs-remote drift and registered four files
    (scaled_structure/model.py, topo_simvp/model.py,
    _e41_struct_metrics.py, _e45_mechanism_probe.py).  That was right
    on 2026-11-15, when the eval side had not been touched.  But the
    v2.52 round landed a FIFTH change -- the eval side's
    ``_trained_model_kwargs`` restore in ``scripts/evaluate.py`` -- and
    nobody re-ran the drift audit afterwards.  Measured byte-for-byte
    this round (normalised md5 on both sides), the drifted set is
    exactly those five, while the controls (amp_monotone_gate,
    amp_hard_support, train.py, _e45_arms.sh, both model yamls) and
    the S3 guard are identical.  This matters operationally: the
    eval-side restore is precisely the half of defect (1) that carries
    ``clamp_mm`` to scoring time, and it is the half that CANNOT be
    recovered from a checkpoint, because clamp_mm is a plain float
    attribute and is not in ``state_dict``.  Syncing four files would
    re-introduce the exact failure mode this project has already
    paid for once (a runtime switch that does not change state_dict
    => the eval side silently uses the old mechanism back).

2.  **A3a's gradient read-out passes on 1/3 seeds, not 0/3** -- and
    the published document already says so; what needed correcting was
    this round's own re-derivation, which had applied A3b's
    arm-specific criterion to A3a.  Under the pre-registered v2.41
    split, A3a's primary read-out is the field maximum and A3b's is
    the gradient concentration; both lines must be reported for both
    arms.  A3a's gradient is 0.5526/0.5666/0.4782 => 1/3 (seed 2
    passes), exactly as MASTER and the E45 experiment file already
    record.  A3a's OWN discarding trigger -- tail_max_over_p99 > 10 --
    fires on 3/3 seeds (26.5/35.9/27.2), and its field maximum stays
    0/3, so the topology branch stays closed and no verdict moves.

Scope of this write-back
------------------------
* MASTER.md -- header version line, the section 4.47 card row, the
  section 4.48 status bullet (two places: the "4 files" claim and the
  open-questions list), the section 9 row, a section 12 changelog row
  (the five places Gate 3 names);
* INDEX.md -- the E45 row;
* E45 experiment file -- the version pointer and the round note;
* MEMORY.md -- the version pointer, the status section, the to-do item;
* HANDOVER_E45.md -- the version pointer and the round note;
* both daily logs -- the round note.

The historical section 12 rows (v2.44-v2.53) are deliberately NOT
rewritten: they are the audit trail.  Every replacement is anchored on a
literal that must occur exactly the expected number of times; a missing
anchor is an error, never a silent skip.  The new section 12 row is
built from its CELLS; the separators are inserted by the code and the
pipe count is compared against the row it follows.  Running the script
twice is a no-op.

Every marker is PER-FILE and distinctive.  The bare version string is
not usable as a guard: earlier rounds left the text "（＝升 v2.51）" in
the handover and the daily logs, and a guard on the bare string would
skip those files WITHOUT writing anything -- a silent false positive.

A pre-flight pass validates EVERY anchor before a single file is
written, so a typo cannot leave the document set half-written.
"""

from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]

MASTER = ROOT / "docs" / "MASTER.md"
INDEX = ROOT / "docs" / "experiments" / "INDEX.md"
E45DOC = ROOT / "docs" / "experiments" / "E45-gate-direction-amount-separation-baseline.md"
MEMORY = Path("R:/1radarNWPprediction/.workbuddy/memory/MEMORY.md")
HANDOVER = Path("R:/1radarNWPprediction/.workbuddy/memory/HANDOVER_E45.md")
DAILY = Path("R:/1radarNWPprediction/.workbuddy/memory/2026-11-11.md")
DAILY_PROJECT = ROOT / ".workbuddy" / "memory" / "2026-11-16.md"

TAG = "v2.54"
MARKER = "v2.54 第 95 轮"
MARK_MASTER = "| 2026-11-16 | **v2.54** |"
MARK_INDEX = "v2.54 第 95 轮"
MARK_MEMORY = "v2.54 第 95 轮"
MARK_HANDOVER = "v2.54 追加（第 95 轮：最小同步集订正为 5 个文件）"
MARK_DAILY = "## v2.54 第 95 轮（本次运行"
MARK_E45DOC = "v2.54 追加（第 95 轮：最小同步集订正为 5 个文件）"


def row(*cells):
    """One markdown table row from its cells (separators inserted by code)."""
    return "| " + " | ".join(c.strip() for c in cells) + " |"


def esc(text):
    """Escape pipes inside a table cell.

    A cell may legitimately contain a '|' -- `max|dev|` does -- but an
    unescaped one splits the cell and the row's cell count no longer
    matches the header, which breaks the whole table on render.  MASTER
    already follows this convention (32 escaped pipes on disk).
    """
    return text.replace("|", "\\|")


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
# the new statement, written once per length
# --------------------------------------------------------------------------
SYNC5 = (
    "**最小同步集订正为 5 个文件**（不是 4 个）："
    "第 71–75 轮登记的漂移集是 `models/scaled_structure/model.py`、"
    "`models/topo_simvp/model.py`、`tools/_e41_struct_metrics.py`、"
    "`tools/_e45_mechanism_probe.py` 这 **4 个** —— 写下时是对的（当时评测侧还没动），"
    "但 **v2.52 把评测侧的 `scripts/evaluate.py`（`_trained_model_kwargs` 复原）也改了**，"
    "此后没人重跑漂移审计 ⇒ 本轮**实测**（双方归一化 md5 逐文件比对）："
    "漂移集**恰为这 5 个**（`models/scaled_structure/model.py`、"
    "`models/topo_simvp/model.py`、`scripts/evaluate.py`、"
    "`tools/_e41_struct_metrics.py`、`tools/_e45_mechanism_probe.py`），"
    "而对照组（`models/amp_monotone_gate/model.py`、`models/amp_hard_support/model.py`、"
    "`scripts/train.py`、`scripts/_e45_arms.sh`、两个 model yaml）**与 S3 护栏 "
    "`tools/_e41_s3_equalbias.py` 双方逐位相同** ⇒ 其余 E45 运行时代码无漂移。"
    "**为什么这第五个是承重的**：评测侧复原正是缺陷① 里把 `clamp_mm` 带到打分时刻的那一半，"
    "且是**唯一无法从 checkpoint 恢复**的一半（`clamp_mm` 是普通 float 属性、**不进 `state_dict`**）"
    "⇒ **只同步 4 个文件＝重新踩本项目已经付过费的那个坑**"
    "（运行时开关不改 `state_dict` ⇒ 评测端漏传 ⇒ 静默用回被关掉的机制）。"
)

A3A = (
    "**A3a 的梯度读数是 1/3 过（不是 0/3），但拓扑支照旧结掉**："
    "按预注册的 v2.41 分工，**A3a 的主攻读数是场最大值、A3b 的是梯度集中**，两条线两臂都要报。"
    "实测 A3a 的 `grad_top1pct_share` ＝ **0.5526／0.5666／0.4782** ⇒ **种子 2 过**（< 0.5）"
    "⇒ **1/3**（MASTER §4.48 与 E45 实验文件第 8 节早已按 1/3 登记，**本轮核对一致**）；"
    "但 **A3a 自己的舍弃触发条件是 `tail_max_over_p99 > 10`，实测 3/3 全过"
    "（26.55／35.93／27.19）**，且场最大值仍 **0/3 过**（六格全 402.4288 mm ＝ 模型 clamp 上限）"
    "⇒ **拓扑支按预注册结掉，裁决不变**。"
)

REMOTE = (
    "**远端状态复核（只读查询，未 sbatch、未同步、未下载）**：E45 主链终态不变"
    "（提交器 273／训练数组 275·283／评测 274·282·290／汇总 291／297·298／314–317／343 "
    "全 `COMPLETED`）、`squeue -u wf` **空**、21 份 `.eval.json` ＋ 14 份 run 级 json 齐；"
    "**作业 343 结束（06:02:26）之后 `runs/e45` 下被修改的文件 = 0 份**（已发布产物零改动）；"
    "**14 份产物远端↔本地 md5 14/14 逐位全同**（本轮实测）⇒ **无新活动**。"
)

AUDIT = (
    "**裁决独立重导**（`tools/_e45_round95.py`，只读本地，**POS 84/84 VERDICT: PASS**）："
    "① 头条表由 **21 份主跑 `*.eval.json`** 复算 ⇒ 七臂中位数与过地板通过率与已发布值 "
    "**max|dev| = 0.000e+00**（对照 B0 中位数 0.121817／2/3 过地板；"
    "A3a 中位数 0.110015、A3b 0.116055 虽各自 3/3 过地板但**中位数均低于同批对照 B0** ⇒ 不晋级）；"
    "② 机制判据逐格重判（自原始探针 JSON）⇒ 门指向 A1／A2 **3/3 过**、"
    "拓扑两臂**场最大值 0/3 过**、量通道 A4 **3/3 过**／A4b **2/3 过**"
    "（种子 2 触发**相关性分支**，其 `scale_cv` 3.0188 为六格最高 ⇒ **量通道全程未退化成常数**）、"
    "`mean(out)=T` 仍 **4/6 格不达**（只记**实现缺陷**、不当机制结论）；"
    "③ S3 **数值级重算**（`ci_lo ≤ 0 ≤ ci_hi`，不读存储布尔）⇒ **12/12 区间跨零**、"
    "存储标志与数值 0 处不自洽、失效模式**分两支报**"
    "（`GAIN_WAS_AREA_OR_AMPLITUDE` 4 臂／`NO_GAIN_TO_BEGIN_WITH` 2 臂）；"
    "④ 种子稳健性扫读 `s3_seedrobust.json` ⇒ **两侧各 6 臂 × 20 个种子 = 240/240 格全含零**、"
    "按预注册晋级判据（点估 > 0 且区间不含零）**合格种子数 = 0**、"
    "注册种子**逐位复现已发布数字（max|dev| = 0）**、六臂 `max_abs_bias_gap` 全 0.0。"
)

VERDICT = "**裁决不变＝六臂判负、无一臂进十 seed、不投十 seed 算力**。"

BODY = REMOTE + SYNC5 + A3A + AUDIT + VERDICT

# short form for the MASTER header cell
HEAD_SHORT = (
    "远端无新活动（主链全 `COMPLETED`、队列空、**343 结束后 `runs/e45` 零改动**、"
    "**14 份产物 md5 14/14 全同**）；**最小同步集订正为 5 个文件**"
    "（补上 v2.52 的 `scripts/evaluate.py` 评测侧复原 —— 那是把 `clamp_mm` 带到打分时刻、"
    "且**唯一无法从 checkpoint 恢复**的一半 ⇒ 只同步 4 个＝重踩旧坑）；"
    "**A3a 梯度读数 1/3 过**（0.4782，种子 2）但 A3a 自身触发条件 `tail_max_over_p99 > 10` "
    "3/3 全过、场最大值 0/3 ⇒ **拓扑支照旧结掉**；"
    "**裁决不变＝六臂判负、不投十 seed 算力**。"
)

# --------------------------------------------------------------------------
# MASTER
# --------------------------------------------------------------------------
HEADER_V53_OLD = "**v2.53（2026-11-16，最新）**"
HEADER_V53_NEW = "**v2.53（2026-11-16）**"
HEADER_TAIL_OLD = "**裁决不变＝六臂判负、不投十 seed 算力**。 |"
HEADER_TAIL_NEW = (
    "**裁决不变＝六臂判负、不投十 seed 算力**；**v2.54（2026-11-16，最新）**：" + HEAD_SHORT + " |"
)

CARD_A_OLD = "**裁决不变＝六臂判负、无一臂进十 seed、不投十 seed 算力**。 |"
CARD_A_NEW = "**裁决不变＝六臂判负、无一臂进十 seed、不投十 seed 算力** ｜ **v2.54**：" + esc(BODY) + " |"
CARD_B_OLD = "v2.53 见 §4.48 状态行 |"
CARD_B_NEW = "v2.53 见 §4.48 状态行；v2.54 见 §4.48 状态行 |"

# the "4 files" claim in the section 4.48 status bullet
S448_SYNC_OLD = "- **状态**：4 个修复文件只在本地，**未同步远端、未重训重测** ⇒ 不改任何裁决。"
S448_SYNC_NEW = "- **状态**：5 个修复文件只在本地，**未同步远端、未重训重测** ⇒ 不改任何裁决。"
S448_TAIL_OLD = "最小同步集仍是这 4 个文件。"
S448_TAIL_NEW = "最小同步集**已订正为 5 个文件**（本轮实测，补上 `scripts/evaluate.py`）。"

S448_OLD = "**唯一阻塞仍是同账号配额**，释放即可跑、**无需重投**。"
S448_NEW = (
    S448_OLD + " **v2.54 另记**：远端无新活动（主链全 `COMPLETED`、队列空、"
    "**343 结束后 `runs/e45` 被修改文件 = 0**、**14 份产物 md5 14/14 全同**）；"
    + SYNC5 + A3A + VERDICT
)

# the open-questions list: the sync item is now five files
S449_OLD = "- ① 4 个修复文件是否按最小集同步远端（会覆盖远端文件，属状态改变，不擅自动）；"
S449_NEW = "- ① **5 个**修复文件是否按最小集同步远端（会覆盖远端文件，属状态改变，不擅自动）；"

S9_OLD = "v2.53 见 §4.48 状态行（**作业 343 已跑完：两侧 20/20 种子全跨零、A2 不翻转**） |"
S9_NEW = (
    "v2.53 见 §4.48 状态行（**作业 343 已跑完：两侧 20/20 种子全跨零、A2 不翻转**）；"
    "v2.54 见 §4.48 状态行（**最小同步集订正为 5 个文件** ＋ **A3a 梯度读数 1/3 过**） |"
)

CHANGELOG_ROW = row(*[esc(c) for c in (
    "2026-11-16",
    f"**{TAG}**",
    "**E45 第 95 轮：远端状态复核 ＋ 最小同步集由 4 个订正为 5 个 ＋ A3a 梯度读数的口径核对**"
    "：远端主链终态不变（提交器 273／训练数组 275·283／评测 274·282·290／汇总 291／297·298／"
    "314–317／343 全 `COMPLETED`）、`squeue -u wf` **空**、21 份 `.eval.json` ＋ 14 份 run 级 json 齐；"
    "**作业 343 结束（2026-11-16 06:02:26）之后 `runs/e45` 下被修改的文件 = 0 份**（已发布产物零改动）；"
    "**14 份产物远端↔本地 md5 14/14 逐位全同**（本轮实测）⇒ **无新活动**。"
    "**远端动作**：只读查询（`squeue`／`sacct`／`md5sum`／`find`／`ls`）；"
    "**未 sbatch、未同步、未下载、未改远端状态**。"
    "**本轮唯一实质发现（对已发布文本的订正）**：第 71–75 轮登记的「最小同步集仍是这 4 个文件」"
    "**已过时** —— **v2.52 把评测侧的 `scripts/evaluate.py`（`_trained_model_kwargs` 复原）也改了**，"
    "此后没人重跑漂移审计 ⇒ 本轮**实测**（双方归一化 md5 逐文件比对）：漂移集**恰为 5 个**"
    "（`models/scaled_structure/model.py`、`models/topo_simvp/model.py`、`scripts/evaluate.py`、"
    "`tools/_e41_struct_metrics.py`、`tools/_e45_mechanism_probe.py`），"
    "对照组与 **S3 护栏 `tools/_e41_s3_equalbias.py` 双方逐位相同** ⇒ 其余 E45 运行时代码无漂移。"
    "**为什么第五个是承重的**：评测侧复原正是缺陷① 里把 `clamp_mm` 带到打分时刻的那一半，"
    "且是**唯一无法从 checkpoint 恢复**的一半（`clamp_mm` 是普通 float 属性、**不进 `state_dict`**）"
    "⇒ **只同步 4 个文件＝重踩本项目已付过费的坑**（运行时开关不改 `state_dict` ⇒ 评测端漏传 ⇒ 静默用回旧机制）。"
    "**口径核对（不改裁决）**：按预注册的 v2.41 分工（A3a 主读场最大值、A3b 主读梯度集中，"
    "两条线两臂都要报），实测 **A3a 的 `grad_top1pct_share` ＝ 0.5526／0.5666／0.4782 ⇒ 1/3 过**"
    "（种子 2；MASTER 与 E45 实验文件早已按 1/3 登记）；但 **A3a 自身的舍弃触发条件 "
    "`tail_max_over_p99 > 10` 3/3 全过**（26.55／35.93／27.19）、场最大值 **0/3 过**"
    "（六格全 402.4288 mm ＝ 模型 clamp 上限）⇒ **拓扑支按预注册结掉，裁决不变**。"
    "**裁决独立重导**（`tools/_e45_round95.py`，只读本地，**POS 84/84 VERDICT: PASS**）："
    "头条表由 21 份主跑 `*.eval.json` 复算 ⇒ **max|dev| = 0**（B0 中位数 0.121817／2/3；"
    "A3a 0.110015、A3b 0.116055 虽 3/3 过地板但**中位数均低于同批对照 B0** ⇒ 不晋级）；"
    "机制判据逐格重判与已发布一致（门指向 3/3、拓扑场最大值 0/3、量通道 A4 3/3／A4b 2/3、"
    "`mean(out)=T` 4/6 不达**只记实现缺陷**）；S3 **数值级重算** ⇒ **12/12 跨零**、"
    "失效模式**分两支报**；种子扫描 **240/240 格全含零**、**合格种子数 0**、"
    "注册种子**逐位复现（max|dev| = 0）**。"
    "**裁决零变化**（六臂仍全部判负、无一臂进十 seed、**不投十 seed 算力**）；"
    "**下一步（不变）**：§4.5 三件实现缺陷修复（① 训练侧再接一次 `--model-kwargs {\"clamp_mm\":20.0}`；"
    "② 量通道改相对护栏破绝对 `1e-12`；③ `maxratio` 改 `np.nanmedian` ＋ 报可用帧数）"
    "＋ **重开须重新提机制族**；**新增一条操作纪律：同步远端时按 5 文件最小集，"
    "且训练侧与评测侧必须同时携带 `clamp_mm`**。",
    "`tools/_e45_round95.py` · `tools/_evals/e45/round95_status_audit.json` · "
    "`docs/experiments/E45-gate-direction-amount-separation-baseline.md`",
)])

# --------------------------------------------------------------------------
# INDEX
# --------------------------------------------------------------------------
INDEX_OLD = "；**裁决不变＝六臂判负、不投十 seed 算力** |"
INDEX_NEW = esc(
    "；**裁决不变＝六臂判负、不投十 seed 算力** ｜ **v2.54 第 95 轮**："
    "远端无新活动（主链全 `COMPLETED`、队列空、**343 结束后 `runs/e45` 零改动**、"
    "**14 份产物 md5 14/14 全同**）；**最小同步集由 4 个订正为 5 个**"
    "（补上 v2.52 的 `scripts/evaluate.py` 评测侧复原 —— 那是把 `clamp_mm` 带到打分时刻、"
    "且**唯一无法从 checkpoint 恢复**的一半 ⇒ 只同步 4 个＝重踩旧坑）；"
    "**A3a 梯度读数 1/3 过**（0.4782，种子 2）但 A3a 自身触发条件 `tail_max_over_p99 > 10` "
    "3/3 全过、场最大值 0/3 ⇒ **拓扑支照旧结掉**；"
    "**裁决不变＝六臂判负、不投十 seed 算力**"
) + " |"

# --------------------------------------------------------------------------
# E45 experiment file
# --------------------------------------------------------------------------
E45DOC_PTR_OLD = "> **2026-11-16（v2.53）**："
E45DOC_PTR_NEW = (
    "> **2026-11-16（v2.54）**：**E45 内容与判据自 v2.49 起未变**（S0–S6 一字未改；"
    "裁决不变：六臂判负、不投十 seed 算力）；第 95 轮远端状态复核无新活动，"
    "并订正一处已过时的登记：**最小同步集由 4 个文件订正为 5 个**"
    "（补上 v2.52 落地的 `scripts/evaluate.py` 评测侧 `_trained_model_kwargs` 复原 —— "
    "那是把 `clamp_mm` 带到打分时刻、且**唯一无法从 checkpoint 恢复**的一半）；"
    "同时核对 **A3a 的 `grad_top1pct_share` 为 1/3 过**（0.5526／0.5666／0.4782，种子 2）——"
    "与本文第 8 节的登记一致；**A3a 自身触发条件 `tail_max_over_p99 > 10` 3/3 全过**、"
    "场最大值 0/3 ⇒ **拓扑支照旧结掉**。"
    "**裁决不变＝六臂判负、不投十 seed 算力**（详见 §8.4 结案段与交接文档）。\n"
    "> **2026-11-16（v2.53）**："
)

E45DOC_TODO_OLD = (
    "⇒ **下一轮臂表**接 `--model-kwargs {\"clamp_mm\":20.0}` 即可（已实测该写法经臂表自己的 "
    "`read -r -a` 分词**仍是一个 token**、且经 `model_kwargs()` 真的进到构造参数）。"
)
E45DOC_TODO_NEW = (
    E45DOC_TODO_OLD + "\n"
    "  **v2.54 追加（第 95 轮：最小同步集订正为 5 个文件）**：同步远端时的最小集是 **5 个文件**"
    "（`models/scaled_structure/model.py`、`models/topo_simvp/model.py`、`scripts/evaluate.py`、"
    "`tools/_e41_struct_metrics.py`、`tools/_e45_mechanism_probe.py`），"
    "**不是原先登记的 4 个** —— 原登记写下时评测侧还没改，v2.52 之后未重跑漂移审计；"
    "本轮实测漂移集恰为这 5 个，对照组与 S3 护栏双方逐位相同。"
    "**第五个（`scripts/evaluate.py`）是承重的**：它是把 `clamp_mm` 带到打分时刻、"
    "且**唯一无法从 checkpoint 恢复**（`clamp_mm` 不进 `state_dict`）的那一半。"
    "**三件待办本身不变**（① 训练侧还需接 `--model-kwargs {\"clamp_mm\":20.0}` 一次；"
    "② 量通道改相对护栏破绝对 `1e-12`；③ `maxratio` 改 `np.nanmedian` ＋ 报可用帧数）。"
)

# --------------------------------------------------------------------------
# MEMORY.md
# --------------------------------------------------------------------------
MEM_PTR_OLD = "> 权威 `docs/MASTER.md` **v2.53**（"
MEM_PTR_NEW = "> 权威 `docs/MASTER.md` **v2.54**（"
MEM_PTR_TAIL_OLD = "**作业 343 已跑完：S3 两侧各 20 个 bootstrap 种子全跨零、A2 不翻转**）；"
MEM_PTR_TAIL_NEW = (
    "**作业 343 已跑完：S3 两侧各 20 个 bootstrap 种子全跨零、A2 不翻转**、"
    "**最小同步集订正为 5 个文件**）；"
)

MEM_STATUS_OLD = (
    "**v2.53 追加**：作业 343 已跑完，**S3 两侧的 bootstrap 种子稳健性收口** ⇒ "
    "**A2 不翻转** ⇒ 待办清单本身**不变**（① 训练侧还需接一次 `--model-kwargs {\"clamp_mm\":20.0}`；"
    "② 量通道改相对护栏；③ `maxratio` 改 `np.nanmedian`）。"
)
MEM_STATUS_NEW = (
    MEM_STATUS_OLD + " **v2.54 第 95 轮（2026-11-16）**：远端无新活动（主链全 `COMPLETED`、队列空、"
    "**343 结束后 `runs/e45` 零改动**、**14 份产物 md5 14/14 全同**）；"
    "**最小同步集由 4 个订正为 5 个文件**（补上 v2.52 的 `scripts/evaluate.py` 评测侧复原 —— "
    "那是把 `clamp_mm` 带到打分时刻、且**唯一无法从 checkpoint 恢复**的一半 ⇒ 只同步 4 个＝重踩旧坑）；"
    "**A3a 梯度读数 1/3 过**（0.4782，种子 2，与已发布一致）但 A3a 自身触发条件 "
    "`tail_max_over_p99 > 10` 3/3 全过、场最大值 0/3 ⇒ **拓扑支照旧结掉**。"
    "**裁决不变＝六臂判负、不投十 seed 算力**；待办清单本身**不变**。"
)

MEM_TODO_OLD = (
    "5. 待用户裁决：① 4 个修复文件按最小集同步远端（会覆盖远端文件）；"
)
MEM_TODO_NEW = (
    "5. 待用户裁决：① **5 个**修复文件按最小集同步远端（会覆盖远端文件；"
    "**v2.54 订正：含 `scripts/evaluate.py` 评测侧复原，第五个是承重的**）；"
)

MEM_R95_NEW = (
    "- **E45 第 95 轮（2026-11-16）**：远端状态复核无新活动（主链全 `COMPLETED`、`squeue` 空、"
    "**作业 343 结束后 `runs/e45` 被修改文件 = 0**、**14 份产物远端↔本地 md5 14/14 全同**）；"
    "**唯一实质发现是订正一处已过时的登记**：**最小同步集由 4 个文件订正为 5 个** —— "
    "**v2.52 落地了评测侧的 `scripts/evaluate.py`（`_trained_model_kwargs` 复原）**，"
    "此后未重跑漂移审计；本轮实测（双方归一化 md5 逐文件比对）漂移集**恰为 5 个**，"
    "对照组与 **S3 护栏双方逐位相同** ⇒ **第五个是承重的**（`clamp_mm` 不进 `state_dict` ⇒ "
    "评测端漏传就静默用回旧机制）。另核对 **A3a 的 `grad_top1pct_share` 为 1/3 过**"
    "（0.5526／0.5666／0.4782，种子 2；与已发布一致），但 **A3a 自身触发条件 "
    "`tail_max_over_p99 > 10` 3/3 全过**、场最大值 0/3 ⇒ **拓扑支照旧结掉**。"
    "独立重导（`tools/_e45_round95.py`，只读本地，**POS 84/84 VERDICT: PASS**）：头条表 max|dev| = 0、"
    "机制判据逐格与已发布一致、S3 **12/12 跨零**、种子扫描 **240/240 全含零**、合格种子数 0。"
    "**裁决不变＝六臂判负、无一臂进十 seed、不投十 seed 算力**；"
    "下一步仍是 §4.5 三件实现缺陷修复 ＋ **重开须重新提机制族**。"
)

# --------------------------------------------------------------------------
# HANDOVER_E45.md
# --------------------------------------------------------------------------
HO_PTR_OLD = "Gate 3 写回已完成（MASTER **v2.53**）。"
HO_PTR_NEW = "Gate 3 写回已完成（MASTER **v2.54**）。"

HO_TODO2_OLD = (
    "⇒ **待办② 从「两侧都要补」缩到「只差训练侧一次」**（下一轮臂表接 "
    "`--model-kwargs {\"clamp_mm\":20.0}`，已实测经臂表的 `read -r -a` 仍是一个 token）。"
)
HO_TODO2_NEW = (
    HO_TODO2_OLD + "\n\n**v2.54 追加（第 95 轮：最小同步集订正为 5 个文件）**："
    "远端状态复核无新活动（主链全 `COMPLETED`、队列空、**343 结束后 `runs/e45` 零改动**、"
    "**14 份产物 md5 14/14 全同**）⇒ **裁决不变＝六臂判负、不投十 seed 算力**。"
    "**最小同步集订正为 5 个文件**（不是 4 个）：第 71–75 轮登记时评测侧还没改，"
    "**v2.52 之后没人重跑漂移审计** ⇒ 本轮实测（双方归一化 md5 逐文件比对）漂移集**恰为 5 个**"
    "（`models/scaled_structure/model.py`、`models/topo_simvp/model.py`、`scripts/evaluate.py`、"
    "`tools/_e41_struct_metrics.py`、`tools/_e45_mechanism_probe.py`），对照组与 S3 护栏双方逐位相同。"
    "**第五个是承重的**（评测侧复原＝把 `clamp_mm` 带到打分时刻、且唯一无法从 checkpoint 恢复的一半）。"
    "另核对 **A3a 的 `grad_top1pct_share` 为 1/3 过**（0.4782，种子 2，与已发布一致），"
    "但 A3a 自身触发条件 `tail_max_over_p99 > 10` 3/3 全过、场最大值 0/3 ⇒ **拓扑支照旧结掉**。"
    "**三件待办本身不变**。"
)

# --------------------------------------------------------------------------
# patchers
# --------------------------------------------------------------------------
def patch_master(text):
    if MARK_MASTER in text:
        print("MASTER already carries the v2.54 changelog row -> skip")
        return text
    # the "4 files" claim must be corrected in the 4.48 status bullet.
    # ORDER MATTERS: the status bullet's PREFIX changes too, so the
    # later anchor (which still starts with "4 个修复文件") must be
    # patched BEFORE the prefix-changing replacement.
    text = replace_in_line(text, "- **状态**：4 个修复文件只在本地", S448_OLD, S448_NEW,
                           "master.s448.status")
    text = replace_in_line(text, "- **状态**：", S448_SYNC_OLD, S448_SYNC_NEW,
                           "master.s448.files")
    text = replace_in_line(text, "- **状态**：", S448_TAIL_OLD, S448_TAIL_NEW,
                           "master.s448.synccount")
    text = replace_in_line(text, "- ① 4 个修复文件是否", S449_OLD, S449_NEW,
                           "master.s449.open")
    text = replace_in_line(text, "| 最后更新 |", HEADER_V53_OLD, HEADER_V53_NEW,
                           "master.header.v53tag")
    text = replace_in_line(text, "| 最后更新 |", HEADER_TAIL_OLD, HEADER_TAIL_NEW,
                           "master.header.tail")
    text = replace_in_line(text, "| §4.47 | 修三处机制缺陷", CARD_A_OLD, CARD_A_NEW,
                           "master.card.body")
    text = replace_in_line(text, "| §4.47 | 修三处机制缺陷", CARD_B_OLD, CARD_B_NEW,
                           "master.card.refs")
    text = replace_in_line(text, "| **E45 / §4.47 修三处机制缺陷 + 摆正基准** |", S9_OLD, S9_NEW,
                           "master.s9.row")
    # insert the v2.54 changelog row right after the v2.53 row
    lines = text.split("\n")
    idx = [i for i, l in enumerate(lines) if l.startswith("| 2026-11-16 | **v2.53** |")]
    if len(idx) != 1:
        raise SystemExit(f"CHANGELOG ANCHOR MISS: found {len(idx)} v2.53 changelog rows")
    i = idx[0]
    prev = unescaped_pipes(lines[i])
    new = unescaped_pipes(CHANGELOG_ROW)
    if prev != new:
        raise SystemExit(f"PIPE COUNT MISMATCH: v2.53 row {prev} cells vs new row {new} cells")
    lines.insert(i + 1, CHANGELOG_ROW)
    return "\n".join(lines)


def patch_index(text):
    if MARK_INDEX in text:
        print("INDEX already carries", MARK_INDEX, "-> skip")
        return text
    return replace_in_line(text, "| E45 | §4.47（v2.49 压缩总览） |", INDEX_OLD, INDEX_NEW,
                           "index.e45")


def patch_e45doc(text):
    if MARK_E45DOC in text:
        print("E45DOC already carries the marker -> skip")
        return text
    text = apply(text, E45DOC_PTR_OLD, E45DOC_PTR_NEW, 1, "e45doc.pointer")
    text = apply(text, E45DOC_TODO_OLD, E45DOC_TODO_NEW, 1, "e45doc.todo")
    return text


def patch_memory(text):
    if MARK_MEMORY in text:
        print("MEMORY already carries", MARK_MEMORY, "-> skip")
        return text
    text = apply(text, MEM_PTR_OLD, MEM_PTR_NEW, 1, "memory.header.version")
    text = apply(text, MEM_PTR_TAIL_OLD, MEM_PTR_TAIL_NEW, 1, "memory.header.tail")
    text = apply(text, MEM_STATUS_OLD, MEM_STATUS_NEW, 1, "memory.status")
    text = apply(text, MEM_TODO_OLD, MEM_TODO_NEW, 1, "memory.todo")
    # append the round-95 bullet after the round-94 one
    lines = text.split("\n")
    r94 = [i for i, l in enumerate(lines) if l.startswith("- **E45 第 94 轮（2026-11-16）**")]
    if len(r94) != 1:
        raise SystemExit(f"MEMORY: expected exactly 1 round-94 bullet, found {len(r94)}")
    lines.insert(r94[0] + 1, MEM_R95_NEW)
    return "\n".join(lines)


def patch_handover(text):
    if MARK_HANDOVER in text:
        print("HANDOVER already carries the marker -> skip")
        return text
    text = apply(text, HO_PTR_OLD, HO_PTR_NEW, 1, "handover.pointer")
    text = apply(text, HO_TODO2_OLD, HO_TODO2_NEW, 1, "handover.todo2")
    return text


ROUND_NOTE = """
<!-- E45-R95-BEGIN -->
### 2026-11-16｜第 95 轮（E45 远端状态复核 ＋ **最小同步集由 4 个订正为 5 个** ＋ **A3a 梯度读数口径核对**）

- **判定**：远端 E45 主链终态不变（提交器 273／训练数组 275·283／评测 274·282·290／汇总 291／297·298／314–317／343 全 `COMPLETED`）、`squeue -u wf` **空**、21 份 `.eval.json` ＋ 14 份 run 级 json 齐；**作业 343 结束（06:02:26）之后 `runs/e45` 下被修改的文件 = 0 份**（已发布产物零改动）；**14 份产物远端↔本地 md5 14/14 逐位全同**（本轮实测）⇒ **无新活动**。**远端动作**：只读查询；**未 sbatch、未同步、未下载、未改远端状态**。
- **本轮唯一实质发现（对已发布文本的订正）**：第 71–75 轮登记的「**最小同步集仍是这 4 个文件**」**已过时** —— **v2.52 把评测侧的 `scripts/evaluate.py`（`_trained_model_kwargs` 复原）也改了**，此后没人重跑漂移审计 ⇒ 本轮**实测**（双方**归一化** md5 逐文件比对）：漂移集**恰为 5 个**（`models/scaled_structure/model.py`、`models/topo_simvp/model.py`、`scripts/evaluate.py`、`tools/_e41_struct_metrics.py`、`tools/_e45_mechanism_probe.py`），对照组（`amp_monotone_gate`、`amp_hard_support`、`scripts/train.py`、`scripts/_e45_arms.sh`、两个 model yaml）与 **S3 护栏 `tools/_e41_s3_equalbias.py` 双方逐位相同** ⇒ 其余 E45 运行时代码无漂移。**为什么第五个是承重的**：评测侧复原正是缺陷① 里把 `clamp_mm` 带到打分时刻的那一半，且是**唯一无法从 checkpoint 恢复**的一半（`clamp_mm` 是普通 float 属性、**不进 `state_dict`**）⇒ **只同步 4 个文件＝重新踩本项目已经付过费的那个坑**（运行时开关不改 `state_dict` ⇒ 评测端漏传 ⇒ 静默用回被关掉的机制）。
- **口径核对（不改裁决）**：按预注册的 **v2.41 分工**（A3a 主读场最大值、A3b 主读梯度集中，两条线两臂都要报），实测 **A3a 的 `grad_top1pct_share` ＝ 0.5526／0.5666／0.4782 ⇒ 1/3 过**（种子 2）—— 与 MASTER §4.48、E45 实验文件第 8 节的登记**一致**；但 **A3a 自身的舍弃触发条件 `tail_max_over_p99 > 10` 3/3 全过**（26.55／35.93／27.19）、场最大值仍 **0/3 过**（六格全 402.4288 mm ＝ 模型 clamp 上限）⇒ **拓扑支按预注册结掉**。
- **裁决独立重导**（`tools/_e45_round95.py`，只读本地，**POS 84/84 VERDICT: PASS**）：① 头条表由 **21 份主跑 `*.eval.json`** 复算 ⇒ 七臂中位数与过地板通过率与已发布值 **max|dev| = 0.000e+00**（对照 B0 中位数 0.121817／2/3 过地板；A3a 0.110015、A3b 0.116055 虽各自 3/3 过地板但**中位数均低于同批对照 B0** ⇒ 不晋级）；② 机制判据逐格重判（自原始探针 JSON）⇒ 门指向 A1／A2 **3/3 过**、拓扑两臂**场最大值 0/3 过**、量通道 A4 **3/3 过**／A4b **2/3 过**（种子 2 触发**相关性分支**，其 `scale_cv` 3.0188 为六格最高 ⇒ **量通道全程未退化成常数**）、`mean(out)=T` 仍 **4/6 格不达**（只记**实现缺陷**、不当机制结论）；③ S3 **数值级重算**（`ci_lo ≤ 0 ≤ ci_hi`，不读存储布尔）⇒ **12/12 区间跨零**、存储标志与数值 0 处不自洽、失效模式**分两支报**（`GAIN_WAS_AREA_OR_AMPLITUDE` 4 臂／`NO_GAIN_TO_BEGIN_WITH` 2 臂）；④ 种子扫描读 `s3_seedrobust.json` ⇒ **两侧各 6 臂 × 20 个种子 = 240/240 格全含零**、按预注册晋级判据（点估 > 0 且区间不含零）**合格种子数 = 0**、注册种子**逐位复现已发布数字（max|dev| = 0）**、六臂 `max_abs_bias_gap` 全 0.0。
- **裁决零变化**（六臂仍全部判负、无一臂进十 seed、**不投十 seed 算力**）；**下一步（不变）**：§4.5 三件实现缺陷修复（① 训练侧再接一次 `--model-kwargs {"clamp_mm":20.0}`；② 量通道改相对护栏破绝对 `1e-12`；③ `maxratio` 改 `np.nanmedian` ＋ 报可用帧数）＋ **重开须重新提机制族**；**新增一条操作纪律：同步远端按 5 文件最小集，且训练侧与评测侧必须同时携带 `clamp_mm`**。
- **本轮自身失误与修正（记录以防重犯）**：新写的审计脚本首跑报 3 条 FAIL，**全部是脚本自身的错误预期** —— ① 拿**远端原始 md5** 比**本地归一化 md5**（`scripts/train.py` 双方仅差行尾，被误判为漂移；改用双方归一化 md5 后该文件逐位相同）；② 把 `pass_rate_over_floor`（存的是 `"2/3"` 字符串）当整数读；③ 把 **A3b 的臂专属判据套到 A3a** 上（应为 A3a 主读场最大值、A3b 主读梯度集中，两条线两臂都要报）。⇒ **纪律：新增审计项必须先核对原始布局与预注册的臂分工，再写断言；跨机比较必须双方同口径（归一化）。**
- **产物**：`tools/_e45_round95.py`、`tools/_evals/e45/round95_status_audit.json`、`tools/_sync_v254.py`。
<!-- E45-R95-END -->
"""


def append_daily(text):
    if MARK_DAILY in text:
        print("daily log already carries", MARK_DAILY, "-> skip")
        return text
    note = (
        f"\n\n## {MARKER}（本次运行，2026-11-16）· E45 状态复核 ＋ "
        "**最小同步集订正为 5 个文件** ＋ **A3a 梯度读数口径核对**\n\n" + ROUND_NOTE
    )
    return text + note


def append_handover_round(text):
    """Append the round note to the handover, under its own marker."""
    if "<!-- E45-R95-BEGIN -->" in text:
        print("HANDOVER already carries the R95 round note -> skip")
        return text
    return text + ROUND_NOTE


PLAN = [
    (MASTER, patch_master),
    (INDEX, patch_index),
    (E45DOC, patch_e45doc),
    (MEMORY, patch_memory),
    (HANDOVER, patch_handover),
    (HANDOVER, append_handover_round),
    (DAILY, append_daily),
    (DAILY_PROJECT, append_daily),
]


def main():
    # ---- pre-flight: validate EVERY anchor before writing anything ----
    print("pre-flight: checking every anchor")
    for path, fn in PLAN:
        if not path.exists():
            raise SystemExit(f"FILE MISSING: {path}")
        original = path.read_text(encoding="utf-8")
        try:
            fn(original)          # dry run: raises on a missing anchor
        except SystemExit as exc:
            raise SystemExit(f"PRE-FLIGHT FAILED for {path}:\n{exc}")
        print(f"  ok: {path.name}")

    print("\nwriting:")
    changed = []
    for path, fn in PLAN:
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
