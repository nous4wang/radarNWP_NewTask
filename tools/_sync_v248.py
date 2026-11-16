"""Write back the E45 scale-branch correction (v2.48).

What v2.48 corrects
-------------------
``scale_is_constant`` is a DERIVED BOOLEAN and, per the probe's own
definition (tools/_e45_mechanism_probe.py, l.63-64 and l.679-681), a
DISJUNCTION of two independent branches:

    CV_BRANCH   : scale_cv < 0.05        -> the amount stopped varying
    CORR_BRANCH : scale_truth_corr < 0.3 -> the amount still varies but no
                                           longer tracks the rain

The E45 conclusion prose reported the A4b seed-2 cell as
"退化成常数" ("degenerated into a constant"), i.e. it asserted the CV
branch fired.  Recomputing both branches from the raw probe artefacts
(``tools/_e45_verify_round43.py``) shows:

* the CV branch fires in NO cell of E45 -- the smallest scale_cv over the
  six cells is 1.2358, i.e. 24.7x the 0.05 floor;
* the CORR branch fires exactly once, in A4b seed 2;
* A4b seed 2 has the HIGHEST scale_cv of the six cells (3.0188) -- its
  amount channel is the MOST variable, not a constant;
* what actually happened there is a COLLAPSE IN MAGNITUDE
  (scale_mean 0.03155, under 1/11 of the smaller A4b peer) plus loss of
  correlation with the truth.

The pre-registered consequence for A4b is already correlation-flavoured
("既有解码头的平均雨强不足以承担量通道", E45 §3.1b/§3.2), so the prose
contradicted the very rule it cites.

The verdict does NOT change: A4b seed 2's flag is true, so the
pre-registered discard fires whichever branch produced it; all six arms
stay negative and no arm enters the 10-seed line.

Scope of this write-back
------------------------
* MASTER.md -- header version line, §4.47 card, §9 row, §12 changelog
  row (four places, Gate 3);
* INDEX.md -- the E45 row;
* the E45 experiment file -- the §8.4 conclusion (item 3);
* MEMORY.md -- the version pointer and the status section;
* HANDOVER_E45.md -- the version pointer, the seed-2 warning block and a
  new registered to-do;
* both daily logs -- the round note.

The historical §12 rows (v2.44-v2.47) are deliberately NOT rewritten:
they are the audit trail, and the new v2.48 row discloses that their
"退化成常数" wording is superseded.  Every replacement is anchored on a
literal that must occur exactly the expected number of times, a missing
anchor is an error, never a silent skip.  The new §12 row is built from
its CELLS; the separators are inserted by the code, and the pipe count
is compared against the row it follows.  Running the script twice is a
no-op.
"""
from pathlib import Path
import re
import sys

MASTER = Path("docs/MASTER.md")
INDEX = Path("docs/experiments/INDEX.md")
E45DOC = Path("docs/experiments/E45-gate-direction-amount-separation-baseline.md")
MEMORY = Path("R:/1radarNWPprediction/.workbuddy/memory/MEMORY.md")
HANDOVER = Path("R:/1radarNWPprediction/.workbuddy/memory/HANDOVER_E45.md")
DAILY = Path("R:/1radarNWPprediction/.workbuddy/memory/2026-11-13.md")
DAILY_PROJECT = Path(".workbuddy/memory/2026-11-13.md")

TAG = "v2.48"
OLD_TAG = "v2.47"


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
    """Replace `old` by `new` inside the single line starting with `line_prefix`.

    Used where the same sentence occurs in more than one table (a
    conclusion row and a historical changelog row) and only the
    conclusion row may be touched.
    """
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


def insert_after_line(text, line_prefix, new_line, label=""):
    """Insert `new_line` right after the single line starting with `line_prefix`."""
    if new_line.strip() in text:
        return text          # already applied
    lines = text.split("\n")
    hits = [i for i, l in enumerate(lines) if l.startswith(line_prefix)]
    if len(hits) != 1:
        raise SystemExit(
            f"LINE ANCHOR MISS ({label}): prefix {line_prefix[:60]!r} "
            f"matched {len(hits)} line(s), expected 1")
    lines.insert(hits[0] + 1, new_line)
    return "\n".join(lines)


# --------------------------------------------------------------------------
# the corrected reading, stated once
# --------------------------------------------------------------------------
BRANCH_NOTE = (
    "**触发的是相关性分支，不是「退化成常数」**：该种子的 `scale_cv` 3.0188 是六个格子里"
    "**最高**的，量通道没有变成常数；真正发生的是**量级塌缩**（`scale_mean` 0.0316，"
    "不足同臂另两个种子 0.407／0.353 的 1/11）**＋ 与真值脱钩**"
    "（`scale_truth_corr` 0.2789 < 0.3）"
)

# --------------------------------------------------------------------------
# MASTER
# --------------------------------------------------------------------------
HEADER_OLD = (
    "2026-11-13（v2.47，**E45 三个种子全部跑完并结案＝负面：六条臂全部判负，"
    "没有一条进得了十 seed 正式跑**；v2.47 首次把**文档里印的数字**与原始产物逐格对账，"
    "查出并订正 §8.2 seed-0 表 A4 `CSI@2.0` 的一个末位笔误（0.175501 → 0.175500），"
    "**裁决不变**）"
)
HEADER_NEW = (
    "2026-11-13（v2.48，**E45 三个种子全部跑完并结案＝负面：六条臂全部判负，"
    "没有一条进得了十 seed 正式跑**；v2.47 首次把**文档里印的数字**与原始产物逐格对账，"
    "查出并订正 §8.2 seed-0 表 A4 `CSI@2.0` 的一个末位笔误（0.175501 → 0.175500）；"
    "v2.48 把**派生布尔读数拆成它的触发分支**再判读，查出并订正结论正文里"
    "「A4b 第三个种子**退化成常数**」的**误标**——**E45 全程没有任何一格真的退化成常数**"
    "（最小 `scale_cv` 1.2358 是 0.05 底线的 24.7 倍），A4b 种子 2 触发的是"
    "**相关性分支**（`scale_truth_corr` 0.2789 < 0.3，且其 `scale_cv` 3.0188 恰是六格最高）"
    "⇒ 实际是**量级塌缩＋与真值脱钩**；**裁决不变**）"
)

HEADER_SCALE_OLD = (
    "**量通道确实在做量**（A4 三个种子都没退化；A4b 第三个种子退化 ⇒ 只在 2/3 seed 成立）"
)
HEADER_SCALE_NEW = (
    "**量通道确实在做量**（A4 三个种子都没退化；A4b 第三个种子**触发退化下限的相关性分支**"
    "⇒ 只在 2/3 seed 成立；**E45 全程没有任何一格真的「退化成常数」**——最小 `scale_cv` "
    "1.2358 是底线的 24.7 倍）"
)

CARD_PREFIX = "| **E45 / §4.47 修三处机制缺陷 + 摆正基准** |"
CARD_OLD = (
    "**A4b 只在 2/3 seed 成立**（`scale_cv` 1.89～3.02、与真值相关 0.66～0.71，"
    "第三个种子 `scale_truth_corr` 0.2789 < 0.3 ⇒ 退化成常数）"
)
CARD_NEW = (
    "**A4b 只在 2/3 seed 成立**（`scale_cv` 1.89～3.02、与真值相关 0.66～0.71，"
    "第三个种子 `scale_truth_corr` 0.2789 < 0.3 ⇒ " + BRANCH_NOTE + "）"
)
CARD_APPEND_OLD = (
    "｜ **v2.47**：文档里的数字已与原始产物逐格对账，订正 §8.2 seed-0 表 A4 "
    "`CSI@2.0` 的末位笔误（0.175501 → 0.175500），**裁决不变**。｜ 交接：`HANDOVER_E45.md` |"
)
CARD_APPEND_NEW = (
    CARD_APPEND_OLD[:-1] +
    "｜ **v2.48 分支订正**：把 `scale_is_constant` 这个派生布尔拆成它的两条触发分支重判"
    "（`tools/_e45_verify_round43.py`）⇒ **cv 分支在 E45 全部六格从未触发**"
    "（最小 `scale_cv` 1.2358 ＝ 底线的 24.7 倍）、**相关性分支只在 A4b 种子 2 触发一次**，"
    "而该格 `scale_cv` 3.0188 恰是六格**最高** ⇒ 「退化成常数」是**误标**，"
    "实际是**量级塌缩**（`scale_mean` 0.0316，不足同臂另两格的 1/11）**＋ 与真值脱钩**；"
    "**A4b 的舍弃理由按预注册本就是相关性措辞**（「既有解码头的平均雨强不足以承担量通道」），"
    "正文与自己引用的规则原本自相矛盾 ⇒ 已订正。**裁决零变化**（A4b 种子 2 的标志仍为真 "
    "⇒ 预注册的舍弃照样触发；六臂仍全部判负、不投十 seed 算力）。｜ 交接：`HANDOVER_E45.md` |"
)

S9_PREFIX = "| §4.47 | 修三处机制缺陷 + 把比较基准摆正"
S9_OLD = (
    "**不构成裁决变化**（三 seed 中位数／过地板通过率／S3 六臂判负一个数都没动）。 |"
)
S9_NEW = (
    S9_OLD[:-2] +
    " **v2.48 分支订正**：`scale_is_constant` 是**析取**（cv<0.05 **或** corr<0.3），"
    "结论正文把它一律写成「退化成常数」＝断言了 cv 分支；逐格重算证明 **cv 分支在 E45 "
    "六格从未触发**（最小 `scale_cv` 1.2358 ＝ 底线的 24.7 倍），**只有相关性分支在 "
    "A4b 种子 2 触发一次**，而该格 `scale_cv` 3.0188 是六格**最高** ⇒ 实为"
    "**量级塌缩＋与真值脱钩**。**裁决不变**（A4b 种子 2 的标志仍为真 ⇒ 预注册舍弃照样触发）。 |"
)

CHANGELOG_ROW = row(
    "2026-11-13",
    f"**{TAG}**",
    "**E45 量通道「退化成常数」的结论表述订正（裁决不变）**：远端状态与前四十二轮完全一致"
    "（提交器 273 与汇总作业 291 全 `COMPLETED`、21 份评测齐、队列仅 4 个非 E45 的 "
    "`qlfn2-curve`、13 份产物与本地 md5 全同）⇒ **六臂判负、不投十 seed 算力**不变。"
    "**本轮新增的检查**：前四十二轮核过产物、文档措辞、文档数字与源码状态，"
    "**从未把派生布尔读数拆成它的触发分支**——本轮补上这一环"
    "（`tools/_e45_verify_round43.py`，只读本地，VERDICT: PASS）。"
    "`scale_is_constant` 按探针自己的定义是**析取**（`scale_cv < 0.05` **或** "
    "`scale_truth_corr < 0.3`，含义是「量通道不再携带雨强信息」），"
    "而 E45 的结论正文一律写作「退化成常数」，等于断言了 **cv 分支**。"
    "**从原始探针 JSON 逐格重算的结果**：**cv 分支在 E45 全部六格从未触发**"
    "（最小 `scale_cv` 1.2358，是 0.05 底线的 24.7 倍）；**只有相关性分支触发过一次，"
    "且只在 A4b 种子 2**——该格 `scale_cv` 3.0188 恰是六格**最高**，"
    "**量通道没有变成常数**，真正发生的是**量级塌缩**"
    "（`scale_mean` 0.0316，不足同臂另两格 0.407／0.353 的 1/11）**＋ 与真值脱钩**"
    "（`scale_truth_corr` 0.2789）。⇒ 「E45 全程没有任何一格真的退化成常数」；"
    "**「量通道不是瓶颈」这条结论反而更硬**（唯一的退化就是相关性那一支）；"
    "**A4b 的舍弃理由按预注册 §3.1b／§3.2 本来就是相关性措辞**"
    "（「既有解码头的平均雨强不足以承担量通道」），正文与它自引的规则原本自相矛盾 ⇒ 已订正。"
    "**裁决零变化**：A4b 种子 2 的 `scale_is_constant` 仍为真 ⇒ 预注册的舍弃照样触发；"
    "六臂仍全部判负、无一臂进十 seed 算力。"
    "**新登记一条待办（不作机制结论）**：A4b 种子 2 同时是 `scale_mean` 最小、`scale_cv` 最大、"
    "`scale_truth_corr` 最低、`mean_out_equals_T` 相对误差最大（0.258）的一格，"
    "且与 v2.46 记的塌陷帧数最多（4272/32000）同格 ⇒ **该格的「退化」可能与 1e-12 护栏缺陷同源**；"
    "但 A4b 种子 0 有 2741 塌陷帧却 `scale_mean` 最高（0.407）⇒ 不是单调关系，"
    "**尚未建立因果** ⇒ 修 `1e-12` 护栏后重测时**须重新判这一格**，"
    "并**分分支报**（cv 分支／相关性分支），不得再合并成一个布尔。"
    "**新增可复用规则**：**派生布尔读数必须按其定义拆成触发分支再落笔**——"
    "同一个布尔值可以由语义完全不同的两条线触发（一个是「不再变化」，一个是「不再携带信息」），"
    "把布尔值直接翻译成其中一条线的语义，会把「量级塌缩」误写成「退化成常数」，"
    "**并让正文与自己引用的预注册规则互相矛盾**。",
    "`docs/experiments/E45-gate-direction-amount-separation-baseline.md` · "
    "`tools/_e45_verify_round43.py` · `tools/_check_v248_row.py` · "
    "`tools/_evals/e45/diag_probe_s{0,1,2}.json`",
)

# --------------------------------------------------------------------------
# INDEX
# --------------------------------------------------------------------------
INDEX_PREFIX = "| E45 | §4.47 |"
INDEX_OLD = (
    "**量通道 A4 三 seed 都没退化、A4b 只在 2/3 seed 成立（第三个种子 "
    "`scale_truth_corr` 0.2789 < 0.3 ⇒ 退化成常数），但也不涨分 ⇒ 量通道不是瓶颈**"
)
INDEX_NEW = (
    "**量通道 A4 三 seed 都没退化、A4b 只在 2/3 seed 成立（第三个种子 "
    "`scale_truth_corr` 0.2789 < 0.3 ⇒ **触发相关性分支**；其 `scale_cv` 3.0188 是六格最高 "
    "⇒ 不是「退化成常数」，实为量级塌缩＋与真值脱钩），但也不涨分 ⇒ 量通道不是瓶颈**"
)
INDEX_APPEND_OLD = (
    "**v2.45 量通道读数订正**：v2.44 的「量通道三 seed 都没退化成常数」**只对 A4 成立**；"
    "A4b 第三个种子 `scale_truth_corr` = 0.2789 < 0.3 ⇒ `scale_is_constant` = True ⇒ 量通道退化。"
)
INDEX_APPEND_NEW = (
    INDEX_APPEND_OLD +
    " **v2.48 分支订正**：`scale_is_constant` 是**析取**（cv<0.05 **或** corr<0.3），"
    "结论正文把它一律写成「退化成常数」＝断言了 cv 分支；逐格重算证明 **cv 分支在 E45 六格从未触发**"
    "（最小 `scale_cv` 1.2358 ＝ 底线的 24.7 倍），**只有相关性分支在 A4b 种子 2 触发一次**，"
    "该格 `scale_cv` 3.0188 恰是六格最高 ⇒ 实为**量级塌缩＋与真值脱钩**。"
    "**裁决不变**（A4b 种子 2 的标志仍为真 ⇒ 预注册舍弃照样触发）。"
)

# --------------------------------------------------------------------------
# E45 experiment file (§8.4 conclusion, item 3)
# --------------------------------------------------------------------------
E45_ITEM3_PREFIX = "3. **量与结构分离：机制在做量，但不涨分。**"
E45_ITEM3_OLD = (
    "A4b 只在 2/3 seed 成立（`scale_cv` 1.89～3.02、与真值相关 0.66～0.71，"
    "第三个种子 `scale_truth_corr` 0.2789 < 0.3 ⇒ `scale_is_constant` 为真 ⇒ 退化成常数）"
)
E45_ITEM3_NEW = (
    "A4b 只在 2/3 seed 成立（`scale_cv` 1.89～3.02、与真值相关 0.66～0.71，"
    "第三个种子 `scale_truth_corr` 0.2789 < 0.3 ⇒ `scale_is_constant` 为真 ⇒ " +
    BRANCH_NOTE + "）"
)
E45_ITEM3B_OLD = (
    "第三个种子甚至量通道退化（`scale_truth_corr` 0.2789 < 0.3 ⇒ 退化成常数）。"
)
E45_ITEM3B_NEW = (
    "第三个种子甚至触发了退化下限（`scale_truth_corr` 0.2789 < 0.3 ⇒ 相关性分支 ⇒ "
    "量与真值脱钩）。"
)
E45_APPEND_OLD = "**留下三件待办（不当机制结论）**"
E45_APPEND_NEW = (
    "**v2.48 分支订正（2026-11-13，判据零改动）**：本条第 3 项原写「第三个种子量通道退化"
    "⇒ 退化成常数」，但 `scale_is_constant` 是**析取**（§3.1b 写明「两条退化线之一被触发」）："
    "`scale_cv < 0.05`（不再变化）**或** `scale_truth_corr < 0.3`（不再携带雨强信息）。"
    "逐格重算（`tools/_e45_verify_round43.py`）证明 **cv 分支在 E45 全部六格从未触发**"
    "（最小 `scale_cv` 1.2358 ＝ 底线的 24.7 倍），**只有相关性分支在 A4b 种子 2 触发一次**，"
    "而该格 `scale_cv` 3.0188 是六格**最高** ⇒ 实为**量级塌缩**"
    "（`scale_mean` 0.0316，不足同臂另两格的 1/11）**＋ 与真值脱钩**。"
    "**裁决不变**（A4b 种子 2 的标志仍为真 ⇒ §3.1b／§3.2 的预注册舍弃照样触发；"
    "六臂仍全部判负、无一臂进十 seed 算力）。**新登记一条待办（不作机制结论）**："
    "A4b 种子 2 是 `scale_mean` 最小、`scale_cv` 最大、`scale_truth_corr` 最低、"
    "`mean_out_equals_T` 相对误差最大（0.258）的一格，与 v2.46 记的塌陷帧数最多"
    "（4272/32000）同格 ⇒ 其「退化」**可能与 1e-12 护栏缺陷同源**；"
    "但 A4b 种子 0 有 2741 塌陷帧却 `scale_mean` 最高（0.407）⇒ 不是单调关系、"
    "**尚未建立因果** ⇒ 修 `1e-12` 护栏后重测时**须重新判这一格，并分分支报**。\n\n"
    "**留下三件待办（不当机制结论）**"
)

# --------------------------------------------------------------------------
# MEMORY.md
# --------------------------------------------------------------------------
MEM_PTR_OLD = "> 权威 `docs/MASTER.md` v2.47；"
MEM_PTR_NEW = "> 权威 `docs/MASTER.md` v2.48；"

MEM_STATUS_PREFIX = "- **E45 已结案＝负面（六臂全部判负，无一臂进十 seed）**"
MEM_STATUS_OLD = (
    "**A4b 只在 2/3 seed 成立**（`scale_cv` 1.89～3.02、与真值相关 0.66～0.71，"
    "第三个种子 `scale_truth_corr` 0.2789 < 0.3 ⇒ 退化成常数）"
)
MEM_STATUS_NEW = (
    "**A4b 只在 2/3 seed 成立**（`scale_cv` 1.89～3.02、与真值相关 0.66～0.71，"
    "第三个种子 `scale_truth_corr` 0.2789 < 0.3 ⇒ **触发相关性分支**；其 `scale_cv` 3.0188 "
    "是六格最高 ⇒ 不是「退化成常数」，实为量级塌缩＋与真值脱钩）"
)

MEM_ROUND42_PREFIX = "- **（2026-11-13 15:5x 第 42 次独立复核"
MEM_ROUND43_LINE = (
    "- **（2026-11-13 17:0x 第 43 次独立复核：裁决不变，新证据＝派生布尔读数的"
    "**分支分解**）** 远端 E45 全部作业 `COMPLETED`（提交器 **273** 05:04:04、训练数组 "
    "275/283、评测 274/282/290、汇总 **291**、只读诊断 297/298）、`runs/e45/evals/` "
    "**21 份**评测齐、`runs/e45/` **13 份** json 齐全；队列仅剩 4 个 `qlfn2-curve`"
    "（302–305，**非 E45，不干预**）；`run_logs/slurm/` 最新仍是 `e45-diag-298.out`"
    "（Nov 13 03:10）、`runs/` 下无 E46+ 新目录 ⇒ **无新活动**；本轮**未下载、未 sbatch、"
    "未改远端状态**。远端 `runs/e45/*.json` 与本地 `tools/_evals/e45/*.json` md5 **13/13 全同**；"
    "八套既有闸门全部 `VERDICT: PASS` ⇒ **MASTER 升 v2.48**（有结论表述订正，非仅复核）。"
    "**本轮新增检查（新脚本 `tools/_e45_verify_round43.py`，只读本地，POS 15/15 VERDICT: PASS）**："
    "**前四十二轮从未把派生布尔读数拆成它的触发分支**——`scale_is_constant` 按探针自己的定义"
    "是**析取**（`scale_cv < 0.05` **或** `scale_truth_corr` < 0.3，含义是「量通道不再携带雨强信息」），"
    "而结论正文一律写作「退化成常数」＝断言了 cv 分支。**逐格重算原始探针 JSON**："
    "**cv 分支在 E45 全部六格从未触发**（最小 `scale_cv` 1.2358 ＝ 底线的 24.7 倍）；"
    "**只有相关性分支触发过一次，且只在 A4b 种子 2**——该格 `scale_cv` 3.0188 恰是六格**最高**，"
    "**量通道没有变成常数**，实际发生的是**量级塌缩**（`scale_mean` 0.0316，"
    "不足同臂另两格 0.407／0.353 的 1/11）**＋ 与真值脱钩**（`scale_truth_corr` 0.2789）。"
    "⇒ 「E45 全程没有任何一格真的退化成常数」；**「量通道不是瓶颈」这条结论反而更硬**；"
    "**A4b 的舍弃理由按预注册 §3.1b／§3.2 本就是相关性措辞**（「既有解码头的平均雨强不足以承担量通道」），"
    "正文与它自引的规则原本自相矛盾 ⇒ 已订正。**裁决零变化**：A4b 种子 2 的标志仍为真 ⇒ "
    "预注册舍弃照样触发；六臂仍全部判负、无一臂进十 seed 算力。"
    "**新登记一条待办（不作机制结论）**：A4b 种子 2 同时是 `scale_mean` 最小、`scale_cv` 最大、"
    "`scale_truth_corr` 最低、`mean_out_equals_T` 相对误差最大（0.258）的一格，"
    "与 v2.46 记的塌陷帧数最多（4272/32000）同格 ⇒ 其「退化」**可能与 1e-12 护栏缺陷同源**；"
    "但 A4b 种子 0 有 2741 塌陷帧却 `scale_mean` 最高（0.407）⇒ 不是单调关系、**尚未建立因果** ⇒ "
    "修护栏后重测时**须重新判这一格并分分支报**。"
    "**新增可复用规则**：**派生布尔读数必须按其定义拆成触发分支再落笔**——同一个布尔值可以由"
    "语义完全不同的两条线触发（一个是「不再变化」，一个是「不再携带信息」），把布尔值直接翻译成"
    "其中一条线的语义，会把「量级塌缩」误写成「退化成常数」，**并让正文与自己引用的预注册规则互相矛盾**。"
)

# --------------------------------------------------------------------------
# HANDOVER_E45.md
# --------------------------------------------------------------------------
HO_PTR_OLD = "Gate 3 写回已完成（MASTER **v2.47**）。"
HO_PTR_NEW = "Gate 3 写回已完成（MASTER **v2.48**）。"

HO_OLD = (
    "> (1) **A4b 的 `scale_truth_corr` = 0.2789 < 0.3 ⇒ `scale_is_constant = True`"
    "（量通道在 seed 2 退化）**\n"
    "> —— 「量通道没有退化成常数」对 A4b 只在 2/3 seed 成立（A4 三 seed 全过）；"
)
HO_NEW = (
    "> (1) **A4b 的 `scale_truth_corr` = 0.2789 < 0.3 ⇒ `scale_is_constant = True`**"
    "（**v2.48 订正：触发的是相关性分支，不是「退化成常数」**——该种子的 `scale_cv` 3.0188 "
    "是六格**最高**，`scale_mean` 0.0316 不足同臂另两格的 1/11 ⇒ 量级塌缩＋与真值脱钩；"
    "**E45 全程没有任何一格真的退化成常数**，最小 `scale_cv` 1.2358 是底线的 24.7 倍）\n"
    "> —— 按预注册 §3.1b／§3.2，**A4b 种子 2 仍触发舍弃**（「既有解码头的平均雨强不足以承担量通道，"
    "量必须由一个专门的头承担」）；「只在 2/3 seed 成立」指的是**过判据**（两条退化线都没触发），"
    "A4 三 seed 全过。"
)

HO_TODO_OLD = "**④ 判据工具要区分「臂不稳」与「地板口径不稳」**"
HO_TODO_NEW = (
    "**④ 判据工具要区分「臂不稳」与「地板口径不稳」**"
    "（**v2.48 追加：还要区分「派生布尔读数的哪条分支触发了」**——`scale_is_constant` 是析取，"
    "cv 分支＝「不再变化」、相关性分支＝「不再携带雨强信息」；E45 六格里 cv 分支从未触发，"
    "只有 A4b 种子 2 触发相关性分支 ⇒ 报告必须**分分支写**，不得再合并成一个布尔）"
)


# --------------------------------------------------------------------------
# patchers
# --------------------------------------------------------------------------
def patch_master(text):
    if TAG in text:
        print("MASTER already carries", TAG, "-> skip")
        return text
    text = apply(text, HEADER_OLD, HEADER_NEW, 1, "master.header.version")
    text = apply(text, HEADER_SCALE_OLD, HEADER_SCALE_NEW, 1, "master.header.scale")
    text = replace_in_line(text, CARD_PREFIX, CARD_OLD, CARD_NEW, "master.card.scale")
    text = replace_in_line(text, CARD_PREFIX, CARD_APPEND_OLD, CARD_APPEND_NEW,
                           "master.card.append")
    text = replace_in_line(text, S9_PREFIX, S9_OLD, S9_NEW, "master.s9.row")
    # insert the v2.48 changelog row right after the v2.47 row
    lines = text.split("\n")
    idx = [i for i, l in enumerate(lines) if l.startswith("| 2026-11-13 | **v2.47** |")]
    if len(idx) != 1:
        raise SystemExit(f"CHANGELOG ANCHOR MISS: found {len(idx)} v2.47 changelog rows")
    i = idx[0]
    prev = unescaped_pipes(lines[i])
    new = unescaped_pipes(CHANGELOG_ROW)
    if prev != new:
        raise SystemExit(f"PIPE COUNT MISMATCH: v2.47 row {prev} cells vs new row {new} cells")
    lines.insert(i + 1, CHANGELOG_ROW)
    return "\n".join(lines)


def patch_index(text):
    if TAG in text:
        print("INDEX already carries", TAG, "-> skip")
        return text
    text = replace_in_line(text, INDEX_PREFIX, INDEX_OLD, INDEX_NEW, "index.scale")
    text = replace_in_line(text, INDEX_PREFIX, INDEX_APPEND_OLD, INDEX_APPEND_NEW,
                           "index.append")
    return text


def patch_e45doc(text):
    if TAG in text:
        print("E45DOC already carries", TAG, "-> skip")
        return text
    text = replace_in_line(text, E45_ITEM3_PREFIX, E45_ITEM3_OLD, E45_ITEM3_NEW,
                           "e45doc.item3")
    text = apply(text, E45_ITEM3B_OLD, E45_ITEM3B_NEW, 1, "e45doc.item3b")
    text = apply(text, E45_APPEND_OLD, E45_APPEND_NEW, 1, "e45doc.append")
    return text


def patch_memory(text):
    if TAG in text:
        print("MEMORY already carries", TAG, "-> skip")
        return text
    text = apply(text, MEM_PTR_OLD, MEM_PTR_NEW, 1, "memory.header.version")
    text = replace_in_line(text, MEM_STATUS_PREFIX, MEM_STATUS_OLD, MEM_STATUS_NEW,
                           "memory.status.scale")
    text = insert_after_line(text, MEM_ROUND42_PREFIX, MEM_ROUND43_LINE,
                             "memory.status.round43")
    return text


def patch_handover(text):
    if TAG in text:
        print("HANDOVER already carries", TAG, "-> skip")
        return text
    text = apply(text, HO_PTR_OLD, HO_PTR_NEW, 1, "handover.pointer")
    text = apply(text, HO_OLD, HO_NEW, 1, "handover.seed2.warning")
    text = apply(text, HO_TODO_OLD, HO_TODO_NEW, 1, "handover.todo")
    return text


def append_daily(text):
    if TAG in text:
        print("daily log already carries", TAG, "-> skip")
        return text
    note = (
        f"\n\n## {TAG}（本次运行，2026-11-13）· E45 状态复核 + 派生布尔的分支分解\n\n"
        "**远端状态无变化**：提交器 **273** `COMPLETED`（05:04:04）、汇总作业 **291** "
        "`COMPLETED`、三 seed 训练数组 275／283 与评测 274／282／290 全 `COMPLETED`、"
        "21 份评测齐；远端 `runs/e45/` 下 13 个 json 与本地 `tools/_evals/e45/` md5 "
        "**13/13 全同**；队列仅剩 4 个 `qlfn2-curve`（非 E45，不干预）；"
        "`run_logs/slurm/` 最新仍是 `e45-diag-298.out`（Nov 13 03:10）、`runs/` 下无 E46+ "
        "新目录 ⇒ **无新活动**。**未改任何远端状态、未 sbatch 训练／评测、未下载**。\n\n"
        "**本轮的新东西：把派生布尔读数拆成它的触发分支**。前四十二轮核过产物、文档措辞、"
        "文档数字与源码状态，**从来没有核过「一个布尔标志是被哪条线触发的」**——"
        "而 E45 结论里正有一个这样的标志被翻译错了。\n\n"
        "`scale_is_constant` 按探针自己的定义（`tools/_e45_mechanism_probe.py` l.63-64 / "
        "l.679-681）是**析取**：`scale_cv < 0.05`（量不再变化）**或** "
        "`scale_truth_corr < 0.3`（量仍在变但不再携带雨强信息），"
        "其含义是「量通道不再携带雨强信息」。**E45 的结论正文一律把它写成「退化成常数」**，"
        "等于断言了 cv 分支。\n\n"
        "**逐格重算原始探针 JSON（`tools/_e45_verify_round43.py`，只读本地，POS 15/15 "
        "VERDICT: PASS）**：\n\n"
        "- **cv 分支在 E45 全部六格从未触发**（最小 `scale_cv` 1.2358 ＝ 底线的 24.7 倍）；\n"
        "- **只有相关性分支触发过一次，且只在 A4b 种子 2**（`scale_truth_corr` 0.2789）；\n"
        "- **该格 `scale_cv` 3.0188 恰是六格最高** ⇒ 量通道没有变成常数，"
        "它反而是六格里最「多变」的一格；\n"
        "- 真正发生的是**量级塌缩**（`scale_mean` 0.0316，不足同臂另两格 0.407／0.353 的 1/11）"
        "**＋ 与真值脱钩**。\n\n"
        "⇒ 「E45 全程没有任何一格真的退化成常数」；**「量通道不是瓶颈」这条结论反而更硬**"
        "（唯一的退化就是相关性那一支）；**A4b 的舍弃理由按预注册 §3.1b／§3.2 本来就是相关性措辞**"
        "（「既有解码头的平均雨强不足以承担量通道」），正文与它自引的规则原本自相矛盾 ⇒ 已订正。\n\n"
        "**裁决零变化**：A4b 种子 2 的标志仍为真 ⇒ 预注册的舍弃照样触发；"
        "**六臂仍全部判负、无一臂进十 seed 算力**。门指向／拓扑支／量通道三条机制线的其余读数"
        "**一个数都没动**。\n\n"
        "**新登记一条待办（不作机制结论）**：A4b 种子 2 同时是 `scale_mean` 最小、`scale_cv` 最大、"
        "`scale_truth_corr` 最低、`mean_out_equals_T` 相对误差最大（0.258）的一格，"
        "与 v2.46 记的塌陷帧数最多（4272/32000）同格 ⇒ 其「退化」**可能与 `1e-12` 护栏缺陷同源**；"
        "但 A4b 种子 0 有 2741 塌陷帧却 `scale_mean` 最高（0.407）⇒ 不是单调关系、**尚未建立因果** ⇒ "
        "修护栏后重测时**须重新判这一格并分分支报**。\n\n"
        "**新增可复用规则**：**派生布尔读数必须按其定义拆成触发分支再落笔**——同一个布尔值可以由"
        "语义完全不同的两条线触发（一个是「不再变化」，一个是「不再携带信息」），"
        "把布尔值直接翻译成其中一条线的语义，会把「量级塌缩」误写成「退化成常数」，"
        "**并让正文与自己引用的预注册规则互相矛盾**。\n\n"
        "**写回范围（Gate 3 四处 + INDEX + E45 文件 + MEMORY + HANDOVER + 两份日志）**："
        "MASTER 升 **v2.48**（头部版本行／§4.47 卡片／§9 行／§12 新行）、INDEX 的 E45 行、"
        "E45 实验文件 §8.4 结案第 3 条、MEMORY.md 版本指针与状态段、HANDOVER 版本指针与 seed-2 警示块。"
        "**历史 §12 行（v2.44–v2.47）刻意不改写**（它们是审计痕迹），"
        "由新的 v2.48 行披露「『退化成常数』措辞已被取代」。\n"
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
