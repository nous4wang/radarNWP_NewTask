"""Write back the round-94 finding (v2.53).

What v2.53 records
------------------
Round 91 re-ran the RAW half of the S3 equal-bias guard locally with 20
alternative bootstrap seeds: 20/20 intervals contained zero for all six
arms, so the negative verdict did not rest on that one draw.  The
EQUALISED half needs the real guard implementation and was therefore
left to remote Slurm job 343, which was still PENDING behind the
account quota.

Job 343 has now COMPLETED (2026-11-16 06:02:25, CPU, rc=0) and produced
``runs/e45/s3_seedrobust.json``.  This write-back records what that
artifact settles:

* the equalised half is now swept over 20 bootstrap seeds too, so BOTH
  halves of the guard are seed-robust;
* the job's ``module_md5`` equals the normalised md5 of the local
  ``tools/_e41_s3_equalbias.py`` -- it imported the very implementation
  that produced the published numbers;
* the canonical seed reproduces the published S3 numbers bit for bit
  (max|dev| = 0 over 36 cells) and every arm's equalisation gap is 0;
* the zero-containment test is re-derived NUMERICALLY
  (``ci_lo <= 0 <= ci_hi``), not read off the stored boolean;
* the pre-registered promotion rule (point > 0 AND the interval excludes
  zero) is applied per arm per half per seed: ZERO qualifying seeds
  anywhere;
* A2 -- the arm round 91 measured as the only one at the flipping edge
  (nearest endpoint 0.001226 from zero, about 1.0x the measured seed
  jitter) -- does NOT flip: 20/20 equalised intervals still contain
  zero.  That closes round 91's open question.

The verdict is unchanged: six arms negative, none to ten seeds, no
ten-seed compute.

Scope of this write-back
------------------------
* MASTER.md -- header version line, the section 4.47 card row, the
  section 4.48 status bullet, the section 9 row, a section 12 changelog
  row (the five places Gate 3 names);
* INDEX.md -- the E45 row;
* E45 experiment file -- the version pointer and the round note;
* MEMORY.md -- the version pointer, the status section, the to-do item;
* HANDOVER_E45.md -- the version pointer and the round note;
* both daily logs -- the round note.

The historical section 12 rows (v2.44-v2.52) are deliberately NOT
rewritten: they are the audit trail.  Every replacement is anchored on a
literal that must occur exactly the expected number of times; a missing
anchor is an error, never a silent skip.  The new section 12 row is
built from its CELLS; the separators are inserted by the code and the
pipe count is compared against the row it follows.  Running the script
twice is a no-op.

Every marker is PER-FILE and distinctive.  The bare version string is not
usable as a guard: earlier rounds left the text "（＝升 v2.51）" in the
handover and the daily logs, and a guard on the bare string would skip
those files WITHOUT writing anything -- a silent false positive.

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

TAG = "v2.53"
MARKER = "v2.53 第 94 轮"
MARK_MASTER = "| 2026-11-16 | **v2.53** |"
MARK_INDEX = "v2.53 第 94 轮"
MARK_MEMORY = "v2.53 第 94 轮"
MARK_HANDOVER = "v2.53 追加（作业 343 已跑完 ＋ 两侧种子稳健性收口）"
MARK_DAILY = "## v2.53 第 94 轮（本次运行"
MARK_E45DOC = "v2.53 追加（作业 343 已跑完 ＋ 两侧种子稳健性收口）"


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
JOB343 = (
    "**作业 343 已跑完**（`scripts/e45_s3_seedrobust.sbatch`，2026-11-16 06:02:25，CPU，rc=0）"
    "⇒ 产出 `runs/e45/s3_seedrobust.json`（远端↔本地 md5 `c08ed2a8b3bb787d43fdfb29fa49abbc` 逐位相同）。"
)

ROBUST = (
    "**两侧种子稳健性收口**：第 91 轮已在本地把 S3 的 **raw 半边**用 20 个 bootstrap 种子重跑"
    "（20/20 全跨零），**等偏置半边**当时只能交给远端作业 343（须真实护栏实现）⇒ 本轮它跑完，"
    "**两侧都收口**。独立重导（`tools/_e45_round94.py`，只读本地，**POS 60/60 VERDICT: PASS**）："
    "① **实现同一性**——作业记录的 `module_md5` ＝ 本地 `tools/_e41_s3_equalbias.py` 的归一化 md5"
    "（`525f9c75910d5d9f0f3b2c4003d7549b`）⇒ 它导入的**正是产出已发布数字的那份护栏**；"
    "② **逐位复现**——注册种子下 raw／等偏置两半边的点估与区间与已发布 `s3_equalbias.json` "
    "**max|dev| = 0.000e+00（36 格）**，六臂 `max_abs_bias_gap` 全为 0.0；"
    "③ **两侧各 20 个种子全部跨零**——**数值级重算**（`ci_lo ≤ 0 ≤ ci_hi`，不读存储布尔）得 "
    "**raw 半边 6 臂 × 20/20 全含零、等偏置半边 6 臂 × 20/20 全含零**，"
    "且存储标志与数值 **0 处不自洽**；④ **点估不随重采样漂移**（max|drift| = 0）"
    "⇒ 区间变化只来自重采样、不来自数据；"
    "⑤ **按预注册的晋级判据**（点估 > 0 **且**区间不含零）逐臂逐半边逐种子判读 ⇒ "
    "**六个臂在两侧、全部 20 个种子上合格数均为 0**。"
)

A2CLOSE = (
    "**A2 不翻转（第 91 轮唯一的开放问题就此收口）**：第 91 轮量出 **A2 是唯一处在翻转边缘的臂**"
    "（等偏置半边最近端点距零 0.001226，**只等于换种子实测抖动 0.001179 的 1.0 倍**），"
    "故当时判定「A2 的等偏置裁决必须由作业 343 收口」。本轮结果："
    "**A2 等偏置半边在 20 个种子上 20/20 全部含零、最小余量 0.001075、合格种子数 0** ⇒ "
    "**A2 不翻转**，其判负**不依赖那一次抽样**。"
)

HEADLINE = (
    "**头条表同步复算**：由 21 份主跑 `*.eval.json` 的 `thresholds[\"2.0\"].csi` 复算 ⇒ "
    "七臂中位数与过地板通过率与已发布值 **max|dev| = 0**"
    "（对照 B0 中位数 0.121817、过地板 2/3；A3a 中位数 0.110015、A3b 中位数 0.116055，"
    "虽各自 3/3 过地板但**中位数均低于同批对照 B0** ⇒ 不晋级）；"
    "**机制判据重判**（自原始探针 JSON）：A1／A2 门指向 3/3 过、A3a／A3b 拓扑 0/3 过、"
    "A4 量通道 3/3 过、A4b 2/3 过 ⇒ 与已发布一致。"
)

VERDICT = "**裁决不变＝六臂判负、无一臂进十 seed、不投十 seed 算力**。"

BODY = JOB343 + ROBUST + A2CLOSE + HEADLINE + VERDICT

# short form for the MASTER header cell
HEAD_SHORT = (
    "作业 343 **已跑完**（2026-11-16 06:02:25）⇒ **S3 的 raw 与等偏置两侧各用 20 个 bootstrap "
    "种子重跑、全部 20/20 跨零**（数值级重算，不读存储布尔）⇒ 两侧都不依赖那一次抽样；"
    "**A2 不翻转**（等偏置半边 20/20 含零、最小余量 0.001075）⇒ 第 91 轮唯一的开放问题收口；"
    "**裁决不变＝六臂判负、不投十 seed 算力**。"
)

# --------------------------------------------------------------------------
# MASTER
# --------------------------------------------------------------------------
HEADER_V52_OLD = "**v2.52（2026-11-16，最新）**"
HEADER_V52_NEW = "**v2.52（2026-11-16）**"
HEADER_TAIL_OLD = (
    "⇒ 训练侧只剩「下一轮臂表接一次」，**裁决不变＝六臂判负、不投十 seed 算力** |"
)
HEADER_TAIL_NEW = (
    "⇒ 训练侧只剩「下一轮臂表接一次」；**v2.53（2026-11-16，最新）**：" + HEAD_SHORT + " |"
)

CARD_A_OLD = "⇒ 训练侧只剩「下一轮臂表接一次」。 |"
# the cell body is escaped; the trailing " |" is the row separator and is not
CARD_A_NEW = "⇒ 训练侧只剩「下一轮臂表接一次」。 ｜ **v2.53**：" + esc(BODY) + " |"
CARD_B_OLD = "v2.52 见 §4.48 状态行 |"
CARD_B_NEW = "v2.52 见 §4.48 状态行；v2.53 见 §4.48 状态行 |"

S448_OLD = "**唯一阻塞仍是同账号配额**，释放即可跑、**无需重投**。"
S448_NEW = (
    S448_OLD + " **v2.53 另记**：作业 343 **已跑完**（2026-11-16 06:02:25、CPU、rc=0）"
    "⇒ 产出 `runs/e45/s3_seedrobust.json`；" + ROBUST + A2CLOSE + VERDICT
)

S9_OLD = (
    "v2.52 见 §4.48 状态行（**21 份运行记录证实幅度上限从未生效** ＋ **评测侧接线已落地**） |"
)
S9_NEW = (
    "v2.52 见 §4.48 状态行（**21 份运行记录证实幅度上限从未生效** ＋ **评测侧接线已落地**）；"
    "v2.53 见 §4.48 状态行（**作业 343 已跑完：两侧 20/20 种子全跨零、A2 不翻转**） |"
)

CHANGELOG_ROW = row(*[esc(c) for c in (
    "2026-11-16",
    f"**{TAG}**",
    "**E45 第 94 轮：远端 Slurm 作业 343 跑完 ⇒ S3 等偏置半边的 bootstrap 种子稳健性收口**"
    "：远端主链终态不变（提交器 273／训练数组 275·283／评测 274·282·290／汇总 291／297·298／314–317 "
    "全 `COMPLETED`、`squeue` 空、21 份 `.eval.json` ＋ 13 份 run 级 json 齐）⇒ **无新活动**。"
    "**唯一新增的是第 91 轮就排队的作业 343**（`scripts/e45_s3_seedrobust.sbatch`，2026-11-16 06:02:25 "
    "跑完、CPU、rc=0）⇒ 产出 `runs/e45/s3_seedrobust.json`（远端↔本地 md5 "
    "`c08ed2a8b3bb787d43fdfb29fa49abbc` 逐位相同）；**结案后（> 11-11 14:20:42）被修改的文件仅此 1 份**"
    "（＝新产物本身，已发布产物零改动）。"
    "**远端动作**：只读查询 ＋ 1 次下载；**未 sbatch、未同步、未改远端状态**。"
    "**裁决独立重导**（`tools/_e45_round94.py`，只读本地，**POS 60/60 VERDICT: PASS**）："
    "**① 实现同一性**——作业 `module_md5` ＝ 本地 `tools/_e41_s3_equalbias.py` 归一化 md5"
    "⇒ 导入的**正是产出已发布数字的那份护栏**；**② 逐位复现**——注册种子下两半边与已发布 "
    "`s3_equalbias.json` **max|dev| = 0.000e+00（36 格）**、六臂 `max_abs_bias_gap` 全 0.0；"
    "**③ 两侧各 20 个种子全部跨零**——**数值级重算**（`ci_lo ≤ 0 ≤ ci_hi`，不读存储布尔）得 "
    "raw 半边 6 臂 × 20/20 全含零、**等偏置半边 6 臂 × 20/20 全含零**，存储标志与数值 0 处不自洽；"
    "**④ 点估不随重采样漂移**（max|drift| = 0）；**⑤ 按预注册晋级判据**（点估 > 0 且区间不含零）"
    "逐臂逐半边逐种子判读 ⇒ **六臂在两侧、全部 20 个种子上合格数均为 0**。"
    "**A2 不翻转**：第 91 轮量出 A2 是唯一处在翻转边缘的臂（等偏置最近端点距零 0.001226 "
    "＝ 换种子实测抖动 0.001179 的 1.0 倍），本轮 **A2 等偏置半边 20/20 含零、最小余量 0.001075、"
    "合格种子数 0** ⇒ **第 91 轮唯一的开放问题就此收口**。"
    "**头条表与机制判据同步复算**（21 份主跑 `*.eval.json`）：七臂中位数与过地板通过率与已发布值 "
    "**max|dev| = 0**（B0 中位数 0.121817／2/3；A3a 0.110015、A3b 0.116055 虽 3/3 过地板但**中位数均低于同批对照 B0** "
    "⇒ 不晋级）；机制判据 A1／A2 3/3 过、A3a／A3b 0/3 过、A4 3/3 过、A4b 2/3 过 ⇒ 与已发布一致。"
    "**裁决零变化**（六臂仍全部判负、无一臂进十 seed、**不投十 seed 算力**）；"
    "**下一步（不变）**：§4.5 三件实现缺陷修复（① 幅度截断接到输出场上——训练侧还需接一次；"
    "② 量通道改相对护栏；③ `maxratio` 改 `np.nanmedian` ＋ 报可用帧数）＋ **重开须重新提机制族**。",
    "`tools/_e45_round94.py` · `tools/_evals/e45/round94_status_audit.json` · "
    "`tools/_evals/e45/s3_seedrobust.json` · `scripts/e45_s3_seedrobust.sbatch` · "
    "`docs/experiments/E45-gate-direction-amount-separation-baseline.md`",
)])

# --------------------------------------------------------------------------
# INDEX
# --------------------------------------------------------------------------
INDEX_OLD = "；**裁决不变＝六臂判负、不投十 seed 算力** |"
# the cell body is escaped; the trailing " |" is the row separator and is not
INDEX_NEW = esc(
    "；**裁决不变＝六臂判负、不投十 seed 算力** ｜ **v2.53 第 94 轮**："
    "第 91 轮就排队的远端作业 343 **已跑完**（2026-11-16 06:02:25、CPU）⇒ 产出 "
    "`runs/e45/s3_seedrobust.json` ⇒ **S3 的 raw 与等偏置两侧各用 20 个 bootstrap 种子重跑、"
    "全部 20/20 跨零**（数值级重算，不读存储布尔；作业 `module_md5` ＝ 本地护栏实现的归一化 md5 "
    "⇒ 同一份实现、注册种子逐位复现 max|dev| = 0）⇒ **两侧都不依赖那一次抽样**；"
    "**A2 不翻转**（等偏置半边 20/20 含零、最小余量 0.001075）⇒ 第 91 轮唯一的开放问题收口；"
    "按预注册晋级判据（点估 > 0 且区间不含零）**六臂合格种子数全为 0**；"
    "**裁决不变＝六臂判负、不投十 seed 算力**"
) + " |"

# --------------------------------------------------------------------------
# E45 experiment file
# --------------------------------------------------------------------------
E45DOC_PTR_OLD = "> **2026-11-16（v2.52）**："
E45DOC_PTR_NEW = (
    "> **2026-11-16（v2.53）**：**E45 内容与判据自 v2.49 起未变**（S0–S6 一字未改；"
    "裁决不变：六臂判负、不投十 seed 算力）；第 94 轮收割远端作业 343（`scripts/e45_s3_seedrobust.sbatch`，"
    "2026-11-16 06:02:25 跑完）⇒ **S3 等偏置半边的 bootstrap 种子稳健性收口**："
    "**raw 与等偏置两侧各 20 个种子全部跨零**（数值级重算、不读存储布尔；同一份护栏实现、"
    "注册种子逐位复现 max|dev| = 0）⇒ **两侧都不依赖那一次抽样**；"
    "**A2 不翻转**（第 91 轮量出的唯一翻转边缘臂，20/20 含零、最小余量 0.001075）⇒ "
    "**第 91 轮唯一的开放问题收口**；按预注册晋级判据**六臂合格种子数全为 0** ⇒ "
    "**裁决不变＝六臂判负、不投十 seed 算力**（详见 §8.4 结案段与交接文档）。\n"
    "> **2026-11-16（v2.52）**："
)

E45DOC_TODO_OLD = (
    "⇒ **下一轮臂表**接 `--model-kwargs {\"clamp_mm\":20.0}` 即可（已实测该写法经臂表自己的 "
    "`read -r -a` 分词**仍是一个 token**、且经 `model_kwargs()` 真的进到构造参数）。"
)
E45DOC_TODO_NEW = (
    E45DOC_TODO_OLD + "\n"
    "  **v2.53 追加（作业 343 已跑完 ＋ 两侧种子稳健性收口）**：作业 343 已跑完，"
    "**S3 等偏置半边的 bootstrap 种子稳健性收口** ⇒ **raw 与等偏置两侧各 20 个种子全部跨零** "
    "（数值级重算）⇒ 判负**两侧都不依赖那一次抽样**；**A2 不翻转** ⇒ 第 91 轮唯一的开放问题收口；"
    "按预注册晋级判据**六臂合格种子数全为 0** ⇒ **裁决不变＝六臂判负、不投十 seed 算力**。"
    "**三件待办本身不变**（① 训练侧还需接 `--model-kwargs {\"clamp_mm\":20.0}` 一次；"
    "② 量通道改相对护栏破绝对 `1e-12`；③ `maxratio` 改 `np.nanmedian` ＋ 报可用帧数）。"
)

# --------------------------------------------------------------------------
# MEMORY.md
# --------------------------------------------------------------------------
MEM_PTR_OLD = "> 权威 `docs/MASTER.md` **v2.52**（"
MEM_PTR_NEW = "> 权威 `docs/MASTER.md` **v2.53**（"
MEM_PTR_TAIL_OLD = "**21 份运行记录证实幅度上限从未生效**、**评测侧接线已落地**）；"
MEM_PTR_TAIL_NEW = (
    "**21 份运行记录证实幅度上限从未生效**、**评测侧接线已落地**、"
    "**作业 343 已跑完：S3 两侧各 20 个 bootstrap 种子全跨零、A2 不翻转**）；"
)

MEM_STATUS_OLD = (
    "**训练侧不动**（E45 臂表是预注册的「E45 跑的是什么」）⇒ 下一轮臂表接 "
    "`--model-kwargs {\"clamp_mm\":20.0}` 即可（已实测经 `read -r -a` 仍是一个 token）。"
    "**裁决不变＝六臂判负、不投十 seed 算力**。"
)
MEM_STATUS_NEW = (
    "**训练侧不动**（E45 臂表是预注册的「E45 跑的是什么」）⇒ 下一轮臂表接 "
    "`--model-kwargs {\"clamp_mm\":20.0}` 即可（已实测经 `read -r -a` 仍是一个 token）。"
    "**v2.53 第 94 轮（2026-11-16）**：第 91 轮就排队的远端作业 343 **已跑完**"
    "（2026-11-16 06:02:25、CPU）⇒ 产出 `runs/e45/s3_seedrobust.json` ⇒ "
    "**S3 的 raw 与等偏置两侧各用 20 个 bootstrap 种子重跑、全部 20/20 跨零**"
    "（数值级重算 `ci_lo ≤ 0 ≤ ci_hi`，不读存储布尔；作业 `module_md5` ＝ 本地护栏实现的归一化 md5 "
    "⇒ 同一份实现、注册种子逐位复现 max|dev| = 0）⇒ **两侧都不依赖那一次抽样**；"
    "**A2 不翻转**（等偏置半边 20/20 含零、最小余量 0.001075）⇒ 第 91 轮唯一的开放问题收口；"
    "按预注册晋级判据（点估 > 0 且区间不含零）**六臂合格种子数全为 0**。"
    "**裁决不变＝六臂判负、不投十 seed 算力**。"
)

MEM_TODO_OLD = (
    "**训练侧按冻结纪律不动**（E45 臂表是预注册的「E45 跑的是什么」，改它＝改 E45 的定义）。"
)
MEM_TODO_NEW = (
    MEM_TODO_OLD + " **v2.53 追加**：作业 343 已跑完，**S3 两侧的 bootstrap 种子稳健性收口** ⇒ "
    "**A2 不翻转** ⇒ 待办清单本身**不变**（① 训练侧还需接一次 `--model-kwargs {\"clamp_mm\":20.0}`；"
    "② 量通道改相对护栏；③ `maxratio` 改 `np.nanmedian`）。"
)

MEM_R91_OLD = (
    "- **E45 第 91 轮（2026-11-16）**：远端主链终态不变（273 等全 `COMPLETED`、产物 md5 13/13、"
    "结案后零改动）；**首次独立重跑 S3 的重采样**——用护栏自身实现，注册种子逐位复现（max|dev| = 0），"
    "20 个 bootstrap 种子下六臂 raw 区间 **20/20 全含零** ⇒ **判负不依赖那一次抽样**；"
    "等偏置半边由远端 Slurm 作业 343（CPU）承担。已发布文档的分分支表述已合规（两分支分写）。"
    "**裁决不变＝六臂判负、不投十 seed 算力**；下一步仍是 §4.5 三件实现缺陷修复 ＋ 重开须重新提机制族。"
)
MEM_R91_NEW = (
    "- **E45 第 91 轮（2026-11-16）**：远端主链终态不变（273 等全 `COMPLETED`、产物 md5 13/13、"
    "结案后零改动）；**首次独立重跑 S3 的重采样**——用护栏自身实现，注册种子逐位复现（max|dev| = 0），"
    "20 个 bootstrap 种子下六臂 raw 区间 **20/20 全含零** ⇒ **判负不依赖那一次抽样**；"
    "等偏置半边由远端 Slurm 作业 343（CPU）承担（**已于第 94 轮跑完并收口，见下条**）。"
    "已发布文档的分分支表述已合规（两分支分写）。"
    "**裁决不变＝六臂判负、不投十 seed 算力**；下一步仍是 §4.5 三件实现缺陷修复 ＋ 重开须重新提机制族。"
)

MEM_R94_NEW = (
    "- **E45 第 94 轮（2026-11-16）**：远端主链终态不变（273 等全 `COMPLETED`、`squeue` 空、"
    "21 份评测 ＋ 13 份 run 级 json 齐、**结案后被修改文件仅 1 份＝新产物 `s3_seedrobust.json` 本身**）"
    "⇒ 无新活动；**唯一新增**：第 91 轮就排队的作业 343 **已跑完**（2026-11-16 06:02:25、CPU、rc=0）"
    "⇒ **S3 的 raw 与等偏置两侧各 20 个 bootstrap 种子全部跨零**（数值级重算 `ci_lo ≤ 0 ≤ ci_hi`，"
    "不读存储布尔；作业 `module_md5` ＝ 本地 `tools/_e41_s3_equalbias.py` 归一化 md5 ⇒ 同一份实现；"
    "注册种子逐位复现 max|dev| = 0，六臂 `max_abs_bias_gap` 全 0.0；点估不随重采样漂移）"
    "⇒ **两侧都不依赖那一次抽样**；**A2 不翻转**（第 91 轮量出的唯一翻转边缘臂：等偏置半边 20/20 含零、"
    "最小余量 0.001075、合格种子数 0）⇒ **第 91 轮唯一的开放问题收口**；"
    "按预注册晋级判据（点估 > 0 且区间不含零）**六臂合格种子数全为 0**；"
    "头条表与机制判据同步复算与已发布值 **max|dev| = 0**。"
    "**裁决不变＝六臂判负、无一臂进十 seed、不投十 seed 算力**；"
    "下一步仍是 §4.5 三件实现缺陷修复 ＋ **重开须重新提机制族**。"
)

# --------------------------------------------------------------------------
# HANDOVER_E45.md
# --------------------------------------------------------------------------
HO_PTR_OLD = "Gate 3 写回已完成（MASTER **v2.52**）。"
HO_PTR_NEW = "Gate 3 写回已完成（MASTER **v2.53**）。"

HO_TODO2_OLD = (
    "⇒ **待办② 从「两侧都要补」缩到「只差训练侧一次」**（下一轮臂表接 "
    "`--model-kwargs {\"clamp_mm\":20.0}`，已实测经臂表的 `read -r -a` 仍是一个 token）。"
)
HO_TODO2_NEW = (
    HO_TODO2_OLD + "\n\n**v2.53 追加（作业 343 已跑完 ＋ 两侧种子稳健性收口）**："
    "作业 343 **已跑完**（2026-11-16 06:02:25、CPU、rc=0）⇒ 产出 `runs/e45/s3_seedrobust.json` "
    "（远端↔本地 md5 逐位相同）⇒ **S3 的 raw 与等偏置两侧各 20 个 bootstrap 种子全部跨零**"
    "（数值级重算、不读存储布尔；作业 `module_md5` ＝ 本地护栏实现的归一化 md5 ⇒ 同一份实现、"
    "注册种子逐位复现 max|dev| = 0）⇒ **两侧都不依赖那一次抽样**；"
    "**A2 不翻转**（等偏置半边 20/20 含零、最小余量 0.001075）⇒ **第 91 轮唯一的开放问题收口**；"
    "按预注册晋级判据**六臂合格种子数全为 0** ⇒ **裁决不变＝六臂判负、不投十 seed 算力**。"
    "**三件待办本身不变**。"
)

# --------------------------------------------------------------------------
# patchers
# --------------------------------------------------------------------------
def patch_master(text):
    if MARK_MASTER in text:
        print("MASTER already carries the v2.53 changelog row -> skip")
        return text
    text = replace_in_line(text, "| 最后更新 |", HEADER_V52_OLD, HEADER_V52_NEW,
                           "master.header.v52tag")
    text = replace_in_line(text, "| 最后更新 |", HEADER_TAIL_OLD, HEADER_TAIL_NEW,
                           "master.header.tail")
    text = replace_in_line(text, "| §4.47 | 修三处机制缺陷", CARD_A_OLD, CARD_A_NEW,
                           "master.card.body")
    text = replace_in_line(text, "| §4.47 | 修三处机制缺陷", CARD_B_OLD, CARD_B_NEW,
                           "master.card.refs")
    text = replace_in_line(text, "- **状态**：4 个修复文件只在本地", S448_OLD, S448_NEW,
                           "master.s448.status")
    text = replace_in_line(text, "| **E45 / §4.47 修三处机制缺陷 + 摆正基准** |", S9_OLD, S9_NEW,
                           "master.s9.row")
    # insert the v2.53 changelog row right after the v2.52 row
    lines = text.split("\n")
    idx = [i for i, l in enumerate(lines) if l.startswith("| 2026-11-16 | **v2.52** |")]
    if len(idx) != 1:
        raise SystemExit(f"CHANGELOG ANCHOR MISS: found {len(idx)} v2.52 changelog rows")
    i = idx[0]
    prev = unescaped_pipes(lines[i])
    new = unescaped_pipes(CHANGELOG_ROW)
    if prev != new:
        raise SystemExit(f"PIPE COUNT MISMATCH: v2.52 row {prev} cells vs new row {new} cells")
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
    # the round-91 bullet was appended three times by an earlier round:
    # collapse the duplicates FIRST, so the count-1 anchors below hold
    lines = text.split("\n")
    r91 = [i for i, l in enumerate(lines) if l.startswith("- **E45 第 91 轮（2026-11-16）**")]
    if len(r91) == 3 and len({lines[i] for i in r91}) == 1:
        keep = set(r91[:1])
        lines = [l for i, l in enumerate(lines) if i not in set(r91[1:])]
        print(f"MEMORY: removed {len(r91) - 1} duplicate round-91 bullets")
        text = "\n".join(lines)
    elif len(r91) > 1:
        raise SystemExit(f"MEMORY: round-91 bullet appears {len(r91)}x and is not identical "
                         "-- refusing to guess which to drop")
    text = apply(text, MEM_PTR_OLD, MEM_PTR_NEW, 1, "memory.header.version")
    text = apply(text, MEM_PTR_TAIL_OLD, MEM_PTR_TAIL_NEW, 1, "memory.header.tail")
    text = apply(text, MEM_STATUS_OLD, MEM_STATUS_NEW, 1, "memory.status")
    text = apply(text, MEM_TODO_OLD, MEM_TODO_NEW, 1, "memory.todo")
    text = apply(text, MEM_R91_OLD, MEM_R91_NEW, 1, "memory.r91.note")
    # append the round-94 bullet right after the round-91 one
    lines = text.split("\n")
    r91 = [i for i, l in enumerate(lines) if l.startswith("- **E45 第 91 轮（2026-11-16）**")]
    if len(r91) != 1:
        raise SystemExit(f"MEMORY: expected exactly 1 round-91 bullet, found {len(r91)}")
    lines.insert(r91[0] + 1, MEM_R94_NEW)
    return "\n".join(lines)


def patch_handover(text):
    if MARK_HANDOVER in text:
        print("HANDOVER already carries the marker -> skip")
        return text
    text = apply(text, HO_PTR_OLD, HO_PTR_NEW, 1, "handover.pointer")
    text = apply(text, HO_TODO2_OLD, HO_TODO2_NEW, 1, "handover.todo2")
    return text


ROUND_NOTE = """
<!-- E45-R94-BEGIN -->
### 2026-11-16｜第 94 轮（E45 远端状态复核 ＋ **作业 343 跑完 ⇒ S3 等偏置半边的种子稳健性收口** ＋ **A2 不翻转**）

- **判定**：远端 E45 主链终态不变（提交器 273／训练数组 275·283／评测 274·282·290／汇总 291／297·298／314–317 全 `COMPLETED`、`squeue -u wf` 空、21 份 `.eval.json` ＋ 13 份 run 级 json 齐）⇒ **无新活动**。**远端动作**：只读查询 ＋ **1 次下载**（`s3_seedrobust.json`）；**未 sbatch、未同步、未改远端状态**。
- **唯一新增**：第 91 轮就排队的 **作业 343（`e45-s3seedrobust`）已跑完**（2026-11-16 06:02:25，CPU，rc=0）⇒ 产出 `runs/e45/s3_seedrobust.json`（远端↔本地 md5 `c08ed2a8b3bb787d43fdfb29fa49abbc` 逐位相同）。**结案后（> 11-11 14:20:42）被修改的文件仅此 1 份**（＝新产物本身）⇒ **已发布产物零改动**。
- **裁决独立重导**（`tools/_e45_round94.py`，只读本地，**POS 60/60 VERDICT: PASS**）：**① 实现同一性**——作业记录的 `module_md5` ＝ 本地 `tools/_e41_s3_equalbias.py` 的归一化 md5（`525f9c75910d5d9f0f3b2c4003d7549b`）⇒ 它导入的**正是产出已发布数字的那份护栏**；**② 逐位复现**——注册种子下 raw／等偏置两半边与已发布 `s3_equalbias.json` **max|dev| = 0.000e+00（36 格）**，六臂 `max_abs_bias_gap` 全为 0.0；**③ 两侧各 20 个种子全部跨零**——**数值级重算**（`ci_lo ≤ 0 ≤ ci_hi`，**不读存储布尔**）得 **raw 半边 6 臂 × 20/20 全含零、等偏置半边 6 臂 × 20/20 全含零**，存储标志与数值 **0 处不自洽**；**④ 点估不随重采样漂移**（max|drift| = 0）⇒ 区间变化只来自重采样、不来自数据。
- **按预注册的晋级判据**（点估 > 0 **且**区间不含零）逐臂逐半边逐种子判读 ⇒ **六个臂在两侧、全部 20 个种子上合格数均为 0**。
- **A2 不翻转（第 91 轮唯一的开放问题就此收口）**：第 91 轮量出 A2 是**唯一处在翻转边缘的臂**（等偏置半边最近端点距零 0.001226，**只等于换种子实测抖动 0.001179 的 1.0 倍**），故当时判定「A2 的等偏置裁决必须由作业 343 收口」。本轮：**A2 等偏置半边 20/20 全部含零、最小余量 0.001075、合格种子数 0** ⇒ **A2 不翻转**，其判负**不依赖那一次抽样**。
- **头条表与机制判据同步复算**（21 份主跑 `*.eval.json`）：七臂中位数与过地板通过率与已发布值 **max|dev| = 0**（对照 B0 中位数 0.121817／过地板 2/3；A3a 中位数 0.110015、A3b 中位数 0.116055，虽各自 3/3 过地板但**中位数均低于同批对照 B0** ⇒ 不晋级）；机制判据 A1／A2 门指向 3/3 过、A3a／A3b 拓扑 0/3 过、A4 量通道 3/3 过、A4b 2/3 过 ⇒ 与已发布一致。
- **裁决零变化**（六臂仍全部判负、无一臂进十 seed、**不投十 seed 算力**）；**下一步（不变）**：§4.5 三件实现缺陷修复（① 幅度截断接到输出场上——训练侧还需接一次；② 量通道改相对护栏破绝对 `1e-12`；③ `maxratio` 改 `np.nanmedian` ＋ 报可用帧数）＋ **重开须重新提机制族**。
- **产物**：`tools/_e45_round94.py`、`tools/_evals/e45/round94_status_audit.json`、`tools/_evals/e45/s3_seedrobust.json`、`tools/_sync_v253.py`、`tools/_check_v253_row.py`。
<!-- E45-R94-END -->
"""


def append_daily(text):
    if MARK_DAILY in text:
        print("daily log already carries", MARK_DAILY, "-> skip")
        return text
    note = (
        f"\n\n## {MARKER}（本次运行，2026-11-16）· E45 状态复核 ＋ "
        "**作业 343 跑完 ⇒ S3 两侧种子稳健性收口** ＋ **A2 不翻转**\n\n" + ROUND_NOTE
    )
    return text + note


def append_handover_round(text):
    """Append the round note to the handover, under its own marker."""
    if "<!-- E45-R94-BEGIN -->" in text:
        print("HANDOVER already carries the R94 round note -> skip")
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
