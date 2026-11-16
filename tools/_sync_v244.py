"""Write the E45 three-seed verdict into MASTER (four places), INDEX and the
E45 experiment file.

Rows are built from their CELLS; the separators are inserted by the code.
Hand-writing the separators inside the text is how a row ends up with the
wrong number of columns -- and a markdown table with the wrong number of
columns silently breaks the whole table.  Every replacement therefore
checks that the pipe count of the new line equals the pipe count of the
line it replaces.

The script is idempotent: a place that already carries the version tag is
left alone, so running it twice cannot insert the same text twice.
"""
from pathlib import Path
import re
import sys

MASTER = Path("docs/MASTER.md")
INDEX = Path("docs/experiments/INDEX.md")
E45DOC = Path("docs/experiments/E45-gate-direction-amount-separation-baseline.md")

TAG = "v2.44"
LINK = "`docs/experiments/E45-gate-direction-amount-separation-baseline.md`"


def row(*cells):
    """One markdown table row from its cells."""
    return "| " + " | ".join(c.strip() for c in cells) + " |"


def pipes(line):
    return line.count("|")


# --------------------------------------------------------------------------
# MASTER: header line / §4.47 card / §9 row / §12 changelog
# --------------------------------------------------------------------------

HEADER_MARK = "| 最后更新 | 2026-11-11（v2.43，"
CARD_MARK = "| §4.47 |"
STATUS_MARK = "| **E45 / §4.47 修三处机制缺陷 + 摆正基准** |"
CHANGELOG_MARK = "| 2026-11-11 | **v2.43** |"

HEADER_NEW = row(
    "最后更新",
    "2026-11-11（v2.44，**E45 三个种子全部跑完并结案＝负面：六条臂全部判负，没有一条进得了十 seed 正式跑**）",
    "**判据一字未改**，三 seed 结论按「中位数＋过地板通过率」报。"
    "**没有任何一条臂的提升在统计上站得住** —— 原始增益的置信区间六条臂**全部跨零**，"
    "等偏置配平到与对照臂相同的偏置后**全部归零或仍跨零** ⇒ 按预注册「配平后跨零即判面积／幅度效应」**六臂舍弃**。"
    "三条机制线的判读：**门指向问题确实被堵住了**（三个种子的门与幅度相关性从 −0.858320 变成 +0.39～+0.69，"
    "峰值处门不再关闭）**但不涨分 ⇒ 门不是瓶颈**；**梯度集中两条治法都没堵住**"
    "（场最大值三个种子仍是 402.43 mm，针尖还在冲天花板）⇒ **拓扑支按预注册结掉**；"
    "**量通道确实在做量**（三个种子都没退化成常数）**但也不涨分 ⇒ 量通道不是瓶颈**。"
    "唯一像样的是过地板通过率：A3a／A3b 三次全过，**但对照臂自己也只有两次过**"
    "（地板 0.0924 是单一种子的口径），而这两条臂的中位数**低于对照** ⇒ 不构成晋级依据。"
    "**下一步：不投十 seed 算力（六臂无一够格），改代码修两处实现缺陷、把诊断结论写进实验文件；"
    "若要重开，须重新提机制族，不是派生**。"
    "**详见 `R:/1radarNWPprediction/.workbuddy/memory/HANDOVER_E45.md`**",
    "**本会话 token 教训**：远端检查合并成一条命令、产物分次下载、写回用单元格构造表格行并核对列数",
)

CARD_NEW = row(
    "§4.47",
    "修三处机制缺陷 + 把比较基准摆正：门指向、梯度集中、量与结构解耦",
    "七臂各只比基线多一处改动。**三个种子全部跑完**：门指向三 seed 同向堵住"
    "（门与幅度相关性 −0.858320 → +0.39～+0.69），梯度集中两条治法都没堵住"
    "（场最大值仍 402.43 mm），量通道三 seed 都没退化成常数；"
    "**但六条臂的提升在统计上全部站不住**（原始增益置信区间全跨零，等偏置配平后全归零），"
    "门与量两条机制**堵住了也不涨分**。",
    "**结案＝负面（六臂全部判负，无一臂进阶十 seed）**。"
    "**门指向问题结构上已解决但不是瓶颈**；**拓扑支按预注册结掉**（不再换损失组合方式重试）；"
    "**量与结构分离支结掉**（显式总量标定没让偏置达标，且配平后无增益）。"
    "**三 seed 全过地板的 A3a／A3b 不晋级**（对照臂自己也只有 2/3 过地板，两臂中位数低于对照，"
    "且两臂的臂级失败读数均触发）。**留下三件待办**：① 幅度截断只作用在拓扑项内部、"
    "没作用在输出场上（输出场仍被钳在 402.43 mm 的模型上限）——下一轮若重开的第一个候选修复点；"
    "② **量通道的实现缺陷**（`mean(out)=T` 不成立：新增标量头那臂一个种子不达、"
    "既有解码头那臂三个种子全不达）⇒ 修代码后重测才算机制结论；"
    "③ **门那组的第三个读数探针没测出来**（三个种子全是空值）⇒ 读数缺失，不是通过也不是失败。",
    LINK,
)

STATUS_NEW = row(
    "**E45 / §4.47 修三处机制缺陷 + 摆正基准**",
    "**已跑完并结案＝负面（2026-11-11 v2.44）**：七臂 × 三 seed 训练评测全部完成，"
    "S3 等偏置对照自动跑完。**六臂的提升统计上全部站不住**：原始增益置信区间六臂全跨零，"
    "等偏置配平后全部归零或仍跨零 ⇒ **没有一臂进阶十 seed**。"
    "三条机制线的判读：**门指向确实堵住了**（三个种子的门与幅度相关性 −0.858320 → +0.39～+0.69，"
    "峰值处门不再关闭）**但不涨分 ⇒ 门不是瓶颈**；"
    "**梯度集中两条治法都没堵住**（场最大值三 seed 仍 402.43 mm ⇒ "
    "**幅度截断只作用在拓扑项内部、没作用在输出场上**，下一轮重开时的第一个候选修复点）"
    "⇒ **拓扑支按预注册结掉**；**量通道三 seed 都没退化成常数**"
    "（变异系数 1.24～3.02、与真值相关 0.55～0.74）**但也不涨分 ⇒ 量通道不是瓶颈**。",
    "**下一步（不投十 seed 算力）**：① 修两处实现缺陷并留证"
    "（量通道 `mean(out)=T` 不成立；门那组的峰值比读数三个种子全是空值）；"
    "② 把「地板是单一种子口径、对照臂自己只有 2/3 过地板」写进判据工具"
    "（过地板通过率不能单独用）；③ **若要重开，必须重新提机制族，不是派生**（八族已封口）。"
    "**A4b 的量通道在第三个种子退化**（相关性 0.2789 < 0.3）——这条只对两个种子成立，不当机制结论。"
    "｜ 交接：`HANDOVER_E45.md`",
)

CHANGELOG_NEW = row(
    "2026-11-11",
    "**v2.44**",
    "**E45 三个种子全部跑完并结案＝负面（六臂全部判负，无一臂进十 seed）**："
    "① 提交器作业 273 与汇总作业 291 全部完成，21 份评测齐；"
    "② **S3 等偏置对照跑完 ⇒ 六条臂全部判负**：原始增益的置信区间六臂**全部跨零**，"
    "配平到与对照臂相同偏置后**全部归零或仍跨零** ⇒ 按预注册「配平后跨零即判面积／幅度效应」**六臂舍弃**；"
    "③ 机制读数三 seed 同向：**门指向确实堵住**（门与幅度相关性 −0.858320 → +0.6103／+0.6145／+0.6881 "
    "与 +0.4035／+0.3113／+0.3879，峰值处门 0.8217／0.7599／0.7224 与 1.0／1.0／1.0）"
    "**但不涨分 ⇒ 门不是瓶颈**；**梯度集中两条治法都没堵住**"
    "（场最大值三 seed 全是 402.43 mm，梯度集中 0.5526／0.5666／0.4782 与 0.9712／0.9915／0.9775，"
    "尾部超出 99 分位 26.5～65.2 倍）⇒ **拓扑支结掉**；"
    "**量通道三 seed 都没退化成常数**（变异系数 1.24～3.02、与真值相关 0.55～0.74）"
    "**但也不涨分 ⇒ 量通道不是瓶颈**；"
    "④ **过地板通过率只有 A3a／A3b 是三次全过，但对照臂自己也只有两次过**"
    "（地板 0.0924 是单一种子口径）且这两臂中位数低于对照 ⇒ **不晋级**；"
    "⑤ **三件待办**：幅度截断没作用在输出场上（仍被钳在模型上限 402.43 mm）／"
    "量通道 `mean(out)=T` 不成立属实现缺陷（A4 一个种子不达、A4b 三个种子全不达）⇒ 修后重测／"
    "门那组的峰值比读数三个种子全是空值属读数缺失；"
    "⑥ **不投十 seed 算力（六臂无一够格）**，**重开须重新提机制族**。",
    LINK + " · `tools/_evals/e45/{scores_summary,s3_equalbias,struct_metrics_all}.json`",
)


def replace_line(lines, marker, new, tag):
    """Rewrite one line, but only once, and keep the column count."""
    hits = [i for i, line in enumerate(lines) if line.startswith(marker)]
    if not hits:
        raise SystemExit(f"NOT FOUND: {marker[:60]}")
    if len(hits) > 1:
        raise SystemExit(f"AMBIGUOUS ({len(hits)}): {marker[:60]}")
    i = hits[0]
    if tag in lines[i]:
        return 0, f"already up to date (line {i + 1})"
    if pipes(lines[i]) != pipes(new):
        raise SystemExit(
            f"COLUMN MISMATCH on line {i + 1}: "
            f"old {pipes(lines[i])} pipes, new {pipes(new)} pipes"
        )
    lines[i] = new
    return 1, f"replaced line {i + 1} ({pipes(new) - 1} columns)"


def insert_changelog(lines, after_marker, new, tag):
    """Insert a new changelog row right after an existing one."""
    for line in lines:
        if tag in line and line.startswith("| 2026-11-11 |"):
            return 0, "changelog already carries the tag"
    hits = [i for i, line in enumerate(lines) if line.startswith(after_marker)]
    if len(hits) != 1:
        raise SystemExit(f"CHANGELOG ANCHOR: found {len(hits)} (expected 1)")
    lines.insert(hits[0] + 1, new)
    return 1, f"inserted changelog row after line {hits[0] + 1}"


def split_cells(line):
    """Split a markdown table row into its cells the way a renderer does.

    The leading and the trailing separator are optional; when they are
    absent the text after the last separator is the last cell.
    """
    parts = line.split("|")
    if parts and not parts[0].strip():
        parts = parts[1:]
    if parts and not parts[-1].strip():
        parts = parts[:-1]
    return [p.strip() for p in parts]


def append_to_row(line, extra, header_cells):
    """Append text inside the last cell of a markdown table row.

    A previous append left the text *after* the final separator, which
    pushed it out of the table and gave the row one cell too many.  This
    rebuilds the row from its cells (the overflow merged back into the
    last one) so the row keeps the same number of cells as the header.
    """
    cells = split_cells(line)
    if len(cells) > header_cells:
        cells = cells[:header_cells - 1] + [" ".join(cells[header_cells - 1:])]
    if len(cells) < header_cells:
        raise SystemExit(
            f"INDEX E45 row has {len(cells)} cells, header has {header_cells}"
        )
    cells[-1] = (cells[-1] + " " + extra).strip()
    return row(*cells), cells


# --------------------------------------------------------------------------
# E45 experiment file: §8.1 算力现状 / §8.4 待完成 -> §8.4 三 seed 判读与结案
# --------------------------------------------------------------------------

E45_S81_OLD = (
    "- 提交器 **job 273**（Slurm 作业形态）串行推进 seed 0 → 1 → 2，每 seed = 训练数组 + 评测。\n"
    "- **seed 0 已全部完成**并取回产物；seed 1 评测中（job 282）；seed 2 待跑。\n"
    "- 已取回：`tools/_evals/e45/{scores_summary_s0,diag_probe_s0,struct_metrics_s0}.json`。\n"
)
E45_S81_NEW = (
    "- 提交器 **job 273**（Slurm 作业形态）串行推进 seed 0 → 1 → 2，每 seed = 训练数组 + 评测。"
    "**2026-11-11 14:20 全部完成**（273 COMPLETED，墙钟 05:04:04），"
    "三 seed 训练数组 275／283 与三 seed 评测 274／282／290 全部 COMPLETED，"
    "汇总作业 **291**（e45-summary，00:14:23）自动完成 S3 等偏置对照。\n"
    "- **21 份评测齐**（7 臂 × 3 seed）；已取回 `tools/_evals/e45/` 下 12 份："
    "`{scores_summary,diag_probe,struct_metrics}_s{0,1,2}.json` ＋ 合并产物 "
    "`scores_summary.json`、`scores_all.json`、`struct_metrics_all.json`、`s3_equalbias.json`。\n"
)

E45_S84_HEAD_OLD = "### 8.4 待完成"
E45_S84_HEAD_NEW = "### 8.4 三 seed 判读与结案（2026-11-11 14:20 全部完成）"

E45_S84_NEW_BODY = """
**判读口径**：中位数 ＋ 过地板通过率（地板 **0.0924**，Oracle 上界 **0.1653**），
单点分数不单独出现；S3 为等偏置纯幅度对照（配平到与对照臂 B0 相同的 `BIAS@2.0`，
10000 次 case 级 bootstrap，538 窗 × 3 seed）。

**分数（3 seed 中位数 ＋ 通过率）**

| 臂 | `CSI@2.0` 中位数 | 三 seed 范围 | `BIAS@2.0` 中位数 | 过地板通过率 | 超 Oracle 次数 |
|---|---|---|---|---|---|
| B0 | 0.1218 | 0.0917–0.1305 | 0.5095 | 2/3 | 0/3 |
| A1 | 0.1184 | 0.0804–0.1577 | 0.4430 | 2/3 | 0/3 |
| A2 | 0.1120 | 0.0795–0.1341 | 0.4490 | 2/3 | 0/3 |
| A3a | 0.1100 | 0.1072–0.1495 | 0.6394 | **3/3** | 0/3 |
| A3b | 0.1161 | 0.0992–0.1697 | 0.6188 | **3/3** | 1/3 |
| A4 | **0.1439** | 0.0787–0.1755 | 0.8362 | 2/3 | 1/3 |
| A4b | 0.1376 | 0.0152–0.1399 | 2.8215 | 2/3 | 0/3 |

中位数高于对照臂的只有 A4、A4b，但两者偏置同时爆表（离散度 2.23× 与 **9.23×**，
A4b 第三个种子直接塌到 0.0152 且偏置转为欠预报 0.0939）⇒ 分数不稳到不能当候选。

**S3 等偏置对照（六臂全部判负）**

| 臂 | 原始增益（点估计与 95 % 区间） | 配平后增益（点估计与 95 % 区间） | 裁决 |
|---|---|---|---|
| A1 | +0.00419［−0.0493, +0.0648］ | +0.00018［−0.0228, +0.0166］ | 跨零 ⇒ 仍是面积／幅度效应 |
| A2 | −0.00610［−0.0124, +0.0033］ | +0.00409［−0.00123, +0.0111］ | 原始即跨零 ⇒ 本来就没有增益 |
| A3a | +0.00758［−0.0201, +0.0270］ | −0.00022［−0.00854, +0.00748］ | 跨零 ⇒ 仍是面积／幅度效应 |
| A3b | +0.01365［−0.0307, +0.0466］ | −0.00284［−0.01068, +0.0071］ | 跨零 ⇒ 仍是面积／幅度效应 |
| A4 | +0.01804［−0.0122, +0.0522］ | −0.00616［−0.0377, +0.0150］ | 跨零 ⇒ 仍是面积／幅度效应 |
| A4b | −0.01710［−0.1146, +0.0467］ | −0.02411［−0.0656, +0.00284］ | 原始即跨零 ⇒ 本来就没有增益 |

⇒ **原始增益六臂全部跨零，配平后全部归零或仍跨零** ⇒ 按 §3.2「配平后跨零即判面积／幅度效应」
**六条臂全部舍弃**；按 §3.2「3 seed 未全过地板 ⇒ 进阶 2 用 ≥10 seed 分线并报告不稳定」，
本轮**没有任何一条臂够格进入阶 2**（A3a／A3b 虽 3/3 过地板，但两臂的臂级失败读数均触发、
中位数低于对照臂，且 S3 配平后跨零）⇒ **不投十 seed 算力**。

**机制读数（三 seed 全部读出）**

| 读数 | 判据 | A1（s0/s1/s2） | A2（s0/s1/s2） |
|---|---|---|---|
| `corr(mask,amp)` | ≥ 0 | +0.6103／+0.6145／+0.6881 ✔ | +0.4035／+0.3113／+0.3879 ✔ |
| `mask_at_peak` | > 0.5 | 0.8217／0.7599／0.7224 ✔ | 1.0／1.0／1.0 ✔ |
| `maxratio_as_trained` | > 0.9 | 空值／空值／空值（读数缺失） | 未产出 |
| `mask_dry_frac_lt_0p1` | — | 0.0000／0.0000／0.0000 | 0.1444／0.1263／0.2435 |
| `corr_mask_resid` | — | 0.9941／0.9980／0.9992 | 0.6769／0.6496／0.7251 |

| 读数 | 判据 | A3a（s0/s1/s2） | A3b（s0/s1/s2） |
|---|---|---|---|
| `grad_top1pct_share` | < 0.5 | 0.5526／0.5666／0.4782 | 0.9712／0.9915／0.9775 ✘ |
| `field_max_mm` | ≤ 30 | 402.43／402.43／402.43 ✘ | 402.43／402.43／402.43 ✘ |
| `tail_max_over_p99` | ≤ 10 | 26.55／35.93／27.19 ✘ | 28.45／65.17／40.37 ✘ |

| 读数 | 判据 | A4（s0/s1/s2） | A4b（s0/s1/s2） |
|---|---|---|---|
| `scale_cv` | ≥ 0.05 | 2.004／1.236／1.705 ✔ | 2.112／1.891／3.019 ✔ |
| `scale_truth_corr` | ≥ 0.3 | 0.7429／0.5521／0.6384 ✔ | 0.6565／0.7080／**0.2789 ✘** |
| `scale_is_constant` | 必须 False | False／False／False ✔ | False／False／**True ✘** |
| `mean_out_equals_T` | 必须 True | True／**False ✘**／True | False／False／False ✘ |

**结案判读（逐机制族）**

1. **门指向：结构上已解决，但不是瓶颈。** 三个种子的门与幅度相关性全部转正（+0.39～+0.69），
   峰值处门不再关闭，A1 的门严格是残差的单调函数（相关 0.994～0.999）⇒
   **E44 定位的「可学门把强峰判成背景」这个病因确实被堵住了**；但两条臂的中位数都低于对照臂、
   S3 配平后归零 ⇒ **门不是这个 task 的瓶颈**（这本身就是一条有价值的定位）。
   ⚠️ A2 的干区比例逐种子扩大（14.4 % → 12.6 % → 24.4 %）⇒ 硬支撑把一部分雨区关掉了。
2. **拓扑支：按预注册结掉。** 两条治法都没堵住针尖：场最大值三个种子都是 **402.43 mm**
   （＝ `expm1(6)` 的钳位上限），尾部超出 99 分位 26.5～65.2 倍（真值 3.303）；
   梯度集中 A3b 三个种子全部 > 0.5（0.9712／0.9915／0.9775）。
   ⇒ 按 §3.2「A3a 与 A3b 都判负 ⇒ 拓扑支结掉」，**不再换损失组合方式重试**。
3. **量与结构分离：机制在做量，但不涨分。** 量通道三个种子都没退化成常数
   （`scale_cv` 1.24～3.02、与真值相关 0.55～0.74）；但 A4 的偏置中位 0.8362（仍 < 0.9）
   ⇒ 按 §3.2「显式总量标定不足以让量达标」**该支结掉**；A4b 偏置中位 2.8215 爆表，
   第三个种子甚至量通道退化（`scale_truth_corr` 0.2789 < 0.3 ⇒ 退化成常数）。
4. **地板口径本身不稳**：对照臂 B0 自己也只有 2/3 过地板（第三个种子的 0.0917 < 地板 0.0924，
   地板是单一种子口径）⇒ **过地板通过率不能单独用来判晋级**。

**留下三件待办（不当机制结论）**

- **幅度截断只作用在拓扑项内部、没作用在输出场上**：场最大值三 seed 全是 402.43 mm（钳位上限）
  ⇒ 下一轮若重开，这是第一个候选修复点（把截断作用在输出场上，而不是只作用在拓扑损失里）。
- **量通道 `mean(out)=T` 不成立是实现缺陷**（`own` 模式下 `T` 取自 `dec[-1]` 的均值，
  而输出由结构算子乘出，两者不同源）：A4 第二个种子不达、A4b 三个种子全不达
  ⇒ 按 §3.2「实现缺陷，修实现后重测，不得当机制结论」。
- **门那组的第三个读数没测出来**：`maxratio_as_trained` 三个种子全是空值、A2 根本没产出
  ⇒ **读数缺失**，既不算通过也不算失败；探针需修。

**下一步**：**不投十 seed 算力**（六臂无一够格）。修两处实现/探针缺陷并留证；
**若要重开，必须重新提机制族，不是派生**（输出参数化、损失几何、低秩＋稀疏、
低秩只约束背景／RPCA 式加性分解／拓扑支／算子学习、生成式条件采样、量子启发、集成
八族已封口）。交接细节见 `R:/1radarNWPprediction/.workbuddy/memory/HANDOVER_E45.md`。
"""


def main() -> int:
    lines = MASTER.read_text(encoding="utf-8").split("\n")
    changed = 0
    for marker, new in (
        (HEADER_MARK, HEADER_NEW),
        (CARD_MARK, CARD_NEW),
        (STATUS_MARK, STATUS_NEW),
    ):
        n, msg = replace_line(lines, marker, new, TAG)
        changed += n
        print(f"  {msg}")
    n, msg = insert_changelog(lines, CHANGELOG_MARK, CHANGELOG_NEW, TAG)
    changed += n
    print(f"  {msg}")
    if changed:
        MASTER.write_text("\n".join(lines), encoding="utf-8")

    # INDEX: the E45 row, extended with the three-seed verdict.
    idx_lines = INDEX.read_text(encoding="utf-8").split("\n")
    header = next(l for l in idx_lines if l.startswith("| 编号 |"))
    header_cells = len(split_cells(header))
    hits = [i for i, line in enumerate(idx_lines) if line.startswith("| E45 |")]
    if len(hits) != 1:
        raise SystemExit(f"INDEX E45 row: found {len(hits)} (expected 1)")
    i = hits[0]
    if TAG in idx_lines[i]:
        print("  INDEX E45 row already carries the tag")
    else:
        extra = (
            "**v2.44 三 seed 结案＝负面（六臂全部判负，无一臂进十 seed）**："
            "**21 份评测齐、汇总作业 291 跑完 S3 等偏置对照**；"
            "**原始增益的置信区间六臂全部跨零，配平后全部归零或仍跨零 ⇒ 六臂按 §3.2 舍弃**；"
            "**门指向确实堵住但不涨分 ⇒ 门不是瓶颈**（相关性 −0.858320 → +0.39～+0.69，峰值处门不再关闭）；"
            "**梯度集中两条治法都没堵住（场最大值三 seed 全是 402.43 mm）⇒ 拓扑支结掉**；"
            "**量通道三 seed 都没退化成常数，但也不涨分 ⇒ 量通道不是瓶颈**；"
            "**A3a／A3b 虽 3/3 过地板但对照臂自己只有 2/3 过（地板是单一种子口径）且中位数低于对照 ⇒ 不晋级**；"
            "**不投十 seed 算力**；留下三件待办："
            "① 幅度截断没作用在输出场上（仍钳在 402.43 mm 的模型上限）；"
            "② 量通道 `mean(out)=T` 不成立属实现缺陷（A4 一个种子不达、A4b 三个种子全不达）⇒ 修后重测；"
            "③ 门那组的峰值比读数三个种子全是空值属读数缺失"
        )
        new_row, cells = append_to_row(idx_lines[i], extra, header_cells)
        if len(cells) != header_cells:
            raise SystemExit(
                f"INDEX E45 row: {len(cells)} cells, header has {header_cells}"
            )
        if pipes(new_row) != pipes(header):
            raise SystemExit("INDEX E45 row: separator count differs from the header")
        idx_lines[i] = new_row
        INDEX.write_text("\n".join(idx_lines), encoding="utf-8")
        print(f"  INDEX E45 row updated (line {i + 1}, {len(cells)} cells)")

    # E45 experiment file: §8.1 算力现状 + §8.4 待完成 -> 三 seed 判读与结案.
    doc = E45DOC.read_text(encoding="utf-8")
    if E45_S84_HEAD_NEW in doc:
        print("  E45 doc already carries the three-seed verdict")
    else:
        if E45_S81_OLD not in doc:
            raise SystemExit("E45 doc: §8.1 block not found")
        doc = doc.replace(E45_S81_OLD, E45_S81_NEW, 1)
        if E45_S84_HEAD_OLD not in doc:
            raise SystemExit("E45 doc: §8.4 heading not found")
        cut = doc.index(E45_S84_HEAD_OLD)
        doc = doc[:cut] + E45_S84_HEAD_NEW + E45_S84_NEW_BODY
        E45DOC.write_text(doc, encoding="utf-8")
        print("  E45 doc §8.1 updated, §8.4 rewritten as the three-seed verdict")

    print(f"done: {changed} MASTER places rewritten")
    return 0


if __name__ == "__main__":
    sys.exit(main())
