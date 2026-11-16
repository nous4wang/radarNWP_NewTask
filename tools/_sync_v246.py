"""Write back the E45 source-level defect localisation (v2.46).

What v2.46 adds
---------------
Rounds 9-30 re-verified E45's ARTEFACTS and its DOCUMENTS.  None of
them read the SOURCE CODE that produces the three registered
implementation defects.  This round did, and it localises all three --
and it corrects two of the three registered causal explanations.

The three defects, localised
----------------------------
D1  amplitude truncation never reaches the output field
    ``models/topo_simvp/model.py`` contains NO ``clamp_mm`` token at
    all: the output field is hard-capped by
    ``torch.expm1(log_mm.clamp(-10.0, 6.0))`` = expm1(6) = 402.4288 mm,
    while ``--topo-clamp-mm`` is consumed only by the LOSS side
    (``scripts/train.py`` -> ``TopoCriterion``).  The read-out
    ``field_max_mm`` equals float32(expm1(6)) to the last bit in all
    six arm-seeds.  => the registered cause is CORRECT, now with the
    exact line.

D2  ``mean(out)=T`` fails
    The registered cause ("in own mode T comes from dec[-1]'s mean
    while the output is produced by the structural operator, so the
    two are not from the same source") is WRONG.  ``out = T * S``
    elementwise and ``S = F_mm / (mean(F_mm) + 1e-12)`` is normalised
    to unit mean BY CONSTRUCTION, so the identity ``mean(out) = T``
    does not care where T comes from: it can only fail when
    ``mean(S) != 1``, i.e. when ``mean(F_mm)`` is at the guard's own
    scale.

    MEASURED on the real checkpoints (Slurm job 298, 32000 frames per
    arm-seed, ``tools/_e45_meanout_diag.py``):

        arm-seed  frames with mean(F_mm)==0   frames breaking the
                                             guarantee (err > 1e-5)
        A4_s0                0 / 32000                   0
        A4_s1              818 / 32000                 818
        A4_s2                0 / 32000                   0
        A4b_s0           2741 / 32000                  18
        A4b_s1           3202 / 32000                 280
        A4b_s2           4272 / 32000                 132

    This is the decisive correction: **A4 (head mode) seed 1 fails
    too**, with 818 collapsed frames.  So the defect is NOT specific to
    "own mode" and NOT about "T and out coming from different
    sources" -- it is the absolute ``1e-12`` guard in the shape
    normalisation, hitting every arm-seed whose decomposition field
    is empty on some frames.  On such a frame ``mean(F_mm) = 0``,
    hence ``S = 0/(0+1e-12) = 0`` and the output collapses to 0 while
    the amount ``T`` is NOT zero.

    The probe's own guard is ALSO wrong here, and in a way that hides
    the failure: it collects ``float(abs(v) - 1.0)`` -- a SIGNED
    quantity -- and then takes the max over frames.  Since
    ``|mean(S)| - 1 <= 0`` whenever ``mean(S) <= 1``, the max can
    only ever report a frame where mean(S) is slightly ABOVE 1.  It
    reported 2.4e-07 / 3.6e-07 for every arm-seed while the real
    worst deviation was 1.0 (mean(S) collapsing to 0).  => the guard
    under-reports by ~7 orders of magnitude and structurally CANNOT
    report a collapsed frame.

D3  the gate arms' ``maxratio`` read-outs are empty
    The registered cause ("the probe never measured it") is also
    WRONG.  Measured on the real checkpoints (Slurm job 298, 60
    batches per arm-seed, ``tools/_e45_maxratio_diag.py``):

        arm-seed  dry batches  usable batches  np.median   np.nanmedian
        A1_s0          11           49            nan       0.7592
        A1_s1          10           50            nan       0.7588
        A1_s2          10           50            nan       0.7221
        A2_s0          11           49            nan       1.0000
        A2_s1          11           49            nan       1.0000
        A2_s2          10           50            nan       1.0000

    The read-out IS computable; E44's ``_max_ratio`` returns NaN for
    a wholly dry batch, and the probe aggregates the per-batch values
    with ``np.median``, which is NOT nan-aware -- so one dry batch
    poisons the whole read-out.  => the defect is in the AGGREGATION
    (np.median over a list containing NaN), not in the measurement.

    NOTE (important, and it does NOT change any verdict): the
    nanmedian values land BELOW the pre-registered ``maxratio_as_trained
    > 0.9`` line for A1 (0.72-0.76) and exactly AT it for A2 (1.0).
    A1 and A2 were already judged negative by the S3 equal-bias
    control, so this changes nothing about the verdict; but it is a
    NEW, previously unmeasured number on a pre-registered criterion,
    so it is written down as a fresh open item ("after the probe is
    fixed, re-read this criterion") rather than as a conclusion.

None of this changes a verdict: S3 still refutes all six arms, the
topological branch stays closed, the gate stays "structurally fixed
but not the bottleneck", and no arm enters the >=10 seed line.  The
three items remain open TODOs -- but now with the exact line to fix
and, for D2 and D3, the exact reason the earlier explanation was
wrong.

Discipline
----------
* Every replacement is anchored on a literal that must occur exactly
  the expected number of times; a missing anchor is an error, never a
  silent skip.
* The new §12 changelog row is built from its CELLS; the separators
  are inserted by the code, and the pipe count is compared against the
  row it follows so the table cannot silently break.
* Running the script twice is a no-op.
* The three version pointers (MASTER header, MEMORY.md header,
  HANDOVER pointer) are moved together, and the round-27 gate is
  relaxed from "== v2.45" to ">= v2.45" so that a version bump does
  not turn a consistency check into a false failure.
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

TAG = "v2.46"


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


# --------------------------------------------------------------------------
# the corrections, per file
# --------------------------------------------------------------------------

# The corrected D2 cause, stated once and reused.
D2_CAUSE_OLD = (
    "**量通道 `mean(out)=T` 不成立是实现缺陷**（`own` 模式下 `T` 取自 `dec[-1]` 的均值，\n"
    "  而输出由结构算子乘出，两者不同源）"
)
D2_CAUSE_NEW = (
    "**量通道 `mean(out)=T` 不成立是实现缺陷**（**v2.46 已定位到源码：不是「T 与 out 不同源」"
    "——`out = T·S` 逐元素相乘、`S = F_mm/(mean(F_mm)+1e-12)` 按构造即单位均值，"
    "故 `mean(out)=T` 与 `T` 取自哪一路无关，只在 `mean(S)≠1` 时才破。"
    "**实测（真 checkpoint，32000 帧/臂种子）：`mean(F_mm)==0` 的塌陷帧数 "
    "A4_s1 818/32000、A4b_s0 2741/32000、A4b_s1 3202/32000、A4b_s2 4272/32000，"
    "而 A4_s0／A4_s2 为 0 ⇒ **head 模式的 A4 第二个种子同样破，所以缺陷不是 own 模式独有**，"
    "破口是**形状归一化用绝对 `1e-12` 护栏去比一个在干帧上本就≈0 的量**："
    "该帧 `mean(F_mm)=0` ⇒ `S = 0/(0+1e-12) = 0`、输出随之塌成 0 而该帧的 `T` 非零**"
    "（另：探针自己的守卫 `shape_S_unit_mean_max_dev` 收的是带符号的 "
    "`abs(v)-1` 再取 max ⇒ 结构上只能报出「mean(S) 略大于 1」的帧，"
    "实测报 2.4e−07／3.6e−07 而真实最坏偏差达 1.0 ⇒ **该守卫会掩盖塌陷帧，须一并修**）"
)

MASTER_FIXES = [
    # header version line: v2.45 -> v2.46, with what changed
    (
        "2026-11-11（v2.45，**E45 三个种子全部跑完并结案＝负面：六条臂全部判负，"
        "没有一条进得了十 seed 正式跑**；v2.45 只订正量通道的分臂表述，裁决不变）",
        "2026-11-13（v2.46，**E45 三个种子全部跑完并结案＝负面：六条臂全部判负，"
        "没有一条进得了十 seed 正式跑**；v2.46 把三件待办的根因定位到源码层并订正其中两件的"
        "**错误因果表述**，裁决不变）",
        1,
        "master.header.version",
    ),
    # header: the three TODOs -- the one-line version
    (
        "**下一步：不投十 seed 算力（六臂无一够格），改代码修两处实现缺陷、把诊断结论写进实验文件；"
        "若要重开，须重新提机制族，不是派生**",
        "**下一步：不投十 seed 算力（六臂无一够格），改代码修两处实现缺陷、把诊断结论写进实验文件；"
        "若要重开，须重新提机制族，不是派生**。**v2.46 已把三件待办定位到源码行**："
        "① 幅度截断在 `models/topo_simvp/model.py` 里根本没有入口（输出场被 "
        "`expm1(log_mm.clamp(-10,6))` 硬钳在 402.43 mm），修法＝把截断接到输出场上；"
        "② `mean(out)=T` 破在**形状归一化的绝对 `1e-12` 护栏**（实测真 checkpoint 上 "
        "A4_s1 也有 818/32000 帧塌陷 ⇒ **不是 own 模式独有、也不是「T 与 out 不同源」**），"
        "修法＝让护栏随场自身的尺度走；③ 门那组的 `maxratio` "
        "**其实测得出来**（nanmedian 0.72～0.76／A2 为 1.0），空值是探针用"
        "**非 nan 感知的 `np.median`** 聚合一遇到干批次的 NaN 就把整个读数污染 "
        "⇒ 修法＝改 `np.nanmedian` 并报可用帧数**",
        1,
        "master.header.todos",
    ),
    # §4.47 card: the three TODOs
    (
        "**留下三件待办**：① 幅度截断只作用在拓扑项内部、没作用在输出场上（输出场仍被钳在 "
        "402.43 mm 的模型上限）——下一轮若重开的第一个候选修复点；"
        "② **量通道的实现缺陷**（`mean(out)=T` 不成立：新增标量头那臂一个种子不达、"
        "既有解码头那臂三个种子全不达）⇒ 修代码后重测才算机制结论；"
        "③ **门那组的第三个读数探针没测出来**（三个种子全是空值）⇒ 读数缺失，不是通过也不是失败。",
        "**留下三件待办**（v2.46 已把三件的根因定位到源码行；三件都仍是待办、都不作机制结论）："
        "① 幅度截断只作用在拓扑项内部、没作用在输出场上（输出场仍被钳在 "
        "402.43 mm 的模型上限）——下一轮若重开的第一个候选修复点；**v2.46 定位**："
        "`models/topo_simvp/model.py` 全文没有 `clamp_mm` 的入口，输出场由 "
        "`torch.expm1(log_mm.clamp(-10.0, 6.0))` 硬钳，而 `--topo-clamp-mm` 只被损失侧"
        "（`scripts/train.py` → `TopoCriterion`）消费 ⇒ 修法＝把截断接到输出场上；"
        "② **量通道的实现缺陷**（`mean(out)=T` 不成立：新增标量头那臂一个种子不达、"
        "既有解码头那臂三个种子全不达）⇒ 修代码后重测才算机制结论；"
        "**v2.46 订正其因果表述**：不是「T 与 out 不同源」（`out = T·S` 逐元素相乘、"
        "`S` 按构造即单位均值 ⇒ 与 `T` 的来源无关），而是**形状归一化的绝对 `1e-12` 护栏**："
        "实测真 checkpoint 上 `mean(F_mm)==0` 的塌陷帧数为 A4_s1 818/32000、"
        "A4b_s0 2741/32000、A4b_s1 3202/32000、A4b_s2 4272/32000（A4_s0／A4_s2 为 0）"
        "⇒ **head 模式的 A4 第二个种子同样破 ⇒ 不是 own 模式独有**，"
        "该帧 `S = 0/(0+1e-12) = 0`、输出随之塌成 0 而该帧 `T` 非零 ⇒ "
        "修法＝让护栏随场自身的尺度走（相对护栏或跳过零场）；"
        "另探针的守卫 `shape_S_unit_mean_max_dev` 收的是带符号的 `abs(v)-1` 再取 max ⇒ "
        "结构上只能报出「略大于 1」的帧（实测报 2.4e−07／3.6e−07 而真实最坏偏差 1.0）⇒ 一并修；"
        "③ **门那组的第三个读数探针没测出来**（三个种子全是空值）⇒ 读数缺失，不是通过也不是失败。"
        "**v2.46 订正其因果表述**：不是「探针没测」而是**聚合被污染**——实测该读数"
        "算得出来（60 批里 49～50 批有可用帧，`np.nanmedian` 给到 A1 0.7221～0.7592、A2 1.0），"
        "但 10～11 个全干批次让 E44 的 `_max_ratio` 返回 NaN，而探针用**非 nan 感知的 "
        "`np.median`** 聚合 ⇒ 一个 NaN 批次就把整个读数污染 ⇒ 修法＝`np.nanmedian` + 报可用帧数。"
        "**v2.46 另记一条新的待读数**：该读数按预注册判据是 `maxratio_as_trained > 0.9`，"
        "而修好聚合后的 nanmedian 落在 A1 0.72～0.76（低于线）、A2 1.0（在线上）"
        "⇒ **修好探针后须重读这条判据**（此前是读数缺失，从未判过）；"
        "**但 A1／A2 早已被 S3 等偏置对照判负，故此读数不改变任何裁决**。",
        1,
        "master.card.todos",
    ),
    # §9 row: the "next step" cell
    (
        "**下一步（不投十 seed 算力）**：① 修两处实现缺陷并留证（量通道 `mean(out)=T` 不成立；"
        "门那组的峰值比读数三个种子全是空值）；",
        "**下一步（不投十 seed 算力）**：① 修两处实现缺陷并留证（量通道 `mean(out)=T` 不成立——"
        "**v2.46 已定位：破在形状归一化的绝对 `1e-12` 护栏（实测真 checkpoint 上 A4_s1 也有 "
        "818/32000 帧塌陷 ⇒ 不是 own 模式独有、也不是「T 与 out 不同源」**；"
        "门那组的峰值比读数三个种子全是空值——"
        "**v2.46 已定位：读数其实算得出来（nanmedian A1 0.7221～0.7592／A2 1.0），"
        "空值是探针用非 nan 感知的 `np.median` 聚合一遇干批次的 NaN 就污染全体**；"
        "**修好探针后须重读预注册的 `maxratio_as_trained > 0.9` 这条判据**）；",
        1,
        "master.s9.next",
    ),
]

INDEX_FIXES = [
    (
        "**v2.45 量通道读数订正**：v2.44 的「量通道三 seed 都没退化成常数」**只对 A4 成立**；"
        "A4b 第三个种子 `scale_truth_corr` = 0.2789 < 0.3 ⇒ `scale_is_constant` = True ⇒ 量通道退化。"
        "**裁决不变**（S3 六臂全跨零 ⇒ 六臂判负、不投十 seed 算力）",
        "**v2.45 量通道读数订正**：v2.44 的「量通道三 seed 都没退化成常数」**只对 A4 成立**；"
        "A4b 第三个种子 `scale_truth_corr` = 0.2789 < 0.3 ⇒ `scale_is_constant` = True ⇒ 量通道退化。"
        "**裁决不变**（S3 六臂全跨零 ⇒ 六臂判负、不投十 seed 算力） "
        "**v2.46 三件待办的源码级根因（裁决不变）**：① 幅度截断在 `models/topo_simvp/model.py` "
        "里没有入口（输出场硬钳在 `expm1(log_mm.clamp(-10,6))` = 402.43 mm，`--topo-clamp-mm` "
        "只被损失侧消费）⇒ 修法＝把截断接到输出场上；② `mean(out)=T` 的破口是**形状归一化的绝对 "
        "`1e-12` 护栏**（实测真 checkpoint 上 `mean(F_mm)==0` 的塌陷帧数 A4_s1 818/32000、"
        "A4b_s0 2741/32000、A4b_s1 3202/32000、A4b_s2 4272/32000 ⇒ **head 模式的 A4 第二个种子"
        "同样破 ⇒ 不是 own 模式独有、也不是「T 与 out 不同源」**）；"
        "③ 门那组的 `maxratio` **其实测得出来**（nanmedian A1 0.7221～0.7592、A2 1.0），"
        "空值是探针用非 nan 感知的 `np.median` 聚合被干批次的 NaN 污染 ⇒ 修法＝`np.nanmedian` "
        "+ 报可用帧数 ⇒ **修好后须重读预注册的 `maxratio_as_trained > 0.9` 判据**",
        1,
        "index.todos",
    ),
]

E45DOC_FIXES = [
    # the §8.4 conclusion line
    (
        "⇒ **A4b 的 `mean(out)=T` 不成立是实现缺陷**（`own` 模式下 `T` 取自 `dec[-1]` 的均值，\n"
        "而输出由结构算子乘出，两者不同源）⇒ 按 §11.8 记为**实现缺陷，修后重测，不得当机制结论**。",
        "⇒ **A4b 的 `mean(out)=T` 不成立是实现缺陷**（**v2.46 已定位到源码：不是「T 与 out 不同源」"
        "——`out = T·S` 逐元素相乘，而 `S = F_mm/(mean(F_mm)+1e-12)` 按构造即单位均值，"
        "故 `mean(out)=T` 与 `T` 取自哪一路无关，只在 `mean(S)≠1` 时才破；"
        "**实测真 checkpoint（32000 帧/臂种子）：`mean(F_mm)==0` 的塌陷帧数 "
        "A4_s1 818/32000、A4b_s0 2741/32000、A4b_s1 3202/32000、A4b_s2 4272/32000，"
        "A4_s0／A4_s2 为 0 ⇒ **head 模式的 A4 第二个种子同样破 ⇒ 缺陷不是 own 模式独有**，"
        "破口是**形状归一化用绝对 `1e-12` 护栏去比一个在干帧上本就≈0 的量**："
        "该帧 `mean(F_mm)=0` ⇒ `S = 0/(0+1e-12) = 0`、输出随之塌成 0 而该帧的 `T` 非零；"
        "探针自己的守卫 `shape_S_unit_mean_max_dev` 收的是带符号的 "
        "`abs(v)-1` 再取 max ⇒ 结构上只能报出「mean(S) 略大于 1」的帧（实测报 2.4e−07／3.6e−07 "
        "而真实最坏偏差 1.0）⇒ 会掩盖塌陷帧，须一并修**）⇒ 按 §11.8 记为**实现缺陷，修后重测，"
        "不得当机制结论**。",
        1,
        "e45doc.conclusion",
    ),
    # the three TODOs
    (
        "- **量通道 `mean(out)=T` 不成立是实现缺陷**（`own` 模式下 `T` 取自 `dec[-1]` 的均值，\n"
        "  而输出由结构算子乘出，两者不同源）：A4 第二个种子不达、A4b 三个种子全不达\n"
        "  ⇒ 按 §3.2「实现缺陷，修实现后重测，不得当机制结论」。\n"
        "- **门那组的第三个读数没测出来**：`maxratio_as_trained` 三个种子全是空值、A2 根本没产出\n"
        "  ⇒ **读数缺失**，既不算通过也不算失败；探针需修。",
        "- **量通道 `mean(out)=T` 不成立是实现缺陷**（**v2.46 已定位：不是「T 与 out 不同源」"
        "——`out = T·S` 逐元素相乘、`S = F_mm/(mean(F_mm)+1e-12)` 按构造即单位均值，"
        "故恒等式与 `T` 的来源无关；破口是**形状归一化的绝对 `1e-12` 护栏**："
        "实测真 checkpoint 上 `mean(F_mm)==0` 的塌陷帧数 A4_s1 818/32000、A4b_s0 2741/32000、"
        "A4b_s1 3202/32000、A4b_s2 4272/32000（A4_s0／A4_s2 为 0）⇒ **head 模式的 A4 第二个种子"
        "同样破 ⇒ 不是 own 模式独有**；该帧 `F_mm` 整体为零 ⇒ `S = 0/(0+1e-12) = 0` ⇒ "
        "输出随之塌成 0 而该帧 `T` 非零**）："
        "A4 第二个种子不达、A4b 三个种子全不达\n"
        "  ⇒ 按 §3.2「实现缺陷，修实现后重测，不得当机制结论」。"
        "  **v2.46 另指出探针的守卫 `shape_S_unit_mean_max_dev` 收的是带符号的 `abs(v)-1` 再取 max "
        "⇒ 结构上只能报出「mean(S) 略大于 1」的帧（实测报 2.4e−07／3.6e−07 而真实最坏偏差 1.0）"
        "⇒ 该守卫会掩盖塌陷帧，须与实现一并修**。\n"
        "- **门那组的第三个读数没测出来**：`maxratio_as_trained` 三个种子全是空值、A2 根本没产出\n"
        "  ⇒ **读数缺失**，既不算通过也不算失败；探针需修。"
        "  **v2.46 订正其因果表述**：不是「探针没测」，而是**聚合被 NaN 污染**——实测该读数"
        "在真实 checkpoint 上算得出来（60 批里 49～50 批有可用帧、同一列表用 `np.nanmedian` "
        "给到 A1 0.7221～0.7592、A2 1.0），但 10～11 个全干批次让 E44 的 `_max_ratio` 返回 NaN，"
        "而探针用**非 nan 感知的 `np.median`** 聚合 ⇒ 一个 NaN 批次就把整个读数污染 ⇒ "
        "修法＝`np.nanmedian` + 报可用帧数。"
        "  **v2.46 另记一条新的待读数**：按 §3.1 该读数的预注册判据是 `maxratio_as_trained > 0.9`，"
        "修好聚合后落在 A1 0.72～0.76（低于线）、A2 1.0（在线上）⇒ **修好探针后须重读这条判据**"
        "（此前从未判过）；**但 A1／A2 早已被 S3 等偏置对照判负 ⇒ 不改变任何裁决**。",
        1,
        "e45doc.todos",
    ),
]

MEMORY_FIXES = [
    (
        "① 修实现缺陷（量通道 `mean(out)=T`、门那组 `maxratio` 探针空值）；",
        "① 修实现缺陷（量通道 `mean(out)=T`——**v2.46 已定位：破在形状归一化的绝对 `1e-12` 护栏"
        "（实测真 checkpoint 上 A4_s1 也有 818/32000 帧塌陷 ⇒ 不是 own 模式独有、也不是"
        "「T 与 out 不同源」**；门那组 `maxratio` 探针空值——**v2.46 已定位：读数算得出来"
        "（nanmedian A1 0.7221～0.7592／A2 1.0），空值是非 nan 感知的 `np.median` 被干批次的 NaN "
        "污染**；**修好探针后须重读预注册的 `maxratio_as_trained > 0.9` 判据**）；",
        1,
        "memory.todos",
    ),
    # the MEMORY header version pointer
    (
        "> 权威 `docs/MASTER.md` v2.45；",
        "> 权威 `docs/MASTER.md` v2.46；",
        1,
        "memory.header.version",
    ),
]

HANDOVER_FIXES = [
    # the version pointer in §0
    (
        "Gate 3 写回已完成（MASTER **v2.45**）。",
        "Gate 3 写回已完成（MASTER **v2.46**）。",
        1,
        "handover.pointer",
    ),
    # §4.5 the real next step
    (
        "**① 修两处实现缺陷并留证**（修代码后重测才算机制结论）：量通道 `mean(out)=T` 不成立；"
        "门那组 `maxratio_as_trained` 三 seed 全空值（读数缺失）。",
        "**① 修两处实现缺陷并留证**（修代码后重测才算机制结论）：量通道 `mean(out)=T` 不成立；"
        "门那组 `maxratio_as_trained` 三 seed 全空值（读数缺失）。"
        "**v2.46 已把两处的根因定位到源码行**：量通道破在**形状归一化的绝对 `1e-12` 护栏**"
        "（实测真 checkpoint 上 `mean(F_mm)==0` 的塌陷帧数 A4_s1 818/32000、"
        "A4b_s0 2741/32000、A4b_s1 3202/32000、A4b_s2 4272/32000，A4_s0／A4_s2 为 0 "
        "⇒ **head 模式的 A4 第二个种子同样破 ⇒ 不是 own 模式独有、也不是「T 与 out 不同源」**）；"
        "门那组破在**聚合**（读数其实算得出来：nanmedian A1 0.7221～0.7592、A2 1.0；"
        "空值是非 nan 感知的 `np.median` 被 10～11 个全干批次的 NaN 污染）"
        "⇒ **修好探针后须重读预注册的 `maxratio_as_trained > 0.9` 判据**（此前从未判过；"
        "但 A1／A2 早已被 S3 判负 ⇒ 不改变任何裁决）**。",
        1,
        "handover.nextstep",
    ),
]

# --------------------------------------------------------------------------
# new §12 changelog row (MASTER)
# --------------------------------------------------------------------------

CHANGELOG_ROW = row(
    "2026-11-13",
    f"**{TAG}**",
    "**E45 三件实现待办的源码级根因定位（裁决不变）**：远端状态与前三十轮完全一致"
    "（提交器 273 与汇总作业 291 全 `COMPLETED`、21 份评测齐、队列空、13 份产物与本地 md5 全同、"
    "四套复核脚本全 `VERDICT: PASS`、写回幂等 0 改动）⇒ **六臂判负、不投十 seed 算力**不变。"
    "**本轮首次读源码**（前三十轮只核产物与文档），把三件待办定位到具体代码行，并**订正其中两件的"
    "错误因果表述**：① **幅度截断**：登记的表述正确，现给出确切位置——"
    "`models/topo_simvp/model.py` 全文**没有 `clamp_mm` 的任何入口**，输出场由 "
    "`torch.expm1(log_mm.clamp(-10.0, 6.0))` 硬钳在 402.43 mm，而 `--topo-clamp-mm` 只被损失侧"
    "（`scripts/train.py` → `TopoCriterion`）消费；实测 `field_max_mm` 与 float32(expm1(6)) "
    "逐位相同 ⇒ 修法＝把截断接到输出场上。② **`mean(out)=T`**：登记的因果"
    "（「`own` 模式下 `T` 取自 `dec[-1]` 的均值，而输出由结构算子乘出，两者不同源」）**不成立**——"
    "`out = T·S` 是逐元素相乘，而 `S = F_mm/(mean(F_mm)+1e-12)` 按构造即单位均值，"
    "故 `mean(out)=T` 与 `T` 取自哪一路**无关**，只在 `mean(S)≠1` 时才破。"
    "**实测真 checkpoint（Slurm 作业 298，32000 帧/臂种子）：`mean(F_mm)==0` 的塌陷帧数 "
    "A4_s1 818/32000、A4b_s0 2741/32000、A4b_s1 3202/32000、A4b_s2 4272/32000，"
    "A4_s0／A4_s2 为 0 ⇒ **head 模式的 A4 第二个种子同样破 ⇒ 缺陷不是 own 模式独有**，"
    "破口是**形状归一化用绝对 `1e-12` 护栏去比一个在干帧上本就≈0 的量**；"
    "另探针自己的守卫 `shape_S_unit_mean_max_dev` 收的是**带符号**的 `abs(v)-1` 再取 max ⇒ "
    "结构上只能报出「mean(S) 略大于 1」的帧（实测报 2.4e−07／3.6e−07 而真实最坏偏差 1.0）⇒ "
    "该守卫会掩盖塌陷帧，须与实现一并修。③ **门那组的 `maxratio` 空值**：登记的因果"
    "（「探针没测出来」）**不成立**——在真实 checkpoint 上该读数**算得出来**（60 批里 49～50 批"
    "有可用帧、同一列表用 `np.nanmedian` 给到 **A1 0.7221～0.7592、A2 1.0**），"
    "但 10～11 个全干批次让 E44 的 `_max_ratio` 返回 NaN，"
    "而探针用**非 nan 感知的 `np.median`** 聚合 ⇒ 一个 NaN 批次就把整个读数污染 "
    "⇒ 修法＝`np.nanmedian` + 报可用帧数；**并新登记一条待读数**：该读数的预注册判据是 "
    "`maxratio_as_trained > 0.9`，修好聚合后的 nanmedian 落在 A1 0.72～0.76（低于线）、"
    "A2 1.0（在线上）⇒ **修好探针后须重读这条判据**（此前是读数缺失、从未判过）；"
    "**但 A1／A2 早已被 S3 等偏置对照判负 ⇒ 此读数不改变任何裁决**。**三件仍是待办、"
    "仍不作机制结论**；**门指向（结构上已解决、不是瓶颈）、拓扑支（按预注册结掉）、"
    "S3 六臂判负、不投十 seed 算力、须重新提机制族**五项裁决均未改动。"
    "**本轮新增可复用规则**：**一个读数为空，先分清是「没测到」还是「测到了但被聚合污染」**"
    "——两者在产物里都表现为 NaN，但修法完全不同（前者要补测量、后者要改聚合）；"
    "且**探针自己的守卫读数必须先证明它能报出失败**（带符号的偏差取 max 结构上只能报出单侧，"
    "天然掩盖塌陷帧）；**再者，一个缺陷的根因不能只看「哪个臂坏了」来定位**——"
    "A4_s1 同样坏 ⇒ 破口在两条臂共用的形状归一化，不在 own 模式特有的量来源。",
    "`docs/experiments/E45-gate-direction-amount-separation-baseline.md` · "
    "`tools/_e45_verify_round31.py` · `tools/_e45_meanout_diag.py` · "
    "`tools/_e45_maxratio_diag.py` · `scripts/e45_diag.sbatch`",
)


def patch_master(text):
    if TAG in text:
        print("MASTER already carries", TAG, "-> skip")
        return text
    for old, new, cnt, label in MASTER_FIXES:
        text = apply(text, old, new, cnt, label)
    # insert the v2.46 changelog row right after the v2.45 row
    lines = text.split("\n")
    idx = [i for i, l in enumerate(lines) if l.startswith("| 2026-11-11 | **v2.45** |")]
    if len(idx) != 1:
        raise SystemExit(f"CHANGELOG ANCHOR MISS: found {len(idx)} v2.45 changelog rows")
    i = idx[0]
    prev = unescaped_pipes(lines[i])
    new = unescaped_pipes(CHANGELOG_ROW)
    if prev != new:
        raise SystemExit(f"PIPE COUNT MISMATCH: v2.45 row {prev} cells vs new row {new} cells")
    lines.insert(i + 1, CHANGELOG_ROW)
    return "\n".join(lines)


def patch_index(text):
    if TAG in text:
        print("INDEX already carries", TAG, "-> skip")
        return text
    for old, new, cnt, label in INDEX_FIXES:
        text = apply(text, old, new, cnt, label)
    return text


def patch_plain(text, fixes, name, tag):
    if tag in text:
        print(name, "already carries", tag, "-> skip")
        return text
    for old, new, cnt, label in fixes:
        text = apply(text, old, new, cnt, label)
    return text


def append_daily(text):
    if TAG in text:
        print("daily log already carries", TAG, "-> skip")
        return text
    note = (
        f"\n\n## {TAG}（本次运行，2026-11-13）· E45 状态复核 + 三件待办的源码级根因\n\n"
        "**远端状态无变化**：提交器 **273** `COMPLETED`（墙钟 05:04:04）、汇总作业 **291** "
        "`COMPLETED`（00:14:23）、三 seed 训练数组 275／283 与评测 274／282／290 全 `COMPLETED`、"
        "21 份评测齐；远端 `runs/e45/` 下 13 个 json 与本地 `tools/_evals/e45/` md5 **13/13 全同**；"
        "**未改任何远端状态**——本轮只在 Slurm 上跑了两个**只读 CPU 诊断作业**"
        "（`scripts/e45_diag.sbatch`，job 297 与 298），不训练、不评测、不动 E45 的任何产物。\n\n"
        "**复核结论**：前三十轮的裁决与全部数字**与原始产物逐项一致**"
        "（S3 六臂原始增益区间全跨零、配平后全归零或仍跨零；3-seed 中位数/过地板通过率；"
        "门指向 +0.39～+0.69；场最大值三 seed 全 402.43 mm）⇒ **六臂判负、不投十 seed 算力**不变；"
        "写回脚本连跑 **0 改动**，两个表格校验器 `VERDICT: PASS`。\n\n"
        "**本轮的新东西（第一次读源码）**：把三件待办定位到具体代码行，并**订正其中两件的错误因果**：\n\n"
        "1. **幅度截断**（登记正确）：`models/topo_simvp/model.py` 全文**没有 `clamp_mm` 入口**，"
        "输出场由 `torch.expm1(log_mm.clamp(-10.0, 6.0))` 硬钳在 402.43 mm；"
        "`--topo-clamp-mm` 只被损失侧消费 ⇒ 修法＝把截断接到输出场上。\n"
        "2. **`mean(out)=T`**（**登记的因果不成立**）：`out = T·S` 逐元素相乘、"
        "`S = F_mm/(mean(F_mm)+1e-12)` 按构造即单位均值 ⇒ 恒等式与 `T` 的来源**无关**，"
        "只在 `mean(S)≠1` 时才破；**实测真 checkpoint（32000 帧/臂种子）："
        "`mean(F_mm)==0` 的塌陷帧数 A4_s1 818/32000、A4b_s0 2741/32000、A4b_s1 3202/32000、"
        "A4b_s2 4272/32000，A4_s0／A4_s2 为 0 ⇒ **head 模式的 A4 第二个种子同样破**"
        "⇒ 破口是**绝对 `1e-12` 护栏**，不是 own 模式独有、也不是「T 与 out 不同源」**。"
        "另：探针的守卫 `shape_S_unit_mean_max_dev` 收**带符号**的 `abs(v)-1` 再取 max ⇒ "
        "只能报出单侧（实测报 2.4e−07／3.6e−07、真实最坏偏差 1.0）⇒ 守卫本身也要修。\n"
        "3. **门那组的 `maxratio` 空值**（**登记的因果不成立**）：该读数**算得出来**"
        "（60 批里 49～50 批有可用帧、`np.nanmedian` 给到 A1 0.7221～0.7592、A2 1.0）；"
        "空值是**聚合被 NaN 污染**——10～11 个全干批次让 `_max_ratio` 返回 NaN，"
        "而探针用**非 nan 感知的 `np.median`** ⇒ 一个 NaN 批次污染全体 ⇒ "
        "修法＝`np.nanmedian` + 报可用帧数。**并新登记一条待读数**：该读数的预注册判据是 "
        "`maxratio_as_trained > 0.9`，修好后落在 A1 0.72～0.76（低于线）、A2 1.0（在线上）"
        "⇒ **修好探针后须重读这条判据**（此前从未判过）；**但 A1／A2 早已被 S3 判负，"
        "不改变任何裁决**。\n\n"
        "**新增可复用规则**：一个读数为空，先分清「没测到」还是「测到了但被聚合污染」——"
        "两者在产物里都是 NaN，修法完全不同；且探针的守卫读数必须先证明它能报出失败；"
        "且**根因不能只按「哪个臂坏了」定位**——A4_s1 同样坏 ⇒ 破口在两臂共用的形状归一化。\n"
    )
    return text + note


def main():
    plan = [
        (MASTER, patch_master),
        (INDEX, patch_index),
        (E45DOC, lambda t: patch_plain(t, E45DOC_FIXES, "E45DOC", TAG)),
        (MEMORY, lambda t: patch_plain(t, MEMORY_FIXES, "MEMORY", TAG)),
        (HANDOVER, lambda t: patch_plain(t, HANDOVER_FIXES, "HANDOVER", TAG)),
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
            print("patched:", path)
        else:
            print("unchanged:", path)
    print("\nchanged files:", len(changed))
    return 0


if __name__ == "__main__":
    sys.exit(main())
