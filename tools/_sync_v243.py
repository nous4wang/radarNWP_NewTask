"""Write the E45 run status into MASTER (four places) and INDEX.

The rows are built from their CELLS, and the separators are inserted by
the code.  Hand-writing the separators inside the text is how a row ends
up with the wrong number of columns -- and a markdown table with the
wrong number of columns silently breaks the whole table.

The script is idempotent: a line that already carries the version tag
is left alone, so running it twice cannot insert the same text twice.
"""
from pathlib import Path

MASTER = Path("docs/MASTER.md")
INDEX = Path("docs/experiments/INDEX.md")

TAG = "v2.43"
LINK = "`docs/experiments/E45-gate-direction-amount-separation-baseline.md`"


def row(*cells):
    """One markdown table row from its cells.

    The trailing separator is part of the row: the rows this file
    replaces all end with one, and a row without it renders as one
    cell short.
    """
    return "| " + " | ".join(c.strip() for c in cells) + " |"


#: The markers are the FIRST few characters of the line, which is what
#: makes them stable against edits further along the line.
HEADER_MARK = "| 最后更新 | 2026-11-11（v2.42，"
CARD_MARK = "| §4.47 |"
STATUS_MARK = "| **E45 / §4.47 修三处机制缺陷 + 摆正基准** |"
CHANGELOG_MARK = "| 2026-11-11 | **v2.42** |"
CHANGELOG_MARK_V43 = "| 2026-11-11 | **v2.43** |"


HEADER_NEW = row(
    "最后更新",
    "2026-11-11（v2.43，**E45 运行中：seed 0 已完成，门指向结构上已解决、"
    "拓扑支两条读数都不过、量通道没有退化成常数**）",
    "**判据一字未改**（S0–S6 原文 + 七臂 + 三 seed），**新增 §8 运行状态节**"
    "（只记事实与读数，不改裁决方向）；"
    "seed 0 分数 A4 0.1755／A3b 0.1697／A3a 0.1495 对 B0 0.1218，"
    "**但 BIAS 同时爆表（1.61／0.92 对 0.51）且 A4／A3b 超过 Oracle 上界 0.1653** ⇒ "
    "**过预报，不写提升，须走 S3 等偏置对照**；"
    "机制读数：`corr(mask,amp)` 从 −0.858320 转到 **+0.6103／+0.4035**、"
    "`mask_at_peak` 到 **0.8217／1.0**（门指向**结构上已解决**，但两臂 CSI 低于 B0）；"
    "`grad_top1pct_share` **0.5526／0.9712**（都 > 0.5）、`field_max_mm` 仍 **402.43 mm** "
    "⇒ 按预注册**拓扑支结掉**；"
    "`scale_cv` **2.00／2.11**、`scale_truth_corr` **0.743／0.657** ⇒ **量通道没有退化成常数**；"
    "A4b 的 `mean(out)=T` 不达（rel 0.0393）⇒ **实现缺陷，修后重测**；"
    "**下一步：等三 seed 跑完 → 取回 → S3 → 按中位数+通过率判读**。"
    "**详见 `R:/1radarNWPprediction/.workbuddy/memory/HANDOVER_E45.md`**（新会话唯一入口）",
    "**本会话 token 教训**：累积上下文达 12M token，每回合工具调用 ≤ 12 次、"
    "远端检查合并成一条命令、同一文件不重复读",
)

CARD_NEW = row(
    "§4.47",
    "修三处机制缺陷 + 把比较基准摆正：门指向、梯度集中、量与结构解耦",
    "七臂各只比基线多一处改动：修「门怎么指向」、治梯度集中、把量与结构解耦。",
    "**运行中（v2.43）**：seed 0 已完成。**门指向结构上已解决**"
    "（`corr(mask,amp)` −0.858320 → **+0.6103／+0.4035**，`mask_at_peak` → **0.8217／1.0**），"
    "但两臂 CSI 低于 B0；**拓扑支两条读数都不过**"
    "（`grad_top1pct_share` 0.5526／0.9712 > 0.5；`field_max_mm` 仍 402.43 mm）"
    "⇒ 按预注册结掉；**量通道没有退化成常数**"
    "（`scale_cv` 2.00／2.11，`scale_truth_corr` 0.743／0.657）；"
    "A4／A3b 的 CSI 超过 Oracle 上界且 BIAS 爆表 ⇒ **过预报，须 S3 才判**。"
    "seed 0 分数与读数见 E45 §8",
    LINK,
)

STATUS_NEW = row(
    "**E45 / §4.47 修三处机制缺陷 + 摆正基准**",
    "**运行中（2026-11-11 v2.43）**：七臂判据零改动，seed 0 已完成并取回。"
    "**门指向结构上已解决**（`corr(mask,amp)` 从 −0.858320 转到 +0.6103／+0.4035，"
    "`mask_at_peak` 到 0.8217／1.0，A1 的门严格是残差的单调函数 0.9941）"
    "**但 A1／A2 的 CSI 低于 B0**；**拓扑支按预注册结掉**"
    "（`grad_top1pct_share` 0.5526／0.9712 都 > 0.5，`field_max_mm` 仍 402.43 mm —— "
    "**幅度截断只作用在拓扑项内部、没作用在输出场上**，是下一轮的候选修复点）；"
    "**量通道没有退化成常数**（`scale_cv` 2.00／2.11 ≥ 0.05，"
    "`scale_truth_corr` 0.743／0.657 ≥ 0.3，`scale_is_constant` 均 False）"
    "——**用户问的「scale 是否变成常数」答案是没有**；"
    "A4b 的 `mean(out)=T` 不达（rel 0.0393）⇒ **实现缺陷，修后重测，不当机制结论**。"
    "seed 0 分数 A4 0.1755／A3b 0.1697／A3a 0.1495 对 B0 0.1218，"
    "**但 A4／A3b 超过 Oracle 上界 0.1653 且 BIAS 爆表 1.61／0.92 对 0.51 ⇒ 过预报，不写提升**。"
    "**下一步**：等三 seed 跑完 → 取回产物 → 跑 S3 等偏置对照 → 按中位数+通过率判读 → 写回四处",
    "｜ 交接：`HANDOVER_E45.md`",
)

CHANGELOG_NEW = row(
    "2026-11-11",
    "**v2.43**",
    "**E45 运行状态写回（判据零改动，只补 §8 运行状态节）**：seed 0 已完成并取回。"
    "**门指向结构上已解决**（`corr(mask,amp)` −0.858320 → **+0.6103／+0.4035**，"
    "`mask_at_peak` → **0.8217／1.0**），但 A1／A2 的 `CSI@2.0` **低于 B0**；"
    "**拓扑支按预注册结掉**（`grad_top1pct_share` 0.5526／0.9712 都 > 0.5，"
    "`field_max_mm` 仍 402.43 mm ⇒ **幅度截断只作用在拓扑项内部、没作用在输出场上**）；"
    "**量通道没有退化成常数**（`scale_cv` 2.00／2.11、`scale_truth_corr` 0.743／0.657）"
    "——**用户问的「scale 是否变成常数」答案是没有**；A4b 的 `mean(out)=T` 不达（rel 0.0393）"
    "⇒ **实现缺陷，修后重测**。A4／A3b 的 `CSI@2.0` 超过 Oracle 上界 0.1653 且 BIAS 爆表 "
    "⇒ **过预报，不写提升，须 S3 等偏置对照**。"
    "**同时完成上下文瘦身与切会话交接**：工作记忆 7977 → 5336 字符（67 %），"
    "新增 `HANDOVER_E45.md` 作新会话唯一入口，历史全文留档于 `MEMORY_v240_FULL_20261111.md`",
    LINK,
)


def replace_line(lines, marker, new, tag):
    """Rewrite one line, but only once.

    The tag is the version string the new line carries.  If the line
    already carries it, the write has happened and is skipped -- so
    running the script twice cannot insert the same text twice.
    """
    hits = [i for i, line in enumerate(lines) if line.startswith(marker)]
    if not hits and marker == CHANGELOG_MARK:
        hits = [i for i, line in enumerate(lines)
                if line.startswith(CHANGELOG_MARK_V43)]
    if not hits:
        return 0, f"NOT FOUND: {marker[:60]}"
    if len(hits) > 1:
        return 0, f"AMBIGUOUS ({len(hits)}): {marker[:60]}"
    if tag in lines[hits[0]]:
        return 0, f"already up to date (line {hits[0] + 1})"
    lines[hits[0]] = new
    return 1, f"replaced line {hits[0] + 1}"


def main() -> int:
    lines = MASTER.read_text(encoding="utf-8").split("\n")
    changed = 0
    for marker, new in (
        (HEADER_MARK, HEADER_NEW),
        (CARD_MARK, CARD_NEW),
        (STATUS_MARK, STATUS_NEW),
        (CHANGELOG_MARK, CHANGELOG_NEW),
    ):
        n, msg = replace_line(lines, marker, new, TAG)
        changed += n
        print(f"  {msg}")
    if changed:
        MASTER.write_text("\n".join(lines), encoding="utf-8")

    # INDEX: the E45 row, extended with the run status.
    idx_lines = INDEX.read_text(encoding="utf-8").split("\n")
    hits = [i for i, line in enumerate(idx_lines) if line.startswith("| E45 |")]
    if len(hits) != 1:
        print(f"  INDEX E45 row: found {len(hits)} (expected 1) -- skipped")
    else:
        i = hits[0]
        if TAG in idx_lines[i]:
            print("  INDEX E45 row already carries the tag")
        else:
            idx_lines[i] = idx_lines[i].rstrip() + (
                "**v2.43 运行状态**：seed 0 已完成；**门指向结构上已解决**"
                "（`corr(mask,amp)` −0.858320 → **+0.6103／+0.4035**）但两臂 CSI 低于 B0；"
                "**拓扑支按预注册结掉**（梯度集中 0.5526／0.9712 > 0.5，场最大值仍 402.43 mm）；"
                "**量通道没有退化成常数**（`scale_cv` 2.00／2.11）；"
                "A4／A3b 超过 Oracle 上界且 BIAS 爆表 ⇒ **过预报，须 S3 才判**；"
                "**新增 §8 运行状态节（判据零改动）**"
            )
            INDEX.write_text("\n".join(idx_lines), encoding="utf-8")
            print(f"  INDEX E45 row updated (line {i + 1})")

    print(f"done: {changed}/4 MASTER places rewritten")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
