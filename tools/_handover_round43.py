"""Add the round-43 item (⑧) to HANDOVER_E45.md section 4.

The handover is the entry file every new session reads first, so the
branch-decomposition finding must be in it, not only in MASTER.

Idempotent: the marker is a literal substring of the inserted text.
"""
from pathlib import Path
import sys

HANDOVER = Path("R:/1radarNWPprediction/.workbuddy/memory/HANDOVER_E45.md")

MARK = "⑧（2026-11-13 第 43 轮新增证据）"

ITEM = (
    "**⑧（2026-11-13 第 43 轮新增证据）派生布尔的分支分解："
    "`scale_is_constant` 是**析取**，E45 六格里 cv 分支从未触发**——"
    "脚本 `tools/_e45_verify_round43.py`（只读本地，POS 15/15 VERDICT: PASS）"
    "把量通道的 `scale_is_constant` 按探针自己的定义"
    "（`tools/_e45_mechanism_probe.py` l.63-64／l.679-681）拆成两条触发分支："
    "`scale_cv < 0.05`（量不再变化）**或** `scale_truth_corr < 0.3`"
    "（量仍在变但不再携带雨强信息）。**逐格重算六格的原始探针 JSON**："
    "**cv 分支在 E45 全部六格从未触发**（最小 `scale_cv` 1.2358 ＝ 底线的 24.7 倍）；"
    "**只有相关性分支触发过一次，且只在 A4b 种子 2**"
    "（`scale_truth_corr` 0.2789）；**而该格 `scale_cv` 3.0188 恰是六格最高** ⇒ "
    "**量通道没有变成常数**，真正发生的是**量级塌缩**"
    "（`scale_mean` 0.0316，不足同臂另两格 0.407／0.353 的 1/11）**＋ 与真值脱钩**。"
    "⇒ 「E45 全程没有任何一格真的退化成常数」；"
    "**「量通道不是瓶颈」这条结论反而更硬**（唯一的退化就是相关性那一支）；"
    "**A4b 的舍弃理由按预注册 §3.1b／§3.2 本就是相关性措辞**"
    "（「既有解码头的平均雨强不足以承担量通道」），"
    "而结案正文把它一律写成「退化成常数」＝断言了 cv 分支 ⇒ **已订正（MASTER v2.48）**。"
    "**裁决零变化**：A4b 种子 2 的标志仍为真 ⇒ 预注册的舍弃照样触发；"
    "六臂仍全部判负、不投十 seed 算力。"
    "**新登记一条待办（不作机制结论）**：A4b 种子 2 同时是 `scale_mean` 最小、"
    "`scale_cv` 最大、`scale_truth_corr` 最低、`mean_out_equals_T` 相对误差最大"
    "（0.258）的一格，与 v2.46 记的塌陷帧数最多（4272/32000）同格 ⇒ "
    "其「退化」**可能与 `1e-12` 护栏缺陷同源**；"
    "但 A4b 种子 0 有 2741 塌陷帧却 `scale_mean` 最高（0.407）⇒ 不是单调关系、"
    "**尚未建立因果** ⇒ 修护栏后重测时**须重新判这一格并分分支报**。\n"
)

ANCHOR = "\n\n---\n\n### 下面是结案前的执行记录（存档用，不必重跑）"


def main():
    text = HANDOVER.read_text(encoding="utf-8")
    if MARK in text:
        print("HANDOVER already carries item ⑧ -> skip")
        return
    if ANCHOR not in text:
        raise SystemExit(
            "ANCHOR MISS: the end of section 4 was not found "
            "(the '下面是结案前的执行记录' separator)")
    text = text.replace(ANCHOR, "\n\n" + ITEM + ANCHOR, 1)
    HANDOVER.write_text(text, encoding="utf-8")
    print("changed: HANDOVER_E45.md (item ⑧ inserted)")

    # verify
    t = HANDOVER.read_text(encoding="utf-8")
    assert t.count(MARK) == 1, f"item ⑧ appears {t.count(MARK)}x"
    assert "**⑧" in t
    print(f"verified: item ⑧ present exactly once "
          f"(file now {len(t)} chars)")


if __name__ == "__main__":
    sys.exit(main())
