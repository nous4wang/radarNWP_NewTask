"""Generate the HQMA results table from the evaluation reports."""
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
RUNS = REPO / ".workbuddy" / "runs_hqma"

#: The spectral bands, measured on the same windows as the scores.
SPECTRA = {
    "simvp_v2": (0.0163, 0.0320),
    "simvp_v2_nwp_concat": (0.0153, 0.0367),
    "hqma_standard": (0.0171, 0.0277),
    "hqma_balanced": (0.0692, 0.0676),
}

ORDER = [
    "simvp_v2",
    "simvp_v2_nwp_concat",
    "hqma_standard",
    "hqma_balanced",
]

LABELS = {
    "simvp_v2": "simvp_v2 (backbone)",
    "simvp_v2_nwp_concat": "simvp_v2_nwp_concat (M1)",
    "hqma_standard": "hqma_standard (new head, standard loss)",
    "hqma_balanced": "hqma_balanced (new head, balanced loss)",
}


def main() -> None:
    rows = []
    for name in ORDER:
        path = RUNS / f"{name}.eval.json"
        if not path.exists():
            continue
        t = json.loads(path.read_text(encoding="utf-8"))["thresholds"]
        high, low = SPECTRA[name]
        rows.append((
            LABELS[name],
            t["0.1"]["csi"], t["0.5"]["csi"], t["1.0"]["csi"],
            t["2.0"]["csi"], t["4.0"]["csi"],
            t["0.5"]["far"], t["0.5"]["pod"], t["0.5"]["bias"],
            high, low,
            t["0.5"]["hits"], t["0.5"]["false_alarms"], t["0.5"]["misses"],
        ))

    lines = [
        "## HQMA 对照实验（洁净划分，n=538，20 epoch，norm-v2）",
        "",
        "| run | CSI@0.1 | CSI@0.5 | CSI@1.0 | CSI@2.0 | CSI@4.0 "
        "| FAR@0.5 | POD@0.5 | BIAS@0.5 | high_k | low_k |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for row in rows:
        lines.append(
            "| " + row[0] + " | "
            + " | ".join(f"{v:.3f}" for v in row[1:11]) + " |"
        )

    lines += [
        "",
        "high_k = λ<10 km 的谱能量保留率（预测/实测）；low_k = λ>30 km。"
        "对照：pysteps_adv（物理外推）high_k = 3.566 / low_k = 0.452。",
        "",
        "命中计数的分解（@0.5 mm）：",
        "",
        "| run | hits | false alarms | misses |",
        "|---|---|---|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row[0]} | {int(row[11]):,} | {int(row[12]):,} | {int(row[13]):,} |"
        )

    lines += [
        "",
        "**判读**",
        "",
        "1. `hqma_balanced` 把 **BIAS@0.5 从 0.124 抬到 0.211**（对照 "
        "simvp_v2），**CSI@2.0 从 0.004 抬到 0.019**（4.8 倍），"
        "**CSI@4.0 从 0.000 抬到 0.007**（从零到有信号）。",
        "2. 谱诊断同步印证：**high_k 从 0.0163 抬到 0.0692（4.2 倍）**，"
        "low_k 从 0.0320 到 0.0676（2.1 倍）。",
        "   → **欠预测 → 抹平小尺度 → 强降水 CSI 塌陷** 这条链在新模型上被反向"
        "验证：抬 BIAS 就抬回了谱能量，谱能量回来强降水分数就回来。",
        "3. 代价是弱降水：CSI@0.1 从 0.500 降到 0.251，FAR@0.5 从 0.440 升到 "
        "0.585。命中 +26 %，虚警 +127 % —— **在弱降水上用虚警换命中率**。",
        "4. `hqma_standard`（新头 + 标准损失）几乎没动 BIAS（0.146）→ "
        "**起作用的是 balanced 损失，不是 POVM 头本身**。头单独用没有效果，"
        "必须配上以分数为目标的损失。",
        "",
        "**结论**：这套组合是**第一个在强降水上超过全部既有基线的深度模型**"
        "（CSI@2.0 = 0.019 > mmf_rnn 0.023? 否 —— 见下）……",
    ]
    # The last claim has to be checked against the matrix, not asserted.
    matrix = REPO / ".workbuddy" / "runs"
    best = None
    best_deep = None
    if matrix.exists():
        for path in matrix.glob("*.eval.json"):
            name = path.stem.replace(".eval", "")
            t = json.loads(path.read_text(encoding="utf-8"))["thresholds"]
            value = t["2.0"]["csi"]
            if value != value:
                continue
            if best is None or value > best[1]:
                best = (name, value)
            # The physical baseline is not a deep model, so the comparison
            # that matters for this table is against the deep ones.
            if name != "pysteps_adv" and (
                best_deep is None or value > best_deep[1]
            ):
                best_deep = (name, value)
    hqma_csi = rows[-1][4] if rows else float("nan")
    if best_deep and best:
        verdict = (
            "**结论**：`hqma_balanced` 的 CSI@2.0 = "
            f"{hqma_csi:.3f}，"
            + (
                "**超过**"
                if hqma_csi > best_deep[1]
                else "**低于**"
            )
            + f"矩阵里最好的深度模型 `{best_deep[0]}` = {best_deep[1]:.3f}，"
            f"也远低于物理外推 `{best[0]}` = {best[1]:.3f}。"
        )
        if hqma_csi <= best_deep[1]:
            verdict += (
                "\n\n它把强降水从\"几乎为零\"提到了可测的水平，但仍未追平"
                "物理外推 —— 这与机理一致：抬 BIAS 只是第一步，"
                "要把 high_k 从 0.069 推到物理外推的 3.566 还需要更彻底的"
                "改动（损失重加权的形式、极端样本重采样、或输出头的分辨率）。"
            )
        lines[-1] = verdict

    text = "\n".join(lines) + "\n"
    (REPO / "docs" / "TGRS_TABLES_HQMA.md").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
