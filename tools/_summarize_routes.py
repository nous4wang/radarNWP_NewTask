"""One table for the route arms, the retrained baselines and the reference.

Reads the downloaded .eval.json files and prints the comparison that the
paper needs: every arm against persistence and against the retrained
baselines, plus the per-lead-time breakdown that shows where each arm
collapses.
"""
from __future__ import annotations

import json
from pathlib import Path

BASE = Path(__file__).resolve().parent / "_evals"
TIERS = ["0.1", "0.5", "2.0", "4.0"]
FLOOR = 0.0924  # persistence CSI@2.0, the strong-rain floor


def load(run: str, arm: str) -> dict:
    return json.loads((BASE / run / f"{arm}.eval.json").read_text(encoding="utf-8"))


def csi(rep: dict, t: str) -> float:
    return rep["thresholds"][t]["csi"]


def main() -> None:
    arms = [
        ("reference", "persistence", "持久性外推", "参考"),
        ("route_baseline", "simvp_v2", "simvp_v2 (重训基线)", "同任务"),
        ("route_baseline", "simvp_v2_nwp_concat", "simvp_v2_nwp_concat (重训基线)", "同任务"),
        ("route_r1r2", "base_concat", "base_concat", "同任务"),
        ("route_r1r2", "qpe_0", "qpe_0 (R1, w=0)", "同任务"),
        ("route_r1r2", "qpe_05", "qpe_05 (R1, w=0.5)", "同任务"),
        ("route_r1r2", "assim_lag11", "assim_lag11 (R2)", "异任务"),
        ("route_r1r2", "assim_lag4", "assim_lag4 (R2, 24min)", "异任务"),
        ("route_r1r2", "assim_lag2", "assim_lag2 (R2, 12min)", "异任务"),
        ("route_r1r2", "assim_lag0", "assim_lag0 (R2, 无延迟)", "异任务"),
    ]

    print("=" * 108)
    print("主表：路线臂 vs 重训基线 vs 持久性（clean test split, n=538, ep19, score-aware）")
    print("=" * 108)
    hdr = f"{'臂':<30}{'组':<7}{'CSI@0.1':>9}{'CSI@0.5':>9}{'CSI@2.0':>9}{'CSI@4.0':>9}{'FAR@0.5':>9}{'BIAS@0.5':>9}"
    print(hdr)
    print("-" * 108)
    reps = {}
    for run, arm, label, grp in arms:
        rep = load(run, arm)
        reps[arm] = rep
        th = rep["thresholds"]
        print(
            f"{label:<30}{grp:<7}"
            f"{th['0.1']['csi']:>9.4f}{th['0.5']['csi']:>9.4f}"
            f"{th['2.0']['csi']:>9.4f}{th['4.0']['csi']:>9.4f}"
            f"{th['0.5']['far']:>9.4f}{th['0.5']['bias']:>9.4f}"
        )

    print()
    print("=" * 108)
    print(f"判据一：CSI@2.0 > {FLOOR}（持久性地板）")
    print("=" * 108)
    for run, arm, label, grp in arms:
        v = csi(reps[arm], "2.0")
        flag = "PASS" if v > FLOOR else "FAIL"
        print(f"{label:<30} {v:.4f}   {v / FLOOR * 100:6.1f}% of floor   [{flag}]")

    print()
    print("=" * 108)
    print("判据二：R1 消融 qpe_05 > qpe_0（历史降水监督是否有害）")
    print("=" * 108)
    for t in TIERS:
        a = csi(reps["qpe_0"], t)
        b = csi(reps["qpe_05"], t)
        print(f"  CSI@{t:<5} qpe_0={a:.4f}  qpe_05={b:.4f}   ratio={b / a:.3f}  "
              f"{'监督更差' if b < a else '监督更好'}")

    print()
    print("=" * 108)
    print("判据三：QPE-QPF 拆分 qpe_0 vs base_concat（是否只是参数变多）")
    print("=" * 108)
    for t in TIERS:
        a = csi(reps["base_concat"], t)
        b = csi(reps["qpe_0"], t)
        print(f"  CSI@{t:<5} base_concat={a:.4f}  qpe_0={b:.4f}   ratio={b / a:.2f}x")

    print()
    print("=" * 108)
    print("逐预报时效 CSI@2.0（结构差异：谁在远期塌掉）")
    print("=" * 108)
    leads = [str(i) for i in range(10)]
    print(f"{'臂':<30}" + "".join(f"{'L' + l:>8}" for l in leads) + f"{'衰减':>9}")
    for run, arm, label, grp in arms:
        rep = reps[arm]
        # lead_csi holds bare csi numbers; lead_scores holds the full metric
        # dict. The persistence reference only carries lead_scores.
        if "lead_csi" in rep:
            row = [rep["lead_csi"][l]["2.0"] for l in leads]
        else:
            row = [rep["lead_scores"][l]["2.0"]["csi"] for l in leads]
        first, last = row[0], row[-1]
        print(f"{label:<30}" + "".join(f"{v:>8.4f}" for v in row) + f"{first / last:>8.1f}x")


if __name__ == "__main__":
    main()
