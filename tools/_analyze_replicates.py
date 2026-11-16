"""The replicate table: how much does one configuration move between runs?

Three arms in the project are the same configuration trained more than once,
under the same seed and the same code. They are the only direct measurement
of the run-to-run spread available, and every threshold in the project is
judged against it.

The evaluation side was checked separately: the same checkpoint evaluated
twice gives a bit-identical report, so all of the spread below comes from
the training side.
"""
from __future__ import annotations

import json
from pathlib import Path

BASE = Path(__file__).resolve().parent / "_evals"
TIERS = ["0.1", "0.5", "2.0", "4.0"]


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def csi(rep: dict, t: str) -> float:
    return rep["thresholds"][t]["csi"]


def lead(rep: dict, t: str = "2.0") -> list[float]:
    if "lead_csi" in rep:
        return [rep["lead_csi"][str(i)][t] for i in range(10)]
    return [rep["lead_scores"][str(i)][t]["csi"] for i in range(10)]


# Same configuration, trained more than once. The learning rate is the
# discriminator: 3e-4 for the matrix and the epoch curve, 2e-4 for the
# routes and the retrained baselines.
REPLICATES = {
    "simvp_v2 @ lr 3e-4": [
        ("score_matrix/simvp_v2", BASE / "score_matrix/simvp_v2.eval.json"),
        ("epoch_curve seed0 ep19", BASE.parent.parent / ".workbuddy/epoch_curve/sa_epoch_19.eval.json"),
    ],
    "simvp_v2_nwp_concat @ lr 2e-4": [
        ("route_r1r2/base_concat", BASE / "route_r1r2/base_concat.eval.json"),
        ("route_baseline/retrained", BASE / "route_baseline/simvp_v2_nwp_concat.eval.json"),
    ],
}


def main() -> None:
    print("=" * 100)
    print("同配置重复训练：score-aware 的运行间离散度")
    print("=" * 100)

    worst = 0.0
    for name, entries in REPLICATES.items():
        print(f"\n--- {name} ---")
        reps = []
        for label, path in entries:
            if not path.exists():
                print(f"  {label:<28} MISSING {path}")
                continue
            rep = load(path)
            reps.append((label, rep))
            row = "  ".join(f"CSI@{t}={csi(rep, t):.4f}" for t in TIERS)
            print(f"  {label:<28} {row}")

        if len(reps) == 2:
            (_, a), (_, b) = reps
            ratio = csi(a, "2.0") / csi(b, "2.0")
            worst = max(worst, ratio)
            print(f"  {'=> CSI@2.0 比值':<28} {ratio:.2f}x  "
                  f"(min {min(csi(a,'2.0'), csi(b,'2.0')):.4f}, "
                  f"max {max(csi(a,'2.0'), csi(b,'2.0')):.4f})")

    print()
    print("=" * 100)
    print(f"实测方差带上界：CSI@2.0 同配置最大比值 = {worst:.2f}x")
    print("=" * 100)

    print()
    print("=" * 100)
    print("这对已有判据意味着什么（判据 = 单次运行的比值）")
    print("=" * 100)
    checks = [
        ("R1 消融 qpe_05 vs qpe_0", 0.0510, 0.0308,
         "§4.9.2 写'超出 1.4x 方差带 -> R1 被证伪'"),
        ("QPE-QPF 拆分 qpe_0 vs base_concat", 0.0510, 0.0258,
         "§4.9.3 写'1.98x -> 拆分有效'"),
        ("qpe_0 vs base_concat(另一次重复)", 0.0510, 0.0145,
         "同一对比在另一个同配置重复上"),
    ]
    for label, hi, lo, note in checks:
        ratio = hi / lo
        inside = ratio < worst
        print(f"  {label:<36} {ratio:.2f}x  "
              f"{'落在方差带内 -> 不可判定' if inside else '超出方差带 -> 可判定'}   ({note})")

    print()
    print("=" * 100)
    print("逐时效衰减（结构对比，比单点比值稳健）")
    print("=" * 100)
    arms = [
        ("持久性外推", BASE / "reference/persistence.eval.json"),
        ("base_concat (r1r2)", BASE / "route_r1r2/base_concat.eval.json"),
        ("base_concat (重复训练)", BASE / "route_baseline/simvp_v2_nwp_concat.eval.json"),
        ("qpe_0", BASE / "route_r1r2/qpe_0.eval.json"),
        ("qpe_05", BASE / "route_r1r2/qpe_05.eval.json"),
    ]
    print(f"{'臂':<26}" + "".join(f"{'L'+str(i):>8}" for i in range(10)) + f"{'衰减':>9}")
    for label, path in arms:
        if not path.exists():
            continue
        row = lead(load(path))
        print(f"{label:<26}" + "".join(f"{v:>8.4f}" for v in row) + f"{row[0]/row[-1]:>8.1f}x")


if __name__ == "__main__":
    main()
