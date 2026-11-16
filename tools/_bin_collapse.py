"""Where the collapse to the bin floor comes from.

The field-wide tier loss produced predictions whose maximum was 1.9998
millimetres, with one and a half per cent of the pixels sitting within
a hundredth of a millimetre below the two-millimetre threshold and none
crossing it. The score at 2 mm was exactly zero while the frequency bias
at 0.5 mm had more than tripled.

The head reconstructs the rain as the expectation over the bin
distribution: the rain of each bin is its lower bound plus a learned
offset, and the prediction is the probability-weighted sum. Two things
could have produced a prediction that sits just under a bin boundary:
the offsets could have collapsed towards zero, or the probability
mass could have concentrated on the bin whose lower bound happens to be
the threshold.

This script separates them, and checks the consistency that the
construction is supposed to guarantee: that a pixel the head believes
exceeds 2 mm has a rain rate above 2 mm.

The consistency check is the important one. The head can believe that
99 per cent of a pixel's rain is heavy and still report a rain rate
below the heavy threshold, because the expectation of a distribution
whose upper bin starts at 2 mm is only 2 mm when the offset that would
push it past the boundary is zero. The probability and the rate are
then inconsistent, and the verification, which sees only the rate,
scores the pixel as a miss.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "/home/wf/radarNWP")

from radar_nwp_bench.data import RadarNWPDataset
from radar_nwp_bench.models import build_model

DATA = "/home/wf/data/radarNWP/BJ_mmap"
RUNS = Path("/home/wf/radarNWP/runs")


def load_model(name: str):
    found = None
    for folder in ("hqma_scan2", "hqma"):
        base = RUNS / folder / name
        if base.exists():
            for sub in sorted(base.iterdir()):
                if (sub / "last.pt").exists():
                    found = (sub / "last.pt", sub / "config.json")
                    break
        if found:
            break
    if found is None:
        return None, "no checkpoint"
    ckpt_path, cfg_path = found

    head_bins = 6
    if cfg_path.exists():
        head_bins = json.loads(cfg_path.read_text()).get("head_bins", 6)

    weights = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    weights = {k: v for k, v in weights.items() if isinstance(v, torch.Tensor)}

    model = build_model(
        "simvp_v2_hqma", in_radar_c=10, in_nwp_c=27, in_len=11, out_len=10,
        head_bins=head_bins,
    )
    try:
        model.load_state_dict(weights)
    except RuntimeError as exc:
        return None, str(exc).splitlines()[0]
    return model.eval().cuda(), None


def main() -> None:
    dataset = RadarNWPDataset(
        DATA, packed=True, split="test", in_len=11, out_len=10,
        patch_size=64, slide_stride=64,
    )
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=8, shuffle=False, num_workers=4,
    )

    runs = [("scan_B2", "B2 全场比值"), ("scan_B1", "B1 a=0.1"), ("scan_B5", "B5 5档")]

    for name, label in runs:
        model, problem = load_model(name)
        if problem:
            print(f"{label}: skipped ({problem})")
            continue
        head = model.quantum_povm_head
        thresholds = list(head.thresholds)

        mass_sum = np.zeros(len(thresholds) + 1)
        count = 0
        confident = 0
        confident_above = 0
        below_max = 0.0

        with torch.no_grad():
            for i, batch in enumerate(loader):
                if i >= 6:
                    break
                pred = model(batch["x_radar"].cuda(), batch["x_nwp"].cuda())
                cum = head.last_cum_probs
                if cum is None:
                    continue
                if cum.dim() == 5:
                    b, t, k_dim, h, w = cum.shape
                    c = cum.reshape(b * t, k_dim, h, w)
                else:
                    c = cum
                    h, w = c.shape[-2:]
                    b, k_dim = c.shape[0], c.shape[1]
                    t = 1

                # The exceedance probabilities are P(R >= threshold_k),
                # k = 0 .. K-1. The bin masses follow from them: the dry
                # bin is what is left over, and each higher bin is the
                # difference of two exceedance probabilities.
                tail = torch.cat([c[:, 1:], torch.zeros_like(c[:, :1])], dim=1)
                masses = c - tail                      # per-threshold gaps
                # Prepend the dry bin.
                dry = (1.0 - c[:, :1]).clamp_min(0.0)
                full = torch.cat([dry, masses], dim=1)
                mass_sum += full.mean(dim=(0, 2, 3)).cpu().numpy()
                count += 1

                # Consistency at 2 mm: the index of the 2.0 threshold.
                idx = thresholds.index(2.0) if 2.0 in thresholds else None
                if idx is None:
                    continue
                p2 = c[:, idx]
                rates = pred.reshape(b * t, h, w)
                sel = p2 > 0.9
                confident += int(sel.sum())
                above = sel & (rates >= 2.0)
                confident_above += int(above.sum())
                miss = sel & (rates < 2.0)
                if miss.any():
                    below_max = max(below_max, float(rates[miss].max()))

        if not count:
            continue
        masses = mass_sum / count
        floors = [0.0] + thresholds
        print(f"=== {label} (head_bins={head.num_bins}) ===")
        print("档      下界    平均概率质量")
        for k in range(len(masses)):
            print(f"  {k}   {floors[k]:6.2f}   {masses[k]:12.6f}")
        print(f"  P(>=2mm) = {masses[1 + thresholds.index(2.0)]:.6f}")
        print()
        print(f"  头确信超过 2mm 的像素: {confident:,}")
        print(f"  其中雨量确实 >= 2mm 的: {confident_above:,} "
              f"({100 * confident_above / max(confident, 1):.3f} %)")
        if below_max:
            print(f"  未跨过者的最大雨量: {below_max:.6f} mm")
        print()

        del model
        torch.cuda.empty_cache()

    print("=== 构造上的一致性：一个数学事实 ===")
    print("预测 = Σ_k p_k · (下界_k + offset_k)")
    print("若 p 集中在下界 2.0 的档而 offset → 0，则预测 → 2.0⁻，")
    print("而超阈概率 P(≥2mm) → 1。概率说'确定超过'，雨量说'差一点'。")
    print("期望的重构无法避免这一点：期望落在档下界之上，")
    print("除非该档的概率是 1，否则期望就低于下界。")


if __name__ == "__main__":
    main()
