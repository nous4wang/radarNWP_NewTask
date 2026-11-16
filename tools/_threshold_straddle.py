"""Is the score at 2 mm zero because the model never reaches 2 mm, or
because it reaches it by a hair and falls short?

The field-wide scan reported a frequency bias of 0.554 at 0.5 mm -- the
model is predicting three times as much water as before -- and yet a
score of exactly zero at 2.0 mm. The two facts are only compatible if
the predictions are piling up just below the threshold. The maximum
prediction in the whole sample was 2.000 and the 99.9th percentile was
1.999, which is what that looks like.

This script measures the distribution of the predictions around each
threshold: how many pixels sit within a hundredth of a millimetre below
it, and how many cross it. A model whose predictions straddle a
threshold by a rounding error is not making a physical statement about
the rain; it has found the boundary of the loss and parked on it, and
the score is a statement about where the boundary was drawn rather
than about the rain.
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

    model_name = "simvp_v2" if name == "simvp_v2" else "simvp_v2_hqma"
    kwargs = dict(in_radar_c=10, in_nwp_c=27, in_len=11, out_len=10)
    if model_name == "simvp_v2_hqma":
        kwargs["head_bins"] = head_bins
    model = build_model(model_name, **kwargs)
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

    runs = [
        ("simvp_v2", "骨干"),
        ("scan_B3", "B3 逐像素"),
        ("scan_B2", "B2 全场比值"),
        ("scan_B1", "B1 alpha0.1"),
        ("scan_B4", "B4 alpha0.5"),
        ("scan_B5", "B5 5档对照"),
    ]

    thresholds = [0.5, 1.0, 2.0, 4.0]

    print("=== 预测值在阈值附近的分布（占全部预测像素的百分比）===")
    print("阈值以下 0.01 内的像素 / 越过阈值的像素")
    for name, label in runs:
        model, problem = load_model(name)
        if problem:
            print(f"{label:14s}  skipped: {problem}")
            continue
        chunks: list[np.ndarray] = []
        with torch.no_grad():
            for i, batch in enumerate(loader):
                if i >= 15:
                    break
                pred = model(batch["x_radar"].cuda(), batch["x_nwp"].cuda())
                chunks.append(pred[:, :, 0].flatten().cpu().numpy())
        flat = np.concatenate(chunks)
        n = flat.size
        print(f"--- {label}  (n={n:,}, max={flat.max():.6f}) ---")
        for thr in thresholds:
            below = np.count_nonzero(
                (flat >= thr - 0.01) & (flat < thr)
            )
            above = np.count_nonzero(flat >= thr)
            print(f"    {thr:4.1f} mm: 差 0.01 以内 {100 * below / n:8.4f} %"
                  f"   越过 {100 * above / n:8.4f} %")
        # The distinct values near the top tell us whether the head has
        # collapsed onto a constant.
        top = np.sort(flat)[-20000:]
        uniq = np.unique(np.round(top, 4))
        print(f"    最高 20000 个像素里的不同取值数: {uniq.size}"
              f"   (取值范围 {uniq.min():.4f} .. {uniq.max():.4f})")
        print(f"    最高值的出现频次: "
              f"{np.count_nonzero(np.isclose(flat, flat.max(), atol=1e-4)):}")
        del model
        torch.cuda.empty_cache()

    print()
    print("=== 每个模型能到达的最高阈值 ===")
    print(f"{'run':14s}{'max_pred':>11s}{'能越过 0.5':>13s}"
          f"{'1.0':>8s}{'2.0':>8s}{'4.0':>8s}")
    for name, label in runs:
        model, problem = load_model(name)
        if problem:
            continue
        maxima: list[float] = []
        chunks = []
        with torch.no_grad():
            for i, batch in enumerate(loader):
                if i >= 15:
                    break
                pred = model(batch["x_radar"].cuda(), batch["x_nwp"].cuda())
                maxima.append(float(pred.max()))
                chunks.append(pred[:, :, 0].flatten().cpu().numpy())
        flat = np.concatenate(chunks)
        crosses = [f"{100 * (flat >= t).mean():8.4f}" for t in (0.5, 1.0, 2.0, 4.0)]
        print(f"{label:14s}{max(maxima):11.4f}{crosses[0]:>13s}"
              f"{crosses[1]:>8s}{crosses[2]:>8s}{crosses[3]:>8s}")
        del model
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
