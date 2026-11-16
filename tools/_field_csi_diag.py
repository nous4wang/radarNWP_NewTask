"""Why the field-wide tier loss lifts the bias but loses the heavy rain.

The scan produced a striking pattern: the frequency bias at 0.5 mm rose
from 0.211 (the soft-CSI run) to 0.554, while the score at 2.0 mm fell
to exactly zero. A model that predicts more rain and yet hits none of
the heavy rain has not learned to predict more rain -- it has learned to
predict rain of a moderate amount in more places.

This script measures what the predictions actually look like: how much
water they carry, how much of it is concentrated, and whether the heavy
pixels are reached at all. It also reports the probability mass the
hierarchical head puts on each rain bin, because the head is where the
answer has to be.

A checkpoint whose architecture does not match the current code is
skipped with a note rather than crashing the whole run: the soft-CSI
baseline was trained before the head and the translator changed, and
comparing it against the scan needs that stated rather than hidden.
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


def find_checkpoint(name: str) -> tuple[str, Path] | None:
    for folder in ("hqma_scan2", "hqma"):
        base = RUNS / folder / name
        if not base.exists():
            continue
        for sub in sorted(base.iterdir()):
            ckpt = sub / "last.pt"
            if ckpt.exists():
                return str(ckpt), sub / "config.json"
    return None


def read_report(name: str) -> dict | None:
    for folder in ("hqma_scan2", "hqma"):
        path = RUNS / folder / f"{name}.eval.json"
        if path.exists():
            return json.loads(path.read_text())
    return None


def load_model(name: str, device: str = "cuda"):
    """Build the model the checkpoint was trained with, or explain why not."""
    found = find_checkpoint(name)
    if found is None:
        return None, None, f"no checkpoint for {name}"
    ckpt_path, cfg_path = found

    head_bins = 6
    if cfg_path.exists():
        cfg = json.loads(cfg_path.read_text())
        head_bins = cfg.get("head_bins", 6)

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
        first = str(exc).splitlines()[0]
        return None, None, f"architecture mismatch ({first})"
    return model.eval().to(device), head_bins, None


def main() -> None:
    dataset = RadarNWPDataset(
        DATA, packed=True, split="test", in_len=11, out_len=10,
        patch_size=64, slide_stride=64,
    )
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=8, shuffle=False, num_workers=4,
    )

    runs = [
        ("simvp_v2", "骨干 (无 NWP)"),
        ("hqma_balanced", "soft-CSI (旧架构)"),
        ("scan_B3", "B3 逐像素"),
        ("scan_B2", "B2 全场比值"),
        ("scan_B5", "B5 5档对照"),
    ]

    print("=== 预测场的分布（test 划分前 12 个 batch）===")
    print(f"{'run':22s}{'pred_max':>10s}{'p99.9':>9s}{'p99':>8s}"
          f"{'mean':>9s}{'>0.5%':>8s}{'>2%':>8s}{'>4%':>8s}{'truth_max':>11s}")
    print("-" * 98)

    for name, label in runs:
        model, head_bins, problem = load_model(name)
        if problem:
            print(f"{label:22s}  skipped: {problem}")
            continue
        maxima: list[float] = []
        truth_max: list[float] = []
        all_pred: list[np.ndarray] = []
        with torch.no_grad():
            for i, batch in enumerate(loader):
                if i >= 12:
                    break
                x_radar = batch["x_radar"].cuda()
                x_nwp = batch["x_nwp"].cuda()
                y = batch["y_rain"].cuda()
                pred = model(x_radar, x_nwp).clamp_min(0.0)
                maxima.append(float(pred.max()))
                truth_max.append(float(y.max()))
                all_pred.append(pred[0, :, 0].flatten().cpu().numpy())
        flat = np.concatenate(all_pred)
        p = np.percentile(flat, [99, 99.9, 100])
        print(f"{label:22s}{p[2]:10.3f}{p[1]:9.3f}{p[0]:8.3f}"
              f"{flat.mean():9.4f}"
              f"{100 * (flat > 0.5).mean():8.3f}"
              f"{100 * (flat > 2.0).mean():8.3f}"
              f"{100 * (flat > 4.0).mean():8.3f}"
              f"{max(truth_max):11.3f}")
        del model
        torch.cuda.empty_cache()

    print()
    print("=== 分级头的概率质量（模型自己认为各档有多少）===")
    print(f"{'run':22s}{'bin0(干)':>11s}{'bin1':>9s}{'bin2':>9s}"
          f"{'bin3':>9s}{'bin4':>9s}{'bin5':>9s}{'P(>=2mm)':>11s}")
    print("-" * 90)
    for name, label in runs:
        model, head_bins, problem = load_model(name)
        if problem or not hasattr(model, "quantum_povm_head"):
            if problem:
                print(f"{label:22s}  skipped: {problem}")
            else:
                print(f"{label:22s}  no hierarchical head")
            continue
        head = model.quantum_povm_head
        n_bins = getattr(head, "num_bins", 6)
        totals = np.zeros(n_bins)
        count = 0
        with torch.no_grad():
            for i, batch in enumerate(loader):
                if i >= 6:
                    break
                pred = model(batch["x_radar"].cuda(), batch["x_nwp"].cuda())
                cum = head.last_cum_probs
                if cum is None:
                    continue
                # cum is (B, T, 5, H, W); the tail sums give the bin masses
                # back: bin k = cum[k] - cum[k+1], with cum[5] = 0.
                c = cum.reshape(-1, cum.shape[2], *cum.shape[3:])
                tail = torch.cat([c[:, 1:], torch.zeros_like(c[:, :1])], dim=1)
                masses = c - tail
                totals += masses.mean(dim=(0, 2, 3)).cpu().numpy()
                count += 1
        if count:
            masses = totals / count
            p_ge2 = float(masses[4:].sum())
            cells = "".join(f"{v:9.4f}" for v in masses)
            print(f"{label:22s}{masses[0]:11.4f}{cells[9:]:>9s}"
                  f"{cells[18:]:>9s}{cells[27:]:>9s}{cells[36:]:>9s}"
                  f"{p_ge2:11.4f}")
        del model
        torch.cuda.empty_cache()

    print()
    for thr in ("2.0", "0.5"):
        print(f"=== 命中分解 @{thr}mm ===")
        print(f"{'run':22s}{'hits':>12s}{'FA':>12s}{'misses':>12s}"
              f"{'CSI':>9s}{'POD':>9s}{'FAR':>9s}{'BIAS':>9s}")
        for name, label in runs:
            r = read_report(name)
            if r is None:
                continue
            a = r["thresholds"].get(thr, {})
            print(f"{label:22s}{int(a.get('hits', 0)):12d}"
                  f"{int(a.get('false_alarms', 0)):12d}"
                  f"{int(a.get('misses', 0)):12d}"
                  f"{a.get('csi', 0):9.3f}{a.get('pod', 0):9.3f}"
                  f"{a.get('far', 0):9.3f}{a.get('bias', 0):9.3f}")
        print()


if __name__ == "__main__":
    main()
