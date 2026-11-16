"""Check the fix on a checkpoint that already exists.

The scan that will confirm the fix takes half an hour. This checks the
same thing on the checkpoints the previous scan left behind, without
retraining: it loads a checkpoint, rebuilds the head with the midpoint
construction, and asks whether the reconstruction can cross a
threshold at all.

The checkpoints are not trained under the new construction, so the
numbers are not scores -- they are a check that the reconstruction is
capable of crossing, which the old one demonstrably was not.
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


def load(name: str, level_mode: str):
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

    cfg = json.loads(cfg_path.read_text()) if cfg_path.exists() else {}
    head_bins = cfg.get("head_bins", 6)

    weights = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    weights = {k: v for k, v in weights.items() if isinstance(v, torch.Tensor)}

    model = build_model(
        "simvp_v2_hqma", in_radar_c=10, in_nwp_c=27, in_len=11, out_len=10,
        head_bins=head_bins, level_mode=level_mode,
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

    thresholds = [0.5, 1.0, 2.0, 4.0]
    print("Does the reconstruction cross the thresholds?")
    print("(The checkpoints were trained under the old construction; these")
    print(" are not scores, they are a check on the reconstruction.)\n")

    for name in ("scan_B2", "scan_B5"):
        for mode in ("floor", "midpoint"):
            model, problem = load(name, mode)
            if problem:
                print(f"{name} / {mode}: skipped ({problem})")
                continue
            chunks = []
            with torch.no_grad():
                for i, batch in enumerate(loader):
                    if i >= 10:
                        break
                    pred = model(batch["x_radar"].cuda(), batch["x_nwp"].cuda())
                    chunks.append(pred[:, :, 0].flatten().cpu().numpy())
            flat = np.concatenate(chunks)
            n = flat.size
            print(f"--- {name} / {mode} ---")
            print(f"    max prediction: {flat.max():.6f} mm")
            for thr in thresholds:
                above = 100 * (flat >= thr).mean()
                below = 100 * ((flat >= thr - 0.01) & (flat < thr)).mean()
                print(f"    {thr:4.1f} mm: crossing {above:8.4f} %   "
                      f"just below {below:8.4f} %")
            del model
            torch.cuda.empty_cache()
        print()


if __name__ == "__main__":
    main()
