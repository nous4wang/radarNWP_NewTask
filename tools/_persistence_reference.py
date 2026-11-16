"""A persistence reference for the analytical baseline.

The physics advection baseline now scores higher than every learned model at
the heavy thresholds.  Before that goes in the paper it has to be shown that
the advection is doing real work, rather than the baseline simply being a
better-calibrated forecast of "the rain stays roughly where it is".

This computes the persistence forecast -- the last observed reflectivity,
held still, converted with the same Z-R relation -- and scores it through
exactly the same path as evaluate.py: the same patches, the same stitching
back into the 301x301 field, the same verifier.

    python tools/_persistence_reference.py --split test
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from radar_nwp_bench.data import NWPAlignment, RadarNWPDataset, build_dataloader
from radar_nwp_bench.metrics import PatchStitcher, PrecipitationVerifier

DATA_ROOT = "/home/wf/data/radarNWP/BJ_mmap"
THRESHOLDS = (0.1, 0.5, 1.0, 2.0, 4.0)


def zr_to_rain(dbz: torch.Tensor) -> torch.Tensor:
    """The same Z-R relation the analytical baseline uses.

    Z = 200 * R^1.6 gives the rain rate in mm/h; the target is a 6-minute
    accumulation, so the rate is scaled by 0.1.
    """
    linear_z = torch.pow(10.0, dbz.clamp_min(0.0) / 10.0)
    rain_rate = torch.pow(linear_z / 200.0, 1.0 / 1.6)
    return rain_rate * 0.1


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="test")
    parser.add_argument("--patch-size", type=int, default=64)
    parser.add_argument("--slide-stride", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=6)
    args = parser.parse_args()

    dataset = RadarNWPDataset(
        DATA_ROOT, packed=True, in_len=11, out_len=10,
        patch_size=args.patch_size, random_crop=False, split=args.split,
        alignment=NWPAlignment("forecast"), slide_stride=args.slide_stride,
    )
    grid_h, grid_w = dataset.grid_shape
    cr_index = list(dataset.radar_vars).index("CR")
    print(f"split={args.split} windows={len(dataset)} "
          f"grid={grid_h}x{grid_w} CR index={cr_index}")

    loader = build_dataloader(
        dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, drop_last=False, seed=0,
    )

    verifier = PrecipitationVerifier(thresholds=THRESHOLDS)
    stitcher = PatchStitcher(
        (grid_h, grid_w), n_frames=10, n_positions=dataset.n_patch_positions,
    )

    with torch.no_grad():
        for batch in loader:
            x_radar = batch["x_radar"]
            truth = batch["y_rain"]
            # The persistence forecast: the last observed reflectivity, held
            # still for every lead time, converted with Z-R.
            last_dbz = torch.expm1(x_radar[:, -1, cr_index:cr_index + 1]
                                   .clamp_min(0.0) * 6.0).clamp(0.0, 75.0)
            rain = zr_to_rain(last_dbz)                      # (B, 1, H, W)
            # repeat, not expand: expand cannot widen a non-singleton axis.
            pred = rain.unsqueeze(1).expand(-1, truth.shape[1], -1, -1, -1)

            pred_np = pred.cpu().numpy()
            truth_np = truth.cpu().numpy()
            sample_idx = batch["case_index"].cpu().numpy()
            position_idx = batch["position_index"].cpu().numpy()
            origin = batch["patch_origin"].cpu().numpy()
            for b in range(pred_np.shape[0]):
                s = int(sample_idx[b])
                stitcher.add(s, int(position_idx[b]),
                             (int(origin[b][0]), int(origin[b][1])),
                             pred_np[b, :, 0], truth_np[b, :, 0])
                if stitcher.is_complete(s):
                    sp, st = stitcher.finalize(s)
                    verifier.update(sp[None], st[None])
                    stitcher.release(s)

    report = verifier.report()
    print(f"\nPERSISTENCE REFERENCE (split={args.split}, "
          f"n={report['n_samples']})")
    print(f"{'threshold':>10}{'CSI':>10}{'POD':>10}{'FAR':>10}")
    for thr in THRESHOLDS:
        t = report["thresholds"][str(thr)]
        far = t["far"]
        print(f"{thr:>10}{t['csi']:>10.4f}{t['pod']:>10.4f}"
              f"{(far if far == far else float('nan')):>10.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
