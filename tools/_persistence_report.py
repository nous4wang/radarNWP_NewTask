"""Produce the persistence reference as a report the table generator reads.

The persistence forecast was measured once, from a log. The table
generator reads reports, so the reference needs to be a report too --
otherwise it sits in a log while the models sit in the table, and the
reader compares a score against a number they cannot find.

The forecast is the last observed reflectivity, held still for every
lead time and converted with the same Z-R relation the analytical
baseline uses. It needs no training and no checkpoint: it is what a
forecaster gets by doing nothing clever.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, "/home/wf/radarNWP")

import numpy as np
import torch

from radar_nwp_bench.data import RadarNWPDataset
from radar_nwp_bench.data.alignment import NWPAlignment
from radar_nwp_bench.metrics.precipitation import PrecipitationVerifier
from radar_nwp_bench.metrics.stitching import PatchStitcher

DATA = "/home/wf/data/radarNWP/BJ_mmap"
OUT = Path("/home/wf/radarNWP/runs/reference")
THRESHOLDS = (0.1, 0.5, 2.0, 4.0)


def zr_to_rain(dbz: torch.Tensor) -> torch.Tensor:
    """The same Z-R relation the analytical baseline uses.

    Z = 200 * R^1.6 gives the rain rate in mm/h; the target is a
    6-minute accumulation, so the rate is scaled by 0.1.
    """
    linear_z = torch.pow(10.0, dbz.clamp_min(0.0) / 10.0)
    rain_rate = torch.pow(linear_z / 200.0, 1.0 / 1.6)
    return rain_rate * 0.1


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)

    dataset = RadarNWPDataset(
        DATA, packed=True, split="test", in_len=11, out_len=10,
        patch_size=64, random_crop=False, slide_stride=64,
        alignment=NWPAlignment("forecast"),
    )
    grid_h, grid_w = dataset.grid_shape
    cr_index = list(dataset.radar_vars).index("CR")

    from radar_nwp_bench.data.dataset import build_dataloader

    loader = build_dataloader(
        dataset, batch_size=16, shuffle=False, num_workers=6,
        drop_last=False, seed=0,
    )

    verifier = PrecipitationVerifier(thresholds=THRESHOLDS)
    stitcher = PatchStitcher(
        (grid_h, grid_w), n_frames=10, n_positions=dataset.n_patch_positions,
    )

    with torch.no_grad():
        for batch in loader:
            x_radar = batch["x_radar"]
            truth = batch["y_rain"]
            last_dbz = torch.expm1(
                x_radar[:, -1, cr_index:cr_index + 1].clamp_min(0.0) * 6.0
            ).clamp(0.0, 75.0)
            rain = zr_to_rain(last_dbz)
            pred = rain.unsqueeze(1).expand(-1, truth.shape[1], -1, -1, -1)

            pred_np = pred.cpu().numpy()
            truth_np = truth.cpu().numpy()
            sample_idx = batch["case_index"].cpu().numpy()
            position_idx = batch["position_index"].cpu().numpy()
            origin = batch["patch_origin"].cpu().numpy()
            for b in range(pred_np.shape[0]):
                s = int(sample_idx[b])
                stitcher.add(
                    s, int(position_idx[b]),
                    (int(origin[b][0]), int(origin[b][1])),
                    pred_np[b, :, 0], truth_np[b, :, 0],
                )
                if stitcher.is_complete(s):
                    sp, st = stitcher.finalize(s)
                    verifier.update(sp[None], st[None])
                    stitcher.release(s)

    report = verifier.report()
    payload = {
        "n_samples": report["n_samples"],
        "thresholds": report["thresholds"],
        "lead_scores": report.get("lead_scores", {}),
        "ablation": {
            "mode": "reference",
            "description": (
                "persistence: the last observed reflectivity, held still for "
                "every lead time and converted with the same Z-R relation the "
                "analytical baseline uses"
            ),
            "radar_vars": list(dataset.radar_vars),
        },
    }
    path = OUT / "persistence.eval.json"
    path.write_text(json.dumps(payload, indent=2))

    print(f"PERSISTENCE REFERENCE  (split=test, n={report['n_samples']})")
    for thr in THRESHOLDS:
        t = report["thresholds"][str(thr)]
        print(f"  {thr:>5} mm : CSI {t['csi']:.4f}  POD {t['pod']:.4f}  "
              f"FAR {t['far']:.4f}")
    print(f"\nwrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
