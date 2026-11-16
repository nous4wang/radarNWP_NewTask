#!/usr/bin/env python3
"""Grid-scan the multiplicative Z-R factor k against CSI (zero-compute rescoring).

§4.17 showed that a *pixelwise-MSE-optimal* k (0.1386) destroys the
persistence score (CSI@2.0 0.0924 -> 0.0009).  That does not answer the
question actually asked: when CSI itself is the objective, what is the best
k, and is k=1.0 (the plain Marshall-Palmer relation) already at the peak?

This script does no fitting at all.  It rescans the analytical persistence
forecast on a full-grid Hann-stitched split for a small grid of k values,
using exactly the same dataset, stitcher, verifier and thresholds as
``tools/_oracle_advection_ceiling.py`` and ``tools/_persistence_reference.py``.

All k arms are scored inside ONE loader pass: each arm owns its own
PatchStitcher and PrecipitationVerifier, so the data is read once and the
arms differ only by the scalar multiplier.  That keeps this a genuinely
zero-compute diagnostic rather than six separate evaluations.

The k=1.0 arm is the anchor: it must reproduce the persistence floor
CSI@2.0 = 0.0924.  If it does not, the whole table is void (§4.18.2 S1-A).

Usage (on the remote, from the repository root)::

    PYTHONPATH=/home/wf/radarNWP python tools/_scan_zr_k.py \
        --data-root /home/wf/data/radarNWP/BJ_mmap --split test \
        --ks 0.5 0.75 1.0 1.25 1.5 2.0 \
        --out-json runs/scan_zr_k_test.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

# ``python tools/xxx.py`` puts tools/ on sys.path[0], not the project root.
# The other analytical tools do the same insertion; keep it here so the
# script works regardless of how it is launched.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from radar_nwp_bench.data import (  # noqa: E402
    NWPAlignment,
    RadarNWPDataset,
    build_dataloader,
)
from radar_nwp_bench.metrics import (  # noqa: E402
    PatchStitcher,
    PrecipitationVerifier,
)

TIERS = ("0.1", "0.5", "2.0", "4.0")
FLOOR = 0.0924


def zr_to_rain(dbz: torch.Tensor) -> torch.Tensor:
    """The analytical persistence baseline's Z-R relation, in mm/6min.

    ``Z = 200 R^1.6`` is an hourly rate; one frame is a six-minute
    accumulation, hence the factor 0.1.  Identical to the definition in
    ``_oracle_advection_ceiling.py``.
    """
    linear_z = torch.pow(10.0, dbz.clamp_min(0.0) / 10.0)
    rain_rate = torch.pow(linear_z / 200.0, 1.0 / 1.6)
    return rain_rate * 0.1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--ks", nargs="*", type=float,
                    default=[0.5, 0.75, 1.0, 1.25, 1.5, 2.0])
    ap.add_argument("--patch-size", type=int, default=64)
    ap.add_argument("--slide-stride", type=int, default=64)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--num-workers", type=int, default=6)
    ap.add_argument("--max-samples", type=int, default=None)
    ap.add_argument("--out-json", default=None)
    ap.add_argument(
        "--per-window-k", type=float, default=None,
        help="also collect per-window contingency tables for this one k "
             "and write them with --per-window-out. Used to bootstrap a "
             "paired comparison against another arm on the same windows.",
    )
    ap.add_argument("--per-window-out", default=None)
    args = ap.parse_args()

    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = RadarNWPDataset(
        args.data_root,
        packed=True,
        in_len=11,
        out_len=10,
        patch_size=args.patch_size,
        alignment=NWPAlignment.FORECAST,
        split=args.split,
        slide_stride=args.slide_stride,
        random_crop=False,
        tolerate_one_missing_frame=True,
        max_samples=args.max_samples,
    )
    loader = build_dataloader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        drop_last=False,
        seed=0,
    )

    grid_h, grid_w = dataset.grid_shape
    cr_index = list(dataset.radar_vars).index("CR")

    arms: dict[float, dict] = {}
    for k in args.ks:
        arms[k] = {
            "stitcher": PatchStitcher(
                (grid_h, grid_w),
                n_frames=10,
                n_positions=dataset.n_patch_positions,
            ),
            "verifier": PrecipitationVerifier(
                thresholds=(0.1, 0.5, 2.0, 4.0),
                per_window=(args.per_window_k is not None
                            and abs(k - args.per_window_k) < 1e-9),
            ),
        }

    with torch.no_grad():
        for batch in loader:
            x_radar = batch["x_radar"].to(device)
            y_true = batch["y_rain"].to(device)
            case_idx = batch["case_index"].cpu().numpy()
            pos_idx = batch["position_index"].cpu().numpy()
            origin = batch["patch_origin"].cpu().numpy()

            # Last observed reflectivity -> dBZ -> Z-R rain.  The 75 dBZ
            # ceiling is part of the persistence reference that produced
            # the 0.0924 floor; it must be kept here too.
            last_dbz = torch.expm1(
                x_radar[:, -1, cr_index:cr_index + 1].clamp_min(0.0) * 6.0
            ).clamp(0.0, 75.0)
            zr_rain = zr_to_rain(last_dbz)  # (B, 1, H, W), mm/6min

            true_np = y_true.cpu().numpy()
            for k, arm in arms.items():
                scaled = zr_rain * float(k)
                pred = scaled.unsqueeze(1).expand(-1, y_true.shape[1], -1, -1, -1)
                pred_np = pred.cpu().numpy()
                for b in range(pred_np.shape[0]):
                    s_i = int(case_idx[b])
                    orig = (int(origin[b][0]), int(origin[b][1]))
                    arm["stitcher"].add(
                        s_i, int(pos_idx[b]), orig,
                        pred_np[b, :, 0], true_np[b, :, 0],
                    )
                    if arm["stitcher"].is_complete(s_i):
                        sp, st = arm["stitcher"].finalize(s_i)
                        arm["verifier"].update(sp[None], st[None])
                        arm["stitcher"].release(s_i)

    reports: dict[str, dict] = {}
    for k, arm in arms.items():
        rep = arm["verifier"].report()
        reports[f"{k:g}"] = {
            "thresholds": {t: rep["thresholds"][t] for t in TIERS},
            "n_samples": rep.get("n_samples"),
        }

    print("=" * 88)
    print(f"Z-R k grid scan | split={args.split} | n={reports[next(iter(reports))]['n_samples']}")
    print("=" * 88)
    print(f"{'k':>7} | {'CSI@0.1':>9} {'CSI@0.5':>9} {'CSI@2.0':>9} "
          f"{'CSI@4.0':>9} | {'BIAS@0.5':>9} {'FAR@2.0':>9} {'POD@2.0':>9}")
    print("-" * 88)
    for key, rep in reports.items():
        t = rep["thresholds"]
        print(f"{key:>7} | {t['0.1']['csi']:>9.4f} {t['0.5']['csi']:>9.4f} "
              f"{t['2.0']['csi']:>9.4f} {t['4.0']['csi']:>9.4f} | "
              f"{t['0.5']['bias']:>9.4f} {t['2.0']['far']:>9.4f} "
              f"{t['2.0']['pod']:>9.4f}")
    print("-" * 88)

    csi2 = {float(k): v["thresholds"]["2.0"]["csi"] for k, v in reports.items()}
    best_k = max(csi2, key=csi2.get)
    print(f"argmax_k(CSI@2.0) = {best_k:g}  -> {csi2[best_k]:.4f}")
    print(f"persistence floor = {FLOOR:.4f} | Oracle ceiling = 0.1653")
    if 1.0 in csi2:
        anchor = csi2[1.0]
        ok = abs(anchor - FLOOR) <= 0.002
        print(f"ANCHOR k=1.0: CSI@2.0 = {anchor:.4f} "
              f"({'PASS' if ok else 'FAIL -- table void per S1-A'}, "
              f"floor {FLOOR:.4f}, delta {anchor - FLOOR:+.4f})")
        print(f"CSI@2.0(1.0) / max = {anchor / max(csi2.values()):.4f} "
              f"(S1-B needs >= 0.95)")
    print("=" * 88)

    if args.out_json:
        out = {
            "split": args.split,
            "ks": args.ks,
            "floor": FLOOR,
            "argmax_k_csi2": best_k,
            "max_csi2": csi2[best_k],
            "reports": reports,
        }
        Path(args.out_json).write_text(
            json.dumps(out, indent=2, default=float), encoding="utf-8"
        )
        print(f"report written to {args.out_json}")
    if args.per_window_out:
        if args.per_window_k is None:
            raise SystemExit("--per-window-out needs --per-window-k")
        arm = arms.get(args.per_window_k)
        if arm is None:
            raise SystemExit(
                f"--per-window-k {args.per_window_k:g} is not in --ks"
            )
        table = arm["verifier"].per_window_table()
        # Match scripts/evaluate.py --per-window-out exactly, so a paired
        # bootstrap can load both arms with the same reader.
        Path(args.per_window_out).write_text(
            json.dumps(
                {
                    "n_windows": len(table),
                    "thresholds": [f"{t}" for t in
                                   arm["verifier"].thresholds],
                    "windows": table,
                },
                indent=2, default=float,
            ),
            encoding="utf-8",
        )
        print(f"per-window table written to {args.per_window_out} "
              f"({len(table)} windows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
