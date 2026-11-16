"""Measure what advection can do on its own, before any network is trained.

The question this answers is narrow and it is the one the whole third tier
depends on: if the rainfall field were known perfectly at the start of the
forecast and the only thing available afterwards were a physical motion
field, how far would that carry the forecast? The answer is not the
theoretical ceiling of advection -- it is the ceiling of *this* flow
estimate, propagated from the last two observed frames.

Three arms are needed to make the number mean anything:

``zero``    the rainfall field is perfect and nothing moves. This is the
             persistence baseline expressed in the same units, so it is the
             zero point of the comparison.
``one``     the flow is used for the first frame only and the forecast then
             copies the last frame. This separates "is the flow estimate
             any good" from "does the flow survive an hour".
``flow``    the flow is propagated over the ten frames. This is the arm of
             interest, run at four truncation levels.

The flow is a first-order optical flow between the last two observed
frames. Its truncation is a physical statement, not a numerical one: the
limit is in pixels per frame, and a frame is six minutes, so a truncation
of two pixels per frame is twenty kilometres per hour. The original plan
proposed six, which is three hundred and sixty kilometres per hour and
therefore clips nothing.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

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
from radar_nwp_bench.models.common import floor_at_zero, make_warp_grid  # noqa: E402

TIERS = ("0.1", "0.5", "2.0", "4.0")
#: The persistence floor, measured by the same verifier and the same
#: stitching on this very split.
FLOOR = 0.0924


def zr_to_rain(dbz: torch.Tensor) -> torch.Tensor:
    """The same Z-R relation the analytical persistence baseline uses.

    ``Z = 200 R^1.6`` gives the rain rate in millimetres per hour; the
    target is a six-minute accumulation, so the rate is scaled by 0.1.
    """
    linear_z = torch.pow(10.0, dbz.clamp_min(0.0) / 10.0)
    rain_rate = torch.pow(linear_z / 200.0, 1.0 / 1.6)
    return rain_rate * 0.1


def compute_rain_flow(
    y_prev: torch.Tensor, y_curr: torch.Tensor
) -> torch.Tensor:
    """First-order optical flow between two rainfall fields.

    Args:
        y_prev: ``(B, 1, H, W)`` rainfall in millimetres, one frame earlier.
        y_curr: ``(B, 1, H, W)`` rainfall in millimetres, the later frame.

    Returns:
        ``(B, 2, H, W)`` displacement in pixels, ordered ``(dy, dx)``.
    """
    dt = y_curr - y_prev
    # Replicate padding: the roll would otherwise wrap the far edge to the
    # near one and manufacture a gradient where there is none.
    cur = F.pad(y_curr, (1, 1, 1, 1), mode="replicate")
    dy = (cur[:, :, 2:, 1:-1] - cur[:, :, :-2, 1:-1]) / 2.0
    dx = (cur[:, :, 1:-1, 2:] - cur[:, :, 1:-1, :-2]) / 2.0
    denom = dx**2 + dy**2 + 1e-4
    u = -(dt * dx) / denom
    v = -(dt * dy) / denom
    return torch.cat([v, u], dim=1)


def advect(
    r0: torch.Tensor, flow: torch.Tensor, steps: int
) -> torch.Tensor:
    """Propagate the initial field along the flow for ``steps`` frames.

    The whole displacement is applied in one resampling, which is the
    semi-Lagrangian form: the field is sampled at the position it would
    have been carried to, not marched there in small increments.
    """
    grid = make_warp_grid(flow * float(steps), (r0.shape[-2], r0.shape[-1]))
    warped = F.grid_sample(
        r0, grid, mode="bilinear", padding_mode="border", align_corners=True
    )
    return floor_at_zero(warped)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True)
    ap.add_argument(
        "--packed",
        action="store_true",
        default=True,
        help="the archive is in packed form; kept for symmetry with the "
        "other entry points, which take the same flag",
    )
    ap.add_argument("--split", default="test")
    ap.add_argument("--patch-size", type=int, default=64)
    ap.add_argument("--slide-stride", type=int, default=64)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--num-workers", type=int, default=6)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--clips", type=float, nargs="*", default=[1.0, 2.0, 4.0, 6.0])
    ap.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="evaluate only the first few cases; for a smoke test. The full "
        "split is needed for a score that can be quoted",
    )
    ap.add_argument("--out-json", default=None)
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0)

    dataset = RadarNWPDataset(
        args.data_root,
        packed=args.packed,
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
    # One stitcher and one verifier per arm: the arms are scored on the
    # same windows, but each has its own forecast.
    #
    # The first arm is the anchor. It reproduces the persistence baseline
    # exactly -- the last reflectivity held still and converted with Z-R --
    # so if it comes back near the floor of 0.0924 then the dataset, the
    # stitching and the verifier are being driven the way the rest of the
    # benchmark drives them, and the other arms can be read as they stand.
    arm_names = ["zr"] + ["ground_zero", "ground_one"] + [
        f"ground_flow{c:g}" for c in args.clips
    ]
    arms: dict[str, dict] = {}
    for name in arm_names:
        arms[name] = {
            "stitcher": PatchStitcher(
                (grid_h, grid_w),
                n_frames=10,
                n_positions=dataset.n_patch_positions,
            ),
            "verifier": PrecipitationVerifier(thresholds=(0.1, 0.5, 2.0, 4.0)),
            "mass_ratio": [],
            "mass_n": 0,
        }

    with torch.no_grad():
        for batch in loader:
            x_radar = batch["x_radar"].to(device)
            y_hist = batch["y_hist"].to(device)  # (B, 11, 1, H, W) in mm
            y_true = batch["y_rain"].to(device)  # (B, 10, 1, H, W) in mm
            r0 = y_hist[:, -1]
            flow_raw = compute_rain_flow(y_hist[:, -2], y_hist[:, -1])

            # The anchor: the last reflectivity, held still, in Z-R rain.
            last_dbz = torch.expm1(
                x_radar[:, -1, cr_index:cr_index + 1].clamp_min(0.0) * 6.0
            ).clamp(0.0, 75.0)
            zr_rain = zr_to_rain(last_dbz)

            case_idx = batch["case_index"].cpu().numpy()
            pos_idx = batch["position_index"].cpu().numpy()
            origin = batch["patch_origin"].cpu().numpy()

            for name, arm in arms.items():
                if name == "zr":
                    pred = zr_rain.unsqueeze(1).expand(
                        -1, y_true.shape[1], -1, -1, -1
                    )
                else:
                    if name == "ground_zero":
                        flow = torch.zeros_like(flow_raw)
                    elif name == "ground_one":
                        flow = flow_raw.clamp(-2.0, 2.0)
                    else:
                        clip = float(name.replace("ground_flow", ""))
                        flow = flow_raw.clamp(-clip, clip)

                    preds = []
                    for step in range(1, 11):
                        if name == "ground_one" and step > 1:
                            # the flow is used for the first frame only
                            warped = r0
                        else:
                            warped = advect(r0, flow, step)
                        preds.append(warped)
                    pred = torch.stack(preds, dim=1)

                # The mass check: bilinear resampling is not conservative, so
                # the total rainfall after the warp is recorded against the
                # arm's own starting field. A ratio far from one means the
                # "spectrum preserving" claim is not earned, and the number
                # has to say so.
                source = zr_rain if name == "zr" else r0
                before = float(source.sum().item())
                after = float(pred[:, 0].sum().item())
                if before > 1e-6:
                    arm["mass_ratio"].append(after / before)
                    arm["mass_n"] += 1
                # Separately: how much rain the Z-R inversion claims against
                # what the gauge network recorded. This is not a skill score
                # but it is the reason the two arms differ at all.
                if name == "zr":
                    g = float(r0.sum().item())
                    if g > 1e-6:
                        arm.setdefault("zr_over_ground", []).append(
                            before / g
                        )

                pred_np = pred.cpu().numpy()
                true_np = y_true.cpu().numpy()
                for b in range(pred_np.shape[0]):
                    s_i, p_i = int(case_idx[b]), int(pos_idx[b])
                    orig = (int(origin[b][0]), int(origin[b][1]))
                    arm["stitcher"].add(
                        s_i, p_i, orig, pred_np[b, :, 0], true_np[b, :, 0]
                    )
                    if arm["stitcher"].is_complete(s_i):
                        sp, st = arm["stitcher"].finalize(s_i)
                        arm["verifier"].update(sp[None], st[None])
                        arm["stitcher"].release(s_i)

    out: dict = {"n_samples": dataset.n_samples if hasattr(dataset, "n_samples") else None,
                 "grid": [grid_h, grid_w], "arms": {}}
    print("=" * 96)
    print(f"the advection ceiling on the {args.split} split, perfect rainfall at the start")
    print("=" * 96)
    print()
    print(
        f"  {'arm':<18} {'CSI@0.1':>8} {'CSI@0.5':>8} {'CSI@2.0':>8} {'CSI@4.0':>8}"
        f" {'FAR@2.0':>8} {'BIAS@0.5':>9} {'mass':>7}"
    )
    for name, arm in arms.items():
        rep = arm["verifier"].report()["thresholds"]
        row = [rep[t]["csi"] for t in TIERS]
        mass = (
            float(np.mean(arm["mass_ratio"]))
            if arm["mass_ratio"]
            else float("nan")
        )
        print(
            f"  {name:<18} {row[0]:>8.4f} {row[1]:>8.4f} {row[2]:>8.4f} {row[3]:>8.4f}"
            f" {rep['2.0']['far']:>8.3f} {rep['0.5']['bias']:>9.3f} {mass:>7.3f}"
        )
        out["arms"][name] = {
            "thresholds": {t: rep[t] for t in TIERS},
            "mass_ratio_mean": mass,
            "lead_csi_2.0": arm["verifier"].report()["lead_csi"],
        }

    print()
    print(f"  the persistence floor is CSI@2.0 = {FLOOR:.4f}")
    for name, arm in arms.items():
        c = arm["verifier"].report()["thresholds"]["2.0"]["csi"]
        print(f"    {name:<18} {c:.4f}   difference from the floor {c - FLOOR:+.4f}")
    print()

    print("  where the flow stops working: CSI@2.0 per lead frame")
    ref = "ground_flow2g" if "ground_flow2g" in arms else arm_names[-1]
    leads = sorted(out["arms"][ref]["lead_csi_2.0"], key=int)
    header = "  lead    " + "".join(f"{name:>17}" for name in arm_names)
    print(header)
    for lead in leads:
        row = "".join(
            f"{out['arms'][name]['lead_csi_2.0'][lead]['2.0']:>17.4f}"
            for name in arm_names
        )
        print(f"  {lead:<7} {row}")
    print()

    print()
    zr_arm = arms["zr"]
    if zr_arm.get("zr_over_ground"):
        ratio = float(np.mean(zr_arm["zr_over_ground"]))
        print(
            f"  the Z-R inversion claims {ratio:.3f} times the rainfall the "
            f"gauge network recorded, averaged over the patches"
        )
        out["zr_over_ground"] = ratio
        print()
    print("  the truncation sweep, CSI@2.0 (a limit of two pixels per frame")
    print("  is twenty kilometres per hour, since a frame is six minutes)")
    for name in arm_names:
        if not name.startswith("ground_flow"):
            continue
        c = out["arms"][name]["thresholds"]["2.0"]["csi"]
        clip = float(name.replace("ground_flow", ""))
        print(f"    {clip:>4.1f} pixels per frame   CSI@2.0 = {c:.4f}")
    print()

    if args.out_json:
        Path(args.out_json).write_text(
            json.dumps(out, indent=2, default=float), encoding="utf-8"
        )
        print(f"  report written to {args.out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
