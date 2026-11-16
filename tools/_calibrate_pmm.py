#!/usr/bin/env python3
"""Probability-Matching-Method (PMM) rescoring of the persistence baseline.

Gate 1 positioning (MASTER section 4.20.1): this is NOT a new "lossless
extreme-value recalibration breakthrough".  ``R = F_Y^{-1}(F_Z(CR))`` is a
*pixelwise monotone map*, so it belongs to the same amplitude / distribution
calibration family as the Z-R scaling of 4.17 and the scalar k grid of 4.18.
Monotone maps cannot reorder pixels, and the same-frame CR-to-LABEL pixel
correlation is only r = +0.17 (load-bearing fact 7), so no spatial structure
can be created here.

The ONE new degree of freedom versus a single scalar k is the *nonlinear,
per-quantile shape*: PMM can lift the upper tail without amplifying the dry
bulk.  The decisive test is therefore D2-B -- a paired bootstrap against the
test-tuned scalar k=2.0 (CSI@2.0 = 0.1199) -- not the BIAS interval proposed
in the source document (which contradicts 4.17.3 and 4.18.5.4).

Seven corrections relative to the document's ``tools/_calibrate_pmm.py``:

1. ``patch_size=64`` and ``slide_stride=64`` are set EXPLICITLY on both
   splits.  The document used the dataset default of 128 with a stride of 64,
   which makes ``len(dataset)`` disagree with ``n_patch_positions`` and breaks
   the full-grid Hann stitching.
2. ``thresholds=(0.1, 0.5, 2.0, 4.0)`` is passed explicitly; the verifier's
   default is the 7-tier DEFAULT_THRESHOLDS, which changes the report shape.
3. The 75 dBZ ceiling of the analytical persistence reference is kept, so the
   k=1.0 anchor still reproduces the 0.0924 floor (S1-A / D2 anchor).
4. The CR channel index is looked up dynamically from ``dataset.radar_vars``.
5. ``shuffle=False`` + a fixed seed on BOTH loaders, and deterministic strided
   pixel subsampling for the quantile tables -- the source used shuffle=True
   with no seed.
6. The quantile table is built from ALL pixels (dry ones included), so
   ``F_Z`` and ``F_Y`` describe the same population that is later mapped.
   The source built a *conditional* table on ``(cr>=15)|(y>=0.05)`` but then
   mapped the full field into it -- a distribution mismatch.  Three variants
   are scored side by side so no single implementation choice can manufacture
   the verdict: ``pmm`` (all pixels), ``pmm_cond`` (the document's conditional
   table + echo cut), and ``pmm_rain`` (operational PMM over raining pixels
   only, the most generous variant).
7. Per-window contingency tables are written for every arm in exactly the
   shape of ``scripts/evaluate.py --per-window-out``, so ``tools/_bootstrap_
   qpe_vs_zr.py`` can run the paired D2-A / D2-B tests on the remote.
8. Tied CR values are resolved at the CDF **upper** edge (standard PMM,
   ``F_Z(z) = P(Z <= z)``), not the lower edge; the latter under-estimates
   rainfall and would fake a FAIL that belongs to the implementation.

Usage (on the remote, from the repository root)::

    PYTHONPATH=/home/wf/radarNWP python tools/_calibrate_pmm.py \
        --data-root /home/wf/data/radarNWP/BJ_mmap \
        --out-json runs/pmm_persistence.json \
        --per-window-prefix runs/pmm
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

# ``python tools/xxx.py`` puts tools/ on sys.path[0], not the project root.
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
FLOOR = 0.0924          # pre-registered analytical persistence floor (k=1.0)
K2_CSI = 0.1199         # test-tuned scalar oracle (leaky upper bound)
ORACLE = 0.1653         # perfect-QPE + no-motion ceiling
ARM_NAMES = ("pmm", "pmm_cond", "pmm_rain", "k1", "k2")


def zr_to_rain(dbz: torch.Tensor) -> torch.Tensor:
    """The analytical persistence Z-R relation, in mm/6min.

    ``Z = 200 R^1.6`` is an hourly rate; one frame is a six-minute
    accumulation, hence the factor 0.1.  Byte-for-byte the definition in
    ``tools/_scan_zr_k.py`` and ``tools/_oracle_advection_ceiling.py``.
    """
    linear_z = torch.pow(10.0, dbz.clamp_min(0.0) / 10.0)
    rain_rate = torch.pow(linear_z / 200.0, 1.0 / 1.6)
    return rain_rate * 0.1


def make_dataset(data_root: str, split: str, patch_size: int,
                 slide_stride: int, max_samples: int | None):
    return RadarNWPDataset(
        data_root,
        packed=True,
        in_len=11,
        out_len=10,
        patch_size=patch_size,
        alignment=NWPAlignment.FORECAST,
        split=split,
        slide_stride=slide_stride,
        random_crop=False,
        tolerate_one_missing_frame=True,
        max_samples=max_samples,
    )


def make_loader(dataset, batch_size: int, num_workers: int, seed: int):
    return build_dataloader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False,
        seed=seed,
    )


def monotone_quantiles(z_q: np.ndarray, y_q: np.ndarray):
    """Make the interpolation x-grid strictly increasing.

    Empirical quantiles contain long runs of identical values (the dry bulk
    is ~97 % of pixels, so most of F_Y is exactly 0).  ``np.interp`` is
    undefined for such ties.

    Standard PMM evaluates ``F_Z(z) = P(Z <= z)``, i.e. the **upper edge** of
    a run of tied CR values.  Taking the FIRST occurrence (the lower edge)
    systematically under-estimates the mapped rainfall and would manufacture
    a FAIL that belongs to the implementation, not to the method.  Use the
    last occurrence of each distinct value instead.
    """
    z_unique = np.unique(z_q)
    last_idx = np.searchsorted(z_q, z_unique, side="right") - 1
    return z_unique, y_q[last_idx]


def collect_quantiles(loader, cr_index: int, num_quantiles: int,
                      quantile_batches: int, pixels_per_batch: int,
                      echo_min_dbz: float) -> dict:
    """Build empirical quantile tables from the TRAIN split only (no leakage)."""
    z_all: list[np.ndarray] = []
    y_all: list[np.ndarray] = []
    z_cond: list[np.ndarray] = []
    y_cond: list[np.ndarray] = []
    z_rain: list[np.ndarray] = []
    y_rain: list[np.ndarray] = []

    print(">>> sampling TRAIN CR / ground-rainfall distributions ...")
    with torch.no_grad():
        for i, batch in enumerate(loader):
            if quantile_batches and i >= quantile_batches:
                break
            last_dbz = torch.expm1(
                batch["x_radar"][:, -1, cr_index].clamp_min(0.0) * 6.0
            ).clamp(0.0, 75.0).cpu().numpy()
            # y_hist[:, -1] is the t0 ground-rainfall truth (mm/6min), the
            # correct QPE target; y_rain holds the ten future frames.
            y0 = batch["y_hist"][:, -1, 0].cpu().numpy()

            cr_f = np.ascontiguousarray(last_dbz.reshape(-1))
            y_f = np.ascontiguousarray(y0.reshape(-1))
            step = max(1, cr_f.size // max(1, pixels_per_batch))
            cr_s, y_s = cr_f[::step], y_f[::step]
            z_all.append(cr_s)
            y_all.append(y_s)

            # Document variant: (echo >= 15 dBZ) OR (rain >= 0.05 mm).
            valid = (cr_s >= echo_min_dbz) | (y_s >= 0.05)
            if valid.any():
                z_cond.append(cr_s[valid])
                y_cond.append(y_s[valid])
            # Operational PMM: match quantiles over RAINING pixels only.
            wet = y_s >= 0.05
            if wet.any():
                z_rain.append(cr_s[wet])
                y_rain.append(y_s[wet])

    z_cat = np.concatenate(z_all)
    y_cat = np.concatenate(y_all)
    z_c_cat = np.concatenate(z_cond) if z_cond else z_cat
    y_c_cat = np.concatenate(y_cond) if y_cond else y_cat
    z_r_cat = np.concatenate(z_rain) if z_rain else z_cat
    y_r_cat = np.concatenate(y_rain) if y_rain else y_cat

    q_grid = np.linspace(0.0, 1.0, num_quantiles)
    z_q, y_q = np.quantile(z_cat, q_grid), np.quantile(y_cat, q_grid)
    z_qc, y_qc = np.quantile(z_c_cat, q_grid), np.quantile(y_c_cat, q_grid)
    z_qr, y_qr = np.quantile(z_r_cat, q_grid), np.quantile(y_r_cat, q_grid)

    z_full, y_full = monotone_quantiles(z_q, y_q)
    z_cond_m, y_cond_m = monotone_quantiles(z_qc, y_qc)
    z_rain_m, y_rain_m = monotone_quantiles(z_qr, y_qr)
    print(f"  pixels sampled: ALL={z_cat.size:,}  COND={z_c_cat.size:,} "
          f"RAIN={z_r_cat.size:,}")
    print(f"  F_Y 99 %% quantile (ALL)  = {y_q[int(0.99 * (num_quantiles - 1))]:.4f} mm/6min")
    print(f"  F_Z 99 %% quantile (ALL)  = {z_q[int(0.99 * (num_quantiles - 1))]:.2f} dBZ")
    print(f"  F_Y 99 %% quantile (RAIN) = {y_qr[int(0.99 * (num_quantiles - 1))]:.4f} mm/6min")
    print(f"  monotone lookup sizes: ALL={z_full.size}, COND={z_cond_m.size}, "
          f"RAIN={z_rain_m.size}")
    return {
        "full": (z_full, y_full),
        "cond": (z_cond_m, y_cond_m),
        "rain": (z_rain_m, y_rain_m),
        "n_pixels_all": int(z_cat.size),
        "n_pixels_cond": int(z_c_cat.size),
        "n_pixels_rain": int(z_r_cat.size),
        "q_grid_size": num_quantiles,
        "y_q_p50_p90_p99_p999": [
            float(y_q[int(p * (num_quantiles - 1))]) for p in (0.5, 0.9, 0.99, 0.999)
        ],
        "y_q_rain_p50_p90_p99_p999": [
            float(y_qr[int(p * (num_quantiles - 1))]) for p in (0.5, 0.9, 0.99, 0.999)
        ],
    }


def pmm_map(cr_dbz: np.ndarray, z_q: np.ndarray, y_q: np.ndarray,
            echo_min_dbz: float, masked: bool) -> np.ndarray:
    """Pixelwise quantile map.  ``masked`` reproduces the document variant."""
    r = np.interp(cr_dbz, z_q, y_q, left=0.0, right=float(y_q[-1]))
    if masked:
        r = np.where(cr_dbz < echo_min_dbz, 0.0, r)
    return np.maximum(r, 0.0)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--out-json", default="runs/pmm_persistence.json")
    ap.add_argument(
        "--per-window-prefix", default=None,
        help="write <prefix>_<arm>.windows.json for every arm, in the shape "
             "of scripts/evaluate.py --per-window-out",
    )
    ap.add_argument("--patch-size", type=int, default=64)
    ap.add_argument("--slide-stride", type=int, default=64)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--num-workers", type=int, default=6)
    ap.add_argument("--num-quantiles", type=int, default=1000)
    ap.add_argument(
        "--quantile-batches", type=int, default=0,
        help="0 = walk the whole TRAIN split (recommended)",
    )
    ap.add_argument("--pixels-per-batch", type=int, default=4096)
    ap.add_argument("--echo-min-dbz", type=float, default=15.0)
    ap.add_argument("--max-samples", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_ds = make_dataset(args.data_root, "train", args.patch_size,
                            args.slide_stride, args.max_samples)
    test_ds = make_dataset(args.data_root, "test", args.patch_size,
                           args.slide_stride, args.max_samples)
    train_loader = make_loader(train_ds, args.batch_size, args.num_workers,
                               args.seed)
    test_loader = make_loader(test_ds, args.batch_size, args.num_workers,
                              args.seed)

    grid_h, grid_w = test_ds.grid_shape
    cr_index = list(test_ds.radar_vars).index("CR")
    print(f">>> CR channel index = {cr_index} | grid = {grid_h}x{grid_w} "
          f"| n_positions = {test_ds.n_patch_positions}")

    tables = collect_quantiles(
        train_loader, cr_index, args.num_quantiles, args.quantile_batches,
        args.pixels_per_batch, args.echo_min_dbz,
    )

    arms: dict[str, dict] = {}
    for name in ARM_NAMES:
        arms[name] = {
            "stitcher": PatchStitcher(
                (grid_h, grid_w), n_frames=10,
                n_positions=test_ds.n_patch_positions,
            ),
            "verifier": PrecipitationVerifier(
                thresholds=(0.1, 0.5, 2.0, 4.0), per_window=True,
            ),
        }

    print(f">>> scoring TEST split (n={len(test_ds) // test_ds.n_patch_positions} "
          f"windows) ...")
    with torch.no_grad():
        for batch in test_loader:
            x_radar = batch["x_radar"].to(device)
            y_true = batch["y_rain"].to(device)
            case_idx = batch["case_index"].cpu().numpy()
            pos_idx = batch["position_index"].cpu().numpy()
            origin = batch["patch_origin"].cpu().numpy()

            last_dbz = torch.expm1(
                x_radar[:, -1, cr_index:cr_index + 1].clamp_min(0.0) * 6.0
            ).clamp(0.0, 75.0)
            cr_np = last_dbz[:, 0].cpu().numpy()
            zr_np = zr_to_rain(last_dbz)[:, 0].cpu().numpy()

            r0_fields = {
                "pmm": pmm_map(cr_np, *tables["full"], args.echo_min_dbz, False),
                "pmm_cond": pmm_map(cr_np, *tables["cond"],
                                    args.echo_min_dbz, True),
                "pmm_rain": pmm_map(cr_np, *tables["rain"],
                                    args.echo_min_dbz, False),
                "k1": zr_np * 1.0,
                "k2": zr_np * 2.0,
            }
            true_np = y_true.cpu().numpy()
            for name, r0 in r0_fields.items():
                # (B, H, W) -> (B, 10, 1, H, W): copy t0 with zero motion.
                pred = np.repeat(r0[:, np.newaxis, np.newaxis, ...], 10, axis=1)
                for b in range(pred.shape[0]):
                    arm = arms[name]
                    arm["stitcher"].add(
                        int(case_idx[b]), int(pos_idx[b]),
                        (int(origin[b][0]), int(origin[b][1])),
                        pred[b, :, 0], true_np[b, :, 0],
                    )
                    if arm["stitcher"].is_complete(int(case_idx[b])):
                        sp, st = arm["stitcher"].finalize(int(case_idx[b]))
                        arm["verifier"].update(sp[None], st[None])
                        arm["stitcher"].release(int(case_idx[b]))

    reports: dict[str, dict] = {}
    for name, arm in arms.items():
        rep = arm["verifier"].report()
        reports[name] = {
            "thresholds": {t: rep["thresholds"][t] for t in TIERS},
            "n_samples": rep.get("n_samples"),
        }

    print("=" * 96)
    print(f"PMM rescoring | TRAIN quantiles -> TEST full-grid Hann stitching "
          f"| n={reports['k1']['n_samples']}")
    print("=" * 96)
    print(f"{'arm':>10} | {'CSI@0.1':>9} {'CSI@0.5':>9} {'CSI@2.0':>9} "
          f"{'CSI@4.0':>9} | {'BIAS@2.0':>10} {'POD@2.0':>9} {'FAR@2.0':>9}")
    print("-" * 96)
    for name, rep in reports.items():
        t = rep["thresholds"]
        print(f"{name:>10} | {t['0.1']['csi']:>9.4f} {t['0.5']['csi']:>9.4f} "
              f"{t['2.0']['csi']:>9.4f} {t['4.0']['csi']:>9.4f} | "
              f"{t['2.0']['bias']:>10.4f} {t['2.0']['pod']:>9.4f} "
              f"{t['2.0']['far']:>9.4f}")
    print("-" * 96)

    csi2 = {n: r["thresholds"]["2.0"]["csi"] for n, r in reports.items()}
    anchor = csi2["k1"]
    anchor_ok = abs(anchor - FLOOR) <= 0.002
    print(f"ANCHOR k1 CSI@2.0 = {anchor:.4f} "
          f"({'PASS' if anchor_ok else 'FAIL -- TABLE VOID per D2 anchor'}, "
          f"floor {FLOOR:.4f}, delta {anchor - FLOOR:+.4f})")
    print(f"k2 reference (test-tuned scalar) = {K2_CSI:.4f} | "
          f"Oracle ceiling = {ORACLE:.4f}")
    print(f"pmm      vs k1 (D2-A point estimate) : {csi2['pmm'] - csi2['k1']:+.4f}")
    print(f"pmm      vs k2 (D2-B point estimate) : {csi2['pmm'] - csi2['k2']:+.4f}")
    print(f"pmm_rain vs k2 (D2-B, operational PMM): {csi2['pmm_rain'] - csi2['k2']:+.4f}")
    print(f"pmm_cond vs pmm (distribution-mismatch diagnostic): "
          f"{csi2['pmm_cond'] - csi2['pmm']:+.4f}")
    best_pmm = max(("pmm", "pmm_cond", "pmm_rain"), key=lambda n: csi2[n])
    print(f"best PMM variant = {best_pmm} (CSI@2.0 = {csi2[best_pmm]:.4f}); "
          f"decisive test is the PAIRED BOOTSTRAP CI, not the point value")
    if csi2["pmm"] <= FLOOR:
        print("D2-C: FAIL -- PMM does not even reach the pre-registered floor.")
    print("=" * 96)

    out = {
        "meta": {
            "positioning": "amplitude/distribution calibration family; "
                           "D2-B paired bootstrap is the decisive test",
            "quantile_split": "train",
            "test_split": "test",
            "patch_size": args.patch_size,
            "slide_stride": args.slide_stride,
            "thresholds": list(TIERS),
            "floor": FLOOR,
            "k2_reference_csi2": K2_CSI,
            "oracle": ORACLE,
            "anchor_k1_csi2": anchor,
            "anchor_pass": bool(anchor_ok),
            "cr_index": cr_index,
        },
        "quantile_tables": {
            "n_pixels_all": tables["n_pixels_all"],
            "n_pixels_cond": tables["n_pixels_cond"],
            "n_pixels_rain": tables["n_pixels_rain"],
            "q_grid_size": tables["q_grid_size"],
            "y_q_p50_p90_p99_p999": tables["y_q_p50_p90_p99_p999"],
            "y_q_rain_p50_p90_p99_p999": tables["y_q_rain_p50_p90_p99_p999"],
            "lookup_size_full": len(tables["full"][0]),
            "lookup_size_cond": len(tables["cond"][0]),
            "lookup_size_rain": len(tables["rain"][0]),
        },
        "reports": reports,
    }
    if args.out_json:
        Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out_json).write_text(
            json.dumps(out, indent=2, default=float), encoding="utf-8"
        )
        print(f"report written to {args.out_json}")

    if args.per_window_prefix:
        prefix = Path(args.per_window_prefix)
        prefix.parent.mkdir(parents=True, exist_ok=True)
        for name, arm in arms.items():
            table = arm["verifier"].per_window_table()
            path = Path(f"{prefix}_{name}.windows.json")
            path.write_text(
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
            print(f"per-window table written to {path} ({len(table)} windows)")
    return 0 if anchor_ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
