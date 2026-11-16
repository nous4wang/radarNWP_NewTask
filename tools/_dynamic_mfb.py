#!/usr/bin/env python3
"""E27 / MASTER 4.29 -- per-window dynamic mean-field bias scalar (oracle).

For every forecast window, estimate ONE scalar from the eleven history
ground-rainfall frames and the analytical radar-derived rain at the same
instants, then scale the current-time radar-derived rain with it and hold it
constant for ten lead frames (zero advection, identical to the persistence
baseline except for the scalar).

Three information lags: 0 / 2 / 4 frames (0 / 12 / 24 min).  Two fixed-k
reference arms (k=1.0 pre-registered floor, k=2.0 test-tuned scalar) are
scored in the SAME pass so the comparison is paired on identical windows.

IMPORTANT (E27 B1-0): the history ground rainfall is TRUTH OBSERVATION.
This is an ideal upper-bound diagnostic, not a deployable method, and its
scores must NOT be compared against the 0.0924 floor as if it were a model.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from radar_nwp_bench.data import (  # noqa: E402
    NWPAlignment, RadarNWPDataset, build_dataloader,
)
from radar_nwp_bench.metrics import PatchStitcher  # noqa: E402
from tools._case_zr_bias import Counter, TIERS, THRESHOLDS  # noqa: E402

ARM_NAMES = ("k1", "k2", "lag0", "lag2", "lag4")
LAGS = {"lag0": 0, "lag2": 2, "lag4": 4}


def zr_to_rain(dbz: torch.Tensor) -> torch.Tensor:
    linear_z = torch.pow(10.0, dbz.clamp_min(0.0) / 10.0)
    return torch.pow(linear_z / 200.0, 1.0 / 1.6) * 0.1


def write_windows(rows: list[dict], path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps({
        "n_windows": len(rows),
        "thresholds": list(TIERS),
        "windows": rows,
    }, indent=2), encoding="utf-8")


def row_from_counts(pred: np.ndarray, truth: np.ndarray) -> dict:
    row = {}
    for thr, key in zip(THRESHOLDS, TIERS):
        p, t = pred >= thr, truth >= thr
        row[key] = {
            "hits": float(np.count_nonzero(p & t)),
            "misses": float(np.count_nonzero(~p & t)),
            "false_alarms": float(np.count_nonzero(p & ~t)),
            "correct_negatives": float(np.count_nonzero(~p & ~t)),
        }
    return row


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--patch-size", type=int, default=301)
    ap.add_argument("--slide-stride", type=int, default=301)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--out-json", default="runs/dynamic_mfb.json")
    ap.add_argument("--windows-prefix", default="runs/mfb")
    args = ap.parse_args()

    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = RadarNWPDataset(
        args.data_root, packed=True, in_len=11, out_len=10,
        patch_size=args.patch_size, alignment=NWPAlignment.FORECAST,
        split=args.split, slide_stride=args.slide_stride,
        random_crop=False, tolerate_one_missing_frame=True,
    )
    loader = build_dataloader(
        dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, drop_last=False, seed=0,
    )
    grid_h, grid_w = dataset.grid_shape
    cr_index = list(dataset.radar_vars).index("CR")
    n_pos = dataset.n_patch_positions
    stitchers = {a: (PatchStitcher((grid_h, grid_w), n_frames=10,
                                   n_positions=n_pos) if n_pos > 1 else None)
                 for a in ARM_NAMES}
    counters = {a: Counter() for a in ARM_NAMES}
    rows = {a: [] for a in ARM_NAMES}
    k_stats = {a: [] for a in LAGS}

    def absorb(name: str, s_i: int, pred: np.ndarray, truth: np.ndarray) -> None:
        counters[name].update(pred, truth)
        rows[name].append(row_from_counts(pred, truth))

    with torch.no_grad():
        for bi, batch in enumerate(loader):
            x_radar = batch["x_radar"].to(device)
            y_hist = batch["y_hist"].to(device)          # (B,11,1,H,W), mm/6min
            y_true = batch["y_rain"].to(device)          # (B,10,1,H,W)
            case_idx = batch["case_index"].cpu().numpy()
            pos_idx = batch["position_index"].cpu().numpy()
            origin = batch["patch_origin"].cpu().numpy()

            dbz_all = torch.expm1(
                x_radar[:, :, cr_index].clamp_min(0.0) * 6.0
            ).clamp(0.0, 75.0)                          # (B,11,H,W)
            zr_all = zr_to_rain(dbz_all)
            now = zr_all[:, -1]                         # (B,H,W)
            obs = y_hist[:, :, 0]                       # (B,11,H,W)
            t_np = y_true.cpu().numpy()

            preds: dict[str, np.ndarray] = {
                "k1": (now * 1.0).cpu().numpy(),
                "k2": (now * 2.0).cpu().numpy(),
            }
            for name, lag in LAGS.items():
                end = obs.shape[1] - lag
                num = obs[:, :end].sum(dim=(1, 2, 3))
                den = zr_all[:, :end].sum(dim=(1, 2, 3)).clamp_min(1e-6)
                k = (num / den).clamp(0.05, 20.0)       # (B,) one scalar / window
                k_stats[name].extend(k.cpu().numpy().tolist())
                preds[name] = (now * k[:, None, None]).cpu().numpy()

            for name in ARM_NAMES:
                p = preds[name]
                exp = p[:, None].repeat(y_true.shape[1], axis=1)  # (B,10,H,W)
                st = stitchers[name]
                for b in range(exp.shape[0]):
                    s_i = int(case_idx[b])
                    if st is None:
                        absorb(name, s_i, exp[b], t_np[b, :, 0])
                        continue
                    st.add(s_i, int(pos_idx[b]),
                           (int(origin[b][0]), int(origin[b][1])),
                           exp[b], t_np[b, :, 0])
                    if st.is_complete(s_i):
                        sp, stt = st.finalize(s_i)
                        absorb(name, s_i, sp, stt)
                        st.release(s_i)
            if (bi + 1) % 50 == 0:
                print(f"  batch {bi + 1}", flush=True)

    reports = {a: counters[a].report() for a in ARM_NAMES}
    result = {
        "meta": {
            "experiment": "E27 / MASTER 4.29",
            "criteria_timestamp": "2026-11-06 21:40",
            "device": str(device),
            "split": args.split,
            "thresholds": list(TIERS),
            "note": "history ground rainfall is TRUTH -> ideal upper bound, "
                    "not comparable with the persistence floor as a method",
        },
        "arms": reports,
        "k_hat_stats": {
            name: {
                "n": len(v), "min": float(np.min(v)), "max": float(np.max(v)),
                "median": float(np.median(v)), "mean": float(np.mean(v)),
                "p10": float(np.percentile(v, 10)),
                "p90": float(np.percentile(v, 90)),
            } for name, v in k_stats.items() if v
        },
        "n_windows": {a: len(rows[a]) for a in ARM_NAMES},
        "pending": {a: (len(stitchers[a].pending) if stitchers[a] else 0)
                    for a in ARM_NAMES},
    }
    Path(args.out_json).write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    for a in ARM_NAMES:
        write_windows(rows[a], f"{args.windows_prefix}_{a}.windows.json")

    print("=" * 92)
    print(f"{'arm':>7} | {'CSI@0.1':>9} {'CSI@0.5':>9} {'CSI@2.0':>9} "
          f"{'CSI@4.0':>9} | {'BIAS@0.5':>9} {'BIAS@2.0':>9}")
    print("-" * 92)
    for a in ARM_NAMES:
        r = reports[a]
        print(f"{a:>7} | " + " ".join(f"{r[t]['csi']:>9.4f}" for t in TIERS)
              + f" | {r['0.5']['bias']:>9.4f} {r['2.0']['bias']:>9.4f}")
    for name, d in result["k_hat_stats"].items():
        print(f"  k_hat {name}: median={d['median']:.4f} "
              f"p10={d['p10']:.4f} p90={d['p90']:.4f} "
              f"range=[{d['min']:.4f},{d['max']:.4f}]")
    print(f"-> {args.out_json}")
    return 0


if __name__ == "__main__":
    main()
